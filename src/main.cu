
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/matrix_interpolation.hpp>
#include <glm/gtx/quaternion.hpp>

#include <cuda_fp16.h>

#include <fpng.h>
#include <argparse/argparse.hpp>

#include "file_reader.h"
#include "renderer.h"
#include "util/cuda_helper_host.h"
#include "util/helper_math_ext.h"
#include "util/image_buffer.h"

#include <string>
#include <fstream>
#include <vector>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

glm::mat4 getIntermediateViewMatrix(glm::mat4& iv_t0, glm::mat4& iv_t1, float t, float spiral_t, float spiral_scale)
{
    return glm::interpolate(iv_t0, iv_t1, t);

    glm::mat4 v_t0 = glm::inverse(iv_t0);
    glm::mat4 v_t1 = glm::inverse(iv_t1);

    glm::vec4 pos_t0 = v_t0[3];
    glm::vec4 pos_t1 = v_t1[3];
    glm::mat3 rot_t0 = v_t0;
    glm::mat3 rot_t1 = v_t1;
    rot_t0[2] *= -1.0f;
    rot_t1[2] *= -1.0f;

    glm::vec3 diff = pos_t1 - pos_t0;
    glm::vec3 dir = glm::normalize(diff);
    glm::vec3 ref(0.0f, 1.0f, 0.0f);

    glm::vec3 right = glm::cross(dir, ref);
    glm::vec3 up = glm::cross(dir, right);

    glm::quat qrot_t0 = glm::quat_cast(rot_t0);
    glm::quat qrot_t1 = glm::quat_cast(rot_t1);
    glm::quat qrot_ti = lerp(qrot_t0, qrot_t1, t);

    glm::mat3 rot_ti = glm::mat3_cast(qrot_t0);
    rot_ti[2] *= -1.0f; 
    glm::vec3 pos_ti = glm::vec3(pos_t0) + diff * t + spiral_scale * ((sin(spiral_t * 2.0f * glm::pi<float>()) * up + cos(spiral_t * 2.0f * glm::pi<float>()) * right));

    glm::mat4 result = rot_ti;
    result[3][0] = pos_ti[0];
    result[3][1] = pos_ti[1];
    result[3][2] = pos_ti[2];
    return glm::inverse(result);
}

int main(int argc, char **argv)
{
    fpng::fpng_init();

    argparse::ArgumentParser program("Splinter - Split-Rendering Frame Interpolation");

    program.add_argument("input-dir").help("path to the input directory");
    program.add_argument("output-dir").help("path to the output directory");
    program.add_argument("-c", "--camera-path").help("input camera path");

    program.add_argument("-w", "--write-images").help("write images to output directory").default_value(false).implicit_value(true);
    program.add_argument("-r", "--resolution").help("rendering resolution").default_value(std::vector<int>{2560, 1440}).nargs(2).scan<'d', int>();
    program.add_argument("-a", "--aux-dirs").help("auxilliary cam directory names").default_value(std::vector<std::string>()).nargs(argparse::nargs_pattern::at_least_one);
    program.add_argument("-l", "--latency").help("delay of server frames by <l> client frames").default_value(0).scan<'d', int>();
    program.add_argument("-m", "--mode").help("rendering mode (for benchmarking; see RenderMode)").default_value(0).choices(0, 1, 2, 3, 4).scan<'d', int>();
    program.add_argument("-lh", "--latency-hold-last").help("hold the last frame when latency sets in").default_value(false).implicit_value(true);
    
    program.add_argument("-ns", "--no-shadows").help("manually disables dynamic shadows, even when information is present").default_value(false).implicit_value(true);
    program.add_argument("-hb", "--holes-black").help("manually disable hole filling; holes will be left black").default_value(false).implicit_value(true);

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error &err)
    {
        std::cout << "Error - Argument parsing failed!" << std::endl;
        std::cout << err.what() << std::endl;
        std::cout << program;
        return 1;
    }

    fs::path input_dir = program.get<std::string>("input-dir");
    if (!fs::exists(input_dir))
    {
        std::cout << "Input directory invalid" << std::endl;
        return 1;
    }

    fs::path output_dir = program.get<std::string>("output-dir");
    if (!fs::exists(output_dir) && !fs::create_directories(output_dir))
    {
        std::cout << "Output directory could not be created" << std::endl;
        return 1;
    }

    
    auto aux_cam_arg = program.get<std::vector<std::string>>("-a");
    std::vector<std::filesystem::path> aux_cam_dirs(aux_cam_arg.size());
    std::transform(aux_cam_arg.begin(), aux_cam_arg.end(), aux_cam_dirs.begin(), [](const std::string& val) { return fs::path(val); });
    
    FrameData frame_data(aux_cam_dirs.size());

    FrameFileReader file_reader;
    file_reader.init(input_dir, aux_cam_dirs, frame_data);

    int latency = program.get<int>("--latency");
    bool latency_hold_last = program.get<bool>("--latency-hold-last");
    bool write_images = program.get<bool>("--write-images");
    std::vector<int> arg_resolution = program.get<std::vector<int>>("--resolution");
    uint2 render_resolution = make_uint2(arg_resolution[0], arg_resolution[1]);

    bool no_shadows = program.get<bool>("no-shadows");
    bool holes_black = program.get<bool>("holes-black");
    RenderMode render_mode = static_cast<RenderMode>(program.get<int>("mode"));

    bool use_input_cam = false;
    FrameFileReader::CameraInfo input_camera_info;
    if (program.is_used("camera-path"))
    {
        use_input_cam = true;
        std::filesystem::path input_camera_path = program.get<std::string>("camera-path");
        file_reader.initCamInfo(input_camera_path, input_camera_info);
    }
    ImageBuffer<uchar4, 4> d_img_final(render_resolution);
    Renderer renderer;

    int fixed_clientFps = 120;
    int fixed_render_duration_seconds = 15;
    int total_n_frames = use_input_cam ? input_camera_info.frames.size() : fixed_clientFps * fixed_render_duration_seconds;
    int clientFps = use_input_cam ? input_camera_info.clientFps : fixed_clientFps;

    int frames_per_anchor = clientFps / file_reader._cam_main_info.serverFps;
    int shadow_frames_per_anchor = file_reader._cam_main_info.shadowFps / file_reader._cam_main_info.serverFps;

    // Iterate over i frames
    int loop_idx = 0;
    for (int i = 0; i < total_n_frames; i += frames_per_anchor, loop_idx++)
    {
        // Render b frames
        for (int b = 0; b < frames_per_anchor; b++)
        {
            if (latency == b)
                file_reader.readNextFrame(frame_data);
                
            float t = float(latency > b ? frames_per_anchor + b : b) / float(frames_per_anchor);
            glm::mat4 view_mat_t = use_input_cam 
                ? input_camera_info.frames[i + b].w2c 
                : getIntermediateViewMatrix(frame_data.fwd->view_mat, frame_data.bwd->view_mat, t, (i + b) / float(fixed_clientFps), 0.0f);
            glm::mat4 viewproj_mat_t = (use_input_cam ? file_reader.normalizeProjMat(input_camera_info.frames[i+b].proj) : frame_data.fwd->proj_mat) * view_mat_t;

            int frame_number = i + b + 1;
            if (frame_number > total_n_frames)
                break;
                
            std::cout << "Frame: " << std::setfill(' ') << std::setw(3) << frame_number
                      << " / "  << std::setfill(' ') << std::setw(3) << total_n_frames << "\r";
            std::flush(std::cout);

            if (holes_black)
                d_img_final.memsetBuffer(make_uchar4(0, 0, 0, 0));

            if (!(b < latency && latency_hold_last)) // hold last actual frame when latency kicks in (do not render)
            {
                RenderInfo render_info{ render_resolution, t, viewproj_mat_t, glm::inverse(viewproj_mat_t), {frames_per_anchor, shadow_frames_per_anchor, no_shadows, render_mode}};
                renderer.render(d_img_final, render_info, frame_data);
            }

            if (write_images)
            {
                std::stringstream ss2;
                ss2 << std::setfill('0') << std::setw(4) << (frame_number - 1);
                std::string str_frame_number = ss2.str();
                d_img_final.writeToFile(output_dir / (str_frame_number + ".png"));
            }
            
            if (latency > b && b == (frames_per_anchor - 1)) // fallback 
                file_reader.readNextFrame(frame_data);
        }
    }
    std::cout << std::endl;

    return 0;
}