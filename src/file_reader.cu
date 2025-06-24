
#include "file_reader.h"

#include "util/helper_common.h"
#include "util/cuda_helper_device.h"
#include "util/cuda_helper_host.h"

#include <json.hpp>
using json = nlohmann::json;

#include <iostream>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

template <bool PROJECTION_LEFT_HANDED, bool PROJECTION_REVERSED_Z, bool PROJECTION_01_DEPTH_RANGE>
__global__ void convertDepthToZ_kernel(uint2 resolution,
                                       cudaSurfaceObject_t buffer)
{
    const uint2 idx2D = {threadIdx.x + blockIdx.x * blockDim.x, threadIdx.y + blockIdx.y * blockDim.y};
    if (idx2D.x >= resolution.x || idx2D.y >= resolution.y)
        return;

    float d = surf2Dread<float>(buffer, idx2D);

    if constexpr (PROJECTION_LEFT_HANDED)
        d = -d;

    if constexpr (PROJECTION_REVERSED_Z)
        d = 1.0f - d;

    if constexpr (PROJECTION_01_DEPTH_RANGE)
        d = 2.0f * d - 1.0f;

    surf2Dwrite(d, buffer, idx2D);
}

namespace glm
{
    void from_json(const json &j, mat4 &m)
    {
        m[0][0] = j.at("e00").get<float>();
        m[0][1] = j.at("e10").get<float>();
        m[0][2] = j.at("e20").get<float>();
        m[0][3] = j.at("e30").get<float>();

        m[1][0] = j.at("e01").get<float>();
        m[1][1] = j.at("e11").get<float>();
        m[1][2] = j.at("e21").get<float>();
        m[1][3] = j.at("e31").get<float>();

        m[2][0] = j.at("e02").get<float>();
        m[2][1] = j.at("e12").get<float>();
        m[2][2] = j.at("e22").get<float>();
        m[2][3] = j.at("e32").get<float>();

        m[3][0] = j.at("e03").get<float>();
        m[3][1] = j.at("e13").get<float>();
        m[3][2] = j.at("e23").get<float>();
        m[3][3] = j.at("e33").get<float>();
    }
}

void from_json(const json &j, FrameFileReader::FrameInfo &f)
{
    j.at("idx").get_to(f.idx);
    f.w2c = j["w2c"].get<glm::mat4>();
    f.proj = j["proj"].get<glm::mat4>();
}

void from_json(const json &j, uint2 &r)
{
    j.at("x").get_to(r.x);
    j.at("y").get_to(r.y);
}

bool FrameFileReader::initCamInfo(fs::path input_path, FrameFileReader::CameraInfo& cam_info)
{
    fs::path cam_json_file = input_path / "cam.json";
    if (!fs::exists(cam_json_file))
    {
        std::cout << "Failed to open camera file: " << cam_json_file << std::endl;
        return false;
    }

    std::ifstream f(cam_json_file);

    json cam_data = json::parse(f);
    cam_info.frames = cam_data.at("frames").get<std::vector<FrameFileReader::FrameInfo>>();
    if (cam_info.frames.size() == 0)
    {
        std::cout << "Could not load any frames from camera file: " << cam_json_file << std::endl;
        return false;
    }

    cam_info.resolution = cam_data.at("resolution").get<uint2>();
    cam_info.clientFps = cam_data.value<int>("clientFps", 60);
    cam_info.serverFps = cam_data.value<int>("serverFps", 10);
    cam_info.shadowFps = cam_data.value<int>("shadowFps", 60);

    bool object_motion = cam_data.value<bool>("objectMotion", true);
    bool shadow_gradients = cam_data.value<bool>("shadowGradients", true);
    cam_info.priority = object_motion * 2 + shadow_gradients * 1;

    return true;
}

bool FrameFileReader::init(fs::path input_path, const std::vector<std::filesystem::path>& aux_cam_dirs, FrameData& frame)
{
    if (!initCamInfo(input_path / main_cam_dir, _cam_main_info))
        return false;

    std::vector<uint2> aux_res;
    std::vector<int> aux_priorities;
    
    for (auto aux_cam_dir : aux_cam_dirs)
    {
        CameraInfo aux_cam_info;
        if (!initCamInfo(input_path / aux_cam_dir, aux_cam_info))
            return false;

        _cam_aux_info.push_back(aux_cam_info);
        aux_res.emplace_back(aux_cam_info.resolution);
        aux_priorities.emplace_back(aux_cam_info.priority);
    }

    this->aux_cam_dirs = aux_cam_dirs;

    _input_path = input_path;

    frame.resize(_cam_main_info.resolution, aux_res, aux_priorities);
    readColorDepth(frame, 0);

    return true;
}

void FrameFileReader::readColorDepthCam(ZBufferImage& frame_image, const CameraInfo& cam_info, const std::filesystem::path& cam_path, int frame_idx)
{
    frame_image.view_mat = cam_info.frames[frame_idx].w2c;
    frame_image.proj_mat = normalizeProjMat(cam_info.frames[frame_idx].proj);
    frame_image.viewproj_mat = frame_image.proj_mat * frame_image.view_mat;
    frame_image.inv_viewproj_mat = glm::inverse(frame_image.viewproj_mat);

    int img_idx = cam_info.frames[frame_idx].idx;
    std::stringstream ss;
    ss << std::setw(4) << std::setfill('0') << img_idx;
    std::string img_idx_s = ss.str();

    frame_image.img.readFromFile(cam_path / color_dir / (img_idx_s + ".png"));
    frame_image.z_buffer.readFromFile(cam_path / depth_dir / (img_idx_s + ".exr"));

    const uint BLOCK_SIZE_2D = 16;
    dim3 block_dim(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 grid_dim_2d = toDim3(divRoundUp(cam_info.resolution, BLOCK_SIZE_2D));
    convertDepthToZ_kernel<projection_left_handed, projection_reversed_z, projection_01_depth_range><<<grid_dim_2d, block_dim>>>(cam_info.resolution, frame_image.z_buffer.surf());
}

void FrameFileReader::readColorDepth(FrameData& frame, int frame_idx)
{
    readColorDepthCam(*(frame.bwd), _cam_main_info, _input_path / main_cam_dir, frame_idx);
    for (int i = 0; i < frame.aux_cam_images.size(); i++)
        readColorDepthCam(*(frame.aux_cam_images[i].next), _cam_aux_info[i], _input_path / aux_cam_dirs[i], frame_idx);
    CUDA_SYNC_CHECK_THROW();
}

bool FrameFileReader::readNextFrame(FrameData& frame)
{
    frame.swap();

    if (_frame_index >= _cam_main_info.frames.size() - 1)
        return false;

    readColorDepth(frame, _frame_index + 1);

    int img_idx = _cam_main_info.frames[_frame_index].idx;
    std::stringstream ss;
    ss << std::setw(4) << std::setfill('0') << img_idx;
    std::string img_idx_s = ss.str();

    auto readIfDirExists = [&]<typename T, int N>(std::optional<ImageBuffer<T, N>>& opt_buffer, fs::path cam_dir, fs::path subdir, std::string file_extension)
    {
        fs::path folder_path = _input_path / cam_dir / subdir;
        if (std::filesystem::exists(folder_path))
            opt_buffer.value().readFromFile(folder_path / (img_idx_s + "." + file_extension));
        else
            opt_buffer = std::nullopt;
    };
    auto readShadowImgs = [&](std::unique_ptr<MotionZBufferImage>& buffer, fs::path cam_dir, bool forward)
    {
        readIfDirExists(buffer->shadowColorA_img, cam_dir, forward ? shadowColorA_f_dir : shadowColorA_b_dir, "png");
        readIfDirExists(buffer->shadowColorB_img, cam_dir, forward ? shadowColorB_f_dir : shadowColorB_b_dir, "png");
        readIfDirExists(buffer->shadowInfo_img, cam_dir, forward ? shadowInfo_f_dir : shadowInfo_b_dir, "png");
    };
    
    // Main flow fields
    readIfDirExists(frame.fwd->object_motion, main_cam_dir, fwd_ff_dir, "exr");
    readIfDirExists(frame.bwd->object_motion, main_cam_dir, bwd_ff_dir, "exr");

    // Main shadow info
    readShadowImgs(frame.fwd, main_cam_dir, true);
    readShadowImgs(frame.bwd, main_cam_dir, false);

    for (int i = 0; i < frame.aux_cam_images.size(); i++)
    {
        // Aux flow fields
        readIfDirExists(frame.aux_cam_images[i].curr->object_motion, aux_cam_dirs[i], fwd_ff_dir, "exr");
        readIfDirExists(frame.aux_cam_images[i].next->object_motion, aux_cam_dirs[i], bwd_ff_dir, "exr");
        
        // Aux shadow Info
        readShadowImgs(frame.aux_cam_images[i].curr, aux_cam_dirs[i], true);
        readShadowImgs(frame.aux_cam_images[i].next, aux_cam_dirs[i], false);
    }

    _frame_index++;
    return true;
}

glm::mat4 FrameFileReader::normalizeProjMat(glm::mat4 proj)
{
    if constexpr (projection_left_handed)
        proj = proj * projection_swap_left_right_handed;
    if constexpr (projection_reversed_z)
        proj = projection_reverse_z * proj;
    if constexpr (projection_01_depth_range)
        proj = projection_01_depth_to_11 * proj;

    proj = projection_unity_2022_3 * proj;
    return proj;
}