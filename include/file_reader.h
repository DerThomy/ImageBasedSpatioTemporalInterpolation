
#pragma once

#include "data.h"

#include <filesystem>
#include <vector>

struct FrameFileReader
{
    struct FrameInfo
    {
        int idx;
        glm::mat4 w2c;
        glm::mat4 proj;
    };

    struct CameraInfo
    {
        std::vector<FrameInfo> frames;
        uint2 resolution;
        int clientFps = 60;
        int serverFps = 10;
        int shadowFps = 60;
        int priority = 0;
    };

    bool init(std::filesystem::path input_path, const std::vector<std::filesystem::path>& aux_cam_dirs, FrameData& frame);
    bool initCamInfo(std::filesystem::path input_path, CameraInfo& cam_info);
    bool readNextFrame(FrameData& frame);

    void readColorDepth(FrameData& frame, int frame_idx);
    void readColorDepthCam(ZBufferImage& frame_image, const CameraInfo& cam_info, const std::filesystem::path& cam_path, int frame_idx);

    glm::mat4 normalizeProjMat(glm::mat4 proj);

    CameraInfo _cam_main_info;
    std::vector<CameraInfo> _cam_aux_info;
    int _frame_index = 0;

    std::filesystem::path _input_path;

    const std::filesystem::path main_cam_dir = "main";
    std::vector<std::filesystem::path> aux_cam_dirs;

    const std::filesystem::path color_dir = "color";
    const std::filesystem::path depth_dir = "depth";
    const std::filesystem::path fwd_ff_dir = "forwardFlowField";
    const std::filesystem::path bwd_ff_dir = "backwardFlowField";
    
    const std::filesystem::path shadowColorA_f_dir = "forwardShadowColorA";
    const std::filesystem::path shadowColorB_f_dir = "forwardShadowColorB";
    const std::filesystem::path shadowInfo_f_dir = "forwardShadowInfo";
    const std::filesystem::path shadowColorA_b_dir = "backwardShadowColorA";
    const std::filesystem::path shadowColorB_b_dir = "backwardShadowColorB";
    const std::filesystem::path shadowInfo_b_dir = "backwardShadowInfo";

    const std::filesystem::path cam_filename = "cam.json";

    static constexpr bool projection_reversed_z = true;
    static constexpr bool projection_01_depth_range = true;
    static constexpr bool projection_left_handed = false;

    const glm::mat4 projection_unity_2022_3             = glm::mat4(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,  0.0f, 1.0f);
    const glm::mat4 projection_reverse_z                = glm::mat4(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f,  1.0f, 1.0f);
    const glm::mat4 projection_01_depth_to_11           = glm::mat4(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,  2.0f, 0.0f, 0.0f, 0.0f, -1.0f, 1.0f);
    const glm::mat4 projection_swap_left_right_handed   = glm::mat4(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f,  0.0f, 1.0f);
};