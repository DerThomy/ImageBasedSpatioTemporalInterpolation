
#pragma once

#include "data.h"
#include "util/image_buffer.h"

typedef unsigned long long int uint64_cu;

enum RenderMode
{
    OURS = 0,
    BACKWARD_BIDIRECTIONAL = 1,
    BACKWARD_UNIDIRECTIONAL = 2,
    SPLATTING = 3,
    TIMEWARP = 4
};

struct AdditionalRenderInfo
{
    int frames_per_anchor;
    int shadow_frames_per_anchor;
    bool no_shadows;
    RenderMode mode;
};

struct RenderInfo
{
    uint2 resolution;
    float alpha;

    glm::mat4 viewproj_mat;
    glm::mat4 inv_viewproj_mat;

    AdditionalRenderInfo additional_info;
};

struct CamReprojInfo
{
    const uint2 resolution;
    const bool forward;
    const int priority;
    cudaTextureObject_t shadowColorA_img;
    cudaTextureObject_t shadowColorB_img;
    cudaTextureObject_t shadowInfo_img;
    cudaTextureObject_t z_buffer;
    cudaTextureObject_t img;
    cudaTextureObject_t cam_ff;
    cudaTextureObject_t full_ff;
    const uint64_cu* splatting_buffer;
};

struct Renderer
{
    Renderer(){};

    void render(ImageBuffer<uchar4, 4> &render_buffer, const RenderInfo render_info, const FrameData &frame_data);

private:
    void resizeBuffers(const RenderInfo render_info, const FrameData &frame_data);

    template <bool INIT_FULL_MOTION_TO>
    void computeFlowFields_sortedSplatting(const float weight, const bool init_linear_flowfields, MotionZBufferImage* input, const glm::mat4 vp_mat_to, const glm::mat4 vp_mat_cam, const glm::mat4 vp_mat_from_main, ImageBuffer<float4, 4>& motion_to_ff, ImageBuffer<float4, 4>& motion_cam_ff, uint64_cu* splatting_buffer);

    ImageBuffer<float4, 4> fwd2bwd_ff;
    std::vector<ImageBuffer<float4, 4>> fwd2bwd_aux_ff;

    ImageBuffer<float4, 4> fwd2cam_ff;
    std::vector<ImageBuffer<float4, 4>> fwd2cam_aux_ff;

    ImageBuffer<float4, 4> fwd_momentum;

    ImageBuffer<float4, 4> bwd2fwd_ff;
    std::vector<ImageBuffer<float4, 4>> bwd2fwd_aux_ff;

    ImageBuffer<float4, 4> bwd2cam_ff;
    std::vector<ImageBuffer<float4, 4>> bwd2cam_aux_ff;

    ImageBuffer<float4, 4> bwd_momentum;

    struct AuxCamSplattingBuffers
    {
        uint64_cu* forward = nullptr;
        uint64_cu* backward = nullptr;
    };

    uint2 _curr_resolution;
    uint64_cu* d_fwd_main_splatting_buffer = nullptr;
    uint64_cu* d_bwd_main_splatting_buffer = nullptr;
    std::vector<AuxCamSplattingBuffers> d_aux_cam_splatting_buffers;

    CamReprojInfo* d_fwd_aux_cam_info = nullptr;
    CamReprojInfo* d_bwd_aux_cam_info = nullptr;
};