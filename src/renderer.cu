#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "renderer.h"

#include "util/cuda_helper_device.h"
#include "util/cuda_helper_host.h"
#include "util/helper_common.h"

// #define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/matrix_transform.hpp>
// #include <glm/gtx/string_cast.hpp>

#include <optional>
#include <tuple>

constexpr float EPS_1_FULL_RES = 0.002f;
constexpr float EPS_1_LOW_RES = EPS_1_FULL_RES * 4.0f;
constexpr float EPS_2_FULL_RES = 0.001f;
constexpr float EPS_2_LOW_RES = 0.004f;
constexpr float EPS_3 = 0.01f;

constexpr int SPLATTING_BUFFER_INIT_VAL = 0xfe; // not 0xff so depth is not "-nan"

constexpr int2 DEBUG_PX = {1665, 928};
constexpr bool DEBUG = false;

constexpr uint SOFTSHADOW_BITS = 2;
constexpr uint SOFTSHADOW_VAL_MASK = (1U << SOFTSHADOW_BITS) - 1U;

struct BidirReprojInfo
{
    CamReprojInfo main_cam_info;
    CamReprojInfo* aux_cam_infos;
    int num_of_aux_cams;
};

struct TimewarpInfo
{
    bool isset;
    glm::mat4 inv_viewproj_mat_from;
    glm::mat4 inv_viewproj_mat_to;
    glm::mat4 viewproj_mat_to;
    float depth_to;

    uint2 resolution_to;
    cudaTextureObject_t img_to;
};

struct ReprojResult
{
    bool valid = false;
    bool visible_in_main = false;
    int priority = 0;
    float error;
    float depth;
    uchar4 color;
};

struct SplattingKeyInfo
{
    float depth;
    uint2 source_pixel;
};

inline __device__ uint64_cu encodeSplattingKey(SplattingKeyInfo info)
{
    return (uint64_cu)(*(uint32_t *)(&info.depth)) << 32 | info.source_pixel.x << 16 | info.source_pixel.y;
}

inline __device__ SplattingKeyInfo decodeSplattingKey(uint64_cu key)
{
    uint32_t depth_int = (uint32_t)(key >> 32);
    return {*((float *)&depth_int), make_uint2((uint32_t)key >> 16, (uint32_t)(key & 0x0000FFFFULL))};
}

template <bool INIT_FULL_MOTION_TO = true, bool OBJECT_MOTION = true>
__global__ void computeFlowFields_sortedSplatting_kernel(uint2 resolution,
                                                         const float weight,
                                                         const bool init_linear_flowfields, 
                                                         cudaTextureObject_t z_buffer_tex,
                                                         cudaTextureObject_t object_motion_tex,
                                                         const glm::mat4 inv_viewproj_mat_from,
                                                         const glm::mat4 viewproj_mat_to,
                                                         const glm::mat4 viewproj_mat_cam,
                                                         const glm::mat4 viewproj_mat_from_main,
                                                         cudaSurfaceObject_t full_motion_to_surf,
                                                         cudaSurfaceObject_t full_motion_cam_surf,
                                                         uint64_cu *splatting_buffer_cam)
{
    const uint2 idx2D = {threadIdx.x + blockIdx.x * blockDim.x, threadIdx.y + blockIdx.y * blockDim.y};
    if (idx2D.x >= resolution.x || idx2D.y >= resolution.y)
        return;

    float z_from = tex2D<float>(z_buffer_tex, idx2D);
    float4 object_motion = OBJECT_MOTION ? tex2D<float4>(object_motion_tex, idx2D) : make_float4(0.f, 0.f, 0.f, 0.f);

    const float2 pixelCS = pixel2CS(idx2D, resolution);
    const glm::vec4 posCS_from(pixelCS.x, pixelCS.y, z_from, 1.0f);
    const glm::vec4 posWS_from = CS2WS(posCS_from, inv_viewproj_mat_from);

    const glm::vec4 posCS_from_moved = posCS_from + glm::vec4(object_motion.x, object_motion.y, object_motion.z, 0.0f);
    const glm::vec4 posWS_to = CS2WS(posCS_from_moved, inv_viewproj_mat_from);

    const glm::vec4 posCS_from_main = WS2CS(posWS_from, viewproj_mat_from_main);
    bool visible_in_main = posCS_from_main.x > -(1.0f - 1e-3f) && posCS_from_main.x < (1.0f - 1e-3f) &&
                           posCS_from_main.y > -(1.0f - 1e-3f) && posCS_from_main.y < (1.0f - 1e-3f) &&
                           posCS_from_main.z > -(1.0f - 1e-3f) && posCS_from_main.z < (1.0f - 1e-3f);

    const glm::vec4 posWS_scaled = posWS_from + weight * (posWS_to - posWS_from);
    const glm::vec4 posCS_to = WS2CS(posWS_to, viewproj_mat_to);
    const glm::vec4 full_motion_to = posCS_to - posCS_from;

    if constexpr (INIT_FULL_MOTION_TO)
    {
        surf2Dwrite(make_float4(full_motion_to.x, full_motion_to.y, full_motion_to.z, visible_in_main ? 1.0f : 0.0f), full_motion_to_surf, idx2D);
    }

    float depth = z_to_depth(posCS_from.z);
    uint2 pix = idx2D;

    if (!init_linear_flowfields)
    {
        const glm::vec4 posCS_cam = WS2CS(posWS_scaled, viewproj_mat_cam);
        const glm::vec4 full_motion_cam = posCS_cam - posCS_from;
        surf2Dwrite(make_float4(full_motion_cam.x, full_motion_cam.y, full_motion_cam.z, visible_in_main ? 1.0f : 0.0f), full_motion_cam_surf, idx2D);

        const float depth_cam = z_to_depth(posCS_cam.z);
        const int2 pixel_cam = CS2pixel(make_float2(posCS_cam.x, posCS_cam.y), resolution);
        if (!inRange(pixel_cam, resolution) || depth_cam < 0.0f)
            return;
        
        // if (DEBUG && idx2D.x == DEBUG_PX.x && idx2D.y == DEBUG_PX.y)
        //     printf("CREATE %d %d - %f -> %f %f %f (%f %f %f)\n", idx2D.x, idx2D.y, weight, full_motion_cam.x, full_motion_cam.y, full_motion_cam.z, object_motion.x, object_motion.y, object_motion.z);

        depth = depth_cam;
        pix = make_uint2(pixel_cam);
    }
    else
    {
        const glm::vec4 full_motion_weighted = full_motion_to * weight;
        surf2Dwrite(make_float4(full_motion_weighted.x, full_motion_weighted.y, full_motion_weighted.z, visible_in_main ? 1.0f : 0.0f), full_motion_cam_surf, idx2D);
    }

    uint64_cu splatting_key = encodeSplattingKey({depth, idx2D});
    atomicMin(&(splatting_buffer_cam[to1D(make_uint2(pix.x, pix.y), resolution)]), splatting_key);
}

template <int WINDOW_SIZE>
inline __device__ SplattingKeyInfo initializeSearchFromSplatting(const uint2 idx2D,
                                                                 const uint2 resolution,
                                                                 const uint64_cu *splatting_buffer)
{
    SplattingKeyInfo lowest_depth_info = decodeSplattingKey(splatting_buffer[to1D(idx2D, resolution)]);

#pragma unroll
    for (int y_offset = -WINDOW_SIZE; y_offset <= WINDOW_SIZE; y_offset++)
    {
#pragma unroll
        for (int x_offset = -WINDOW_SIZE; x_offset <= WINDOW_SIZE; x_offset++)
        {
            int2 new_idx2D = make_int2(idx2D) + make_int2(x_offset, y_offset);
            if (!inRange(new_idx2D, resolution))
                continue;

            uint2 new_idx2D_uint = make_uint2(new_idx2D);
            SplattingKeyInfo current_info = decodeSplattingKey(splatting_buffer[to1D(new_idx2D_uint, resolution)]);
            if (current_info.depth > 0.0f && (!isfinite(lowest_depth_info.depth) || lowest_depth_info.depth < 0.0f || current_info.depth < lowest_depth_info.depth))
            {
                lowest_depth_info = current_info;
            }
        }
    }

    return lowest_depth_info;
}

template <int SEARCH_ITERATIONS>
__device__ uint2 iterativeSearch(uint2 search_start,
                                 uint2 idx2D,
                                 uint2 resolution,
                                 uint2 low_resolution,
                                 cudaTextureObject_t ff,
                                 const bool forward)
{
    uint2 result = search_start;

#pragma unroll
    for (int i = 0; i < SEARCH_ITERATIONS; i++)
    {
        float4 full_motion = tex2D<float4>(ff, result);
        int2 tmp_result = CS2pixel(pixel2CS(idx2D, resolution) - make_float2(full_motion.x, full_motion.y), low_resolution);
        if (!inRange(tmp_result, low_resolution))
            continue;

        if (DEBUG && idx2D.x == DEBUG_PX.x && idx2D.y == DEBUG_PX.y)
            printf("(fwd? %d) %d %d - %d: %d %d - %f %f\n", (int) forward, idx2D.x, idx2D.y, i, result.x, result.y, full_motion.x, full_motion.y);

        result = make_uint2(tmp_result);
    }

    return result;
}

inline __device__ float computeReprojError(const uint2 pixel_from,
                                           const uint2 resolution_from,
                                           const float2 pixelCS_cam,
                                           const float4 motion)
{
    float2 pixelCS_from = pixel2CS(pixel_from, resolution_from);
    float2 error = pixelCS_from + make_float2(motion.x, motion.y) - pixelCS_cam;

    // scale by resolution(EPS is in the smaller resolution dimension)
    error *= make_float2(resolution_from) / min(resolution_from.x, resolution_from.y);    
    return length(error);
}

__device__ void getMinMaxShadowValues(uint softshadow_gradient, uint shadow_frames_per_anchor, int2& minmax_vals)
{
    minmax_vals = { SOFTSHADOW_VAL_MASK, 0 };
    for (int i = 0; i < shadow_frames_per_anchor; i++)
    {
        minmax_vals.x = min(minmax_vals.x, softshadow_gradient >> (SOFTSHADOW_BITS * i) & SOFTSHADOW_VAL_MASK);
        minmax_vals.y = max(minmax_vals.y, softshadow_gradient >> (SOFTSHADOW_BITS * i) & SOFTSHADOW_VAL_MASK);
    }
}

inline __device__ int frameIdx2Idx(bool forward, int frame_idx, int frames_per_anchor)
{
    return forward ? frame_idx : (frames_per_anchor - frame_idx);
}

inline __device__ uchar4 colorFromShadow(uint shadow_val_idx,
                                         const int shadow_frames_per_anchor,
                                         const CamReprojInfo &reproj_info,
                                         const uint2 pixel,
                                         const uint2 idx2D)
{
    uchar4 main_color = tex2D<uchar4>(reproj_info.img, pixel);
    if (shadow_val_idx < shadow_frames_per_anchor && shadow_val_idx > 0)
    {
        uchar4 shadow_info_uchar4 = tex2D<uchar4>(reproj_info.shadowInfo_img, pixel);
        uint shadow_info_bitfield = (uint(shadow_info_uchar4.x) << 16) | (uint(shadow_info_uchar4.y) << 8) | uint(shadow_info_uchar4.z);

        if (reproj_info.shadowColorB_img != (cudaTextureObject_t) -1)
        {
            if (shadow_info_bitfield != 0)
            {
                int init_val = shadow_info_bitfield & SOFTSHADOW_VAL_MASK;
                int my_val = shadow_info_bitfield >> (SOFTSHADOW_BITS * shadow_val_idx) & SOFTSHADOW_VAL_MASK;
                
                int2 minmax_vals;
                getMinMaxShadowValues(shadow_info_bitfield, shadow_frames_per_anchor, minmax_vals);
                int min_val = minmax_vals.x;
                int max_val = minmax_vals.y;

                float t = (my_val - min_val) / float(max_val - min_val);
                uchar4 min_shadow_color = min_val == init_val ? main_color : tex2D<uchar4>(reproj_info.shadowColorA_img, pixel);
                uchar4 max_shadow_color = max_val == init_val ? main_color : tex2D<uchar4>(reproj_info.shadowColorB_img, pixel);

                float4 interpol_color = make_float4(min_shadow_color) * (1.0 - t) + make_float4(max_shadow_color) * t;
                
                return make_uchar4(interpol_color.x, interpol_color.y, interpol_color.z, 255U);
            }
        }
    }

    return main_color;
}

inline __device__ uchar4 colorFromShadow(const uint2 shadow_frame_idcs,
                                         const float shadow_frame_t,
                                         const AdditionalRenderInfo settings,
                                         const CamReprojInfo &reproj_info,
                                         const uint2 pixel,
                                         const uint2 idx2D)
{
    if (settings.no_shadows || shadow_frame_idcs.x >= settings.shadow_frames_per_anchor || shadow_frame_idcs.y == 0 || reproj_info.shadowColorA_img == (cudaTextureObject_t)-1)
        return tex2D<uchar4>(reproj_info.img, pixel);

    uint2 shadow_val_idcs = {
        frameIdx2Idx(reproj_info.forward, shadow_frame_idcs.x, settings.shadow_frames_per_anchor),
        frameIdx2Idx(reproj_info.forward, shadow_frame_idcs.y, settings.shadow_frames_per_anchor),
    };

    if ((shadow_val_idcs.x == shadow_val_idcs.y) || shadow_val_idcs.y >= settings.shadow_frames_per_anchor)
        return colorFromShadow(shadow_val_idcs.x, settings.shadow_frames_per_anchor, reproj_info, pixel, idx2D);
    if (shadow_val_idcs.x >= settings.shadow_frames_per_anchor)
        return colorFromShadow(shadow_val_idcs.y, settings.shadow_frames_per_anchor, reproj_info, pixel, idx2D);
        
    uchar4 shadow_color_a = colorFromShadow(shadow_val_idcs.x, settings.shadow_frames_per_anchor, reproj_info, pixel, idx2D);    
    uchar4 shadow_color_b = colorFromShadow(shadow_val_idcs.y, settings.shadow_frames_per_anchor, reproj_info, pixel, idx2D);
    
    float4 interpol_shadow_color = lerp_color(shadow_color_a, shadow_color_b, shadow_frame_t);
    return color_to_uchar4(interpol_shadow_color);
}

template <float EPS_2>
__device__ uchar4 secondChanceAndBlend_kernel(const float weight,
                                              const uint frame_idx,
                                              const uint2 shadow_frame_idcs,
                                              const float shadow_frame_t,
                                              const SplattingKeyInfo splatting_info,
                                              const AdditionalRenderInfo settings,
                                              const CamReprojInfo& info_from,
                                              const CamReprojInfo& info_to,
                                              const uint2 idx2D)
{
    const float4 motion_from2to = tex2D<float4>(info_from.full_ff, splatting_info.source_pixel);
    const float3 sourceCS = make_float3(pixel2CS(splatting_info.source_pixel, info_from.resolution), tex2D<float>(info_from.z_buffer, splatting_info.source_pixel));
    const float3 projCS = sourceCS + make_float3(motion_from2to);
    const int2 projPixel = CS2pixel(make_float2(projCS), info_to.resolution);

    uchar4 out_color = colorFromShadow(shadow_frame_idcs, shadow_frame_t, settings, info_from, splatting_info.source_pixel, idx2D);
    if (frame_idx >= settings.frames_per_anchor)
        return out_color;

    if (inRange(projPixel, info_to.resolution))
    {
        float actual_projCS_z = tex2D<float>(info_to.z_buffer, make_uint2(projPixel));
        float3 destinationCS = make_float3(pixel2CS(make_uint2(projPixel), info_to.resolution), actual_projCS_z);

        float4 motion_to2from = tex2D<float4>(info_to.full_ff, make_uint2(projPixel));
        if (distance(make_float3(motion_from2to), make_float3(-motion_to2from)) < EPS_3 && fabs(actual_projCS_z - projCS.z) < EPS_2)
            out_color = color_to_uchar4(lerp_color(colorFromShadow(shadow_frame_idcs, shadow_frame_t, settings, info_to, make_uint2(projPixel), idx2D), out_color, weight));
    }
    return out_color;
}

template <int SEARCH_WINDOW_SIZE, int SEARCH_ITERATIONS, bool ALWAYS_VALID, float EPS_1, float EPS_2>
__device__ ReprojResult bidirectionalReprojection(const uint2 idx2D, const uint2 full_res_idx, const uint2 full_res, const float alpha, const float2 pixelCS, AdditionalRenderInfo settings, const CamReprojInfo fwd_info, const CamReprojInfo bwd_info)
{
    float fwd_e = FLT_MAX;
    bool fwd_visible_in_main = false;
    SplattingKeyInfo fwd_search_init = initializeSearchFromSplatting<SEARCH_WINDOW_SIZE>(idx2D, fwd_info.resolution, fwd_info.splatting_buffer);
    if (fwd_search_init.depth > 0.0f && alpha < 1.0f)
    {
        fwd_search_init.source_pixel = iterativeSearch<SEARCH_ITERATIONS>(fwd_search_init.source_pixel, full_res_idx, full_res, fwd_info.resolution, fwd_info.cam_ff, true);
        float4 fwd_motion = tex2D<float4>(fwd_info.cam_ff, fwd_search_init.source_pixel);
        fwd_search_init.depth = z_to_depth(tex2D<float>(fwd_info.z_buffer, fwd_search_init.source_pixel) + fwd_motion.z);
        fwd_e = computeReprojError(fwd_search_init.source_pixel, fwd_info.resolution, pixelCS, fwd_motion);
        fwd_visible_in_main = fwd_motion.w > 0.0f;
    }

    float bwd_e = FLT_MAX;
    bool bwd_visible_in_main = false;
    SplattingKeyInfo bwd_search_init = initializeSearchFromSplatting<SEARCH_WINDOW_SIZE>(idx2D, bwd_info.resolution, bwd_info.splatting_buffer);
    if (bwd_search_init.depth > 0.0f)
    {
        bwd_search_init.source_pixel = iterativeSearch<SEARCH_ITERATIONS>(bwd_search_init.source_pixel, full_res_idx, full_res, bwd_info.resolution, bwd_info.cam_ff, false);
        float4 bwd_motion = tex2D<float4>(bwd_info.cam_ff, bwd_search_init.source_pixel);
        bwd_search_init.depth = z_to_depth(tex2D<float>(bwd_info.z_buffer, bwd_search_init.source_pixel) + bwd_motion.z);
        bwd_e = computeReprojError(bwd_search_init.source_pixel, bwd_info.resolution, pixelCS, bwd_motion);
        bwd_visible_in_main = bwd_motion.w > 0.0f;
    }
    
    if (DEBUG && full_res_idx.x == DEBUG_PX.x && full_res_idx.y == DEBUG_PX.y)
        printf("BEFORE %f vs. %f (below %f), depth: %f vs %f\n", fwd_e, bwd_e, EPS_1, fwd_search_init.depth, bwd_search_init.depth);
        
    uint frame_idx = __float2uint_rn((float)settings.frames_per_anchor * alpha);
    uint2 shadow_frame_idcs = {
        __float2uint_rd((float)settings.shadow_frames_per_anchor * alpha + 1e-9f),
        __float2uint_ru((float)settings.shadow_frames_per_anchor * alpha - 1e-9f)
    };
    float shadow_lerp_t = settings.shadow_frames_per_anchor * alpha - shadow_frame_idcs.x;

    ReprojResult result;
    if (fwd_e < EPS_1 || bwd_e < EPS_1)
    {
        bool both_corresponding = fwd_e < EPS_1 && bwd_e < EPS_1;
        bool similar_depth = fabs(depth_to_z(fwd_search_init.depth) - depth_to_z(bwd_search_init.depth)) < EPS_2;

        bool choose_forward = (both_corresponding && !similar_depth && fwd_search_init.depth < bwd_search_init.depth) ||
            ((!both_corresponding || similar_depth) && fwd_e < bwd_e);
            
        if (DEBUG && full_res_idx.x == DEBUG_PX.x && full_res_idx.y == DEBUG_PX.y)
            printf("%f vs. %f (below %f): corresponding %d similar %d, depth: %f vs %f, choose fwd?: %d\n", fwd_e, bwd_e, EPS_1, (int) both_corresponding, (int) similar_depth, fwd_search_init.depth, bwd_search_init.depth, (int) choose_forward);
        
        if (choose_forward)
        {
            result = { true, fwd_visible_in_main || (both_corresponding && similar_depth && bwd_visible_in_main), fwd_info.priority, fwd_e, fwd_search_init.depth };
            result.color = secondChanceAndBlend_kernel<EPS_2>(1.0f - alpha, frame_idx, shadow_frame_idcs, shadow_lerp_t, fwd_search_init, settings, fwd_info, bwd_info, idx2D);
        }
        else
        {
            result = { true, bwd_visible_in_main || (both_corresponding && similar_depth && fwd_visible_in_main), bwd_info.priority, bwd_e, bwd_search_init.depth };
            result.color = secondChanceAndBlend_kernel<EPS_2>(alpha, frame_idx, shadow_frame_idcs, shadow_lerp_t, bwd_search_init, settings, bwd_info, fwd_info, idx2D);
        }
    }
    else if (ALWAYS_VALID)
    {
        result.valid = true;

        uchar4 color_fwd = secondChanceAndBlend_kernel<EPS_2>(1.0f - alpha, frame_idx, shadow_frame_idcs, shadow_lerp_t, fwd_search_init, settings, fwd_info, bwd_info, idx2D);
        uchar4 color_bwd = secondChanceAndBlend_kernel<EPS_2>(alpha, frame_idx, shadow_frame_idcs, shadow_lerp_t, bwd_search_init, settings, bwd_info, fwd_info, idx2D);
        result.color = color_to_uchar4(lerp_color(color_fwd, color_bwd, 0.5f));
    }
    return result;
}

template<int SEARCH_WINDOW_SIZE, int SEARCH_ITERATIONS, bool MAIN_ALWAYS_VALID>
__global__ void bidirectionalReprojection_kernel(
    const uint2 resolution,
    const float alpha,
    const AdditionalRenderInfo settings,
    const BidirReprojInfo fwd_info,
    const BidirReprojInfo bwd_info,
    cudaSurfaceObject_t out_img)
{
    const uint2 idx2D = {threadIdx.x + blockIdx.x * blockDim.x, threadIdx.y + blockIdx.y * blockDim.y};
    if (idx2D.x >= resolution.x || idx2D.y >= resolution.y)
        return;

    const float2 pixelCS = pixel2CS(idx2D, resolution);

    ReprojResult main_reproj_result = bidirectionalReprojection<SEARCH_WINDOW_SIZE, SEARCH_ITERATIONS, MAIN_ALWAYS_VALID, EPS_1_FULL_RES, EPS_2_FULL_RES>(idx2D, idx2D, resolution, alpha, pixelCS, settings, fwd_info.main_cam_info, bwd_info.main_cam_info);
    if (main_reproj_result.valid)
    {
        surf2Dwrite(main_reproj_result.color, out_img, idx2D);
    }

    if (fwd_info.num_of_aux_cams <= 0)
        return;
        
    uint2 aux_res = fwd_info.aux_cam_infos[0].resolution;
    uint2 aux_idx = make_uint2(idx2D.x / (resolution.x / aux_res.x), idx2D.y / (resolution.y / aux_res.y));
    float2 aux_pixelCS = pixel2CS(aux_idx, aux_res);


    ReprojResult best_aux_reproj_result;
    for (int i = 0; i < fwd_info.num_of_aux_cams; i++)
    {
        uint2 aux_res = fwd_info.aux_cam_infos[i].resolution;
        uint2 aux_idx = make_uint2(idx2D.x / (resolution.x / aux_res.x), idx2D.y / (resolution.y / aux_res.y));
        float2 aux_pixelCS = pixel2CS(aux_idx, aux_res);

        ReprojResult aux_reproj_result = bidirectionalReprojection<SEARCH_WINDOW_SIZE, SEARCH_ITERATIONS, false, EPS_1_LOW_RES, EPS_2_LOW_RES>(aux_idx, idx2D, resolution, alpha, aux_pixelCS, settings, fwd_info.aux_cam_infos[i], bwd_info.aux_cam_infos[i]);
        if (!best_aux_reproj_result.valid)
        {
            best_aux_reproj_result = aux_reproj_result;
        }
        else if (aux_reproj_result.valid)
        {
            bool similar_depth = fabs(depth_to_z(best_aux_reproj_result.depth) - depth_to_z(aux_reproj_result.depth)) < EPS_2_LOW_RES;

            bool choose_new = (!similar_depth && aux_reproj_result.depth < best_aux_reproj_result.depth) || 
                              (similar_depth && aux_reproj_result.priority >= best_aux_reproj_result.priority && aux_reproj_result.error < best_aux_reproj_result.error);

            if (choose_new)
                best_aux_reproj_result = aux_reproj_result;
        }
    }

    if (best_aux_reproj_result.valid)
    {
        bool choose_aux = !main_reproj_result.valid;
        if (main_reproj_result.valid && !best_aux_reproj_result.visible_in_main) // currently not used. Leads to problems at depth discontinuities
        {
            bool similar_depth = fabs(depth_to_z(best_aux_reproj_result.depth) - depth_to_z(main_reproj_result.depth)) < EPS_2_LOW_RES;
            choose_aux = !similar_depth && best_aux_reproj_result.depth < main_reproj_result.depth;
        }

        if (choose_aux)
            surf2Dwrite(best_aux_reproj_result.color, out_img, idx2D);
    }
}

__global__ void timewarp_kernel(
    const uint2 resolution,
    const float alpha,
    const TimewarpInfo tw_info_fwd,
    const TimewarpInfo tw_info_fwd_aux,
    cudaSurfaceObject_t out_img)
{
    const uint2 idx2D = {threadIdx.x + blockIdx.x * blockDim.x, threadIdx.y + blockIdx.y * blockDim.y};
    if (idx2D.x >= resolution.x || idx2D.y >= resolution.y)
        return;

    float2 pixCS = pixel2CS(idx2D, resolution);
    glm::vec4 nearCS(pixCS.x, pixCS.y, 0.0f, 1.0f);
    glm::vec4 farCS(pixCS.x, pixCS.y, 1.0f, 1.0f);

    glm::vec4 nearWS = CS2WS(nearCS, tw_info_fwd.inv_viewproj_mat_from);
    glm::vec4 farWS  = CS2WS(farCS, tw_info_fwd.inv_viewproj_mat_from);

    glm::vec3 ray_origin = glm::vec3(nearWS);
    glm::vec3 ray_dir = glm::normalize(glm::vec3(farWS - nearWS));

    glm::vec4 centerCS_to(0.0f, 0.0f, tw_info_fwd.depth_to, 1.0f);
    glm::vec4 centerWS_to = CS2WS(centerCS_to, tw_info_fwd.inv_viewproj_mat_to);
    glm::vec4 campos_to = tw_info_fwd.inv_viewproj_mat_to[3];

    glm::vec3 viewdirWS_to = glm::normalize(centerWS_to - campos_to);

    glm::vec3 plane_point = campos_to;
    glm::vec3 plane_normal = viewdirWS_to;

    float denom = glm::dot(ray_dir, plane_normal);
    if (fabs(denom) < 1e-6f) // The ray is parallel to the plane
        return;
    
    float t = glm::dot(plane_point - ray_origin, plane_normal) / denom;
    glm::vec3 intersection = ray_origin + t * ray_dir;

    glm::vec4 projCS = WS2CS(glm::vec4(intersection, 1.0f), tw_info_fwd.viewproj_mat_to);
    int2 pixel = CS2pixel(make_float2(projCS.x, projCS.y), tw_info_fwd.resolution_to);

    uchar4 out_color;
    if (inRange(pixel, tw_info_fwd.resolution_to) || !tw_info_fwd_aux.isset)
    {
        uint2 pixel_clamped = make_uint2(clamp(pixel, make_int2(0, 0), make_int2(tw_info_fwd.resolution_to) - 1));
        out_color = tex2D<uchar4>(tw_info_fwd.img_to, pixel_clamped);
    }
    else
    {
        glm::vec4 projCS_aux = WS2CS(glm::vec4(intersection, 1.0f), tw_info_fwd_aux.viewproj_mat_to);
        int2 pixel_aux = CS2pixel(make_float2(projCS_aux.x, projCS_aux.y), tw_info_fwd_aux.resolution_to);

        uint2 pixel_clamped = make_uint2(clamp(pixel_aux, make_int2(0, 0), make_int2(tw_info_fwd_aux.resolution_to) - 1));
        out_color = tex2D<uchar4>(tw_info_fwd_aux.img_to, pixel_clamped);
    }
    surf2Dwrite(out_color, out_img, idx2D);
}

template <bool INIT_FULL_MOTION_TO>
void Renderer::computeFlowFields_sortedSplatting(const float weight, const bool init_linear_flowfields, MotionZBufferImage *input, const glm::mat4 vp_mat_to, const glm::mat4 vp_mat_cam, const glm::mat4 viewproj_mat_from_main, ImageBuffer<float4, 4>& motion_to_ff, ImageBuffer<float4, 4>& motion_cam_ff, uint64_cu* splatting_buffer)
{
    const uint BLOCK_SIZE_2D = 16;
    dim3 block_dim_2d(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 grid_dim_2d = toDim3(divRoundUp(input->dims, BLOCK_SIZE_2D));

#define CALL_FF_SPLAT(OM, OM_TEX) computeFlowFields_sortedSplatting_kernel<INIT_FULL_MOTION_TO, OM><<<grid_dim_2d, block_dim_2d>>>( \
    input->dims, weight, init_linear_flowfields, \
    input->z_buffer.tex(), OM_TEX, input->inv_viewproj_mat, \
    vp_mat_to, vp_mat_cam, viewproj_mat_from_main, \
    motion_to_ff.surf(), \
    motion_cam_ff.surf(), \
    splatting_buffer);

    if (input->object_motion.has_value())
    {
        CALL_FF_SPLAT(true, input->object_motion.value().tex());
    }
    else
    {
        CALL_FF_SPLAT(false, 0);
    }
}

void Renderer::resizeBuffers(const RenderInfo render_info, const FrameData &frame_data)
{
    fwd2bwd_ff.resize(frame_data.fwd->dims);
    fwd2cam_ff.resize(frame_data.fwd->dims);
    fwd_momentum.resize(frame_data.fwd->dims);

    bwd2fwd_ff.resize(frame_data.bwd->dims);
    bwd2cam_ff.resize(frame_data.bwd->dims);
    bwd_momentum.resize(frame_data.bwd->dims);

    fwd2bwd_aux_ff.resize(frame_data.aux_cam_images.size());
    fwd2cam_aux_ff.resize(frame_data.aux_cam_images.size());
    bwd2fwd_aux_ff.resize(frame_data.aux_cam_images.size());
    bwd2cam_aux_ff.resize(frame_data.aux_cam_images.size());
    d_aux_cam_splatting_buffers.resize(frame_data.aux_cam_images.size());

    for (int i = 0; i < frame_data.aux_cam_images.size(); i++)
        fwd2bwd_aux_ff[i].resize(frame_data.aux_cam_images[i].curr->dims);
    for (int i = 0; i < frame_data.aux_cam_images.size(); i++)
        fwd2cam_aux_ff[i].resize(frame_data.aux_cam_images[i].curr->dims);
    for (int i = 0; i < frame_data.aux_cam_images.size(); i++)
        bwd2fwd_aux_ff[i].resize(frame_data.aux_cam_images[i].next->dims);
    for (int i = 0; i < frame_data.aux_cam_images.size(); i++)
        bwd2cam_aux_ff[i].resize(frame_data.aux_cam_images[i].next->dims);

    size_t render_img_size = render_info.resolution.x * render_info.resolution.y;
    if (_curr_resolution.x != render_info.resolution.x || _curr_resolution.y != render_info.resolution.y)
    {
        if (d_fwd_main_splatting_buffer != nullptr)
            CUDA_CHECK_THROW(cudaFree(d_fwd_main_splatting_buffer));
        if (d_bwd_main_splatting_buffer != nullptr)
            CUDA_CHECK_THROW(cudaFree(d_bwd_main_splatting_buffer));

        CUDA_CHECK_THROW(cudaMalloc(&d_fwd_main_splatting_buffer, sizeof(uint64_cu) * render_img_size));
        CUDA_CHECK_THROW(cudaMalloc(&d_bwd_main_splatting_buffer, sizeof(uint64_cu) * render_img_size));

        for (int i = 0; i < d_aux_cam_splatting_buffers.size(); i++)
        {
            if (d_aux_cam_splatting_buffers[i].forward != nullptr)
                CUDA_CHECK_THROW(cudaFree(d_aux_cam_splatting_buffers[i].forward));
            if (d_aux_cam_splatting_buffers[i].backward != nullptr)
                CUDA_CHECK_THROW(cudaFree(d_aux_cam_splatting_buffers[i].backward));

            CUDA_CHECK_THROW(cudaMalloc(&(d_aux_cam_splatting_buffers[i].forward), sizeof(uint64_cu) * frame_data.aux_cam_images[i].curr->dims.x * frame_data.aux_cam_images[i].curr->dims.y));
            CUDA_CHECK_THROW(cudaMalloc(&(d_aux_cam_splatting_buffers[i].backward), sizeof(uint64_cu) * frame_data.aux_cam_images[i].next->dims.x * frame_data.aux_cam_images[i].next->dims.y));
        }

        if (d_fwd_aux_cam_info != nullptr)
            CUDA_CHECK_THROW(cudaFree(d_fwd_aux_cam_info));
        if (d_bwd_aux_cam_info != nullptr)
            CUDA_CHECK_THROW(cudaFree(d_bwd_aux_cam_info));

        CUDA_CHECK_THROW(cudaMalloc(&d_fwd_aux_cam_info, sizeof(CamReprojInfo) * d_aux_cam_splatting_buffers.size()));
        CUDA_CHECK_THROW(cudaMalloc(&d_bwd_aux_cam_info, sizeof(CamReprojInfo) * d_aux_cam_splatting_buffers.size()));
    }
    _curr_resolution = render_info.resolution;
}

void Renderer::render(ImageBuffer<uchar4, 4> &render_buffer, const RenderInfo render_info, const FrameData &frame_data)
{
    const uint BLOCK_SIZE_2D = 16;
    dim3 block_dim_2d(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 grid_dim_2d = toDim3(divRoundUp(render_info.resolution, BLOCK_SIZE_2D));

    resizeBuffers(render_info, frame_data);

    if (render_info.additional_info.mode == RenderMode::TIMEWARP)
    {
        float avg_z =  frame_data.fwd->z_buffer.average();
        TimewarpInfo tw_info_fwd{true, render_info.inv_viewproj_mat, frame_data.fwd->inv_viewproj_mat, frame_data.fwd->viewproj_mat, avg_z, frame_data.fwd->dims, frame_data.fwd->img.tex()};
        TimewarpInfo tw_info_fwd_lowres_lfov{false};
        if (frame_data.aux_cam_images.size() > 0)
        {
            auto& aux_img = frame_data.aux_cam_images[0].curr;
            tw_info_fwd_lowres_lfov = {true, render_info.inv_viewproj_mat, aux_img->inv_viewproj_mat, aux_img->viewproj_mat, avg_z, aux_img->dims, aux_img->img.tex()};
        }
        
        timewarp_kernel<<<grid_dim_2d, block_dim_2d>>>(render_info.resolution, render_info.alpha, tw_info_fwd, tw_info_fwd_lowres_lfov, render_buffer.surf());
        CUDA_SYNC_CHECK_THROW();
        return;
    }

    size_t render_img_size = render_info.resolution.x * render_info.resolution.y;
    CUDA_CHECK_THROW(cudaMemset(d_fwd_main_splatting_buffer, SPLATTING_BUFFER_INIT_VAL, render_img_size * sizeof(uint64_cu)));
    CUDA_CHECK_THROW(cudaMemset(d_bwd_main_splatting_buffer, SPLATTING_BUFFER_INIT_VAL, render_img_size * sizeof(uint64_cu)));
    for (int i = 0; i < d_aux_cam_splatting_buffers.size(); i++)
    {
        CUDA_CHECK_THROW(cudaMemset(d_aux_cam_splatting_buffers[i].forward, SPLATTING_BUFFER_INIT_VAL, sizeof(uint64_cu) * frame_data.aux_cam_images[i].curr->dims.x * frame_data.aux_cam_images[i].curr->dims.y));
        CUDA_CHECK_THROW(cudaMemset(d_aux_cam_splatting_buffers[i].backward, SPLATTING_BUFFER_INIT_VAL, sizeof(uint64_cu) * frame_data.aux_cam_images[i].next->dims.x * frame_data.aux_cam_images[i].next->dims.y));
    }

    bool init_linear_flowfields = render_info.additional_info.mode == RenderMode::BACKWARD_BIDIRECTIONAL;

    // Splatting Main and Support Cam + FlowFields
    computeFlowFields_sortedSplatting<true>(render_info.alpha, init_linear_flowfields, frame_data.fwd.get(), frame_data.bwd->viewproj_mat, render_info.viewproj_mat, frame_data.fwd.get()->viewproj_mat, fwd2bwd_ff, fwd2cam_ff, d_fwd_main_splatting_buffer);
    computeFlowFields_sortedSplatting<true>(1.0f - render_info.alpha, init_linear_flowfields, frame_data.bwd.get(), frame_data.fwd->viewproj_mat, render_info.viewproj_mat, frame_data.bwd.get()->viewproj_mat, bwd2fwd_ff, bwd2cam_ff, d_bwd_main_splatting_buffer);
    for (int i = 0; i < frame_data.aux_cam_images.size(); i++)
    {
        computeFlowFields_sortedSplatting<true>(render_info.alpha, init_linear_flowfields, frame_data.aux_cam_images[i].curr.get(), frame_data.aux_cam_images[i].next->viewproj_mat, render_info.viewproj_mat, frame_data.fwd.get()->viewproj_mat, fwd2bwd_aux_ff[i], fwd2cam_aux_ff[i], d_aux_cam_splatting_buffers[i].forward);
        computeFlowFields_sortedSplatting<true>(1.0f - render_info.alpha, init_linear_flowfields, frame_data.aux_cam_images[i].next.get(), frame_data.aux_cam_images[i].curr->viewproj_mat, render_info.viewproj_mat, frame_data.bwd.get()->viewproj_mat, bwd2fwd_aux_ff[i], bwd2cam_aux_ff[i], d_aux_cam_splatting_buffers[i].backward);
    }
    CUDA_SYNC_CHECK_THROW();

    auto texOrDefault = []<typename T, int N>(std::optional<ImageBuffer<T, N>>& opt_buffer) { return opt_buffer.has_value() ? opt_buffer.value().tex() : (cudaTextureObject_t)-1; };

    CamReprojInfo fwd_main_cam_info{frame_data.fwd->dims, true, 0, texOrDefault(frame_data.fwd->shadowColorA_img), texOrDefault(frame_data.fwd->shadowColorB_img), texOrDefault(frame_data.fwd->shadowInfo_img), frame_data.fwd->z_buffer.tex(), frame_data.fwd->img.tex(), fwd2cam_ff.tex(), fwd2bwd_ff.tex(), d_fwd_main_splatting_buffer};
    std::vector<CamReprojInfo> fwd_aux_cam_info;
    for (int i = 0; i < frame_data.aux_cam_images.size(); i++)
    {
        const auto& img_buf = frame_data.aux_cam_images[i].curr;
        fwd_aux_cam_info.push_back(CamReprojInfo{img_buf->dims, true, frame_data.aux_cam_images[i].priority, texOrDefault(img_buf->shadowColorA_img), texOrDefault(img_buf->shadowColorB_img), texOrDefault(img_buf->shadowInfo_img), img_buf->z_buffer.tex(), img_buf->img.tex(), fwd2cam_aux_ff[i].tex(), fwd2bwd_aux_ff[i].tex(), d_aux_cam_splatting_buffers[i].forward});
    }

    CamReprojInfo bwd_main_cam_info{frame_data.bwd->dims, false, 0, texOrDefault(frame_data.bwd->shadowColorA_img), texOrDefault(frame_data.bwd->shadowColorB_img), texOrDefault(frame_data.bwd->shadowInfo_img), frame_data.bwd->z_buffer.tex(), frame_data.bwd->img.tex(), bwd2cam_ff.tex(), bwd2fwd_ff.tex(), d_bwd_main_splatting_buffer};
    std::vector<CamReprojInfo> bwd_aux_cam_info;
    for (int i = 0; i < frame_data.aux_cam_images.size(); i++)
    {
        const auto& img_buf = frame_data.aux_cam_images[i].next;
        bwd_aux_cam_info.push_back(CamReprojInfo{img_buf->dims, false, frame_data.aux_cam_images[i].priority, texOrDefault(img_buf->shadowColorA_img), texOrDefault(img_buf->shadowColorB_img), texOrDefault(img_buf->shadowInfo_img), img_buf->z_buffer.tex(), img_buf->img.tex(), bwd2cam_aux_ff[i].tex(), bwd2fwd_aux_ff[i].tex(), d_aux_cam_splatting_buffers[i].backward});
    }

    cudaMemcpy(d_fwd_aux_cam_info, fwd_aux_cam_info.data(), fwd_aux_cam_info.size() * sizeof(CamReprojInfo), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bwd_aux_cam_info, bwd_aux_cam_info.data(), bwd_aux_cam_info.size() * sizeof(CamReprojInfo), cudaMemcpyHostToDevice);

    BidirReprojInfo fwd_info{fwd_main_cam_info, d_fwd_aux_cam_info, static_cast<int>(frame_data.aux_cam_images.size())};
    BidirReprojInfo bwd_info{bwd_main_cam_info, d_bwd_aux_cam_info, static_cast<int>(frame_data.aux_cam_images.size())};

    #define CALL_BIDIR_KERNEL(SW, SI, MAV) bidirectionalReprojection_kernel<SW, SI, MAV><<<grid_dim_2d, block_dim_2d>>>(render_info.resolution, render_info.alpha, render_info.additional_info, fwd_info, bwd_info, render_buffer.surf());

    if (render_info.additional_info.mode == RenderMode::OURS || render_info.additional_info.mode == RenderMode::BACKWARD_UNIDIRECTIONAL)
    {
        CALL_BIDIR_KERNEL(1, 3, false);
    }
    else if (render_info.additional_info.mode == RenderMode::SPLATTING)
    {
        CALL_BIDIR_KERNEL(1, 0, false);
    }
    else if (render_info.additional_info.mode == RenderMode::BACKWARD_BIDIRECTIONAL)
    {
        CALL_BIDIR_KERNEL(3, 5, true);
    }
    else
    {
        std::cout << "Cannot happen" << std::endl;
    }
}