
#pragma once

#include <cuda_runtime.h>

#include "helper_math_ext.h"

inline __device__ __host__ bool inRange(int2 idx2D, uint2 dims)
{
    return idx2D.x >= 0 && idx2D.x < dims.x && idx2D.y >= 0 && idx2D.y < dims.y;
}

inline __device__ __host__ uint to1D(uint2 idx2D, uint2 dims)
{
    return idx2D.y * dims.x + idx2D.x;
}

// ------------------------------------------------------------------
// COLOR
// ------------------------------------------------------------------

inline __device__ __host__ float4 lerp_color(uchar4 a, uchar4 b, float t)
{
    return lerp(make_float4(a), make_float4(b), t) / 255.f;
}

inline __device__ __host__ unsigned char to_8bit(const float f)
{
    return min(255, max(0, int(f * 255.f)));
}
inline __device__ __host__ uchar4 color_to_uchar4(const float4 color)
{
    uchar4 data;
    data.x = to_8bit(color.x);
    data.y = to_8bit(color.y);
    data.z = to_8bit(color.z);
    data.w = to_8bit(color.w);
    return data;
}
inline __device__ __host__ uchar4 color_to_uchar4(const float3 color)
{
    return color_to_uchar4(make_float4(color, 1.f));
}
inline __device__ __host__ uint color_to_uint(const float3 color)
{
    return (to_8bit(color.x) << 0) +
      (to_8bit(color.y) << 8) +
      (to_8bit(color.z) << 16) +
      (255U << 24);
}
inline __device__ __host__ float3 uchar4_to_color(const uchar4 color)
{
    float3 data;
    data.x = (float)color.x / 255.f;
    data.y = (float)color.y / 255.f;
    data.z = (float)color.z / 255.f;

    return data;
}


// ------------------------------------------------------------------
// TRANSFORMATIONS
// ------------------------------------------------------------------

inline __device__ __host__ float2 pixel2CS(uint2 pixel, uint2 resolution)
{
    return 2.0f * (make_float2(pixel) + 0.5f) / make_float2(resolution) - 1.0f;
}

inline __device__ __host__ int2 CS2pixel(float2 posCS, uint2 resolution)
{
    // Pixel centers are at .5 pixels, but flooring gives pixel index
    return make_int2(floorf((posCS + 1.0f) * make_float2(resolution) * 0.5f));
}

inline __device__ __host__ glm::vec4 CS2WS(glm::vec4 posCS, glm::mat4 inv_viewproj_mat)
{
    glm::vec4 posWS = inv_viewproj_mat * posCS;
    return posWS / posWS.w;
}

inline __device__ __host__ glm::vec4 WS2CS(glm::vec4 posWS, glm::mat4 viewproj_mat)
{
    glm::vec4 posCS = viewproj_mat * posWS;
    return posCS / posCS.w;
}

inline __device__ __host__ float z_to_depth(float z)
{
    return (z + 1.0f) * 0.5f;
}
inline __device__ __host__ float depth_to_z(float depth)
{
    return depth * 2.0f - 1.0f;
}