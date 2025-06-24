
#pragma once

#include "helper_math.h"


inline __host__ __device__ uchar4 make_uchar4(unsigned char s)
{
    return make_uchar4(s, s, s, s);
}
inline __host__ __device__ uchar4 make_uchar4(uchar3 a)
{
    return make_uchar4(a.x, a.y, a.z, 0);
}
inline __host__ __device__ uchar4 make_uchar4(uchar3 a, unsigned char w)
{
    return make_uchar4(a.x, a.y, a.z, w);
}
inline __host__ __device__ uchar4 make_uchar4(float4 a)
{
    return make_uchar4((unsigned char)(a.x), (unsigned char)(a.y), (unsigned char)(a.z), (unsigned char)(a.w));
}

inline __host__ __device__ float2 make_float2(uchar2 a)
{
    return make_float2(float(a.x), float(a.y));
}
inline __host__ __device__ float3 make_float3(uchar3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}
inline __host__ __device__ float4 make_float4(uchar4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

// ------------------------------------------------------------------
// dim3
// ------------------------------------------------------------------

inline __host__ __device__ dim3 toDim3(const uint x) { return dim3{x, x, x}; }
inline __host__ __device__ dim3 toDim3(const uint2 a) { return dim3{a.x, a.y, 1}; }
inline __host__ __device__ dim3 toDim3(const uint3 a) { return dim3{a.x, a.y, a.z}; }

inline __host__ __device__ uint divRoundUp(const uint a, const uint b)
{
    return (a + b - 1) / b;
}
inline __host__ __device__ uint3 divRoundUp(const uint3 a, const int b)
{
    return make_uint3(divRoundUp(a.x, b), divRoundUp(a.y, b), divRoundUp(a.z, b));
}
inline __host__ __device__ uint2 divRoundUp(const uint2 a, const int b)
{
    return make_uint2(divRoundUp(a.x, b), divRoundUp(a.y, b));
}