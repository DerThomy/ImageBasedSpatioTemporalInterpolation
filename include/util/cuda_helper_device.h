
#pragma once

#include <cuda_runtime.h>

template <typename T>
inline __device__ T tex2D(cudaTextureObject_t tex, uint2 idx2D)
{
    return tex2D<T>(tex, idx2D.x, idx2D.y);
}

template <typename T>
inline __device__ T surf2Dread(cudaSurfaceObject_t surf, uint2 idx2D)
{
    return surf2Dread<T>(surf, idx2D.x * sizeof(T), idx2D.y);
}

template <typename T>
inline __device__ void surf2Dwrite(T val, cudaSurfaceObject_t surf, uint2 idx2D)
{
    surf2Dwrite(val, surf, idx2D.x * sizeof(T), idx2D.y);
}