
#pragma once

#include "cuda_runtime.h"

#include <filesystem>
#include <vector>

template <typename T, int N_CHANNELS>
struct ImageBuffer
{
public:
    ImageBuffer() {}

    ImageBuffer(uint2 resolution)
    {
        resize(resolution);
    }

    void resize(uint2 resolution);
    void init(cudaTextureAddressMode address_mode = cudaAddressModeBorder);

    void memsetBuffer(T val);
    void download(T* h_buffer);
    void upload(T* h_buffer);

    void readFromFile(std::filesystem::path filename);
    
    void writeImageToFile(std::vector<T>& image, int w, int h, int channels, std::filesystem::path filename);
    void writeToFile(std::filesystem::path filename);
    
    cudaTextureObject_t tex() { return _texture; }
    cudaSurfaceObject_t surf() { return _surface; }    
    
    T average();

    uint2 resolution() { return _resolution; }

private:
    cudaTextureObject_t createTexture(cudaTextureFilterMode filter_mode, cudaTextureAddressMode address_mode = cudaAddressModeBorder);
    cudaSurfaceObject_t createSurface();

    uint2 _resolution = make_uint2(0U, 0U);
    
    cudaArray_t _array;

    cudaSurfaceObject_t _surface = -1U;
    cudaTextureObject_t _texture = -1U;
};