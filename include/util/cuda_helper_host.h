

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)

            // for rethrowing the CUDA error add this to catch:
            // throw std::runtime_error(std::string("CUDA | " FILE_LINE " " #x " failed with error: ") + cudaGetErrorString(result));     \

#define CUDA_CHECK_THROW(x)                                                                                                            \
    do                                                                                                                                 \
    {                                                                                                                                  \
        cudaError_t result = x;                                                                                                        \
        try {                                                                                                                          \
            if (result != cudaSuccess)                                                                                                 \
                throw std::runtime_error(std::string("CUDA | " FILE_LINE " " #x " failed with error: ") + cudaGetErrorString(result)); \
        } catch (std::exception& ex) {                                                                                                 \
            std::cout << ex.what() << std::endl;                                                                                       \
        }                                                                                                                              \
    } while (0)

#define CUDA_SYNC_CHECK_THROW()                                                                                                             \
    do                                                                                                                                      \
    {                                                                                                                                       \
        cudaStreamSynchronize(cudaStreamDefault);                                                                                           \
        cudaError_t result = cudaGetLastError();                                                                                            \
        try {                                                                                                                               \
            if (result != cudaSuccess)                                                                                                      \
                throw std::runtime_error(std::string("CUDA | " FILE_LINE " Synchronize failed with error: ") + cudaGetErrorString(result)); \
        } catch (std::exception& ex) {                                                                                                      \
            std::cout << ex.what() << std::endl;                                                                                            \
        }                                                                                                                                   \
    } while (0)

#define CUDA_SYNC_CHECK_THROW_ASYNC(s)                                                                                                      \
    do                                                                                                                                      \
    {                                                                                                                                       \
        cudaStreamSynchronize(s);                                                                                                           \
        cudaError_t result = cudaGetLastError();                                                                                            \
        try {                                                                                                                               \
            if (result != cudaSuccess)                                                                                                      \
                throw std::runtime_error(std::string("CUDA | " FILE_LINE " Synchronize failed with error: ") + cudaGetErrorString(result)); \
        } catch (std::exception& ex) {                                                                                                      \
            std::cout << ex.what() << std::endl;                                                                                            \
        }                                                                                                                                   \
    } while (0)

inline void cudaPrintMemInfo()
{
    size_t free_byte, total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);

    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;

    std::cout << "GPU memory usage: used = " << used_db / 1024.0 / 1024.0 << " MB"
              << ", free = " << free_db / 1024.0 / 1024.0 << " MB"
              << ", total = " << total_db / 1024.0 / 1024.0 << " MB" << std::endl;
}