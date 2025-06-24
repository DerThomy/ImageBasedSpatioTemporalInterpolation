
#include "util/image_buffer.h"

#include "util/cuda_helper_device.h"
#include "util/cuda_helper_host.h"
#include "util/helper_math_ext.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_STB_ZLIB 1
#define TINYEXR_USE_THREAD 1
#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>

#include <fpng.h>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <filesystem>
namespace fs = std::filesystem;

template <typename T>
__global__ void sumFloatImage_kernel(cudaTextureObject_t tex, const uint2 resolution, T* accum_val)
{    
    const uint2 idx2D = {threadIdx.x + blockIdx.x * blockDim.x, threadIdx.y + blockIdx.y * blockDim.y};
    if (idx2D.x >= resolution.x || idx2D.y >= resolution.y)
        return;

    T val = tex2D<T>(tex, idx2D);
    atomicAdd(accum_val, val);
}

template <typename T, int N_CHANNELS>
void ImageBuffer<T, N_CHANNELS>::resize(uint2 resolution)
{
    if (resolution.x == _resolution.x && resolution.y == _resolution.y)
        return;
    else if (_resolution.x != 0 || _resolution.y != 0)
        throw std::runtime_error("Resizing of ImageBuffer currently not supported");

    _resolution = resolution;

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();
    CUDA_CHECK_THROW(cudaMallocArray(&(this->_array), &desc, resolution.x, resolution.y, cudaArraySurfaceLoadStore));

    init();
}

template <typename T, int N_CHANNELS>
void ImageBuffer<T, N_CHANNELS>::init(cudaTextureAddressMode address_mode)
{
    _surface = createSurface();
    _texture = createTexture(cudaFilterModePoint, address_mode);
}

template <typename T, int N_CHANNELS>
void ImageBuffer<T, N_CHANNELS>::memsetBuffer(T val)
{
    std::vector<T> tmp(_resolution.x * _resolution.y, val);
    upload(tmp.data());
}

template <typename T, int N_CHANNELS>
void ImageBuffer<T, N_CHANNELS>::download(T* h_buffer)
{
    CUDA_CHECK_THROW(cudaMemcpy2DFromArray(h_buffer, _resolution.x * sizeof(T), _array, 0, 0, _resolution.x * sizeof(T), _resolution.y, cudaMemcpyDeviceToHost));
}

template <typename T, int N_CHANNELS>
void ImageBuffer<T, N_CHANNELS>::upload(T* h_buffer)
{
    CUDA_CHECK_THROW(cudaMemcpy2DToArray(_array, 0, 0, h_buffer, _resolution.x * sizeof(T), _resolution.x * sizeof(T), _resolution.y, cudaMemcpyHostToDevice));
}

template <typename T, int N_CHANNELS>
T ImageBuffer<T, N_CHANNELS>::average()
{
    T tmp = T();
    if constexpr (!std::is_same_v<T, float>) {
        return tmp;
    }

    float* d_accum_val;
    cudaMalloc(&d_accum_val, sizeof(float));
    cudaMemset(d_accum_val, 0, sizeof(float));

    const uint32_t BLOCK_SIZE_2D = 16;
    dim3 block_dim_2d(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 grid_dim_2d = toDim3(divRoundUp(_resolution, BLOCK_SIZE_2D));
    sumFloatImage_kernel<<<grid_dim_2d, block_dim_2d>>>(_texture, _resolution, d_accum_val);

    float h_accum_val;
    cudaMemcpy(&h_accum_val, d_accum_val, sizeof(float), cudaMemcpyDeviceToHost);
    h_accum_val /= (_resolution.x * _resolution.y);
    return (T) h_accum_val;
}

template <typename T, int N_CHANNELS>
void ImageBuffer<T, N_CHANNELS>::readFromFile(fs::path filename)
{
    if (filename.extension() == ".exr")
    {
        float* h_img_tmp = nullptr;

        int2 image_resolution;
        const char* err = nullptr;
        int ret = LoadEXR(&h_img_tmp, &image_resolution.x, &image_resolution.y, filename.string().c_str(), &err);
        if (ret != TINYEXR_SUCCESS) 
        {
            std::stringstream ss;
            ss << "Error loading EXR file " << filename;
            if (err)
            {
                ss << ": " << err << std::endl;
                FreeEXRErrorMessage(err);
            }
            throw std::runtime_error(ss.str());
        }
        else
        {
            if (image_resolution.x != _resolution.x || image_resolution.y != _resolution.y)
                throw std::runtime_error("Image resolution has to be set in advance: " + filename.string());
            
            upload((T*) h_img_tmp);
            free(h_img_tmp);
        }
    }
    else
    {
        std::vector<uint8_t> out;

        int2 image_resolution;
        uchar4* h_img = nullptr;
        
        if (filename.extension() == ".png")
        {
            uint2 image_resolution_uint;
            uint32_t comp = 0;
            int success = fpng::fpng_decode_file(filename.string().c_str(), out, image_resolution_uint.x, image_resolution_uint.y, comp, 4);

            if (success == fpng::FPNG_DECODE_NOT_FPNG)
            {
                int comp = 0;
                h_img = (uchar4*) stbi_load(filename.string().c_str(), &image_resolution.x, &image_resolution.y, &comp, 4);

                bool overwrite_pngs_with_fpng_encoding = false;
                if (overwrite_pngs_with_fpng_encoding && h_img != nullptr)
                    fpng::fpng_encode_image_to_file(filename.string().c_str(), (T*) h_img, image_resolution.x, image_resolution.y, 4, fpng::FPNG_ENCODE_SLOWER);
            }
            else if (success == fpng::FPNG_DECODE_SUCCESS)
            {
                h_img = (uchar4*) out.data();
                image_resolution.x = image_resolution_uint.x;
                image_resolution.y = image_resolution_uint.y;
            }
        }
        else
        {
            int comp = 0;
            h_img = (uchar4*) stbi_load(filename.string().c_str(), &image_resolution.x, &image_resolution.y, &comp, 4);
        }

        if (h_img == nullptr)
            throw std::runtime_error("Could not load image " + filename.string());
            
        if (image_resolution.x != _resolution.x || image_resolution.y != _resolution.y)
            throw std::runtime_error("Image resolution has to be set in advance: " + filename.string());

        upload((T*) h_img);
        if (out.size() == 0)
            free(h_img);
    }
}

template <typename T, int N_CHANNELS>
void ImageBuffer<T, N_CHANNELS>::writeImageToFile(std::vector<T>& image, int w, int h, int channels, fs::path filename)
{
    if (filename.extension() == ".png")
    {
        fpng::fpng_encode_image_to_file(filename.string().c_str(), image.data(), w, h, channels);
    }
    else if (filename.extension() == ".bmp")
    {
        stbi_write_bmp(filename.string().c_str(), w, h, channels, image.data());
    }
    else if (filename.extension() == ".jpg")
    {
        stbi_write_jpg(filename.string().c_str(), w, h, channels, image.data(), 100);
    }
    else if (filename.extension() == ".exr")
    {
        const char* err = nullptr;
        int ret = SaveEXR((float*) image.data(), w, h, channels, 0, filename.string().c_str(), &err);

        if (ret != TINYEXR_SUCCESS) 
        {
            std::stringstream ss;
            ss << "Error writing EXR file " << filename;
            if (err)
            {
                ss << ": " << err << std::endl;
                FreeEXRErrorMessage(err);
            }
            throw std::runtime_error(ss.str());
        }
    }
    else
    {
        throw std::runtime_error("Image file extension not supported!");
    }
}

template <typename T, int N_CHANNELS>
void ImageBuffer<T, N_CHANNELS>::writeToFile(std::filesystem::path filename)
{
    std::vector<T> image(_resolution.x * _resolution.y);
    download(image.data());

    writeImageToFile(image, _resolution.x, _resolution.y, N_CHANNELS, filename);
}

template <typename T, int N_CHANNELS>
cudaTextureObject_t ImageBuffer<T, N_CHANNELS>::createTexture(cudaTextureFilterMode filter_mode, cudaTextureAddressMode address_mode)
{
    cudaResourceDesc tex_res_desc;
    memset(&tex_res_desc, 0, sizeof(cudaResourceDesc));
    tex_res_desc.resType = cudaResourceTypeArray;
    tex_res_desc.res.array.array = _array;

    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(cudaTextureDesc));

    tex_desc.normalizedCoords = false;
    tex_desc.filterMode = filter_mode;
    tex_desc.addressMode[0] = address_mode;
    tex_desc.addressMode[1] = address_mode;
    tex_desc.addressMode[2] = address_mode;
    tex_desc.readMode = cudaReadModeElementType;

    cudaTextureObject_t texture;
    CUDA_CHECK_THROW(cudaCreateTextureObject(&texture, &tex_res_desc, &tex_desc, NULL));

    return texture;
}

template <typename T, int N_CHANNELS>
cudaSurfaceObject_t ImageBuffer<T, N_CHANNELS>::createSurface()
{
    cudaResourceDesc surf_res_desc;
    memset(&surf_res_desc, 0, sizeof(cudaResourceDesc));
    surf_res_desc.resType = cudaResourceTypeArray;
    surf_res_desc.res.array.array = _array;

    cudaSurfaceObject_t surface;
    CUDA_CHECK_THROW(cudaCreateSurfaceObject(&surface, &surf_res_desc));

    return surface;
}

template struct ImageBuffer<uchar4, 4>;
template struct ImageBuffer<float, 1>;
template struct ImageBuffer<float4, 4>;