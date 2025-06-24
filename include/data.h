
#pragma once

#include "util/image_buffer.h"

#include <glm/glm.hpp>

#include <memory>
#include <vector>
#include <optional>

struct FrameImage
{
    ImageBuffer<uchar4, 4> img;
    std::optional<ImageBuffer<uchar4, 4>> shadowColorA_img = std::make_optional<ImageBuffer<uchar4, 4>>();
    std::optional<ImageBuffer<uchar4, 4>> shadowColorB_img = std::make_optional<ImageBuffer<uchar4, 4>>();
    std::optional<ImageBuffer<uchar4, 4>> shadowInfo_img = std::make_optional<ImageBuffer<uchar4, 4>>();

    uint2 dims;
    float t;

    glm::mat4 view_mat;
    glm::mat4 proj_mat;
    glm::mat4 viewproj_mat;
    glm::mat4 inv_viewproj_mat;

    void resize(uint2 dims)
    {
        if (shadowColorA_img.has_value()) shadowColorA_img.value().resize(dims);
        if (shadowColorB_img.has_value()) shadowColorB_img.value().resize(dims);
        if (shadowInfo_img.has_value()) shadowInfo_img.value().resize(dims);

        img.resize(dims);
        this->dims = dims;
    }
};

struct ZBufferImage : public FrameImage
{
    ImageBuffer<float, 1> z_buffer;

    void resize(uint2 dims)
    {
        z_buffer.resize(dims);
        FrameImage::resize(dims);
    }
};

struct MotionZBufferImage : public ZBufferImage
{
    std::optional<ImageBuffer<float4, 4>> object_motion = std::make_optional<ImageBuffer<float4, 4>>();

    void resize(uint2 dims)
    {
        if (object_motion.has_value())
        {
            object_motion.value().resize(dims);
            object_motion.value().memsetBuffer(make_float4(0.0f, 0.0f, 0.0f, 0.0f));
        }
        ZBufferImage::resize(dims);
    }
};

struct AuxCamData
{
    AuxCamData() : curr(std::make_unique<MotionZBufferImage>()),
                   next(std::make_unique<MotionZBufferImage>())
    {}

    void swap()
    {
        std::swap(curr, next);
    }

    std::unique_ptr<MotionZBufferImage> curr, next;
    int priority = 0;
};

struct FrameData
{
    FrameData(uint8_t number_of_aux_cams)
                : fwd(std::make_unique<MotionZBufferImage>())
                , bwd(std::make_unique<MotionZBufferImage>())
                , aux_cam_images(number_of_aux_cams)
    {}
    
    std::unique_ptr<MotionZBufferImage> fwd;
    std::unique_ptr<MotionZBufferImage> bwd;

    std::vector<AuxCamData> aux_cam_images;

    std::vector<ZBufferImage> depth_imgs;

    void resize(uint2 main_dims, const std::vector<uint2>& aux_res, const std::vector<int>& aux_priorities)
    {
        fwd->resize(main_dims);
        bwd->resize(main_dims);

        for (int i = 0; i < aux_cam_images.size(); i++)
        {
            aux_cam_images[i].curr->resize(aux_res[i]);
            aux_cam_images[i].next->resize(aux_res[i]);
            aux_cam_images[i].priority = aux_priorities[i];
        }

        for (auto& di : depth_imgs)
            di.resize(main_dims);
    }
    
    void swap()
    {
        std::swap(fwd, bwd);

        for (auto& aux_img : aux_cam_images)
            aux_img.swap();
    }
};