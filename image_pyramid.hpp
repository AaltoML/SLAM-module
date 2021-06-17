#ifndef SLAM_SLAM_IMAGE_PYRAMID_HPP
#define SLAM_SLAM_IMAGE_PYRAMID_HPP

#include <memory>
#include <vector>

#include <accelerated-arrays/image.hpp>
#include <accelerated-arrays/standard_ops.hpp>

#include "../tracker/image.hpp"

namespace cv { class Mat; }
namespace slam {
struct StaticSettings;

struct ImagePyramid {
    static std::unique_ptr<ImagePyramid> build(const StaticSettings &settings, tracker::Image &modelImage);
    virtual ~ImagePyramid();

    virtual void update(tracker::Image &image) = 0;
    virtual std::size_t numberOfLevels() const = 0;
    virtual bool isGpu() const = 0;

    virtual accelerated::Image &getLevel(std::size_t level) = 0; // CPU
    virtual accelerated::Image &getBlurredLevel(std::size_t level) = 0; // CPU
    virtual accelerated::Image &getGpuLevel(std::size_t level) = 0; // GPU, not blurred

    // for debugging, render as OpenCV matrix
    virtual void debugVisualize(cv::Mat &target) = 0;
};
}

#endif
