#ifndef SLAM_SLAM_ORB_EXTRACTOR_HPP
#define SLAM_SLAM_ORB_EXTRACTOR_HPP

#include "static_settings.hpp"
#include "key_point.hpp"
#include <memory>

namespace cv { class Mat; }
namespace tracker { class Camera; struct Image; }
namespace slam {
struct OrbExtractor {
    constexpr static int DESCRIPTOR_COLS = 32;

    virtual ~OrbExtractor() {};

    virtual void detectAndExtract(tracker::Image &img,
        const tracker::Camera &camera,
        const std::vector<tracker::Feature> &tracks,
        KeyPointVector &keyPoints,
        std::vector<int> &keyPointTrackIds) = 0;

    static std::unique_ptr<OrbExtractor> build(const StaticSettings &settings);

    // debug visualization stuff
    enum class VisualizationMode { IMAGE_PYRAMID };
    virtual void debugVisualize(
        const tracker::Image &img,
        cv::Mat &target,
        VisualizationMode mode) const = 0;
};
}

#endif
