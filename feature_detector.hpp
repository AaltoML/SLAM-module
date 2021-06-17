#ifndef SLAM_MULTI_FAST_DETECTOR_HPP
#define SLAM_MULTI_FAST_DETECTOR_HPP

#include <memory>
#include <vector>

#include "key_point.hpp"

namespace tracker { struct Image; }

namespace slam {
struct StaticSettings;
struct ImagePyramid;

struct FeatureDetector {
    static std::unique_ptr<FeatureDetector> build(const StaticSettings &settings, tracker::Image &modelImage);
    virtual ~FeatureDetector();

    /** @return the total number of detected keypoints */
    virtual std::size_t detect(
        ImagePyramid &imagePyramid,
        std::vector< KeyPointVector > &keypointsPerLevel) = 0;

};
}

#endif
