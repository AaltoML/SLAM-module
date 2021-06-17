#ifndef SLAM_FEATURE_SEARCH_HPP
#define SLAM_FEATURE_SEARCH_HPP

#include <memory>
#include <vector>

#include "key_point.hpp"

namespace slam {

/**
 * An acceleration data structure for 2D keypoint search. Once constructed
 * with a fixed set of 2D points, can find indices of these points within
 * a given radius from a given point fast.
 */
class FeatureSearch {
public:
    static std::unique_ptr<FeatureSearch> create(const KeyPointVector &keypoints);
    virtual ~FeatureSearch() = default;
    virtual void getFeaturesAround(float x, float y, float r, std::vector<size_t> &output) const = 0;
};

}

#endif
