#ifndef SLAM_KEY_POINT_HPP
#define SLAM_KEY_POINT_HPP

#include <array>
#include <cstdint>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include "../tracker/track.hpp"

namespace slam {
struct KeyPoint {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    tracker::Feature::Point pt;
    float angle;
    int octave;
    Eigen::Vector3d bearing;

    using Descriptor = std::array<std::uint32_t, 8>;
    Descriptor descriptor;

    template<class Archive>
    void serialize(Archive &ar) {
        ar(pt.x, pt.y, angle, octave, octave, bearing, descriptor);
    }
};

using KeyPointVector = std::vector<KeyPoint, Eigen::aligned_allocator<KeyPoint>>;
}

#endif
