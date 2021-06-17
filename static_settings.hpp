#ifndef SLAM_STATIC_SETTINGS_HPP
#define SLAM_STATIC_SETTINGS_HPP

#include <vector>

namespace odometry { struct Parameters; }
namespace slam {
struct StaticSettings {
    const odometry::Parameters &parameters;
    std::vector<float> scaleFactors;
    std::vector<float> levelSigmaSq;
    StaticSettings(const odometry::Parameters &p);

    static constexpr unsigned ORB_PATCH_RADIUS = 19;
    static constexpr unsigned ORB_FAST_PATCH_SIZE = 31;
    static constexpr unsigned ORB_FAST_PATCH_HALF_SIZE = ORB_FAST_PATCH_SIZE / 2;



    std::vector<std::size_t> maxNumberOfKeypointsPerLevel() const;

};
}

#endif
