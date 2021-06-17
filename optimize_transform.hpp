#ifndef SLAM_OPTIMIZE_TRANFORM_HPP
#define SLAM_OPTIMIZE_TRANFORM_HPP

#include "keyframe.hpp"
#include "map_point.hpp"
#include "static_settings.hpp"

#include <g2o/types/sim3/sim3.h>

namespace slam {

class MapDB;

unsigned int OptimizeSim3Transform(
        const Keyframe &kf1,
        const Keyframe &kf2,
        const std::vector<std::pair<MpId, MpId>> &matches,
        MapDB &mapDB,
        g2o::Sim3 &transform12,
        const StaticSettings &settings);

} // namespace slam

#endif // SLAM_OPTIMIZE_TRANFORM_HPP
