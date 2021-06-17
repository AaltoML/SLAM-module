#ifndef SLAM_RELOCATION_HPP
#define SLAM_RELOCATION_HPP

#include "id.hpp"
#include "mapdb.hpp"
#include "bow_index.hpp"

namespace slam {

void tryRelocation(
    KfId currentKf,
    MapKf candidate,
    MapDB &mapDB,
    const Atlas &atlas,
    const odometry::ParametersSlam &parameters,
    const StaticSettings &settings
);

} // namespace slam

#endif // SLAM_RELOCATION_HPP
