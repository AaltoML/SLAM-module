#ifndef SLAM_BUNDLE_ADJUSTER_HPP
#define SLAM_BUNDLE_ADJUSTER_HPP

#include <memory>
#include <set>

#include "id.hpp"
#include "static_settings.hpp"
#include "ba_stats.hpp"

namespace slam {

class Keyframe;
class MapDB;

struct WorkspaceBA {
    std::set<MpId> localMpIds;
    std::set<KfId, std::greater<KfId>> localKfIds;

    BaStats baStats;

    WorkspaceBA(bool enableBaStats) :
        baStats(enableBaStats)
    {}
};

/**
* Do a local bundle adjustment
*/
std::set<MpId> localBundleAdjust(
    Keyframe &keyframe,
    WorkspaceBA &workspace,
    MapDB &mapDB,
    int problemMaxSize,
    const StaticSettings &settings
);

/**
* Do a local bundle adjustment for a single pose
*/
bool poseBundleAdjust(
    Keyframe &keyframe,
    MapDB &mapDB,
    const StaticSettings &settings
);

void globalBundleAdjust(
    KfId currentKfId,
    MapDB &mapDB,
    const StaticSettings &settings
);

}

#endif
