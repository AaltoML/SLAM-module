#include "relocation.hpp"

#include "../util/logging.hpp"
#include "keyframe_matcher.hpp"
#include "loop_ransac.hpp"

namespace slam {

void tryRelocation(
    KfId currentKfId,
    MapKf candidate,
    MapDB &currentMapDB,
    const Atlas &atlas,
    const odometry::ParametersSlam &parameters,
    const StaticSettings &settings
) {
    const MapDB &candidateMapDB = atlas[candidate.mapId.v];

    const Keyframe &currentKf = *currentMapDB.keyframes.at(currentKfId);
    const Keyframe &candidateKf = *candidateMapDB.keyframes.at(candidate.kfId);

    std::vector<int> matchedFeatureIds;
    matchForLoopClosures(
        currentKf,
        candidateKf,
        currentMapDB,
        candidateMapDB,
        matchedFeatureIds,
        parameters);

    std::vector<std::pair<MpId, MpId>> matches;
    for (unsigned i = 0; i < matchedFeatureIds.size(); ++i) {
        const int kfIdx2 = matchedFeatureIds.at(i);
        if (kfIdx2 >= 0) {
            const MpId mpId1 = currentKf.mapPoints.at(i);
            const MpId mpId2 = candidateKf.mapPoints.at(kfIdx2);
            if (mpId1.v != -1 && mpId2.v != -1) {
                matches.emplace_back(mpId1, mpId2);
            }
        }
    }

    if (matches.size() < parameters.minLoopClosureFeatureMatches) {
        return;
    }
    currentMapDB.loopStages[candidate] = LoopStage::RELOCATION_MAP_POINT_MATCHES;

    LoopRansac loopRansac(
        currentKf,
        candidateKf,
        matches,
        currentMapDB,
        candidateMapDB,
        settings
    );
    loopRansac.ransacSolve(parameters.loopClosureRansacIterations, LoopRansac::DoF::SIM3);
    if (!loopRansac.solutionOk) {
        return;
    }
    currentMapDB.loopStages[candidate] = LoopStage::RELOCATION_MAP_POINT_RANSAC;
}

} // namespace slam
