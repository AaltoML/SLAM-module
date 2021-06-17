#ifndef SLAM_MAPPER_HELPERS_HPP
#define SLAM_MAPPER_HELPERS_HPP

#include <set>
#include <string>

#include "bow_index.hpp"
#include "../api/slam.hpp"
#include "map_point.hpp"
#include "bundle_adjuster.hpp"
#include "../odometry/util.hpp"
#include "../util/util.hpp"
#include "viewer_data_publisher.hpp"
#include "static_settings.hpp"

class CommandQueue;

namespace slam {

class ViewerDataPublisher;
class LoopCloser;
struct MapperInput;
struct OrbExtractor;

bool makeKeyframeDecision(
    const Keyframe &currentKeyframe,
    const Keyframe *previousKeyframe,
    const std::vector<tracker::Feature> &currentTracks,
    const odometry::ParametersSlam &parameters
);

void matchTrackedFeatures(
    Keyframe &currentKeyframe,
    MapDB &mapDB,
    const StaticSettings &settings
);

/**
 * Compute a set of spatially nearby keyframes. This function attempts to improve over
 * the "neighbors" concept from ORBSLAM/OpenVSLAM, which considers only visual "neighborness".
 * This tends to leave holes in the keyframe sequences, which in some situations (local BA) is
 * especially bad for our method that has strong inertial information about successive keyframes.
 *
 * @param currentKeyframe The keyframe to search around, won't be included in return vector.
 * @param minCovisibilities Number of shared map point observations required to include a keyframe from a different "island".
 * @param maxKeyframes Return up to this many keyframes, dropping more distant ones.
 * @param mapDB
 * @param settings
 * @param visualize Produce visualizations from this call.
 * @return Vector of keyframes, sorted by ascending distance, not including the argument keyframe.
 */
std::vector<KfId> computeAdjacentKeyframes(
    const Keyframe &currentKeyframe,
    int minCovisibilities,
    int maxKeyframes,
    const MapDB &mapDB,
    const StaticSettings &settings,
    bool visualize = false
);

void matchLocalMapPoints(
    Keyframe &currentKeyframe,
    const std::vector<KfId> &adjacentKfIds,
    MapDB &mapDB,
    const StaticSettings &settings,
    ViewerDataPublisher *dataPublisher
);

void createNewMapPoints(
    Keyframe &currentKeyframe,
    const std::vector<KfId> &adjacentKfIds,
    MapDB &mapDB,
    const StaticSettings &settings,
    ViewerDataPublisher *dataPublisher
);

void deduplicateMapPoints(
    Keyframe &currentKeyframe,
    const std::vector<KfId> &adjacentKfIds,
    MapDB &mapDB,
    const StaticSettings &settings
);

void cullMapPoints(
    Keyframe &currentKeyframe,
    MapDB &mapDB,
    const odometry::ParametersSlam &parameters
);

void cullKeyframes(
    const std::vector<KfId> &adjacentKfIds,
    MapDB &mapDB,
    BowIndex &bowIndex,
    const odometry::ParametersSlam &parameters
);

void checkConsistency(const MapDB &mapDB);

bool checkPositiveDepth(
    const Eigen::Vector3d &positionW,
    const Eigen::Matrix4d &poseCW
);

bool checkTriangulationAngle(const vecVector3d &raysW, double minAngleDeg);

int getFocalLength(const Keyframe &kf);

bool checkReprojectionError(
    const Eigen::Vector3d& pos,
    const Keyframe &kf,
    const StaticSettings &settings,
    KpId kpId,
    float relativeReprojectionErrorThreshold
);

void triangulateMapPoint(
    MapDB &mapDB,
    MapPoint &mapPoint,
    const StaticSettings &settings,
    TriangulationMethod method = TriangulationMethod::TME
);

void triangulateMapPointFirstLastObs(
    MapDB &mapDB,
    MapPoint &mapPoint,
    const StaticSettings &settings
);

void publishMapForViewer(
    ViewerDataPublisher &dataPublisher,
    const WorkspaceBA *workspaceBA,
    const MapDB &mapDB,
    const odometry::ParametersSlam &parameters
);

Eigen::MatrixXd odometryPriorStrengths(
    KfId kfId1,
    KfId kfId2,
    const odometry::ParametersSlam &parameters,
    const slam::MapDB &mapDB
);

MapDB loadMapDB(
    MapId mapId,
    BowIndex &bowIndex,
    const std::string &loadPath
);

void addKeyframeFrontend(
    MapDB &mapDB,
    std::unique_ptr<Keyframe> keyframePtr,
    bool keyFrameDecision,
    const MapperInput &mapperInput,
    const StaticSettings &settings,
    Eigen::Matrix4d &resultPose,
    Slam::Result::PointCloud *resultPointCloud
);

KfId addKeyframeBackend(
    MapDB &mapDB,
    std::unique_ptr<Keyframe> keyframePtr,
    bool keyFrameDecision,
    const MapperInput &mapperInput,
    const StaticSettings &settings,
    WorkspaceBA &workspaceBA,
    LoopCloser &loopCloser,
    OrbExtractor &orbExtractor,
    BowIndex &bowIndex,
    CommandQueue *commands,
    ViewerDataPublisher *dataPublisher,
    Eigen::Matrix4d &resultPose,
    Slam::Result::PointCloud *resultPointCloud);

ViewerAtlasMap mapDBtoViewerAtlasMap(const MapDB &mapDB);

} // namespace slam

#endif //SLAM_MAPPER_HELPERS_HPP
