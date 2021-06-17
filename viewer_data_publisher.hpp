#ifndef SLAM_DATA_PUBLISHER_HPP
#define SLAM_DATA_PUBLISHER_HPP

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <opencv2/core.hpp>

#include <mutex>
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <unordered_map>

#include "../codegen/output/cmd_parameters.hpp"
#include "id.hpp"
#include "bow_index.hpp"
#include "loop_closer.hpp"

namespace slam {

class MapDB;
class Keyframe;

enum class MatchType {
    MAPPER, LOOP
};

struct ViewerLoopClosure {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Camera-to-world.
    Eigen::Matrix4f currentPose;
    Eigen::Matrix4f candidatePose;
    Eigen::Matrix4f updatedPose;
};

struct ViewerMapPoint {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Vector3f position;
    Eigen::Vector3f normal;
    Eigen::Vector3f color;
    int status;
    bool localMap;
    bool nowVisible;
};

struct ViewerKeyframe {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    KfId id;
    bool localMap;
    bool current;
    Eigen::Matrix4f poseWC;
    Eigen::Matrix4f origPoseWC;
    std::vector<size_t> neighbors;
    std::shared_ptr<const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>> stereoPointCloud;
    std::shared_ptr<const std::vector<cv::Vec3b>> stereoPointCloudColor;
};

struct ViewerAtlasMapPoint {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Vector3f position;
};

struct ViewerAtlasKeyframe {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    KfId id;
    Eigen::Matrix4f poseWC;
};

using ViewerAtlasMapPointVector = std::vector<ViewerAtlasMapPoint, Eigen::aligned_allocator<ViewerAtlasMapPoint>>;
using ViewerAtlasKeyframeVector = std::vector<ViewerAtlasKeyframe, Eigen::aligned_allocator<ViewerAtlasKeyframe>>;
struct ViewerAtlasMap {
    // EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ViewerAtlasMapPointVector mapPoints;
    ViewerAtlasKeyframeVector keyframes;
};

using ViewerAtlas = std::vector<ViewerAtlasMap>;

struct SearchedMapPoint {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Vector2d mp;
    Eigen::Vector2d kp;
};

class ViewerDataPublisher {
public:
    using MapPointVector = std::vector<ViewerMapPoint, Eigen::aligned_allocator<ViewerMapPoint>>;
    using KeyframeVector = std::vector<ViewerKeyframe, Eigen::aligned_allocator<ViewerKeyframe>>;
    using SearchedMapPointVector = std::vector<SearchedMapPoint, Eigen::aligned_allocator<SearchedMapPoint>>;
    using Vector2dVector = std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>;

    ViewerDataPublisher(const cmd::ParametersSlam &parameters);

    std::vector<ViewerLoopClosure> getLoopClosures();
    void addLoopClosure(const ViewerLoopClosure &lc);

    void setMap(
        const MapPointVector &mps,
        const KeyframeVector &kfs,
        const std::map<MapKf, LoopStage> &loopStages,
        double age
    );

    MapPointVector getMapPoints();
    void getKeyframes(
        KeyframeVector &keyframes,
        std::map<MapKf, LoopStage> &loopStages,
        double &age
    );

    void setAtlas(std::unique_ptr<ViewerAtlas> atlas);
    std::unique_ptr<ViewerAtlas> takeAtlas();

    const cmd::ParametersSlam &getParameters() const;

    // TODO: not the ideal design making this abstract, but the quickest fix
    // for separating OpenCV GUI code from cross-platform production code
    virtual void visualizeKeyframe(
        const MapDB &mapDB,
        const cv::Mat &grayFrame,
        const Keyframe &kf
    ) = 0;
    virtual void visualizeOrbs(
        const cv::Mat &frame,
        const Keyframe &kf
    ) = 0;
    virtual void visualizeMapPointSearch(
        const cv::Mat &frame,
        const SearchedMapPointVector &searched,
        const Vector2dVector &mps,
        const Vector2dVector &kps
    ) = 0;
    virtual void showMatches(
        const Keyframe &kf1,
        const Keyframe &kf2,
        const std::vector<std::pair<KpId, KpId>> &matches,
        MatchType matchType
    ) = 0;
    virtual void visualizeOther(const cv::Mat &mat) = 0;
    virtual std::map<std::string, cv::Mat> pollVisualizations() = 0;
    virtual void setFrameRotation(int rotation) = 0;

protected:
    std::mutex m;
    std::vector<ViewerLoopClosure> loopClosures;
    MapPointVector mapPoints;
    KeyframeVector keyframes;
    std::map<MapKf, LoopStage> loopStages;
    double age = 0.0;
    std::unique_ptr<ViewerAtlas> atlas;
    cmd::ParametersSlam parameters;
};

} // namespace slam

#endif // SLAM_DATA_PUBLISHER_HPP
