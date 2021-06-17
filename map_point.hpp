#ifndef SLAM_MAP_POINT_HPP
#define SLAM_MAP_POINT_HPP

#include <map>
#include <set>
#include <memory>

#include <cereal/types/map.hpp>
#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include "id.hpp"
#include "key_point.hpp"
#include "static_settings.hpp"

namespace slam {

class Keyframe;
class MapDB;

enum class MapPointStatus { TRIANGULATED, NOT_TRIANGULATED, UNSURE, BAD };

class MapPoint {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    MapPoint(MpId id, KfId keyframeId, KpId keyPointId);
    MapPoint();
    MapPoint(const MapPoint mapPoint, const std::set<KfId> &activeKeyframes); // Copy MapPoint and remove links to inactive keyframes

    KfId getFirstObservation() const;
    KfId getLastObservation() const;

    void addObservation(KfId keyframeId, KpId keyPointId);

    void eraseObservation(KfId keyframeId);

    /**
     * Update mapPoint descriptor to the featuredescriptor closest to the median
     * of all featuredescriptors.
     *
     * Implementation from `openvslam::landmark::compute_descriptor()`
     */
    void updateDescriptor(const MapDB &mapDB);

    void updateDistanceAndNorm(const MapDB &mapDB, const StaticSettings &settings);

    void replaceWith(MapDB &mapDB, MapPoint &otherMp);

    int predictScaleLevel(float dist, const StaticSettings &settings) const;

    /** Unique ID of this map point */
    MpId id;

    /** Which odometry track ID does this correspond to, -1 if none */
    TrackId trackId = TrackId(-1);

    MapPointStatus status = MapPointStatus::NOT_TRIANGULATED;
    Eigen::Vector3d position;

    // Viewing direction
    Eigen::Vector3f norm;
    float minViewingDistance = 0;
    float maxViewingDistance = 30;

    KeyPoint::Descriptor descriptor;

    /**
     * Observations of this map_point (keyframeId, keyPoint Id in keyframe)
     */
    std::map<KfId, KpId> observations;

    KfId referenceKeyframe;

    cv::Vec3b color; // for visualization / debugging purposes

    template<class Archive>
    void serialize(Archive &ar) {
        ar(
            id,
            trackId,
            status,
            position,
            norm,
            minViewingDistance,
            maxViewingDistance,
            descriptor,
            observations,
            referenceKeyframe,
            color
        );
    }
};

} // namespace slam

#endif //SLAM_MAP_POINT_HPP
