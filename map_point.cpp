#include "map_point.hpp"
#include "mapdb.hpp"
#include "keyframe.hpp"

#include "openvslam/match_base.h"

using Eigen::Vector3d;

namespace slam {

MapPoint::MapPoint(MpId id, KfId keyframeId, KpId keyPointId) :
    id(id), referenceKeyframe(keyframeId) {
    assert(keyframeId.v != -1 && "Cannot create MapPoint without reference keyframe");
    addObservation(keyframeId, keyPointId);
    position = Eigen::Vector3d::Zero();
    norm = Eigen::Vector3f::Zero();
}

// For cereal.
MapPoint::MapPoint() {}

MapPoint::MapPoint(const MapPoint mapPoint, const std::set<KfId> &activeKeyframes) {
    id = mapPoint.id;
    trackId = mapPoint.trackId;
    status = mapPoint.status;
    position = mapPoint.position;
    norm = mapPoint.norm;
    minViewingDistance = mapPoint.minViewingDistance;
    maxViewingDistance = mapPoint.maxViewingDistance;
    descriptor = mapPoint.descriptor;

    std::copy_if(mapPoint.observations.begin(), mapPoint.observations.end(), std::inserter(observations, observations.begin()),
        [activeKeyframes](std::pair<KfId, KpId> const& pair) {
            return activeKeyframes.count(pair.first);
        }
    );

    if (activeKeyframes.count(mapPoint.referenceKeyframe)) {
        referenceKeyframe = mapPoint.referenceKeyframe;
    } else {
        referenceKeyframe = observations.begin()->first; // TODO: Just gets first observation, could do better here?
    }
}

static KfId getFirstOrLastObservation(const MapPoint &mp, bool first = true) {
    assert(!mp.observations.empty() && "Every MapPoint should have at least one observation");

    using P = decltype(mp.observations)::value_type;

    return std::min_element(
        mp.observations.begin(),
        mp.observations.end(),
        [first](const P &p1, const P &p2) { return (p1.first.v < p2.first.v) == first; }
    )->first;
}

KfId MapPoint::getFirstObservation() const {
    return getFirstOrLastObservation(*this, true);
}

KfId MapPoint::getLastObservation() const {
    return getFirstOrLastObservation(*this, false);
}

void MapPoint::addObservation(KfId keyframeId, KpId keyPointId) {
    assert(!observations.count(keyframeId));
    observations.emplace(keyframeId, keyPointId);
}

void MapPoint::eraseObservation(KfId keyframeId) {
    assert(observations.count(keyframeId));
    observations.erase(keyframeId);
}

void MapPoint::updateDescriptor(const MapDB &mapDB) {
    std::vector<KeyPoint::Descriptor> descriptors;
    descriptors.reserve(observations.size());
    for (const auto& obs : observations) {
        const auto &kf = *mapDB.keyframes.at(obs.first);
        if (kf.hasFeatureDescriptors()) {
            const auto &kp = kf.shared->keyPoints.at(obs.second.v);
            descriptors.push_back(kp.descriptor);
        }
    }

    if (descriptors.empty()) return;

    // Get median of Hamming distance
    // Calculate all the Hamming distances between every pair of the features
    const auto num_descs = descriptors.size();
    std::vector<std::vector<unsigned int>> hamm_dists(num_descs, std::vector<unsigned int>(num_descs));
    for (unsigned int i = 0; i < num_descs; ++i) {
        hamm_dists.at(i).at(i) = 0;
        for (unsigned int j = i + 1; j < num_descs; ++j) {
            const auto dist = openvslam::match::compute_descriptor_distance_32(descriptors.at(i).data(), descriptors.at(j).data());
            hamm_dists.at(i).at(j) = dist;
            hamm_dists.at(j).at(i) = dist;
        }
    }

    // Get the nearest value to median
    unsigned int best_median_dist = openvslam::match::MAX_HAMMING_DIST;
    unsigned int best_idx = 0;
    for (unsigned idx = 0; idx < num_descs; ++idx) {
        std::vector<unsigned int> partial_hamm_dists(hamm_dists.at(idx).begin(), hamm_dists.at(idx).begin() + num_descs);
        std::sort(partial_hamm_dists.begin(), partial_hamm_dists.end());
        const auto median_dist = partial_hamm_dists.at(static_cast<unsigned int>(0.5 * (num_descs - 1)));

        if (median_dist < best_median_dist) {
            best_median_dist = median_dist;
            best_idx = idx;
        }
    }

    descriptor = descriptors.at(best_idx);
}

void MapPoint::replaceWith(MapDB &mapDB, MapPoint &otherMp) {
    assert(this->id.v != -1);
    assert(mapDB.mapPoints.count(this->id));
    assert(otherMp.id.v != -1);
    assert(mapDB.mapPoints.count(otherMp.id));

    if (otherMp.id == this->id) {
        return;
    }

    if (trackId.v != -1) {
        if (otherMp.trackId.v == -1) {
            mapDB.trackIdToMapPoint.at(trackId) = otherMp.id;
            otherMp.trackId = trackId;
        } else {
            mapDB.trackIdToMapPoint.erase(trackId);
        }
    }

    for (const auto& kfIdKeypointId : observations) {
        KfId kfId = kfIdKeypointId.first;
        KpId keyPointId = kfIdKeypointId.second;
        Keyframe &kf = *mapDB.keyframes.at(kfId);

        if (kf.keyPointToTrackId.count(keyPointId)) {
            kf.keyPointToTrackId.erase(keyPointId);
        }

        if (!otherMp.observations.count(kfId)) {
            kf.mapPoints[keyPointId.v] = otherMp.id;
            otherMp.addObservation(kfId, keyPointId);
        } else {
            kf.mapPoints[keyPointId.v] = MpId(-1);
        }
    }

    status = MapPointStatus::BAD;
    mapDB.mapPoints.erase(this->id);
}

void MapPoint::updateDistanceAndNorm(const MapDB &mapDB, const StaticSettings &settings) {
    Vector3d normSum = Vector3d::Zero();
    for (const auto &kfIdKp : observations) {
        const auto &kf = *mapDB.keyframes.at(kfIdKp.first);
        normSum += (kf.cameraCenter() - position).normalized();
    }
    norm = normSum.cast<float>() / observations.size();

    const auto &firstKf = *mapDB.keyframes.at(getFirstObservation());
    const float dist = (firstKf.cameraCenter() - position).norm();
    const auto &kp = firstKf.shared->keyPoints[observations.at(firstKf.id).v];

    maxViewingDistance = dist * settings.scaleFactors[kp.octave];
    minViewingDistance = dist * settings.scaleFactors[kp.octave] / settings.scaleFactors.back();
}

int MapPoint::predictScaleLevel(float dist, const StaticSettings &settings) const {
    const float ratio = maxViewingDistance / dist;

    int scale = std::ceil(std::log(ratio)/std::log(settings.parameters.slam.orbScaleFactor));

    return std::min(
        std::max(scale, 0),
        static_cast<int>(settings.scaleFactors.size() - 1)
    );
}

} // namespace slam
