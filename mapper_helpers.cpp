#include "mapper_helpers.hpp"

#include <cereal/archives/binary.hpp>

#include "../util/util.hpp"
#include "../util/logging.hpp"
#include "../util/timer.hpp"
#include "theia/theia.h"

#include "keyframe.hpp"
#include "bundle_adjuster.hpp"
#include "keyframe_matcher.hpp"
#include "mapdb.hpp"
#include "serialization.hpp"

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Matrix3x4d = Eigen::Matrix<double,3,4>;
using Eigen::Vector4d;
using Eigen::Vector3d;
using Eigen::Vector2d;
using Eigen::Vector2f;

namespace slam {

static const double CHI2_INV2D = 5.991;

bool makeKeyframeDecision(
    const Keyframe &currentKeyframe,
    const Keyframe *previousKeyframe,
    const std::vector<tracker::Feature> &currentTracks,
    const odometry::ParametersSlam &parameters
) {
    if (!previousKeyframe) return true;

    double age = currentKeyframe.t - previousKeyframe->t;
    assert(age >= 0.0);
    if (age < parameters.keyframeDecisionMinIntervalSeconds) return false;

    const double distance = (currentKeyframe.origPoseCameraCenter() - previousKeyframe->origPoseCameraCenter()).norm();
    // log_debug("distance check: %g", distance);
    if (distance > parameters.keyframeDecisionDistanceThreshold) {
        return true;
    }

    int prevCovisiblities = 0;
    unsigned nTracks = 0;

    std::set<TrackId> prevTrackIds; // some heap use
    for (const auto &it : previousKeyframe->keyPointToTrackId)
        prevTrackIds.insert(it.second);

    // NOTE: currentKeyframe.keyPointToTrackId has not been populated yet!
    assert(currentKeyframe.keyPointToTrackId.empty());
    for (const auto &track : currentTracks) {
        nTracks++;
        if (prevTrackIds.count(TrackId(track.id))) prevCovisiblities++;
    }

    // log_debug("covisibility check: %d/%d", nTracks, prevCovisiblities);
    const float maxCovis = static_cast<float>(nTracks) * parameters.keyframeDecisionCovisibilityRatio;
    if (prevCovisiblities <= maxCovis) return true;

    return false;
}

void matchTrackedFeatures(
    Keyframe &currentKeyframe,
    MapDB &mapDB,
    const StaticSettings &settings
) {
    timer(slam::TIME_STATS, __FUNCTION__);
    std::size_t total = 0, newPoints = 0, justTriangulated = 0, frustumFail = 0, reproFail = 0, success = 0;
    const auto &parameters = settings.parameters.slam;

    for (size_t v = 0; v < currentKeyframe.shared->keyPoints.size(); v++) {
        KpId keyPointId(v);
        if (currentKeyframe.keyPointToTrackId.count(keyPointId)) {
            TrackId trackId = currentKeyframe.keyPointToTrackId.at(keyPointId);

            if (mapDB.trackIdToMapPoint.count(trackId)) {
                total++;
                MapPoint &mapPoint = mapDB.mapPoints.at(mapDB.trackIdToMapPoint.at(trackId));

                if (mapPoint.status != MapPointStatus::TRIANGULATED) {
                    justTriangulated++;
                    mapPoint.addObservation(currentKeyframe.id, keyPointId);
                    currentKeyframe.addObservation(mapPoint.id, keyPointId);

                    triangulateMapPointFirstLastObs(mapDB, mapPoint, settings);
                    // triangulateMapPoint(*mapPoint);
                } else {
                    // minimal sanity checks: note that odometry should have
                    // already checked at least the epipolar constraint
                    // in the RANSAC filters.
                    if (!currentKeyframe.isInFrustum(mapPoint)) {
                        frustumFail++;
                        continue;
                    }

                    // Odometry triangulation will also eventually check
                    // reprojection error, but this may not have happened
                    // recently for all features
                    if (!checkReprojectionError(
                        mapPoint.position,
                        currentKeyframe,
                        settings,
                        keyPointId,
                        parameters.relativeReprojectionErrorThreshold)
                    ) {
                        reproFail++;
                        continue;
                    }

                    mapPoint.addObservation(currentKeyframe.id, keyPointId);
                    currentKeyframe.addObservation(mapPoint.id, keyPointId);
                }

                // skip descriptor update if this is not a keyframe, but this
                // operation is performed for pose-only bundle adjustment
                if (mapPoint.status == MapPointStatus::TRIANGULATED) {
                    if (currentKeyframe.hasFeatureDescriptors()) mapPoint.updateDescriptor(mapDB);
                    mapPoint.updateDistanceAndNorm(mapDB, settings);
                    success++;
                }
            } else if (currentKeyframe.hasFeatureDescriptors()) {
                // Create new mappoint
                MpId mpId = mapDB.nextMpId();
                MapPoint mapPoint(mpId, currentKeyframe.id, keyPointId);
                currentKeyframe.addObservation(mapPoint.id, keyPointId);

                mapPoint.updateDescriptor(mapDB);
                mapPoint.trackId = trackId;
                mapPoint.color = currentKeyframe.getKeyPointColor(keyPointId);

                mapDB.trackIdToMapPoint.emplace(trackId, mapPoint.id);
                mapDB.mapPoints.emplace(mpId, std::move(mapPoint));
                newPoints++;
            }
        }
    }
}

std::vector<KfId> computeAdjacentKeyframes(
    const Keyframe &currentKeyframe,
    int minCovisibilities,
    int maxKeyframes,
    const MapDB &mapDB,
    const StaticSettings &settings,
    bool visualize
) {
    std::set<KfId> adjacentSet;

    // In the following comments, "consecutive" means those that can be reached
    // via finite number of steps following the next and prev KF pointers.

    // Collect consecutive keyframes and their neighbors, call them `parents`.
    std::set<KfId> parents;
    int i = 0;
    {
        KfId backwards = currentKeyframe.id;
        while (backwards.v != -1) {
            adjacentSet.insert(backwards);
            const Keyframe &keyframe = *mapDB.keyframes.at(backwards);
            // getNeighbors() is somewhat slow, do not call for every keyframe.
            if (i % 2 == 0) {
                for (KfId kfId : keyframe.getNeighbors(mapDB, minCovisibilities, false)) {
                    parents.insert(kfId);
                }
            }
            if (++i >= maxKeyframes) {
                break;
            }
            backwards = keyframe.previousKfId;
        }
    }

    // Return keyframes consecutive to some `parent`.
    for (KfId parent : parents) {
        KfId backwards = parent;
        i = 0;
        while (backwards.v != -1) {
            adjacentSet.insert(backwards);
            if (++i >= maxKeyframes / 2) {
                break;
            }
            const Keyframe &keyframe = *mapDB.keyframes.at(backwards);
            backwards = keyframe.previousKfId;
        }
        KfId forwards = parent;
        i = 0;
        while (forwards.v != -1) {
            adjacentSet.insert(forwards);
            if (++i >= maxKeyframes / 2) {
                break;
            }
            const Keyframe &keyframe = *mapDB.keyframes.at(forwards);
            forwards = keyframe.nextKfId;
        }
    }

    adjacentSet.erase(currentKeyframe.id);
    std::vector<KfId> adjacent(adjacentSet.begin(), adjacentSet.end());

    // Sort by distance.
    Eigen::Vector3d currentPos = currentKeyframe.cameraCenter();
    auto dist2 = [&](KfId kfId) {
        return (mapDB.keyframes.at(kfId)->cameraCenter() - currentPos).squaredNorm();
    };
    std::sort(adjacent.begin(), adjacent.end(), [&](KfId a, KfId b) { return dist2(a) < dist2(b); });

    // Keep N closest keyframes.
    if (static_cast<int>(adjacent.size()) > maxKeyframes) {
        assert(adjacent.begin() + maxKeyframes < adjacent.end());
        adjacent.erase(adjacent.begin() + maxKeyframes, adjacent.end());
    }

    if (visualize && settings.parameters.slam.kfAsciiAdjacent) {
        auto status = [&](KfId kfId) {
            if (std::find(adjacent.begin(), adjacent.end(), kfId) != adjacent.end()) {
                return 'a';
            }
            return ' ';
        };
        asciiKeyframes(status, mapDB, settings.parameters.slam.kfAsciiWidth);
    }

    return adjacent;
}

void matchLocalMapPoints(
    Keyframe &currentKeyframe,
    const std::vector<KfId> &adjacentKfIds,
    MapDB &mapDB,
    const StaticSettings &settings,
    ViewerDataPublisher *dataPublisher
) {
    timer(slam::TIME_STATS, __FUNCTION__);
    const auto &parameters = settings.parameters.slam;

    // Collect map points.
    std::set<MpId> uniqueMps;
    for (KfId kfId : adjacentKfIds) {
        const auto &kf = mapDB.keyframes.at(kfId);
        for (MpId mpId : kf->mapPoints) {
            if (mpId.v != -1) {
                uniqueMps.insert(mpId);
            }
        }
    }
    std::vector<MpId> localMps;
    for (MpId mpId : uniqueMps) {
        MapPoint &mp = mapDB.mapPoints.at(mpId);
        if (mp.status != MapPointStatus::NOT_TRIANGULATED &&
            mp.status != MapPointStatus::BAD &&
            !mp.observations.count(currentKeyframe.id) &&
            currentKeyframe.isInFrustum(mp))
        {
            localMps.push_back(mp.id);
        }
    }

    if (localMps.empty()) return;

    const float r = getFocalLength(currentKeyframe) * parameters.relativeReprojectionErrorThreshold;

    // Search the map points in the current keyframe.
    searchByProjection(currentKeyframe, localMps, mapDB, dataPublisher, r, settings);
}

void createNewMapPoints(
    Keyframe &currentKeyframe,
    const std::vector<KfId> &adjacentKfIds,
    MapDB &mapDB,
    const StaticSettings &settings,
    ViewerDataPublisher *dataPublisher
) {
    timer(slam::TIME_STATS, __FUNCTION__);

    for (KfId kfId : adjacentKfIds) {
        if (kfId == currentKeyframe.id) continue;

        Keyframe &keyframe = *mapDB.keyframes.at(kfId);

        // double baselineDistance = (keyframe.cameraCenter() - currentKeyframe.cameraCenter()).norm();
        // If the scene scale is much smaller than the baseline, abort the triangulation
        // Ratio from openvslam -- should be tested TODO(jhnj)
        // if (baselineDistance < 0.02 * keyframe.computeMedianDepth(mapDB)) {
        //     continue;
        // }

        std::vector<std::pair<KpId, KpId>> matches = matchForTriangulationDBoW(
            currentKeyframe, keyframe, settings);

        if (dataPublisher && dataPublisher->getParameters().visualizeOrbMatching
                && !currentKeyframe.shared->imgDbg.empty() && !keyframe.shared->imgDbg.empty()) {
            if (keyframe.id == currentKeyframe.previousKfId) {
                dataPublisher->showMatches(currentKeyframe, keyframe, matches, MatchType::MAPPER);
            }
        }

        for (const auto &match : matches) {
            MpId mpId = mapDB.nextMpId();
            MapPoint mapPoint(mpId, keyframe.id, match.second);
            mapPoint.color = keyframe.getKeyPointColor(match.second);
            mapPoint.addObservation(currentKeyframe.id, match.first);

            triangulateMapPoint(mapDB, mapPoint, settings);
            if (mapPoint.status != MapPointStatus::NOT_TRIANGULATED) {
                currentKeyframe.addObservation(mapPoint.id, match.first);
                keyframe.addObservation(mapPoint.id, match.second);
                mapPoint.updateDescriptor(mapDB);

                mapDB.mapPoints.emplace(mpId, std::move(mapPoint));
            }
        }
    }
}

void deduplicateMapPoints(
    Keyframe &currentKeyframe,
    const std::vector<KfId> &adjacentKfIds,
    MapDB &mapDB,
    const StaticSettings &settings
) {
    timer(slam::TIME_STATS, __FUNCTION__);

    float margin = getFocalLength(currentKeyframe) * settings.parameters.slam.relativeReprojectionErrorThreshold;

    // MapPoints from current to adjacents.
    for (KfId kfId : adjacentKfIds) {
        Keyframe &adjacent = *mapDB.keyframes.at(kfId);
        replaceDuplication(adjacent, currentKeyframe.mapPoints, margin, mapDB, settings);
    }

    // MapPoints from adjacents to current.
    std::set<MpId> adjacentMapPointsSet;
    for (KfId kfId : adjacentKfIds) {
        Keyframe &adjacent = *mapDB.keyframes.at(kfId);
        for (MpId mapPointId : adjacent.mapPoints) {
            if (mapPointId.v != -1) {
                adjacentMapPointsSet.insert(mapPointId);
            }
        }
    }
    replaceDuplication(currentKeyframe, adjacentMapPointsSet, margin, mapDB, settings);
}

void cullMapPoints(
    Keyframe &currentKeyframe,
    MapDB &mapDB,
    const odometry::ParametersSlam &parameters
) {
    timer(slam::TIME_STATS, __FUNCTION__);

    // TODO: should not iterate over the whole map
    for (auto it = mapDB.mapPoints.begin(); it != mapDB.mapPoints.end(); ) {
        const MapPoint &mp = it->second;
        if (mp.observations.empty()) {
            // TODO: apparently empty map points can be created by BA (?)
            it = mapDB.removeMapPoint(mp);
            continue;
        }
        const int obsAge = currentKeyframe.t - mapDB.keyframes.at(mp.getFirstObservation())->t;
        // do not remove currently visible map points, they still have
        // a chance of being triangulated again
        if (!mp.observations.count(currentKeyframe.id) && obsAge > parameters.minMapPointCullingAge && mp.status != MapPointStatus::TRIANGULATED) {
            it = mapDB.removeMapPoint(mp);
        } else {
            ++it;
        }
    }
}

static void removeKeyframe(
    KfId kfId,
    MapDB &mapDB,
    BowIndex *bowIndex
) {
    const Keyframe &keyframe = *mapDB.keyframes.at(kfId);

    for (const LoopClosureEdge &l : mapDB.loopClosureEdges) {
        // Global BA assumes keyframes involved in loop closures continue to exist.
        assert(kfId != l.kfId1 && kfId != l.kfId2);
    }

    if (bowIndex != nullptr) bowIndex->remove(MapKf { CURRENT_MAP_ID, kfId });

    std::set<MpId> mapPointsToErase;
    const KfId prev = keyframe.previousKfId;
    const KfId next = keyframe.nextKfId;
    assert(prev.v != -1 && "Cannot delete first keyframe");

    for (MpId mpId : keyframe.mapPoints) {
        if (mpId.v != -1) {
            auto &mp = mapDB.mapPoints.at(mpId);
            mp.eraseObservation(keyframe.id);
            if (mp.observations.empty()) {
                // remove orphaned map points
                mapPointsToErase.insert(mpId);
            }
        }
    }

    for (MpId mpId : mapPointsToErase) {
        auto mp = mapDB.mapPoints.at(mpId);
        mapDB.removeMapPoint(mp);
    }

    // Accumulate odometry uncertainty
    if (next.v != -1) {
        Keyframe& nextKf = *mapDB.keyframes.at(next);
        nextKf.uncertainty = nextKf.uncertainty + keyframe.uncertainty;
    }

    // Update KF pointers.
    if (next.v != -1) {
        mapDB.keyframes.at(next)->previousKfId = prev;
    }
    if (prev.v != -1) {
        mapDB.keyframes.at(prev)->nextKfId = next;
    }

    for (auto &it : mapDB.mapPoints) {
        if (it.second.referenceKeyframe == keyframe.id) {
            it.second.referenceKeyframe = prev;
        }
    }

    mapDB.keyframes.erase(mapDB.keyframes.find(kfId));
}

void cullKeyframes(
    const std::vector<KfId> &adjacentKfIds,
    MapDB &mapDB,
    BowIndex &bowIndex,
    const odometry::ParametersSlam &parameters
) {
    timer(slam::TIME_STATS, __FUNCTION__);

    KfId currentKfId = mapDB.keyframes.rbegin()->first;

    // Sort by ascending id to remove newest possible keyframes first.
    std::vector<KfId> sortedKfIds = adjacentKfIds;
    std::sort(sortedKfIds.rbegin(), sortedKfIds.rend());

    for (KfId kfId : sortedKfIds) {
        assert(kfId != currentKfId);

        Keyframe &kf = *mapDB.keyframes.at(kfId);

        // no previous keyframe = first keyframe, don't remove
        if (kf.previousKfId.v < 0) continue;

        // Hack. Do not remove keyframes involved in loop closures because
        // that prevents from placing an optimization constraint between them.
        bool canRemove = true;
        for (const LoopClosureEdge &l : mapDB.loopClosureEdges) {
            if (kfId == l.kfId1 || kfId == l.kfId2) {
                canRemove = false;
                break;
            }
        }
        if (!canRemove) continue;

        unsigned nMapPoints = 0;
        int nCritical = 0;
        // check what ratio of the map points depend on this KF in the
        // sense that if this KF was removed, they could not be used in BA
        // anymore
        for (MpId mpId : kf.mapPoints) {
            if (mpId.v == -1) continue;
            nMapPoints++;
            if (mapDB.mapPoints.at(mpId).observations.size() <= parameters.minObservationsForBA)
                nCritical++;
        }

        if (nCritical < nMapPoints * parameters.keyframeCullMaxCriticalRatio) {
            removeKeyframe(kf.id, mapDB, &bowIndex);
        }
    }
}

static void setPointCloudOutput(const MapDB &mapDB, const Keyframe &kf, Slam::Result::PointCloud &out) {
    out.clear();
    for (MpId mpId : kf.mapPoints) {
        if (mpId.v == -1) continue;
        const MapPoint &mp = mapDB.mapPoints.at(mpId);
        if (mp.status == MapPointStatus::TRIANGULATED) {
            out.push_back(Slam::Result::MapPoint {
                .id = mp.id.v,
                .trackId = mp.trackId.v,
                .position = mp.position,
            });
        }
    }
}

void checkConsistency(const MapDB &mapDB) {
    for (const auto &kfP : mapDB.keyframes) {
        assert(kfP.first == kfP.second->id);
        for (MpId mpId : kfP.second->mapPoints) {
            if (mpId.v != -1) {
                const MapPoint &mp = mapDB.mapPoints.at(mpId);
                assert(mp.observations.count(kfP.first) &&
                    "Keyframe has reference to MapPoint but MapPoint not to Keyframe");
                (void) mp;
            }
        }
    }

    for (const auto &mpP : mapDB.mapPoints) {
        assert(mpP.first == mpP.second.id);
        for (const auto & kfIdkp : mpP.second.observations) {
            const auto &kf = *mapDB.keyframes.at(kfIdkp.first);
            auto it = std::find(kf.mapPoints.begin(), kf.mapPoints.end(), mpP.first);
            assert(it != kf.mapPoints.end() &&
                "MapPoint has reference to Keyframe but Keyframe not to MapPoint");
            (void) it;
        }
    }

    // Test previousKfId and nextKfId id pointers, and that ids are unique.
    if (!mapDB.keyframes.empty()) {
        std::set<KfId> ids;
        KfId kfId = mapDB.keyframes.rbegin()->first;
        assert(kfId.v != -1);
        while (true) {
            assert(!ids.count(kfId));
            ids.insert(kfId);

            KfId next = mapDB.keyframes.at(kfId)->previousKfId;
            if (next.v == -1) break;
            kfId = next;
        }
        assert(kfId == mapDB.keyframes.begin()->first);

        ids.clear();
        while (true) {
            assert(!ids.count(kfId));
            ids.insert(kfId);

            KfId next = mapDB.keyframes.at(kfId)->nextKfId;
            if (next.v == -1) break;
            kfId = next;
        }
        assert(kfId == mapDB.keyframes.rbegin()->first);
    }
}

bool checkPositiveDepth(
    const Vector3d &positionW,
    const Matrix4d &poseCW
) {
    double z = (Eigen::Affine3d(poseCW) * positionW)(2);
    return z > 0;
}

bool checkTriangulationAngle(const vecVector3d &raysW, double minAngleDeg) {
    double cosMinAngle = std::cos(minAngleDeg * M_PI / 180.0);
    for (unsigned i = 0; i < raysW.size(); i++) {
        for (unsigned j = i + 1; j < raysW.size(); j++) {
            if (raysW[i].dot(raysW[j]) < cosMinAngle) {
                return true;
            }
        }
    }
    return false;
}

// (approx) focal length can be used as a proxy for "image size"
int getFocalLength(const Keyframe &kf) {
    return kf.shared->camera->getFocalLength();
}

bool checkReprojectionError(
    const Vector3d& pos,
    const Keyframe &kf,
    const StaticSettings &settings,
    KpId kpId,
    float relativeReprojectionErrorThreshold
) {
    Eigen::Vector2f reprojected = Eigen::Vector2f::Zero();
    if (!kf.reproject(pos, reprojected)) {
        return false;
    }

    const auto &kp = kf.shared->keyPoints[kpId.v];
    Vector2f point(kp.pt.x, kp.pt.y);

    // Try to come up with a suitable reprojection error threshold based
    // on the feature octave (in the image pyramid) and image resolution.
    // Not based on very hard science, replace if you find a better scheme
    const double relSigmaBase = getFocalLength(kf) * relativeReprojectionErrorThreshold;
    const auto REF_SCALE_FACTOR = settings.scaleFactors.size() / 2;
    const double sigma2 = settings.levelSigmaSq.at(kp.octave) / settings.levelSigmaSq.at(REF_SCALE_FACTOR) * relSigmaBase * relSigmaBase;
    return (reprojected - point).squaredNorm() <= CHI2_INV2D * sigma2;
}

void triangulateMapPoint(
    MapDB &mapDB,
    MapPoint &mapPoint,
    const StaticSettings &settings,
    TriangulationMethod method
) {
    const auto &parameters = settings.parameters.slam;
    const bool wasTriangulated = mapPoint.status != MapPointStatus::NOT_TRIANGULATED;
    mapPoint.status = MapPointStatus::NOT_TRIANGULATED;

    int observationCount = mapPoint.observations.size();
    if (observationCount < 2) {
        return;
    }

    vecVector3d raysW;
    bool depthTriangulated = false;
    for (const auto &kfIdKeypointId : mapPoint.observations) {
        const Keyframe &kf = *mapDB.keyframes.at(kfIdKeypointId.first);
        const auto &kp = kf.shared->keyPoints.at(kfIdKeypointId.second.v);
        float depth = kf.keyPointDepth.at(kfIdKeypointId.second.v);
        if (depth > 0 && !wasTriangulated) {
            mapPoint.position = depth * kf.cameraToWorldRotation() * kp.bearing + kf.cameraCenter();
            depthTriangulated = true;
            break;
        }
        raysW.push_back(kf.cameraToWorldRotation() * kp.bearing);
    }

    MapPointStatus statusIfOk = MapPointStatus::UNSURE;
    if (!depthTriangulated) {
        if (observationCount > 2 && checkTriangulationAngle(raysW, parameters.minTriangulationAngleMultipleObs)) {
            statusIfOk = MapPointStatus::TRIANGULATED;
        } else if (!checkTriangulationAngle(raysW, parameters.minTriangulationAngleTwoObs)) {
            return;
        }
    }

    Vector4d triangulatedPointH;
    bool triangulationResult;

    if (depthTriangulated) {
        triangulatedPointH = mapPoint.position.homogeneous();
        triangulationResult = true;
    }
    else if (method == TriangulationMethod::MIDPOINT) {
        // Not aligned because theia::TriangulateMidpoint() is not compatible.
        std::vector<Vector3d> raysW;
        std::vector<Vector3d> origins;

        for (const auto &kfIdKeypointId : mapPoint.observations) {
            Keyframe &kf = *mapDB.keyframes.at(kfIdKeypointId.first);
            origins.push_back(kf.cameraCenter());
            const auto &kp = kf.shared->keyPoints.at(kfIdKeypointId.second.v);
            raysW.push_back(kf.cameraToWorldRotation() * kp.bearing);
        }

        triangulationResult = theia::TriangulateMidpoint(origins, raysW, &triangulatedPointH);
    } else {
        std::vector<Matrix3x4d, Eigen::aligned_allocator<Matrix3x4d>> poses;
        std::vector<Vector2d, Eigen::aligned_allocator<Vector2d>> normalizedPoints;

        for (const auto &kfIdKeypointId : mapPoint.observations) {
            Keyframe &kf = *mapDB.keyframes.at(kfIdKeypointId.first);

            const auto &kp = kf.shared->keyPoints.at(kfIdKeypointId.second.v);
            Eigen::Vector2d normalizedPoint;
            if (kf.shared->camera->normalizePixel(Eigen::Vector2d(kp.pt.x, kp.pt.y), normalizedPoint)) {
                normalizedPoints.push_back(normalizedPoint);
                poses.push_back(kf.poseCW.topRows<3>());
            }
        }

        if (normalizedPoints.size() < 2) {
            triangulationResult = false;
        } else if (normalizedPoints.size() == 2) {
            triangulationResult = theia::Triangulate(
                    poses[0],
                    poses[1],
                    normalizedPoints[0],
                    normalizedPoints[1],
                    &triangulatedPointH);
        } else {
            triangulationResult
                    = theia::TriangulateNView(
                        // The aligned-allocator-vector should be compatible with the non-aligned-version,
                        // but C++ can't express this fact. However, it should be safe to reinterpret_cast
                        // to fix the missing alignemnt specifiers in Theia.
                        reinterpret_cast<const std::vector<Matrix3x4d>&>(poses),
                        reinterpret_cast<const std::vector<Vector2d>&>(normalizedPoints),
                        &triangulatedPointH);
        }
    }

    if (!triangulationResult) {
        return;
    }

    Vector3d triangulatedPoint = triangulatedPointH.head<3>() / triangulatedPointH(3);

    for (const auto &kfIdKeypointId : mapPoint.observations) {
        Keyframe &kf = *mapDB.keyframes.at(kfIdKeypointId.first);

        if (!checkPositiveDepth(triangulatedPoint, kf.poseCW)) {
            return;
        }

        if (!checkReprojectionError(
            triangulatedPoint,
            kf,
            settings,
            kfIdKeypointId.second,
            parameters.relativeReprojectionErrorThreshold)
        ) {
            return;
        }

        // TODO check scale. See openvslam two_view_triangulator.cc
    }

    mapPoint.position = triangulatedPoint;
    mapPoint.status = statusIfOk;
}

void triangulateMapPointFirstLastObs(
    MapDB &mapDB,
    MapPoint &mapPoint,
    const StaticSettings &settings
) {
    const auto &parameters = settings.parameters.slam;
    mapPoint.status = MapPointStatus::NOT_TRIANGULATED;

    int observationCount = mapPoint.observations.size();
    if (observationCount < 2) {
        return;
    }
    // First non-deleted observation
    Keyframe &firstKf = *mapDB.keyframes.at(mapPoint.getFirstObservation());
    Keyframe &lastKf = *mapDB.keyframes.at(mapPoint.getLastObservation());

    const auto &firstKp = firstKf.shared->keyPoints.at(mapPoint.observations.at(firstKf.id).v);
    const auto lastKpId = mapPoint.observations.at(lastKf.id);
    const auto &lastKp = lastKf.shared->keyPoints.at(lastKpId.v);

    float depth = lastKf.keyPointDepth.at(lastKpId.v);
    if (depth > 0) {
        mapPoint.position = depth * lastKf.cameraToWorldRotation() * lastKp.bearing + lastKf.cameraCenter();
    } else {
        if (settings.parameters.tracker.computeDenseStereoDepth) return; // skpping depth free points
        vecVector3d raysW;
        raysW.push_back(firstKf.cameraToWorldRotation() * firstKp.bearing);
        raysW.push_back(lastKf.cameraToWorldRotation() * lastKp.bearing);

        if (!checkTriangulationAngle(raysW, parameters.minTriangulationAngleTwoObs)) {
            return;
        }

        Vector2d firstNormalizedPoint, lastNormalizedPoint;

        if (!firstKf.shared->camera->normalizePixel(Eigen::Vector2d(firstKp.pt.x, firstKp.pt.y), firstNormalizedPoint) ||
            !lastKf.shared->camera->normalizePixel(Eigen::Vector2d(lastKp.pt.x, lastKp.pt.y), lastNormalizedPoint)) {
            return;
        }

        bool triangulationResult;
        Vector4d triangulatedPointH;
        triangulationResult = theia::Triangulate(
            firstKf.poseCW.topRows<3>(),
            lastKf.poseCW.topRows<3>(),
            firstNormalizedPoint,
            lastNormalizedPoint,
            &triangulatedPointH);

        if (!triangulationResult)
            return;

        mapPoint.position = triangulatedPointH.head<3>() / triangulatedPointH(3);
    }

    const auto checkObservation = [&](const KfId kfId) {
        KpId kpId = mapPoint.observations.at(kfId);
        const Keyframe &kf = *mapDB.keyframes.at(kfId);

        // if (!checkPositiveDepth(mapPoint.position, kf.poseCW)){
        //     return false;
        // }

        if (!checkReprojectionError(
            mapPoint.position,
            kf,
            settings,
            kpId,
            parameters.relativeReprojectionErrorThreshold)
        ) {
            return false;
        }

        return true;
    };

    int nNew = 0;
    for (const auto &it : mapPoint.observations) {
        if (checkObservation(it.first)) {
            assert(it.first.v != -1);
            nNew++;
        }
    }

    // Fail triangulation if fewer than two observations match (e.g. only first and last)
    if (nNew < 2) return;
    mapPoint.status = mapPoint.observations.size() > 2 ? MapPointStatus::TRIANGULATED : MapPointStatus::UNSURE;
    mapPoint.updateDescriptor(mapDB);
}

void publishMapForViewer(
    ViewerDataPublisher &dataPublisher,
    const WorkspaceBA *workspaceBA,
    const MapDB &mapDB,
    const odometry::ParametersSlam &parameters
) {
    if (mapDB.keyframes.empty()) return;
    const Keyframe &currentKeyframe = *mapDB.keyframes.rbegin()->second;

    std::set<MpId> visibleIds;
    for (MpId mpId : currentKeyframe.mapPoints) {
        if (mpId.v != -1) visibleIds.insert(mpId);
    }
    ViewerDataPublisher::MapPointVector mps;
    for (const auto &idMp : mapDB.mapPoints) {
        const MapPoint &mp = idMp.second;
        if (mp.status == MapPointStatus::NOT_TRIANGULATED) continue;
        mps.push_back(ViewerMapPoint {
            .position = mp.position.cast<float>(),
            .normal = mp.norm,
            .color = Eigen::Vector3f(mp.color[0], mp.color[1], mp.color[2]) / 255.0f,
            .status = int(mp.status),
            .localMap = workspaceBA && workspaceBA->localMpIds.count(mp.id) > 0,
            .nowVisible = visibleIds.count(mp.id) > 0
        });
    }

    std::map<KfId, size_t> inds;
    ViewerDataPublisher::KeyframeVector kfs;
    size_t i = 0;
    for (auto it = mapDB.keyframes.begin(); it != mapDB.keyframes.end(); ++it, ++i) {
        const Keyframe &kf = *it->second;
        inds.emplace(kf.id, i);

        const auto &adjacent = mapDB.adjacentKfIds;
        bool isAdjacent = std::find(adjacent.begin(), adjacent.end(), kf.id) != adjacent.end();
        kfs.push_back(ViewerKeyframe {
            .id = kf.id,
            .localMap = isAdjacent,
            .current = kf.id == currentKeyframe.id,
            .poseWC = kf.poseCW.inverse().cast<float>(),
            .origPoseWC = kf.origPoseCW.inverse().cast<float>(),
            .neighbors = {},
            .stereoPointCloud = kf.shared->stereoPointCloud,
            .stereoPointCloudColor = kf.shared->stereoPointCloudColor
        });
    }
    i = 0;
    for (auto it = mapDB.keyframes.begin(); it != mapDB.keyframes.end(); ++it, ++i) {
        // Convert KfIds into vector indices.
        std::vector<KfId> ids = it->second->getNeighbors(mapDB, parameters.minNeighbourCovisiblitities);
        kfs[i].neighbors.reserve(ids.size());
        std::transform(
            ids.begin(),
            ids.end(),
            back_inserter(kfs[i].neighbors),
            [&inds](KfId kfId) -> int { return inds.at(kfId); }
        );
    }

    // The data can be published at any time, but at the rendering side we
    // may want to know if the thing is actually new.
    double age = static_cast<double>(mapDB.lastKeyframeCandidateId().v);

    dataPublisher.setMap(mps, kfs, mapDB.loopStages, age);
}

void updatePointCloudRecording(
    float t,
    std::map<MpId, MapPointRecord> &mapPointRecords,
    const std::map<MpId, MapPoint> &mapPoints
) {
    for (const auto &it : mapPoints) {
        const MapPoint &mp = it.second;
        // Try to reduce spread of points by requiring more observations.
        if (mp.observations.size() < 4) continue;
        Eigen::Vector3f p = mp.position.cast<float>();

        if (!mapPointRecords.count(mp.id)) {
            mapPointRecords.insert({ mp.id, MapPointRecord(t, p, mp.norm, MapPointRecord::Type::SLAM) });
        }
        else if (mapPointRecords.at(mp.id).positions.back().p != p) {
            mapPointRecords.at(mp.id).positions.push_back(MapPointRecord::Position { .t = t, .p = p });
            mapPointRecords.at(mp.id).normal = mp.norm;
        }
    }

    // Record also removal of map points.
    Eigen::Vector3f p0 = Eigen::Vector3f::Zero();
    for (auto &it : mapPointRecords) {
        if (!it.second.removed && !mapPoints.count(it.first)) {
            it.second.removed = true;
            it.second.positions.push_back(MapPointRecord::Position { .t = t, .p = p0 });
        }
    }
}

Eigen::MatrixXd odometryPriorStrengths(KfId kfId1, KfId kfId2, const odometry::ParametersSlam &parameters, const slam::MapDB &mapDB) {
    double p = parameters.odometryPriorStrengthPosition;
    double r = parameters.odometryPriorStrengthRotation;
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();

    // The variance of sum of i.i.d. variances is the sum of the variances. The difference
    // between the KF id numbers is basically proportional to the time difference, so we
    // use that to scale the error covariance. Information is the inverse of the covariance
    // so we divide.
    //
    // If we are using uncertainty matrix, it is accmuluated when Keyframes are deleted, so
    // this scaling facor is not required.
    assert(kfId2.v > kfId1.v);

    const Keyframe *kf1 = mapDB.keyframes.at(kfId1).get();
    const Keyframe *kf2 = mapDB.keyframes.at(kfId2).get();

    double s = 0.26667 / (kf2->t - kf1->t);

    // Rotation
    if (parameters.odometryPriorFixed) {
        information.block(0, 0, 3, 3) *= s * r * r;
    // TODO: For now uncertainty matrix from odometry is identity * average uncertainty i.e. same as simple
    // } else if (parameters.odometryPriorSimpleUncertainty) {
    } else {
        // Scaling factor to average uncertainty roughly to 1.0 for individual keyframes
        information.block(0, 0, 3, 3) = r * r / 135000. * kf2->uncertainty.block(0, 0, 3, 3).inverse();
    }

    // Position
    if (parameters.odometryPriorFixed) {
        information.block(3, 3, 3, 3) *= s * p * p;
    } else if (parameters.odometryPriorSimpleUncertainty) {
        float meanUncertainty = (
            1. / kf2->uncertainty.row(0).norm()
            + 1. / kf2->uncertainty.row(1).norm()
            + 1. / kf2->uncertainty.row(2).norm()
        ) / 3.;
        // Scaling factor to average uncertainty roughly to 1.0 for individual keyframes
        information.block(3, 3, 3, 3) *= p * p / 5000. * meanUncertainty;
    } else {
        // Scaling factor to average uncertainty roughly to 1.0 for individual keyframes
        information.block(3, 3, 3, 3) = p * p / 5000. * kf2->uncertainty.block(0, 3, 3, 3).inverse();
    }
    return information;
}

MapDB loadMapDB(
    MapId mapId,
    BowIndex &bowIndex,
    const std::string &loadPath
) {
    util::TimeStats timeStats;

    MapDB mapDB;
    std::ifstream mapStream;
    mapStream.open(loadPath, std::ios::in | std::ios::binary);
    cereal::BinaryInputArchive iarchive(mapStream);
    {
        auto t = timeStats.time("deserialize map");
        iarchive(mapDB);
    }

    {
        auto t = timeStats.time("build bow index");
        for (auto &it : mapDB.keyframes) {
            Keyframe &keyframe = *it.second;
            bowIndex.transform(keyframe.shared->keyPoints, keyframe.shared->bowVec, keyframe.shared->bowFeatureVec);
            bowIndex.add(keyframe, mapId);
        }
    }

    {
        auto t = timeStats.time("build feature search");
        for (auto &it : mapDB.keyframes) {
            Keyframe &keyframe = *it.second;
            keyframe.shared->featureSearch = FeatureSearch::create(keyframe.shared->keyPoints);
        }
    }

    log_debug("%s", timeStats.previousTimings().c_str());
    return mapDB;
}

ViewerAtlasMap mapDBtoViewerAtlasMap(const MapDB &mapDB) {
    ViewerAtlasMap v;
    for (const auto &it : mapDB.keyframes) {
        v.keyframes.push_back(ViewerAtlasKeyframe {
            .id = it.second->id,
            .poseWC = it.second->poseCW.inverse().cast<float>(),
        });
    }
    for (const auto &it : mapDB.mapPoints) {
        v.mapPoints.push_back(ViewerAtlasMapPoint {
            .position = it.second.position.cast<float>(),
        });
    }
    return v;
}

void addKeyframeCommonInner(
    MapDB &mapDB,
    Keyframe &currentKeyframe,
    bool kfDecision,
    const StaticSettings &settings,
    WorkspaceBA *workspaceBAPtr,
    LoopCloser *loopCloser,
    BowIndex *bowIndex,
    CommandQueue *commands,
    ViewerDataPublisher *dataPublisher)
{
    const odometry::ParametersSlam &ps = settings.parameters.slam;

    // Add accumulated uncertainty from discarded keyframes since previous keyframe
    currentKeyframe.uncertainty += mapDB.discardedUncertainty;

    const bool isBackend = loopCloser != nullptr;
    matchTrackedFeatures(currentKeyframe, mapDB, settings);

    constexpr int minCovisibilities = 5;
    std::vector<KfId> adjacentKfIds = computeAdjacentKeyframes(
        currentKeyframe,
        minCovisibilities,
        ps.adjacentSpaceSize,
        mapDB,
        settings,
        true
    );
    mapDB.adjacentKfIds = adjacentKfIds;

    if (kfDecision && isBackend) {
        matchLocalMapPoints(currentKeyframe, adjacentKfIds, mapDB, settings, dataPublisher);
    } else {
        timer(slam::TIME_STATS, "poseBundleAdjust");
        if (ps.nonKeyFramePoseAdjustment) {
            if (poseBundleAdjust(currentKeyframe, mapDB, settings)) {
                if (isBackend) workspaceBAPtr->baStats.update(BaStats::Ba::POSE);
            }
        }
        return;
    }

    if (!isBackend) return;

    assert(workspaceBAPtr && loopCloser && bowIndex);
    auto &workspaceBA = *workspaceBAPtr;

    createNewMapPoints(currentKeyframe, adjacentKfIds, mapDB, settings, dataPublisher);
    deduplicateMapPoints(currentKeyframe, adjacentKfIds, mapDB, settings);

    // Update mapPoints
    for (MpId mpId : currentKeyframe.mapPoints) {
        if (mpId.v == -1) continue;

        MapPoint &mp = mapDB.mapPoints.at(mpId);
        if (mp.status == MapPointStatus::NOT_TRIANGULATED || mp.status == MapPointStatus::BAD)
            continue;

        mp.updateDescriptor(mapDB);
        mp.updateDistanceAndNorm(mapDB, settings);

        if (mp.observations.size() >= ps.minObservationsForBA) {
            mp.status = MapPointStatus::TRIANGULATED;
        } else {
            mp.status = MapPointStatus::UNSURE;
        }
    }

    if (ps.applyLocalBundleAdjustment) {
        timer(slam::TIME_STATS, "localBundleAdjust");
        localBundleAdjust(currentKeyframe, workspaceBA, mapDB, ps.localBAProblemSize, settings);

        // Retriangulate non-BA'd points in the current KF after BA,
        // which should help reprojecting them to the next KF
        for (MpId mpId : currentKeyframe.mapPoints) {
            if (mpId.v == -1) continue;
            MapPoint &mp = mapDB.mapPoints.at(mpId);
            if (mp.status != MapPointStatus::TRIANGULATED || mp.observations.size() >= 2) {
                triangulateMapPoint(mapDB, mp, settings);
                // may change status but that will be checked later
            }
        }
    }

    cullMapPoints(currentKeyframe, mapDB, ps);
    cullKeyframes(adjacentKfIds, mapDB, *bowIndex, ps);

    {
        bowIndex->add(currentKeyframe, CURRENT_MAP_ID);
        bool closedLoop;
        {
            timer(slam::TIME_STATS, "Loop closing");
            closedLoop = loopCloser->tryLoopClosure(currentKeyframe, adjacentKfIds);
        }

        if (closedLoop){
            timer(slam::TIME_STATS, "Loop closing BA");
            // The algorithm seems to work well even if no BA is done after a loop closure.
            if (ps.globalBAAfterLoop) {
                globalBundleAdjust(currentKeyframe.id, mapDB, settings);
                workspaceBA.baStats.update(BaStats::Ba::GLOBAL);
            } else {
                localBundleAdjust(currentKeyframe, workspaceBA, mapDB, ps.loopClosureLocalBAProblemSize, settings);
            }

            if (dataPublisher && commands && commands->getStepMode() == CommandQueue::StepMode::SLAM) {
                publishMapForViewer(*dataPublisher, &workspaceBA, mapDB, ps);
                log_debug("Bundle adjustment after loop closure done");
                commands->waitForAnyKey();
            }
        }
    }

    if (!settings.parameters.slam.pointCloudSavePath.empty()) {
        updatePointCloudRecording(currentKeyframe.t, mapDB.mapPointRecords, mapDB.mapPoints);
    }

    if (dataPublisher) {
        publishMapForViewer(*dataPublisher, &workspaceBA, mapDB, ps);
    }
}

static KfId addKeyframeCommonOuter(
    MapDB &mapDB,
    std::unique_ptr<Keyframe> keyframePtr,
    bool keyFrameDecision,
    const MapperInput mapperInput,
    const StaticSettings &settings,
    WorkspaceBA *workspaceBA,
    LoopCloser *loopCloser,
    OrbExtractor *orbExtractor,
    BowIndex *bowIndex,
    CommandQueue *commands,
    ViewerDataPublisher *dataPublisher,
    Eigen::Matrix4d &resultPose,
    Slam::Result::PointCloud *resultPointCloud)
{
    const auto &poseTrail = mapperInput.poseTrail;
    if (settings.parameters.slam.useFullPoseTrail) {
        // 0 is the new keyframe, 1 is the previous keyframe etc. Update all keyframes that exist
        for (unsigned i = 1; i < poseTrail.size(); i++) {
            const auto &pose = poseTrail.at(i);
            KfId kfId(pose.frameNumber);
            if (mapDB.keyframes.count(kfId)) {
                Keyframe &kf = *mapDB.keyframes.at(kfId);
                // double moveDist = (mapperInput.poseTrail[i].pose.block<3, 1>(0, 3)
                    // - kf.origPoseCW.block<3, 1>(0, 3)).norm();
                // log_debug("Updated pose for  KfId %d, delay %u, move dist %f", kf.id.v, i, moveDist);
                kf.origPoseCW = pose.pose;
                // TODO: When updating uncertainty it may have been accumulated from deleted keyframes. To
                // properly account for this, uncertainty from other keyframes should be kept separate and
                // summed here. In some cases the posetrail might also contain the uncretainty over deleted
                // keyframes.
                // if (i < mapperInput.poseTrail.size() - 1) {
                    // kf.uncertainty = mapperInput.poseTrail[i].uncertainty;
                // }
            }
            // even if some pose trail frames were missing from the graph,
            // it should be safe to update the ones before that
        }

        // Remove all keyframes which have been removed from odometry pose trail
        KfId lastFrame = KfId(poseTrail.back().frameNumber);
        auto  *kf = mapDB.latestKeyframe();
        while (kf && kf->nextKfId.v != -1 && kf->id <= lastFrame) {
            const int frameNumber = kf->id.v;
            auto it = std::find_if(poseTrail.begin(), poseTrail.end(),
                [frameNumber] (const slam::Pose &p) { return p.frameNumber == frameNumber; } );
            kf = mapDB.keyframes.at(kf->nextKfId).get();
            if (it == poseTrail.end()) {
                removeKeyframe(KfId(frameNumber), mapDB, bowIndex);
            }
        }
    }

    const bool isBackend = orbExtractor != nullptr; // hacky-ish
    // also hacky, basically disables the "shared" part and makes a separate
    // copy for the front and backend
    keyframePtr->shared = keyframePtr->shared->clone();
    if (keyFrameDecision && isBackend) {
        auto &kf = *keyframePtr;
        kf.addFullFeatures(mapperInput, *orbExtractor);
        timer(slam::TIME_STATS, "Bow index transform");
        bowIndex->transform(kf.shared->keyPoints, kf.shared->bowVec, kf.shared->bowFeatureVec);
    } else {
        keyframePtr->addTrackerFeatures(mapperInput);
    }

    auto currentKeyframe = mapDB.insertNewKeyframeCandidate(
        std::move(keyframePtr),
        keyFrameDecision,
        poseTrail,
        settings.parameters.slam);

    addKeyframeCommonInner(
        mapDB,
        *currentKeyframe,
        keyFrameDecision,
        settings,
        workspaceBA,
        loopCloser,
        bowIndex,
        commands,
        dataPublisher);

    mapDB.updatePrevPose(*currentKeyframe, keyFrameDecision, poseTrail, settings.parameters);
    KfId currentFrameNumber = currentKeyframe->id;
    resultPose = currentKeyframe->poseCW;
    if (resultPointCloud != nullptr) {
        setPointCloudOutput(mapDB, *currentKeyframe, *resultPointCloud);
    }

    if (!keyFrameDecision) {
        // Update discardedUncertainty with added uncertainty from this frame
        mapDB.discardedUncertainty = currentKeyframe->uncertainty;
        removeKeyframe(currentKeyframe->id, mapDB, bowIndex);
    } else {
        // Reset discarded uncertainty to zero
        mapDB.discardedUncertainty = Eigen::MatrixXd::Zero(3, 6);
    }

    return currentFrameNumber;
}

void addKeyframeFrontend(
    MapDB &mapDB,
    std::unique_ptr<Keyframe> keyframePtr,
    bool kfDecision,
    const MapperInput &mapperInput,
    const StaticSettings &settings,
    Eigen::Matrix4d &resultPose,
    Slam::Result::PointCloud *resultPointCloud
) {
    addKeyframeCommonOuter(mapDB, std::move(keyframePtr), kfDecision, mapperInput,
        settings, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        resultPose, resultPointCloud);
}

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
    Slam::Result::PointCloud *resultPointCloud)
{
    return addKeyframeCommonOuter(
        mapDB,
        std::move(keyframePtr),
        keyFrameDecision,
        mapperInput,
        settings,
        &workspaceBA,
        &loopCloser,
        &orbExtractor,
        &bowIndex,
        commands,
        dataPublisher,
        resultPose,
        resultPointCloud);
}

} // slam
