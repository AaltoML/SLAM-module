#include "loop_closer.hpp"

#include <g2o/types/sim3/sim3.h>

#include "../odometry/parameters.hpp"
#include "../util/logging.hpp"

#include "keyframe.hpp"
#include "keyframe_matcher.hpp"
#include "mapper_helpers.hpp"
#include "loop_ransac.hpp"
#include "../api/slam.hpp"
#include "viewer_data_publisher.hpp"
#include "bow_index.hpp"
#include "map_point.hpp"
#include "optimize_transform.hpp"
#include "relocation.hpp"

#include "loop_closer_stats.hpp"

namespace slam {
namespace {

using PoseMap = std::map<
    KfId,
    Eigen::Matrix4d,
    std::less<KfId>,
    Eigen::aligned_allocator<std::pair<const KfId, Eigen::Matrix4d> >
>;

struct LoopClosure {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    KfId candidateKfId;

    g2o::Sim3 candToCurr;

    std::vector<std::pair<KpId, KpId>> keyPointMatches;
    std::vector<std::pair<MpId, MpId>> mapPointMatches;
};

struct ConsistentSet {
    std::set<KfId> kfIds;
    unsigned int observations;

    bool idsInCommon(const std::set<KfId> &otherSet) const {
        return std::any_of(otherSet.begin(), otherSet.end(), [this](KfId id) {
            return kfIds.count(id);
        });
    }
};

g2o::Sim3 se3ToSim3(const Eigen::Matrix4d &se3) {
    return g2o::Sim3(
        se3.topLeftCorner<3,3>(),
        se3.block<3,1>(0,3),
        1
    );
}

Eigen::Matrix4d sim3ToSe3(const g2o::Sim3 &sim3) {
    Eigen::Matrix4d se3 = Eigen::Matrix4d::Identity();
    se3.topLeftCorner<3,3>() = sim3.rotation().toRotationMatrix();
    se3.block<3,1>(0, 3) = sim3.translation();
    return se3;
}

// 0.0 lambda gives T0, 1.0 lambda gives T1.
g2o::Sim3 interpolateSim3(const g2o::Sim3 &T0, const g2o::Sim3 &T1, double lambda) {
    assert(lambda >= 0.0 && lambda <= 1.0);
    return g2o::Sim3(
        T0.rotation().slerp(lambda, T1.rotation()),
        T0.translation() + lambda * (T1.translation() - T0.translation()),
        T0.scale() + lambda * (T1.scale() - T0.scale())
    );
}

// Stored KfIds can become invalid if they are culled. Also the next valid
// KfId is typically not "current + 1" KfIds are based on tracker frame
// numbers. This finds the next valid id.
KfId nextValidKfId(KfId kfId, const MapDB &mapDB) {
    KfId maxKfId = mapDB.keyframes.rbegin()->first;
    while (kfId <= maxKfId) {
        if (mapDB.keyframes.count(kfId)) return kfId;
        kfId = KfId(kfId.v + 1);
    }
    // Not found.
    return KfId(-1);
}

} // namespace

class LoopCloserImplementation : public LoopCloser {
private:
    const StaticSettings &settings;

    BowIndex &bowIndex;
    LoopCloserStats stats;
    MapDB &mapDB;
    const Atlas &atlas;

    KfId prevLoopClosureKfId = KfId(-1);
    double previousClosureT;

    /**
     * Optional reference to a viewer data publisher. Not owned by this object.
     */
    ViewerDataPublisher *dataPublisher = nullptr;
    CommandQueue *commands = nullptr;

public:
    LoopCloserImplementation(
        const StaticSettings &settings,
        BowIndex &bowIndex,
        MapDB &mapDB,
        const Atlas &atlas
    ) :
        settings(settings),
        bowIndex(bowIndex),
        stats(settings.parameters.slam),
        mapDB(mapDB),
        atlas(atlas),
        previousClosureT(-1.0)
    {}

    bool tryLoopClosure(
        Keyframe &currentKf,
        const std::vector<KfId> &adjacent
    ) {
        const auto &parameters = settings.parameters.slam;

        std::vector<BowSimilar> candidates = bowIndex.getBowSimilar(mapDB, atlas, currentKf);

        int heavyComputations = 0;
        mapDB.loopStages.clear();
        std::vector<LoopClosure> loopClosureCandidates;
        for (const auto &candidate : candidates) {
            mapDB.loopStages[candidate.mapKf] = LoopStage::BOW_MATCH;

            if (candidate.mapKf.mapId != CURRENT_MAP_ID) {
                tryRelocation(currentKf.id, candidate.mapKf, mapDB, atlas, parameters, settings);
                continue;
            }

            KfId kfId = candidate.mapKf.kfId;

            // Number of candidates given by `getBowSimilar()` varies greatly, so we process them
            // sorted by score and stop after too many candidates.
            if (heavyComputations > 10) {
                break;
            }

            stats.newLoop();

            Keyframe &candidateKf = *mapDB.keyframes.at(kfId);

            // Do not correct into early parts of the map because currently the odometry has
            // a tendency to estimate spatial scale badly in the beginning.
            // if (candidateKf.t - firstKfT < 5.0) {
            //     stats.update(LoopCloserStats::Loop::EARLY_MAP_IGNORED);
            //     continue;
            // }

            // Do not make very short corrections (either too recent candidateKf or previous loop closure).
            // Do not correct beyond previous correction.
            double correctionLength = currentKf.t - std::max(candidateKf.t, previousClosureT);
            if (correctionLength < 5.0) {
                stats.update(LoopCloserStats::Loop::TOO_CLOSE_TIME);
                continue;
            }

            if ((currentKf.t - candidateKf.t) < 2.15) {
                stats.update(LoopCloserStats::Loop::TOO_CLOSE_TIME);
                continue;
            }

            // Do a more strict version of the later UNNECESSARY test by using the keyframe
            // positions rather than the actual correction distance. This often saves us from
            // the rather heavy computations below needed to get the correction vector.
            bool isAdjacent = std::find(adjacent.begin(), adjacent.end(), kfId) != adjacent.end();
            double kfDistance = (candidateKf.cameraCenter() - currentKf.cameraCenter()).norm();
            constexpr double unnecessarilyCloseDistance = 0.75;
            constexpr double unnecessarilyCloseDistanceKf = 0.5;
            if (isAdjacent && kfDistance < unnecessarilyCloseDistanceKf) {
                stats.update(LoopCloserStats::Loop::UNNECESSARY_EARLY);
                continue;
            }

            // We are now past the fast rejections.
            heavyComputations++;

            mapDB.loopStages[candidate.mapKf] = LoopStage::QUICK_TESTS;

            std::vector<int> matchedFeatureIds;
            matchForLoopClosures(
                currentKf,
                candidateKf,
                mapDB,
                mapDB,
                matchedFeatureIds,
                parameters);

            std::vector<std::pair<MpId, MpId>> matches;
            for (unsigned i = 0; i < matchedFeatureIds.size(); ++i) {
                const int kfIdx2 = matchedFeatureIds.at(i);
                if (kfIdx2 >= 0) {
                    const MpId mpId1 = currentKf.mapPoints.at(i);
                    const MpId mpId2 = candidateKf.mapPoints.at(kfIdx2);
                    if (mpId1.v != -1 && mpId2.v != -1 && mpId1 != mpId2) {
                        matches.emplace_back(mpId1, mpId2);
                    }
                }
            }

            if (matches.size() < parameters.minLoopClosureFeatureMatches) {
                stats.update(LoopCloserStats::Loop::TOO_FEW_FEATURE_MATCHES);
                continue;
            }

            // Estimate transformation candidateKf->current
            LoopRansac loopRansac(
                currentKf,
                candidateKf,
                matches,
                mapDB,
                mapDB,
                settings
            );

            loopRansac.ransacSolve(parameters.loopClosureRansacIterations, LoopRansac::DoF::SIM3);
            if (!loopRansac.solutionOk) {
                stats.update(LoopCloserStats::Loop::RANSAC_FAILED);
                continue;
            }

            mapDB.loopStages[candidate.mapKf] = LoopStage::MAP_POINT_MATCHES;

            std::vector<std::pair<MpId, MpId>> ransacInlierMatches;
            assert(loopRansac.bestInliers.size() == matches.size());
            for (unsigned i = 0; i < loopRansac.bestInliers.size(); i++) {
                if (loopRansac.bestInliers.at(i))
                    ransacInlierMatches.push_back(matches.at(i));
            }

            Eigen::Matrix4d transform12 = Eigen::Matrix4d::Identity();
            transform12.topLeftCorner<3,3>() = loopRansac.bestScale12 * loopRansac.bestR12;
            transform12.block<3,1>(0,3) = loopRansac.bestT12;

            // Match further map points
            matchMapPointsSim3(
                    currentKf,
                    candidateKf,
                    transform12,
                    mapDB,
                    ransacInlierMatches,
                    settings);

            std::vector<std::pair<KpId, KpId>> keyPointMatches;
            for (const auto &p : ransacInlierMatches) {
                keyPointMatches.emplace_back(
                    mapDB.mapPoints.at(p.first).observations.at(currentKf.id),
                    mapDB.mapPoints.at(p.second).observations.at(candidateKf.id)
                );
            }

            // Visualize keypoint matches (even if the loop closure is ultimately rejected).
            if (dataPublisher && dataPublisher->getParameters().visualizeLoopOrbMatching
                    && !currentKf.shared->imgDbg.empty() && !candidateKf.shared->imgDbg.empty()) {
                dataPublisher->showMatches(currentKf, candidateKf, keyPointMatches, MatchType::LOOP);
            }

            // Optimize transformation using all inliers
            g2o::Sim3 g2oSim3CandToCurr(
                loopRansac.bestR12,
                loopRansac.bestT12,
                loopRansac.bestScale12);
            OptimizeSim3Transform(currentKf, candidateKf, ransacInlierMatches, mapDB, g2oSim3CandToCurr, settings);

            // Compute new pose for current keyframe.
            Eigen::Matrix4d updatedPose = sim3ToSe3(g2oSim3CandToCurr * se3ToSim3(candidateKf.poseCW));

            // To continue considering the loop closure candidate, require either that the
            // candidate keyframe is disconnected from the current map, or that the correction
            // distance is considerable (possibly indicating some problem with local tracking).
            double correctionDistance = (worldToCameraMatrixCameraCenter(currentKf.poseCW) - worldToCameraMatrixCameraCenter(updatedPose)).norm();
            if (isAdjacent && correctionDistance < unnecessarilyCloseDistance) {
                stats.update(LoopCloserStats::Loop::UNNECESSARY);
                continue;
            }

            // Sometimes a keyframe just outside the adjacency set is matched for loop
            // closure because the scene can be recognized from a surprising distance.
            constexpr double distanceFactor = 1.0;
            double distanceFromCandidate = (worldToCameraMatrixCameraCenter(candidateKf.poseCW) - worldToCameraMatrixCameraCenter(updatedPose)).norm();
            if (distanceFromCandidate > distanceFactor * correctionDistance) {
                stats.update(LoopCloserStats::Loop::UNNECESSARY);
                continue;
            }

            // constexpr double maxSim3translation = 10.0;
            // if (translationCorrection.norm() > maxSim3translation) {
            //     stats.update(LoopCloserStats::Loop::SUSPICIOUS);
            //     continue;
            // }

            Eigen::Matrix4d U = sim3ToSe3(g2oSim3CandToCurr);
            Eigen::Matrix3d R = (U * candidateKf.poseCW).inverse().topLeftCorner<3, 3>().transpose() * currentKf.poseCW.inverse().topLeftCorner<3, 3>();
            Eigen::AngleAxisd aa(R);
            double angleChange = aa.angle();

            // TODO: It would be better to measure shortest distance/time in a graph where loop closures connect nodes, since those fix drift
            double distanceTraveled = 0.0;
            KfId curr = currentKf.id;
            while(curr != candidateKf.id) {
                Keyframe& kf1 = *mapDB.keyframes.at(curr);
                curr = kf1.previousKfId;
                Keyframe& kf2 = *mapDB.keyframes.at(curr);
                distanceTraveled += (kf1.cameraCenter() - kf2.cameraCenter()).norm();
            }
            double timeBetweenKf = currentKf.t - candidateKf.t;
            if (correctionDistance / timeBetweenKf > parameters.maximumDriftMetersPerSecond
                || correctionDistance / distanceTraveled > parameters.maximumDriftMetersPerTraveled) {
                // log_debug("Drift between candidate pose and current pose (%.5f m, %.5f m/s, %.5f m/m) exceeds maximum drift "
                //     "(%.5f m/s, %.5f m/m) and loop closure was rejected.",
                //     correctionDistance, correctionDistance / timeBetweenKf, correctionDistance / distanceTraveled,
                //     parameters.maximumDriftMetersPerSecond, parameters.maximumDriftMetersPerTraveled);
                stats.update(LoopCloserStats::Loop::TOO_LARGE_POSITION_DRIFT);
                continue;
            }
            if (angleChange / timeBetweenKf > parameters.maximumDriftRadiansPerSecond
                || angleChange / distanceTraveled > parameters.maximumDriftRadiansPerTraveled) {
                // log_debug("Drift between candidate pose and current pose (%.5f degrees, %.5f degrees/s, %.5f degrees/m) exceeds maximum drift "
                //     "(%.5f degrees/s, %.5f degrees/m) and loop closure was rejected.",
                //     angleChange * 2. * M_PI, angleChange / timeBetweenKf * 2. * M_PI, angleChange / distanceTraveled * 2. * M_PI,
                //     parameters.maximumDriftRadiansPerSecond * 2. * M_PI, parameters.maximumDriftRadiansPerTraveled * 2. * M_PI);
                stats.update(LoopCloserStats::Loop::TOO_LARGE_ANGLE_DRIFT);
                continue;
            }

            if (dataPublisher) {
                dataPublisher->addLoopClosure(ViewerLoopClosure {
                    .currentPose = currentKf.poseCW.cast<float>().inverse(),
                    .candidatePose = candidateKf.poseCW.cast<float>().inverse(),
                    .updatedPose = updatedPose.cast<float>().inverse(),
                });
            }

            loopClosureCandidates.push_back(LoopClosure {
                .candidateKfId = candidateKf.id,
                .candToCurr = g2oSim3CandToCurr,
                .keyPointMatches = std::move(keyPointMatches),
                .mapPointMatches = std::move(ransacInlierMatches)
            });

            mapDB.loopStages[candidate.mapKf] = LoopStage::ACCEPTED;
            stats.update(LoopCloserStats::Loop::OK);
        }
        stats.finishFrame();

        // Return only here so that the above visualizations work.
        if (!parameters.applyLoopClosures) {
            return false;
        }

        // Sort loop closures, most recent first.
        auto idCmp = [](const LoopClosure &a, const LoopClosure &b) {
            return a.candidateKfId.v > b.candidateKfId.v;
        };
        std::sort(loopClosureCandidates.begin(), loopClosureCandidates.end(), idCmp);

        for (const LoopClosure &loopClosure : loopClosureCandidates) {
            correctLoop(currentKf, loopClosure);
            prevLoopClosureKfId = currentKf.id;
            return true;
        }

        return false;
    }

    void correctLoop(Keyframe &currentKf, const LoopClosure &loopClosure) {
        const auto &parameters = settings.parameters.slam;
        if (dataPublisher && commands && commands->getStepMode() == CommandQueue::StepMode::SLAM) {
            publishMapForViewer(*dataPublisher, {}, mapDB, parameters);
            log_debug("Starting loop closure.");
            commands->waitForAnyKey();
        }
        // logMapPointMatchDiff(loopClosure.mapPointMatches, true);

        const Keyframe &candidateKf = *mapDB.keyframes.at(loopClosure.candidateKfId);

        // Correct up to previous loop closure point or the loop closure candidate.
        KfId firstKfId = mapDB.keyframes.begin()->first;
        if (prevLoopClosureKfId.v >= 0) {
            prevLoopClosureKfId = nextValidKfId(prevLoopClosureKfId, mapDB);
        }
        KfId correctionStartKfId = std::max(firstKfId, std::max(prevLoopClosureKfId, candidateKf.id));

        PoseMap prevPoses;
        for (auto kfIt = mapDB.keyframes.rbegin(); kfIt != mapDB.keyframes.rend(); kfIt++) {
            prevPoses.emplace(kfIt->first, kfIt->second->poseCW);
        }

        // Transformation to change the current keyframe to its new pose.
        g2o::Sim3 T0; // Identity.
        g2o::Sim3 T = se3ToSim3(currentKf.poseCW).inverse() * loopClosure.candToCurr * se3ToSim3(candidateKf.poseCW);

        // TODO Try setting rotation parts of `T` and/or `Tl` to identity. Note that it
        // requires modifying the translation part also.

        std::vector<KfId> rigidlyTransformedKfIds;
        if (parameters.loopClosureRigidTransform) {
            // This tends to cause problems when the neighborhood contains parts from previous loops.
            // Probably only the last "island" should be used.
            rigidlyTransformedKfIds = currentKf.getNeighbors(mapDB, parameters.minNeighbourCovisiblitities);
        }
        rigidlyTransformedKfIds.push_back(currentKf.id);

        std::map<MpId, KfId> localMapPoints;

        // Correct the whole rigid set using the same transformation T.
        for (unsigned i = 0; i < rigidlyTransformedKfIds.size(); i++) {
            Keyframe &kf = *mapDB.keyframes.at(rigidlyTransformedKfIds.at(i));
            if (kf.id < correctionStartKfId) {
                continue;
            }

            kf.poseCW = sim3ToSe3(se3ToSim3(kf.poseCW) * T);

            for (MpId mpId : kf.mapPoints) {
                if (mpId.v != -1 && !localMapPoints.count(mpId)) {
                    localMapPoints.emplace(mpId, kf.id);
                }
            }
        }

        if (dataPublisher && commands && commands->getStepMode() == CommandQueue::StepMode::SLAM) {
            publishMapForViewer(*dataPublisher, {}, mapDB, parameters);
            log_debug("After rigid transform.");
            commands->waitForAnyKey();
        }

        // Correct poses linearly to remove/slighten the discontinuity between the
        // rigidly moved part and rest, to help following non-linear optimization / BA.
        // This code is more accurate with parameters.loopClosureRigidTransform=false.
        double t0 = mapDB.keyframes.at(correctionStartKfId)->t;
        double t1 = currentKf.t;
        for (auto kfIt = mapDB.keyframes.rbegin(); kfIt != mapDB.keyframes.rend(); kfIt++) {
            Keyframe &kf = *kfIt->second;
            if (kf.id < correctionStartKfId) {
                break;
            }
            if (std::find(rigidlyTransformedKfIds.begin(), rigidlyTransformedKfIds.end(), kf.id) != rigidlyTransformedKfIds.end()) {
                continue;
            }

            double t = kf.t;
            assert(t >= t0 && t <= t1);
            double lambda = (t - t0) / (t1 - t0);
            g2o::Sim3 Tl = interpolateSim3(T0, T, lambda);
            // TODO Would this be more correct?
            // g2o::Sim3 Tl = interpolateSim3(T0, T.inverse(), lambda).inverse();

            kf.poseCW = sim3ToSe3(se3ToSim3(kf.poseCW) * Tl);

            for (MpId mpId : kf.mapPoints) {
                if (mpId.v != -1 && !localMapPoints.count(mpId)) {
                    localMapPoints.emplace(mpId, kf.id);
                }
            }
        }

        mapDB.loopClosureEdges.push_back(LoopClosureEdge {
            .kfId1 = candidateKf.id,
            .kfId2 = currentKf.id,
            .poseDiff = candidateKf.poseCW * currentKf.poseCW.inverse(),
        });

        if (dataPublisher && commands && commands->getStepMode() == CommandQueue::StepMode::SLAM) {
            publishMapForViewer(*dataPublisher, {}, mapDB, parameters);
            log_debug("After linear correction.");
            commands->waitForAnyKey();
        }

        for (auto kfIt = mapDB.keyframes.rbegin(); kfIt != mapDB.keyframes.rend(); kfIt++) {
            Keyframe &kf = *kfIt->second;
            if (kf.id < correctionStartKfId) {
                break;
            }
        }

        // Update map point poses relative to keyframes.
        for (auto &p : mapDB.mapPoints) {
            MapPoint &mp = p.second;

            if (!localMapPoints.count(mp.id)) {
                continue;
            }
            KfId refKf = localMapPoints.at(mp.id);

            g2o::Sim3 correctedCW = se3ToSim3(mapDB.keyframes.at(refKf)->poseCW);
            g2o::Sim3 previousCW = se3ToSim3(prevPoses.at(refKf));

            mp.position = (correctedCW.inverse() * previousCW).map(mp.position.eval());
            mp.updateDescriptor(mapDB);
            mp.updateDistanceAndNorm(mapDB, settings);
        }

        // Triangulate map points.
        int all = 0, changed = 0, failed = 0;
        for (auto &p : mapDB.mapPoints) {
            MapPoint &mp = p.second;
            Eigen::Vector3d position = mp.position;
            MapPointStatus status = mp.status;
            triangulateMapPoint(mapDB, mp, settings);

            if (position != mp.position) {
                changed++;
            }
            if (status == MapPointStatus::TRIANGULATED && mp.status != MapPointStatus::TRIANGULATED) {
                failed++;
            }
            all++;
        }
        (void)changed, (void)all, (void)failed;
        // log_debug( "Retriangulation moved %d/%d map points. Failed: %d/%d",
        //     changed, all, failed, all);

        assert(loopClosure.mapPointMatches.size() == loopClosure.keyPointMatches.size());
        // logMapPointMatchDiff(loopClosure.mapPointMatches, false);

        // Merge moved mappoints
        std::set<MpId> merged;
        for (const auto &mpMatch : loopClosure.mapPointMatches) {
            if (mpMatch.first == mpMatch.second) {
                continue;
            }
            if (merged.count(mpMatch.first) || merged.count(mpMatch.second)) {
                // This seems very rare and not worth implementing a map so that chained merges
                // would work properly.
                continue;
            }
            merged.insert(mpMatch.first);

            // TODO check distance before merging?
            mapDB.mapPoints.at(mpMatch.first).replaceWith(mapDB, mapDB.mapPoints.at(mpMatch.second));
        }

        // Might be useful to run for the current island because depending on a parameter,
        // `rigidlyTransformedKfIds` is usually just the current keyframe.
        searchAndDeduplicate(candidateKf, rigidlyTransformedKfIds);

        if (dataPublisher && commands && commands->getStepMode() == CommandQueue::StepMode::SLAM) {
            publishMapForViewer(*dataPublisher, {}, mapDB, parameters);
            log_debug("After map point manipulation.");
            commands->waitForAnyKey();
        }

        previousClosureT = currentKf.t;

        log_debug("Loop corrected [%d -> %d]", currentKf.id.v, loopClosure.candidateKfId.v);
    }

    /*
     * Find matches between MapPoints in the neighbourhood of #candidateKf and
     * keyframes in #rigidlyTransformedKfIds
     */
    void searchAndDeduplicate(const Keyframe &candidateKf, std::vector<KfId> rigidlyTransformedKfIds) {
        const auto &parameters = settings.parameters.slam;
        std::set<MpId> loopMapPoints;
        // Instead of getNeighbors(), this should perhaps use adjacent keyframes of candidate
        // keyframe prior to merging the map.
        for (KfId kfId : candidateKf.getNeighbors(mapDB, parameters.minNeighbourCovisiblitities, false)) {
            Keyframe &kf = *mapDB.keyframes.at(kfId);
            for (MpId mpId : kf.mapPoints) {
                if (mpId.v != -1) {
                    const auto &mp = mapDB.mapPoints.at(mpId);
                    if (mp.status == MapPointStatus::BAD
                        || mp.status == MapPointStatus::NOT_TRIANGULATED) {
                        continue;
                    }
                    loopMapPoints.insert(mpId);
                }
            }
        }

        int fused = 0;
        for (const KfId &kfId : rigidlyTransformedKfIds) {
            Keyframe &kf = *mapDB.keyframes.at(kfId);
            fused += replaceDuplication(kf, loopMapPoints, 4, mapDB, settings);
        }
    }

    void logMapPointMatchDiff(
        const std::vector<std::pair<MpId, MpId>> &mapPointMatches,
        bool before
    ) {
        double diff1 = 0.0;
        for (const auto &mpMatch : mapPointMatches) {
            assert(mapDB.mapPoints.count(mpMatch.first));
            assert(mapDB.mapPoints.count(mpMatch.second));
            diff1 += (mapDB.mapPoints.at(mpMatch.first).position
                      - mapDB.mapPoints.at(mpMatch.second).position)
                         .norm();
        }
        log_debug(
            "Local map correction %s, diff: %f",
            before ? "before" : "after",
            diff1 / static_cast<double>(mapPointMatches.size()));
    }

    double getTotalReprojectionError() {
        double totalErr = 0;
        for (const auto &p : mapDB.keyframes) {
            auto &kf = *p.second;
            for (unsigned i = 0; i < kf.mapPoints.size(); i++) {
                MpId mpId = kf.mapPoints.at(i);
                if (mpId.v == -1) continue;
                auto mp = mapDB.mapPoints.at(mpId);
                Eigen::Vector2f reproj;
                kf.reproject(mp.position, reproj);

                auto kp = kf.shared->keyPoints.at(i);
                Eigen::Vector2f corr(kp.pt.x, kp.pt.y);
                double err = (corr - reproj).squaredNorm();
                totalErr += err;
            }
        }

        return totalErr;
    }


    void setViewerDataPublisher(ViewerDataPublisher *dataPublisher_) {
        dataPublisher = dataPublisher_;
    }

    void setCommandQueue(CommandQueue *commands_) {
        commands = commands_;
    }
};

std::unique_ptr<LoopCloser> LoopCloser::create(
    const StaticSettings &s,
    BowIndex &bowIndex,
    MapDB &mapDB,
    const Atlas &atlas
) {
    return std::unique_ptr<LoopCloser>(new LoopCloserImplementation(s, bowIndex, mapDB, atlas));
}

} // namespace slam
