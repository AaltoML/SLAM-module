#include <opencv2/features2d.hpp>
#include "openvslam/match_angle_checker.h"
#include "openvslam/match_base.h"
#include "openvslam/essential_solver.h"

#include "keyframe_matcher.hpp"
#include "map_point.hpp"
#include "mapdb.hpp"
#include "../util/logging.hpp"

#include <set>

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector2f;

static constexpr float SQRT_CHI2_INV2D = 2.4477; // for p = 0.05

namespace slam {

namespace {

bool check_epipolar_constraint(
        const Eigen::Vector3d& bearing_1, const Eigen::Vector3d& bearing_2,
        const Eigen::Matrix3d& E_12,
        const float bearing_1_scale_factor,
        const float residual_deg_thr)
{
    // keyframe1上のtエピポーラ平面の法線ベクトル
    const Eigen::Vector3d epiplane_in_1 = E_12 * bearing_2;

    // 法線ベクトルとbearingのなす角を求める
    const auto cos_residual = epiplane_in_1.dot(bearing_1) / epiplane_in_1.norm();
    const auto residual_rad = M_PI / 2.0 - std::abs(std::acos(cos_residual));

    // The original Japanese commnt said the fixed threshold 0.2 was equivalent
    // to 2 pixesls on a 90deg FOV camera with resolution 900px and had a
    // to-do note about parametrization
    const double residual_rad_thr = residual_deg_thr * M_PI / 180.0;

    // 特徴点スケールが大きいほど閾値を緩くする
    // TODO: thresholdの重み付けの検討
    return residual_rad < residual_rad_thr * bearing_1_scale_factor;
}

} // anonymous namespace

static constexpr bool check_orientation = true;

unsigned int matchForLoopClosures(
    const Keyframe &kf1,
    const Keyframe &kf2,
    const MapDB &mapDB1,
    const MapDB &mapDB2,
    std::vector<int> &matchedMapPoints,
    const odometry::ParametersSlam &parameters
) {
    unsigned int num_matches = 0;
    openvslam::match::angle_checker<int> angle_checker;

    matchedMapPoints.resize(kf1.shared->keyPoints.size(), -1);

    std::vector<bool> alreadyMatchedInKf2(kf2.shared->keyPoints.size(), false);

    DBoW2::FeatureVector::const_iterator it1 = kf1.shared->bowFeatureVec.begin();
    DBoW2::FeatureVector::const_iterator it2 = kf2.shared->bowFeatureVec.begin();

    // Iterate through all matching bow nodes in the bowFeatureVec:s
    // Relies on the iterator of std::map being ordered
    while (it1 != kf1.shared->bowFeatureVec.end() && it2 != kf2.shared->bowFeatureVec.end()) {
        DBoW2::NodeId nodeId1 = it1->first;
        DBoW2::NodeId nodeId2 = it2->first;

        if (nodeId1 == nodeId2) {
            const std::vector<unsigned int>& kf1IndicesInNode = it1->second;
            const std::vector<unsigned int>& kf2IndicesInNode = it2->second;

            for (const auto idx_1 : kf1IndicesInNode) {
                MpId mpId1 = kf1.mapPoints.at(idx_1);
                if (mpId1.v == -1) continue;
                // Optionally don't require triangulation for current map points, still require it for historical map points
                if (parameters.requireTringulationForLoopClosures && mapDB1.mapPoints.at(mpId1).status != MapPointStatus::TRIANGULATED) {
                    continue;
                }

                const auto &kp1 = kf1.shared->keyPoints.at(idx_1);
                const auto &desc1 = kp1.descriptor;

                unsigned int best_hamm_dist = MAX_HAMMING_DIST;
                int best_idx_2 = -1;
                unsigned int second_best_hamm_dist = MAX_HAMMING_DIST;

                for (const auto idx_2 : kf2IndicesInNode) {
                    MpId mpId2 = kf2.mapPoints.at(idx_2);
                    if (mpId2.v == -1 || mapDB2.mapPoints.at(mpId2).status != MapPointStatus::TRIANGULATED)
                        continue;

                    if (alreadyMatchedInKf2.at(idx_2)) {
                        continue;
                    }

                    const auto &kp2 = kf2.shared->keyPoints.at(idx_2);
                    const auto desc2 = kp2.descriptor;
                    const auto hamm_dist = openvslam::match::compute_descriptor_distance_32(desc1.data(), desc2.data());

                    if (hamm_dist < best_hamm_dist) {
                        second_best_hamm_dist = best_hamm_dist;
                        best_hamm_dist = hamm_dist;
                        best_idx_2 = idx_2;
                    } else if (hamm_dist < second_best_hamm_dist) {
                        second_best_hamm_dist = hamm_dist;
                    }
                }

                if (HAMMING_DIST_THR_LOW < best_hamm_dist) {
                    continue;
                }

                // Lowe ratio test.
                if (parameters.loopClosureFeatureMatchLoweRatio * second_best_hamm_dist < static_cast<float>(best_hamm_dist)) {
                    continue;
                }

                // The idea here is just to record the best matching feature
                // index in KF2 for each feature in KF1
                matchedMapPoints.at(idx_1) = best_idx_2;
                // Here we make sure the same feature is not matched twice
                alreadyMatchedInKf2.at(best_idx_2) = true;

                num_matches++;

                if (check_orientation) {
                    const auto delta_angle = kp1.angle - kf2.shared->keyPoints.at(best_idx_2).angle;
                    angle_checker.append_delta_angle(delta_angle, idx_1);
                }
            }

            ++it1;
            ++it2;
        }
        else if (nodeId1 < nodeId2) {
            it1 = kf1.shared->bowFeatureVec.lower_bound(nodeId2);
        }
        else {
            it2 = kf2.shared->bowFeatureVec.lower_bound(nodeId1);
        }
    }

    if (check_orientation) {
        const auto invalid_matches = angle_checker.get_invalid_matches();
        for (const auto invalid_idx : invalid_matches) {
            matchedMapPoints.at(invalid_idx) = -1;
            --num_matches;
        }
    }

    return num_matches;
}

std::vector<std::pair<KpId, KpId>> matchForTriangulationDBoW(
    Keyframe& kf1, Keyframe& kf2,
    const StaticSettings &settings)
{
    using Eigen::Matrix3d;
    using Eigen::Vector3d;

    constexpr bool checkOrientation = true;
    const float residual_deg_thr = settings.parameters.slam.epipolarCheckThresholdDegrees;

    Eigen::Matrix3d essentialMatrix;
    essentialMatrix = openvslam::solve::essential_solver::create_E_21(
        kf2.poseCW.topLeftCorner<3, 3>(),
        kf2.poseCW.block<3, 1>(0, 3),
        kf1.poseCW.topLeftCorner<3, 3>(),
        kf1.poseCW.block<3, 1>(0, 3));

    unsigned int num_matches = 0;

    openvslam::match::angle_checker<int> angle_checker;

    const Vector3d cam_center_1 = kf1.cameraCenter();
    const Matrix3d rot_2w = kf2.poseCW.topLeftCorner<3,3>();
    const Vector3d trans_2w = kf2.poseCW.block<3,1>(0,3);
    Vector3d epiplane_in_keyfrm_2;
    reprojectToBearing(*kf2.shared->camera, rot_2w, trans_2w, cam_center_1, epiplane_in_keyfrm_2);

    std::vector<bool> is_already_matched_in_keyfrm_2(kf2.shared->keyPoints.size(), false);
    std::vector<int> matched_indices_2_in_keyfrm_1(kf1.shared->keyPoints.size(), -1);

    DBoW2::FeatureVector::const_iterator itr_1 = kf1.shared->bowFeatureVec.begin();
    DBoW2::FeatureVector::const_iterator itr_2 = kf2.shared->bowFeatureVec.begin();
    const DBoW2::FeatureVector::const_iterator itr_1_end = kf1.shared->bowFeatureVec.end();
    const DBoW2::FeatureVector::const_iterator itr_2_end = kf2.shared->bowFeatureVec.end();

    // This looping structure goes through the higher level nodes of the vocabulary tree
    // (typically this node count is 100), and does work whenever both keyframes have some
    // features in the same node.
    while (itr_1 != itr_1_end && itr_2 != itr_2_end) {
        if (itr_1->first == itr_2->first) {
            // Keypoint indices for the node.
            const auto& keyfrm_1_indices = itr_1->second;
            const auto& keyfrm_2_indices = itr_2->second;

            for (const auto idx_1 : keyfrm_1_indices) {
                auto lm_1 = kf1.mapPoints.at(idx_1);
                // skip features that are already associated with a 3D map point
                if (lm_1.v != -1) {
                    continue;
                }

                const auto& keypt_1 = kf1.shared->keyPoints.at(idx_1);
                const auto &desc1 = keypt_1.descriptor;
                unsigned int best_hamm_dist = HAMMING_DIST_THR_LOW;
                int best_idx_2 = -1;

                for (const auto idx_2 : keyfrm_2_indices) {
                    auto lm_2 = kf2.mapPoints.at(idx_2);
                    // skip features that are already associated with a 3D map point
                    if (lm_2.v != -1) {
                        continue;
                    }

                    // Do not match multiple keypoints from first keyframe to a single keypoint of the second.
                    if (is_already_matched_in_keyfrm_2.at(idx_2)) {
                        continue;
                    }

                    const auto& keypt_2 = kf2.shared->keyPoints.at(idx_2);
                    const auto &desc2 = keypt_2.descriptor;
                    const auto hamm_dist = openvslam::match::compute_descriptor_distance_32(desc1.data(), desc2.data());
                    if (hamm_dist > HAMMING_DIST_THR_LOW || hamm_dist > best_hamm_dist) {
                        continue;
                    }

                    // TODO: try std::max(keypt_1.octave, keypt_2.octave);
                    const int octave = keypt_1.octave;
                    const bool is_inlier = check_epipolar_constraint(keypt_1.bearing, keypt_2.bearing, essentialMatrix,
                                                                     settings.scaleFactors.at(octave), residual_deg_thr);
                    if (is_inlier) {
                        best_idx_2 = idx_2;
                        best_hamm_dist = hamm_dist;
                    }
                }

                if (best_idx_2 < 0) {
                    continue;
                }

                is_already_matched_in_keyfrm_2.at(best_idx_2) = true;
                matched_indices_2_in_keyfrm_1.at(idx_1) = best_idx_2;
                ++num_matches;

                if (checkOrientation) {
                    const auto delta_angle
                        = keypt_1.angle - kf2.shared->keyPoints.at(best_idx_2).angle;
                    angle_checker.append_delta_angle(delta_angle, idx_1);
                }
            }

            ++itr_1;
            ++itr_2;
        }
        else if (itr_1->first < itr_2->first) {
            itr_1 = kf1.shared->bowFeatureVec.lower_bound(itr_2->first);
        }
        else {
            itr_2 = kf2.shared->bowFeatureVec.lower_bound(itr_1->first);
        }
    }

    if (checkOrientation) {
        const auto invalid_matches = angle_checker.get_invalid_matches();
        for (const auto invalid_idx : invalid_matches) {
            matched_indices_2_in_keyfrm_1.at(invalid_idx) = -1;
            --num_matches;
        }
    }

    std::vector<std::pair<KpId, KpId>> matches;
    matches.reserve(num_matches);

    for (unsigned int idx_1 = 0; idx_1 < matched_indices_2_in_keyfrm_1.size(); ++idx_1) {
        if (matched_indices_2_in_keyfrm_1.at(idx_1) < 0) {
            continue;
        }
        matches.emplace_back(std::make_pair(
            KpId(idx_1),
            KpId(matched_indices_2_in_keyfrm_1.at(idx_1))
        ));
    }

    return matches;
}

int searchByProjection(
    Keyframe &kf,
    const std::vector<MpId> &mps,
    MapDB &mapDB,
    ViewerDataPublisher *dataPublisher,
    const float threshold,
    const StaticSettings &settings
) {
    int matchCount=0;
    constexpr float viewAngleLimitCos = 0.5;

    bool visualize = dataPublisher
        && dataPublisher->getParameters().visualizeMapPointSearch
        && !kf.shared->imgDbg.empty();
    ViewerDataPublisher::SearchedMapPointVector matched;
    ViewerDataPublisher::Vector2dVector projectedMps;

    std::vector<size_t> indices;
    for (MpId mpId : mps) {
        MapPoint &mp = mapDB.mapPoints.at(mpId);
        Eigen::Vector2f reprojection;
        const bool inImage = kf.reproject(mp.position, reprojection);
        if (!inImage) {
            continue;
        }

        if (visualize) projectedMps.push_back(reprojection.cast<double>());

        Eigen::Vector3f mpToKf = (kf.cameraCenter() - mp.position).cast<float>();
        float dist = mpToKf.norm();
        if (dist < mp.minViewingDistance || mp.maxViewingDistance < dist)
            continue;

        const float viewingAngleCos = mpToKf.normalized().dot(mp.norm);
        if (viewingAngleCos < viewAngleLimitCos)
            continue;

        const float predictedScaleLevel = mp.predictScaleLevel(dist, settings);

        // TODO: parametrize and check if needed
        constexpr float SMALL_ANGLE_MUL = 2.5 / 4.0;
        constexpr float SMALL_ANGLE_COS = 0.998; // cos(4) =~ 0.998
        float r = viewingAngleCos > SMALL_ANGLE_COS ? SMALL_ANGLE_MUL : 1.0;
        const auto REF_SCALE_FACTOR = settings.scaleFactors.size() / 2;

        kf.getFeaturesAround(
            reprojection,
            r * threshold * settings.scaleFactors.at(predictedScaleLevel) / settings.scaleFactors.at(REF_SCALE_FACTOR),
            indices
        );

        if(indices.empty())
            continue;

        int bestDist = 256;
        int bestLevel = -1;
        int bestDist2 = 256;
        int bestLevel2 = -1;
        int bestIdx = -1;

        // Get best and second best matches
        for(size_t idx : indices) {
            // Ignore already matched features
            if (kf.mapPoints[idx].v != -1 && mapDB.mapPoints.at(kf.mapPoints.at(idx)).observations.size() > 0) {
                continue;
            }

            const auto &kp = kf.shared->keyPoints.at(idx);
            const auto &desc = kp.descriptor;
            const int level = kp.octave;

            const int dist = openvslam::match::compute_descriptor_distance_32(mp.descriptor.data(), desc.data());

            if (dist < bestDist) {
                bestDist2 = bestDist;
                bestDist = dist;
                bestLevel2 = bestLevel;
                bestLevel = level;
                bestIdx = idx;
            } else if (dist < bestDist2) {
                bestLevel2 = level;
                bestDist2 = dist;
            }
        }

        if (bestIdx == -1) continue;

        int HAMMING_DIST_THR_HIGH = 100;
        if (bestDist <= HAMMING_DIST_THR_HIGH) {
            // Apply ratio to second match (only if best and second are in the same scale level)
            if (bestLevel == bestLevel2 && bestDist > 0.8 * bestDist2)
                continue;

            kf.addObservation(mp.id, KpId(bestIdx));
            mp.addObservation(kf.id, KpId(bestIdx));
            matchCount++;

            if (visualize) {
                auto pt = kf.shared->keyPoints[bestIdx].pt;
                matched.push_back(SearchedMapPoint {
                    .mp = reprojection.cast<double>(),
                    .kp = Eigen::Vector2d(pt.x, pt.y),
                });
            }
        }
    }

    if (visualize) {
        ViewerDataPublisher::Vector2dVector unmatchedKps;
        for (size_t i = 0; i < kf.mapPoints.size(); ++i) {
            if (kf.mapPoints[i] == MpId(-1)) {
                auto pt = kf.shared->keyPoints[i].pt;
                unmatchedKps.push_back(Eigen::Vector2d(pt.x, pt.y));
            }
        }
        dataPublisher->visualizeMapPointSearch(kf.shared->imgDbg, matched, projectedMps, unmatchedKps);
    }

    return matchCount;
}

template <typename T>
unsigned int replaceDuplication(
    Keyframe &kf,
    const T &mapPoints,
    const float margin,
    MapDB& mapDB,
    const StaticSettings &settings
) {
    std::set<MpId> erasedMapPointIds;
    unsigned int fusedCount = 0;

    std::vector<size_t> indices;
    for (const MpId &mpId : mapPoints) {
        if (mpId.v == -1 || erasedMapPointIds.count(mpId)) {
            continue;
        }

        MapPoint &mp = mapDB.mapPoints.at(mpId);
        if (mp.observations.count(kf.id)) {
            continue;
        }

        if (mp.status == MapPointStatus::BAD || mp.status == MapPointStatus::NOT_TRIANGULATED) {
            continue;
        }

        // Check that depth is positive and reprojected mapPoint is in the frame
        Eigen::Vector2f reproj;
        const bool inImage = kf.reproject(mp.position, reproj);
        if (!inImage) {
            continue;
        }

        const Eigen::Vector3f mpToKf = (kf.cameraCenter() - mp.position).template cast<float>();
        const float mpToKfDist = mpToKf.norm();

        const float maxDist = mp.maxViewingDistance;
        const float minDist = mp.minViewingDistance;

        if (mpToKfDist < minDist || maxDist < mpToKfDist) {
            continue;
        }

        // TODO This triggers often. Intentional?
        if (mp.norm.isZero(0)) {
            continue;
        }
        // Viewing angle less than 60deg from normal (cos(60) = 0.5)
        if (mpToKf.normalized().dot(mp.norm) < 0.5) {
            continue;
        }

        int predictedScaleLevel = mp.predictScaleLevel(mpToKfDist, settings);

        const float baseScale = settings.scaleFactors[settings.scaleFactors.size()/2];
        float r = margin * settings.scaleFactors[predictedScaleLevel] / baseScale * SQRT_CHI2_INV2D;

        kf.getFeaturesAround(reproj, r, indices);

        if (indices.empty()) {
            continue;
        }

        unsigned int bestDist = MAX_HAMMING_DIST;
        int bestInd = -1;

        for (int ind : indices) {
            // Reprojection error has already been checked by "getFeaturesAround".
            // TODO Skip if keypoint scale level differs too much?

            const auto &kp = kf.shared->keyPoints.at(ind);
            const auto &desc = kp.descriptor;
            const auto hammingDist = openvslam::match::compute_descriptor_distance_32(mp.descriptor.data(), desc.data());

            if (hammingDist < bestDist) {
                bestDist = hammingDist;
                bestInd = ind;
            }
        }
        KpId bestKpId(bestInd);

        if (bestInd == -1 || bestDist > HAMMING_DIST_THR_LOW) {
            continue;
        }

        MpId matchedMpId = kf.mapPoints[bestKpId.v];
        if (matchedMpId.v == -1) {
            // MapPoint corresponding to best match doesn't exist yet
            mp.addObservation(kf.id, bestKpId);
            kf.addObservation(mp.id, bestKpId);
        } else {
            MapPoint &matchedMp = mapDB.mapPoints.at(matchedMpId);

            // Replace the point with fewer observations
            if (mp.observations.size() < matchedMp.observations.size()) {
                if (matchedMp.status == MapPointStatus::NOT_TRIANGULATED) {
                    matchedMp.eraseObservation(kf.id);
                    kf.mapPoints[bestKpId.v] = mp.id;
                    mp.addObservation(kf.id, bestKpId);
                } else {
                    mp.replaceWith(mapDB, matchedMp);
                }
                erasedMapPointIds.insert(mpId);
            } else {
                matchedMp.replaceWith(mapDB, mp);
                erasedMapPointIds.insert(matchedMpId);
            }
        }

        ++fusedCount;
    }

    return fusedCount;
}

template unsigned int replaceDuplication<std::vector<MpId>>(
    Keyframe& kf, const std::vector<MpId>& mapPointIdsToCheck, const float margin, MapDB& mapDB, const StaticSettings &settings);
template unsigned int replaceDuplication<std::set<MpId>>(
    Keyframe& kf, const std::set<MpId>& mapPointIdsToCheck, const float margin, MapDB& mapDB, const StaticSettings &settings);

/**
 *  Find matches in keyframe B for map points seen in keyframe A.
 *  The map points are first transformed W -> A -> B and then reprojected.
 *  The A -> B tranformation is the Sim3 tranformation calculated for a potential
 *  loop closure.
 *
 *  @param mpIdsA MapPoint ids in keyframe A
 *  @param alreadyMatchedInA already matched MapPoints in keyframe A
 *  @param kfB keyframe B
 *  @param rotBAW rotation component of W -> A -> B
 *  @param transBAW translation component of W -> A -> B
 *  @param MapDB
 *  @param margin
 *
 *  @return vector matches, s.t. matches[i_A] = i_B, iff the features at i_A and i_B match
 */
std::vector<int> findMatchesTranformedMps(
    std::vector<MpId> mpIdsA,
    std::vector<bool> alreadyMatchedInA,
    Keyframe &kfB,
    const Eigen::Matrix3d &rotBAW,
    const Eigen::Vector3d &transBAW,
    MapDB &mapDB,
    float margin,
    const StaticSettings &settings
) {
    std::vector<int> matchesAtoB(mpIdsA.size(), -1);

    for (unsigned int indA = 0; indA < mpIdsA.size(); ++indA) {
        if (alreadyMatchedInA.at(indA)) continue;

        MpId mpId = mpIdsA.at(indA);
        if (mpId.v == -1) continue;

        const MapPoint &mp = mapDB.mapPoints.at(mpId);
        if (mp.status != MapPointStatus::TRIANGULATED) continue;

        const Eigen::Vector3d posW = mp.position;
        const Eigen::Vector3d posB = rotBAW * posW + transBAW;

        Eigen::Vector2d reproj = Eigen::Vector2d::Zero();
        float x_right = 0.0;
        const bool in_image = reprojectToImage(*kfB.shared->camera, rotBAW, transBAW, posW, reproj, x_right);

        if (!in_image) {
            continue;
        }

        // Withing ORB scale
        double viewingDistance = posB.norm();
        float maxViewingDistance = mp.maxViewingDistance;
        float minViewingDistance = mp.minViewingDistance;

        if (viewingDistance < minViewingDistance || maxViewingDistance < viewingDistance) {
            continue;
        }

        // Get keyPoints close to projection
        int predScaleLevel = mp.predictScaleLevel(viewingDistance, settings);
        std::vector<size_t> indices;
        kfB.getFeaturesAround(reproj.cast<float>(), margin * settings.scaleFactors.at(predScaleLevel), indices);

        if (indices.empty()) continue;

        // Find closest descriptor
        unsigned int bestHammingDist = MAX_HAMMING_DIST;
        int bestIndB = -1;

        for (size_t indB : indices) {
            // TODO(jhnj): should we only look at mapPoints here? (not just any keyPoint)

            const auto &kpB = kfB.shared->keyPoints.at(indB);

            // Comment from openvslam (translated):
            // TODO: judge scale by keyfrm-> get_keypts_in_cell ()
            if (kpB.octave < predScaleLevel - 1 || kpB.octave > predScaleLevel) {
                continue;
            }

            const auto &desc = kpB.descriptor;
            unsigned int hammingDist =
                openvslam::match::compute_descriptor_distance_32(mp.descriptor.data(), desc.data());

            if (hammingDist < bestHammingDist) {
                bestHammingDist = hammingDist;
                bestIndB = indB;
            }
        }

        if (bestHammingDist <= HAMMING_DIST_THR_HIGH) {
            matchesAtoB.at(indA) = bestIndB;
        }
    }

    return matchesAtoB;
}

void matchMapPointsSim3(
        Keyframe &kf1,
        Keyframe &kf2,
        const Eigen::Matrix4d &transform12,
        MapDB &mapDB,
        std::vector<std::pair<MpId, MpId>> &matches,
        const StaticSettings &settings) {

    constexpr float margin = 7.5;
    std::vector<bool> alreadyMatchedInKf1(kf1.mapPoints.size(), false);
    std::vector<bool> alreadyMatchedInKf2(kf2.mapPoints.size(), false);

    for (const auto &match : matches) {
        alreadyMatchedInKf1.at(mapDB.mapPoints.at(match.first).observations.at(kf1.id).v) = true;
        alreadyMatchedInKf2.at(mapDB.mapPoints.at(match.second).observations.at(kf2.id).v) = true;
    }

    Eigen::Matrix4d transform21w = transform12.inverse() * kf1.poseCW;
    std::vector<int> matched_indices_2_in_keyfrm_1 = findMatchesTranformedMps(
            kf1.mapPoints,
            alreadyMatchedInKf1,
            kf2,
            transform21w.topLeftCorner<3,3>(),
            transform21w.block<3,1>(0,3),
            mapDB,
            margin,
            settings);

    Eigen::Matrix4d transform12W = transform12 * kf2.poseCW;
    std::vector<int> matched_indices_1_in_keyfrm_2 = findMatchesTranformedMps(
            kf2.mapPoints,
            alreadyMatchedInKf2,
            kf1,
            transform12W.topLeftCorner<3,3>(),
            transform12W.block<3,1>(0,3),
            mapDB,
            margin,
            settings);

    // Add only matches that agree, where 1->2 and 2->1 are the same
    unsigned int num_matches = 0;

    for (unsigned i = 0; i < matched_indices_2_in_keyfrm_1.size(); ++i) {
        const auto idx_2 = matched_indices_2_in_keyfrm_1.at(i);
        if (idx_2 < 0) {
            continue;
        }

        if (matched_indices_1_in_keyfrm_2.at(idx_2) == static_cast<int>(i)) {
            matches.emplace_back(kf1.mapPoints.at(i), kf2.mapPoints.at(idx_2));
            ++num_matches;
        }
    }
}

} // namespace slam
