#ifndef SLAM_KEYFRAME_MATCHER_HPP
#define SLAM_KEYFRAME_MATCHER_HPP

#include "keyframe.hpp"
#include "static_settings.hpp"
#include "viewer_data_publisher.hpp"

namespace slam {

constexpr unsigned int HAMMING_DIST_THR_LOW = 50;
constexpr unsigned int HAMMING_DIST_THR_HIGH = 100;
constexpr unsigned int MAX_HAMMING_DIST = 256;

/**
 * Compares descriptors of two keyframes, finding the best match for each feature
 * of the first keyframe in the second keyframe.
 *
 * The output is vector with the length of number of keypoints in `kf1`. Each element
 * is a keypoint index of `kf2`, or -1 if no match was found.
 *
 * Based on openvslam::match::bow_tree::match_frame_and_keyframe.
 *
 * Utilizes DBow index tree to only compare descriptors that are similar to each other. Should
 * be about the same functionality as cv::BFMatcher::knnMatch(....., 2) and selecting only
 * matches where `match1.distance < lowe_ratio * match2.distance`
 *
 * @param kf1 first keyframe
 * @param kf2 second keyframe
 * @param mapDB1 the map first keyframe is part of
 * @param mapDB2 the map second keyframe is part of
 * @param matchedMapPoints the output
 */
unsigned int matchForLoopClosures(
    const Keyframe &kf1,
    const Keyframe &kf2,
    const MapDB &mapDB1,
    const MapDB &mapDB2,
    std::vector<int> &matchedMapPoints,
    const odometry::ParametersSlam &parameters
);

/**
 *  Based on openvslam::match::robust::match_for_triangulation.
 *
 *  Match features not associated with `slam::MapPoint`s between two keyframes. Checks epipolar constraint
 *  and possibly orientation of features.
 *
 *  @param kf1 first keyframe
 *  @param kf2 second keyframe
 *  @param checkOrientation return only matches where the orientation of features matches
 *  @return matches between keyframes
 */
std::vector<std::pair<KpId, KpId>> matchForTriangulationDBoW(Keyframe &kf1, Keyframe &kf2, const StaticSettings &settings);

/**
 * For a (fresh) keyframe, projects the given map points and adds observations for close matches
 * for those keypoints that have no associated map points (MpId == -1).
 */
int searchByProjection(
    Keyframe &kf,
    const std::vector<MpId> &mps,
    MapDB &mapDB,
    ViewerDataPublisher *dataPublisher,
    const float threshold,
    const StaticSettings &settings
);

/**
 * Takes a collection of map points, projects them on the given keyframe and looks for close matches
 * to existing keypoints/map points and merges them.
 */
template <typename T>
unsigned int replaceDuplication(
    Keyframe &kf,
    const T &mapPoints,
    const float margin,
    MapDB &mapDB,
    const StaticSettings &settings);

/**
 *  Match MapPoints between #kf1 and #kf2 using the provided Sim3 transformation
 *
 *  @param matches Vector with current matches. Updated with new matches found by projection.
 */
void matchMapPointsSim3(
        Keyframe &kf1,
        Keyframe &kf2,
        const Eigen::Matrix4d &transform12,
        MapDB &mapDB,
        std::vector<std::pair<MpId, MpId>> &matches,
        const StaticSettings &settings);

} // namespace slam

#endif //SLAM_KEYFRAME_MATCHER_HPP
