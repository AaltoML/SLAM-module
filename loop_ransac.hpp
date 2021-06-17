#ifndef SLAM_LOOPRANSAC_HPP
#define SLAM_LOOPRANSAC_HPP

#include "keyframe.hpp"
#include "mapdb.hpp"
#include "static_settings.hpp"

#include "../odometry/util.hpp"

namespace slam {

class LoopRansac {
public:
    LoopRansac(
        const Keyframe &kf1,
        const Keyframe &kf2,
        const std::vector<std::pair<MpId, MpId>> &matches,
        const MapDB &mapDB1,
        const MapDB &mapDB2,
        const StaticSettings &settings
    );

    enum class DoF {
        SIM3, ZROT
    };

    void ransacSolve(const unsigned int max_num_iter, DoF dof = DoF::ZROT);

    unsigned int count_inliers(
        const Eigen::Matrix3d &rot_12,
        const Eigen::Vector3d &trans_12,
        const float scale_12,
        const Eigen::Matrix3d &rot_21,
        const Eigen::Vector3d &trans_21,
        const float scale_21,
        std::vector<bool> &inliers
    );

    vecVector2d reproject_to_same_image(
        const vecVector3d &lm_coords_in_cam,
        const tracker::Camera &camera,
        std::vector<bool> &visible
    );

    vecVector2d reproject_to_other_image(
        const vecVector3d &lm_coords_in_cam_1,
        const Eigen::Matrix3d &rot_21,
        const Eigen::Vector3d &trans_21,
        const float scale_21,
        const tracker::Camera &camera,
        std::vector<bool> &visible
    );

    const tracker::Camera &camera1;
    const tracker::Camera &camera2;

    // Local coordinates in kf1 and kf2 of matched mapPoints
    vecVector3d commonPtsInKeyframe1;
    vecVector3d commonPtsInKeyframe2;

    std::vector<bool> visibleSame1;
    std::vector<bool> visibleSame2;

    // Chi-square values with two degrees of freedom for reprojection errors
    std::vector<float> chiSqSigmaSq1;
    std::vector<float> chiSqSigmaSq2;

    unsigned int matchCount = 0;

    //! solution is valid or not
    bool solutionOk = false;
    //! most reliable rotation from keyframe 2 to keyframe 1
    Eigen::Matrix3d bestR12;
    //! most reliable translation from keyframe 2 to keyframe 1
    Eigen::Vector3d bestT12;
    //! most reliable scale from keyframe 2 to keyframe 1
    float bestScale12;
    std::vector<bool> bestInliers;
    unsigned int bestInlierCount;

    // Image coordinates of reporjected mapPoints
    vecVector2d reprojected1;
    vecVector2d reprojected2;

    const StaticSettings &settings;
};

/**
 *  Compute Sim3 transformation between two sets of points.
 *  pts2 = s21 * r21 * pts1 + t21
 */
void computeSim3(
    const Eigen::Matrix3d &pts_1,
    const Eigen::Matrix3d &pts_2,
    Eigen::Matrix3d &rot_21,
    Eigen::Vector3d &trans_21,
    float &scale_21);

/**
 *  Compute transformation between two sets of points.
 *  Transformation limited to only: rotation around z-axis + translation
 *  pts2 = r21 * pts1 + t21
 */
void computeRotZ(
    const Eigen::Matrix3d &pts1W,
    const Eigen::Matrix3d &pts2W,
    Eigen::Matrix3d &r21,
    Eigen::Vector3d &t21,
    float &s21);

} // namespace slam

#endif // SLAM_LOOPRANSAC_HPP
