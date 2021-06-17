#include "loop_ransac.hpp"
#include "map_point.hpp"
#include "openvslam/random_array.h"
#include "../util/logging.hpp"

namespace slam {

LoopRansac::LoopRansac(
    const Keyframe &kf1,
    const Keyframe &kf2,
    const std::vector<std::pair<MpId, MpId>> &matches,
    const MapDB &mapDB1,
    const MapDB &mapDB2,
    const StaticSettings &settings
) : camera1(*kf1.shared->camera), camera2(*kf2.shared->camera), settings(settings) {

    const auto size = matches.size();
    commonPtsInKeyframe1.reserve(size);
    commonPtsInKeyframe2.reserve(size);
    chiSqSigmaSq1.reserve(size);
    chiSqSigmaSq2.reserve(size);
    visibleSame1.reserve(size);
    visibleSame2.reserve(size);

    matchCount = matches.size();

    // For p = 0.01
    constexpr float CHI_SQ_2D = 9.21034;

    for (const auto &match : matches) {
        const MapPoint &mp1 = mapDB1.mapPoints.at(match.first);
        const MapPoint &mp2 = mapDB2.mapPoints.at(match.second);

        commonPtsInKeyframe1.emplace_back(Eigen::Isometry3d(kf1.poseCW) * mp1.position);
        commonPtsInKeyframe2.emplace_back(Eigen::Isometry3d(kf2.poseCW) * mp2.position);

        int octave1 = kf1.shared->keyPoints.at(mp1.observations.at(kf1.id).v).octave;
        int octave2 = kf2.shared->keyPoints.at(mp2.observations.at(kf2.id).v).octave;
        chiSqSigmaSq1.push_back(CHI_SQ_2D * settings.levelSigmaSq.at(octave1));
        chiSqSigmaSq2.push_back(CHI_SQ_2D * settings.levelSigmaSq.at(octave2));
    }

    reprojected1 = reproject_to_same_image(commonPtsInKeyframe1, camera1, visibleSame1);
    reprojected2 = reproject_to_same_image(commonPtsInKeyframe2, camera2, visibleSame2);
}

void LoopRansac::ransacSolve(const unsigned int max_num_iter, DoF solveType) {
    solutionOk = false;
    bestInlierCount = 0;
    const unsigned minInlierCount = settings.parameters.slam.loopClosureRansacMinInliers;

    if (matchCount < 3 || matchCount < minInlierCount) {
        return;
    }

    // Variables used in RANSAC loop
    Eigen::Matrix3d R12Ransac;
    Eigen::Vector3d t12Ransac;
    float s12Ransac;
    Eigen::Matrix3d R21Ransac;
    Eigen::Vector3d t21Ransac;
    float s21Ransac;

    std::vector<bool> inliers;

    // RANSAC loop
    for (unsigned i = 0; i < max_num_iter; ++i) {

        // Randomly sample 3 points
        Eigen::Matrix3d pts_1, pts_2;
        const auto random_indices = openvslam::util::create_random_array(3, 0, static_cast<int>(matchCount - 1));
        for (unsigned int i = 0; i < 3; ++i) {
            pts_1.block(0, i, 3, 1) = commonPtsInKeyframe1.at(random_indices.at(i));
            pts_2.block(0, i, 3, 1) = commonPtsInKeyframe2.at(random_indices.at(i));
        }

        if (solveType == DoF::ZROT) {
            computeRotZ(pts_1, pts_2, R21Ransac, t21Ransac, s21Ransac);
        } else if (solveType == DoF::SIM3) {
            computeSim3(pts_1, pts_2, R21Ransac, t21Ransac, s21Ransac);
        } else {
            assert(false && "Unknown solveType");
        }

        if (settings.parameters.slam.loopClosureRansacFixScale)
            s21Ransac = 1;

        s12Ransac = 1 / s21Ransac;

        R12Ransac = R21Ransac.transpose();
        t12Ransac = -s12Ransac * R12Ransac * t21Ransac;


        unsigned int num_inliers = count_inliers(R12Ransac, t12Ransac, s12Ransac,
                                               R21Ransac, t21Ransac, s21Ransac,
                                               inliers);

        if (bestInlierCount < num_inliers) {
            bestInlierCount = num_inliers;
            bestR12 = R12Ransac;
            bestT12 = t12Ransac;
            bestScale12 = s12Ransac;
            bestInliers = inliers;
        }
    }

    if (bestInlierCount >= minInlierCount) {
        solutionOk = true;
    }
}

void computeSim3(
    const Eigen::Matrix3d &pts_1,
    const Eigen::Matrix3d &pts_2,
    Eigen::Matrix3d &rot_21,
    Eigen::Vector3d &trans_21,
    float &scale_21
) {
    // Based on "Closed-form solution of absolute orientation using unit quaternions"
    // http://people.csail.mit.edu/bkph/papers/Absolute_Orientation.pdf

    // 各点集合のcentroidを求める
    const Eigen::Vector3d centroid_1 = pts_1.rowwise().mean();
    const Eigen::Vector3d centroid_2 = pts_2.rowwise().mean();

    // 分布の中心をcentroidに動かす
    Eigen::Matrix3d ave_pts_1 = pts_1;
    ave_pts_1.colwise() -= centroid_1;
    Eigen::Matrix3d ave_pts_2 = pts_2;
    ave_pts_2.colwise() -= centroid_2;

    // 4.A Matrix of Sums of Products

    // 行列Mを求める
    const Eigen::Matrix3d M = ave_pts_1 * ave_pts_2.transpose();

    // 行列Nを求める
    const double& Sxx = M(0, 0);
    const double& Syx = M(1, 0);
    const double& Szx = M(2, 0);
    const double& Sxy = M(0, 1);
    const double& Syy = M(1, 1);
    const double& Szy = M(2, 1);
    const double& Sxz = M(0, 2);
    const double& Syz = M(1, 2);
    const double& Szz = M(2, 2);
    Eigen::Matrix4d N;
    N << (Sxx + Syy + Szz), (Syz - Szy), (Szx - Sxz), (Sxy - Syx),
        (Syz - Szy), (Sxx - Syy - Szz), (Sxy + Syx), (Szx + Sxz),
        (Szx - Sxz), (Sxy + Syx), (-Sxx + Syy - Szz), (Syz + Szy),
        (Sxy - Syx), (Szx + Sxz), (Syz + Szy), (-Sxx - Syy + Szz);

    // 4.B Eigenvector Maximizes Matrix Product

    // Nを固有値分解する
    Eigen::EigenSolver<Eigen::Matrix4d> eigensolver(N);

    // 最大固有値を探す
    const auto& eigenvalues = eigensolver.eigenvalues();
    int max_idx = -1;
    double max_eigenvalue = -INFINITY;
    for (int idx = 0; idx < 4; ++idx) {
        if (max_eigenvalue <= eigenvalues(idx, 0).real()) {
            max_eigenvalue = eigenvalues(idx, 0).real();
            max_idx = idx;
        }
    }
    const auto max_eigenvector = eigensolver.eigenvectors().col(max_idx);

    // 複素数なので実数のみ取り出す
    Eigen::Vector4d eigenvector;
    eigenvector << max_eigenvector(0, 0).real(), max_eigenvector(1, 0).real(), max_eigenvector(2, 0).real(), max_eigenvector(3, 0).real();
    eigenvector.normalize();

    // unit quaternionにする
    Eigen::Quaterniond q_rot_21(eigenvector(0), eigenvector(1), eigenvector(2), eigenvector(3));

    // 回転行列に変換
    rot_21 = q_rot_21.normalized().toRotationMatrix();

    // 2.D Finding the Scale

    // averaged points 1をpoints 2の座標系に変換(回転のみ)
    const Eigen::Matrix3d ave_pts_1_in_2 = rot_21 * ave_pts_1;

    // 分母
    const double denom = ave_pts_1.squaredNorm();
    // 分子
    const double numer = ave_pts_2.cwiseProduct(ave_pts_1_in_2).sum();
    // スケール
    scale_21 = numer / denom;

    // 2.C Centroids of the Sets of Measurements

    trans_21 = centroid_2 - scale_21 * rot_21 * centroid_1;
}

unsigned int LoopRansac::count_inliers(const Eigen::Matrix3d& rot_12, const Eigen::Vector3d& trans_12, const float scale_12,
                                        const Eigen::Matrix3d& rot_21, const Eigen::Vector3d& trans_21, const float scale_21,
                                        std::vector<bool>& inliers) {
    unsigned int num_inliers = 0;
    inliers.resize(matchCount, false);
    std::vector<bool> visible1;
    std::vector<bool> visible2;

    vecVector2d reprojected_1_in_cam_2 =
        reproject_to_other_image(commonPtsInKeyframe1, rot_21, trans_21, scale_21, camera2, visible1);

    vecVector2d reprojected_2_in_cam_1 =
        reproject_to_other_image(commonPtsInKeyframe2, rot_12, trans_12, scale_12, camera1, visible2);

    for (unsigned int i = 0; i < matchCount; ++i) {
        if (!visible1[i] || !visible2[i] || !visibleSame1[i] || !visibleSame2[i]) {
            continue;
        }
        const Eigen::Vector2d dist_in_2 = (reprojected_1_in_cam_2.at(i) - reprojected2.at(i));
        const Eigen::Vector2d dist_in_1 = (reprojected_2_in_cam_1.at(i) - reprojected1.at(i));

        const double error_in_2 = dist_in_2.dot(dist_in_2);
        const double error_in_1 = dist_in_1.dot(dist_in_1);

        if (error_in_2 < chiSqSigmaSq2.at(i) && error_in_1 < chiSqSigmaSq1.at(i)) {
            inliers.at(i) = true;
            ++num_inliers;
        }
    }

    return num_inliers;
}

vecVector2d
LoopRansac::reproject_to_other_image(
    const vecVector3d &lm_coords_in_cam_1,
    const Eigen::Matrix3d &rot_21,
    const Eigen::Vector3d &trans_21,
    const float scale_21,
    const tracker::Camera &camera,
    std::vector<bool> &visible
) {
    visible.clear();
    vecVector2d reprojected_in_cam_2;
    reprojected_in_cam_2.reserve(lm_coords_in_cam_1.size());

    for (const auto &lm_coord_in_cam_1 : lm_coords_in_cam_1) {
        Eigen::Vector2d reproj_in_cam_2 = Eigen::Vector2d::Zero();
        float x_right = 0.0;
        bool v = reprojectToImage(camera, scale_21 * rot_21, trans_21, lm_coord_in_cam_1, reproj_in_cam_2, x_right);

        visible.push_back(v);
        reprojected_in_cam_2.push_back(reproj_in_cam_2);
    }

    return reprojected_in_cam_2;
}

vecVector2d
LoopRansac::reproject_to_same_image(
    const vecVector3d &lm_coords_in_cam,
    const tracker::Camera &camera,
    std::vector<bool> &visible
) {
    visible.clear();
    vecVector2d reprojected;
    reprojected.reserve(lm_coords_in_cam.size());

    for (const auto &lm_coord_in_cam : lm_coords_in_cam) {
        Eigen::Vector2d reproj = Eigen::Vector2d::Zero();
        float x_right = 0.0;
        bool v = reprojectToImage(camera, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), lm_coord_in_cam, reproj, x_right);

        visible.push_back(v);
        reprojected.push_back(reproj);
    }
    return reprojected;
}

void computeRotZ(
    const Eigen::Matrix3d &pts1,
    const Eigen::Matrix3d &pts2,
    Eigen::Matrix3d &r21,
    Eigen::Vector3d &t21,
    float &s21
) {
    // Based on 5.A "Coplanar points" in
    // http://people.csail.mit.edu/bkph/papers/Absolute_Orientation.pdf

    const Eigen::Vector3d centroid1 = pts1.rowwise().mean();
    const Eigen::Vector3d centroid2 = pts2.rowwise().mean();

    Eigen::Matrix3d centered1 = pts1.colwise() - centroid1;
    Eigen::Matrix3d centered2 = pts2.colwise() - centroid2;

    double C = (centered1.topRows<2>().cwiseProduct(centered2.topRows<2>())).sum();

    double S = 0;
    for (int i = 0; i < 3; i++) {
        auto p1 = centered1.col(i).head<2>();
        auto p2 = centered2.col(i).head<2>();
        S += p1.x() * p2.y() - p1.y() * p2.x();
    }

    double cosTheta = C / std::sqrt(C*C + S*S);
    double sinTheta = S / std::sqrt(C*C + S*S);

    r21.setZero();
    r21(2,2) = 1;
    r21.topLeftCorner<2,2>() << cosTheta, -sinTheta, sinTheta, cosTheta;

    s21 =  (centered2.cwiseProduct(r21 * centered1)).sum() / centered1.squaredNorm();

    t21 = centroid2 - s21 * r21 * centroid1;
}

} // namespace slam
