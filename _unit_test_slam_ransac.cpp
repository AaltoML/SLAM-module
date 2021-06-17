#include "catch2/catch.hpp"

#ifdef USE_SLAM

#include <Eigen/Dense>
#include <iostream>
#include <stdlib.h>
#include <random>

#include "loop_ransac.hpp"


TEST_CASE("computeSim3") {
    Eigen::Matrix3d pts1;
    pts1 << 3.40188, 2.9844, -1.64777,
            -1.05617, 4.11647, 2.6823,
            2.83099, -3.02449, -2.22225;

    Eigen::Matrix3d rotZ = (Eigen::AngleAxisd(0.653 * M_PI, Eigen::Vector3d::UnitX())
                            * Eigen::AngleAxisd(-1.02 * M_PI, Eigen::Vector3d::UnitY())
                            * Eigen::AngleAxisd(0.13 * M_PI, Eigen::Vector3d::UnitZ()))
                               .toRotationMatrix();

    Eigen::Vector3d trans(3.13321, -1.05617, 2.83099);

    Eigen::Matrix3d pts2 = (rotZ * pts1).colwise() + trans;

    Eigen::Matrix3d pts3 = (rotZ.transpose() * pts2).colwise() - rotZ.transpose() * trans;
    CAPTURE(pts1);
    CAPTURE(pts2);
    CAPTURE(pts3);

    Eigen::Matrix3d rotRes;
    Eigen::Vector3d transRes;
    float scaleRes;
    slam::computeSim3(pts1, pts2, rotRes, transRes, scaleRes);

    CAPTURE(rotRes);
    CAPTURE(rotZ);
    double diff = (rotRes - rotZ).squaredNorm();
    REQUIRE(diff < 0.001);

    CAPTURE(transRes);
    CAPTURE(trans);
    REQUIRE((transRes - trans).norm() < 0.001);

    CAPTURE(scaleRes);
    REQUIRE(scaleRes == 1);
}

TEST_CASE("computeRotZ") {
    Eigen::Matrix3d pts1;
    pts1 << 3.40188, 2.9844, -1.64777,
            -1.05617, 4.11647, 2.6823,
            2.83099, -3.02449, -2.22225;

    double trueRad = 0.653 * M_PI;
    Eigen::Matrix3d rotZ = Eigen::AngleAxisd(trueRad, Eigen::Vector3d::UnitZ()).toRotationMatrix();

    Eigen::Vector3d trans(3.13321, -1.05617, 2.83099);

    Eigen::Matrix3d pts2 = (rotZ * pts1).colwise() + trans;

    Eigen::Matrix3d pts3 = (rotZ.transpose() * pts2).colwise() - rotZ.transpose() * trans;
    CAPTURE(pts1);
    CAPTURE(pts2);
    CAPTURE(pts3);

    Eigen::Matrix3d rotRes;
    Eigen::Vector3d transRes;
    float scaleRes;
    slam::computeRotZ(pts1, pts2, rotRes, transRes, scaleRes);

    CAPTURE(rotRes);
    CAPTURE(rotZ);
    double diff = (rotRes - rotZ).squaredNorm();
    REQUIRE(diff < 0.001);

    CAPTURE(transRes);
    CAPTURE(trans);
    REQUIRE((transRes - trans).norm() < 0.001);

    CAPTURE(scaleRes);
    REQUIRE(scaleRes == 1);
}

TEST_CASE("computeRotZ scale changed") {
    Eigen::Matrix3d pts1;
    pts1 << 3.40188, 2.9844, -1.64777,
            -1.05617, 4.11647, 2.6823,
            2.83099, -3.02449, -2.22225;

    double trueRad = 0.653 * M_PI;
    Eigen::Matrix3d rotZ = Eigen::AngleAxisd(trueRad, Eigen::Vector3d::UnitZ()).toRotationMatrix();

    Eigen::Vector3d trans(3.13321, -1.05617, 2.83099);

    float scale = 1.3211;

    Eigen::Matrix3d pts2 = (scale * rotZ * pts1).colwise() + trans;

    Eigen::Matrix3d pts3 = (rotZ.transpose() * pts2).colwise() - rotZ.transpose() * trans;
    CAPTURE(pts1);
    CAPTURE(pts2);
    CAPTURE(pts3);

    Eigen::Matrix3d rotRes;
    Eigen::Vector3d transRes;
    float scaleRes;
    slam::computeRotZ(pts1, pts2, rotRes, transRes, scaleRes);

    CAPTURE(rotRes);
    CAPTURE(rotZ);
    double diff = (rotRes - rotZ).squaredNorm();
    REQUIRE(diff < 0.001);

    CAPTURE(transRes);
    CAPTURE(trans);
    REQUIRE((transRes - trans).norm() < 0.001);

    CAPTURE(scaleRes);
    REQUIRE(scaleRes == scale);
}

TEST_CASE("computeRotZ random cases") {
    // std::random_device rd;
    std::mt19937 rng(3249);
    std::uniform_real_distribution<float> dist(-1, 1);

    for (int i = 0; i < 100; i++) {
        Eigen::Matrix3d pts1 = Eigen::Matrix3d::Random() * 5;

        double trueRad = dist(rng) * M_PI;
        Eigen::Matrix3d rotZ = Eigen::AngleAxisd(trueRad, Eigen::Vector3d::UnitZ()).toRotationMatrix();

        Eigen::Vector3d trans = Eigen::Vector3d::Random() * 5;

        Eigen::Matrix3d pts2 = (rotZ * pts1).colwise() + trans;

        Eigen::Matrix3d rotRes;
        Eigen::Vector3d transRes;
        float scaleRes;
        slam::computeRotZ(pts1, pts2, rotRes, transRes, scaleRes);

        CAPTURE(rotRes);
        CAPTURE(rotZ);
        double diff = (rotRes - rotZ).squaredNorm();
        REQUIRE(diff < 0.001);

        CAPTURE(transRes);
        CAPTURE(trans);
        REQUIRE((transRes - trans).norm() < 0.001);

        CAPTURE(scaleRes);
        REQUIRE(scaleRes == 1);
    }
}

TEST_CASE("computeRotZ random cases + noise") {
    // std::random_device rd;
    std::mt19937 rng(2432);
    std::uniform_real_distribution<float> dist(-1, 1);

    for (int i = 0; i < 100; i++) {
        Eigen::Matrix3d pts1 = Eigen::Matrix3d::Random() * 5;

        double trueRad = dist(rng) * M_PI;
        Eigen::Matrix3d rotZ = Eigen::AngleAxisd(
            trueRad + dist(rng) * 0.05,
            Eigen::Vector3d::UnitZ()
        ).toRotationMatrix();

        Eigen::Vector3d trans = Eigen::Vector3d::Random() * 5;

        Eigen::Matrix3d pts2 = (rotZ * pts1).colwise() + (trans + Eigen::Vector3d::Random() * 0.1);

        Eigen::Matrix3d rotRes;
        Eigen::Vector3d transRes;
        float scaleRes;
        slam::computeRotZ(pts1, pts2, rotRes, transRes, scaleRes);

        CAPTURE(rotRes);
        CAPTURE(rotZ);
        double diff = (rotRes - rotZ).squaredNorm();
        REQUIRE(diff < 0.2);

        CAPTURE(transRes);
        CAPTURE(trans);
        REQUIRE((transRes - trans).norm() < 0.2);

        CAPTURE(scaleRes);
        REQUIRE(scaleRes == 1);
    }
}

#endif
