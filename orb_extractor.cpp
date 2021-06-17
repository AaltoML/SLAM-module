/*******************************************************************************

                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2009, Willow Garage Inc., all rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*******************************************************************************/

// This file is based on orb_extractor.cc and orb_extractor.h in OpenVSLAM
// which is licensed under the above BSD License, apparently inherited from
// OpenCV

#include "orb_extractor.hpp"
#include "feature_detector.hpp"
#include "image_pyramid.hpp"
#include "../odometry/parameters.hpp"
#include "../tracker/camera.hpp"
#include "../tracker/image.hpp"

#include <vector>
#include <list>

#include "openvslam/orb_point_pairs.h"
#include "openvslam/trigonometric.h"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <accelerated-arrays/opencv_adapter.hpp>

#ifdef USE_SSE_ORB
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif // USE_SSE_ORB

namespace slam {
namespace {
using namespace openvslam;
using namespace openvslam::feature;

class OrbExtractorImplementation : public OrbExtractor {
public:
    void detectAndExtract(
        tracker::Image &trackerImage,
        const tracker::Camera &camera,
        const std::vector<tracker::Feature> &tracks,
        KeyPointVector &keypts,
        std::vector<int> &keyptTrackIds) final
    {
        if (!imagePyramid) imagePyramid = ImagePyramid::build(settings, trackerImage);
        if (!featureDetector) featureDetector = FeatureDetector::build(settings, trackerImage);

        imagePyramid->update(trackerImage);

        // convert tracker keypoints & compute descriptors
        keypts.clear();
        keyptTrackIds.clear();

        for (const auto &track : tracks) {
            const auto &pt = track.points[0];
            const unsigned level = parameters.orbLkTrackLevel;

            const float scale = scale_factors_.at(level);
            const cv::Mat levelImg = accelerated::opencv::ref(imagePyramid->getLevel(level));
            const int x = cvRound(pt.x / scale);
            const int y = cvRound(pt.y / scale);
            const int margin = orb_patch_radius_;

            if (x >= margin && y >= margin &&
                x < levelImg.cols - margin && y < levelImg.rows - margin &&
                camera.isValidPixel(pt.x, pt.y))
            {
                keypts.push_back({
                    .pt = {
                        static_cast<float>(x),
                        static_cast<float>(y)
                    },
                    .angle = ic_angle(levelImg, x, y),
                    .octave = int(level)
                });
                const cv::Mat blurred_img = accelerated::opencv::ref(imagePyramid->getBlurredLevel(level));

                assert(!blurred_img.empty());
                assert(blurred_img.cols == levelImg.cols && blurred_img.rows == levelImg.rows);

                compute_orb_descriptor(
                    x, y, keypts.back().angle,
                    blurred_img,
                    keypts.back().descriptor);

                keypts.back().pt = pt; // correct scale
                keyptTrackIds.push_back(track.id);
            }
        }

        // find other keypoints
        auto &all_keypts = work.keypoints;
        featureDetector->detect(*imagePyramid, all_keypts);
        const unsigned num_keypts = dropInvalidKeypoints(camera, all_keypts) + keypts.size();

        // Compute orientations
        for (unsigned int level = 0; level < parameters.orbScaleLevels; ++level) {
            compute_orientation(accelerated::opencv::ref(imagePyramid->getLevel(level)), all_keypts.at(level));
        }

        unsigned int offset = keypts.size(); // number of tracker keypoints
        if (num_keypts == 0) return;

        for (unsigned int level = 0; level < parameters.orbScaleLevels; ++level) {
            auto& keypts_at_level = all_keypts.at(level);
            const auto num_keypts_at_level = keypts_at_level.size();

            if (num_keypts_at_level == 0) {
                continue;
            }

            const cv::Mat blurred_image = accelerated::opencv::ref(imagePyramid->getBlurredLevel(level));
            assert(!blurred_image.empty());
            compute_orb_descriptors(blurred_image, keypts_at_level, level);

            offset += num_keypts_at_level;

            const float scale_at_level = scale_factors_.at(level);
            for (const auto &kp : keypts_at_level) {
                keypts.push_back({
                    .pt = { kp.pt.x * scale_at_level, kp.pt.y * scale_at_level },
                    .angle = kp.angle,
                    .octave = kp.octave,
                    .descriptor = kp.descriptor
                });
                keyptTrackIds.push_back(-1);
            }
        }
    }

    OrbExtractorImplementation(const StaticSettings &settings) :
        settings(settings),
        parameters(settings.parameters.slam)
    {
        // compute scale pyramid information
        scale_factors_ = settings.scaleFactors;

        // Preparate  for computation of orientation
        u_max_.resize(fast_half_patch_size_ + 1);
        const unsigned int vmax = std::floor(fast_half_patch_size_ * std::sqrt(2.0) / 2 + 1);
        const unsigned int vmin = std::ceil(fast_half_patch_size_ * std::sqrt(2.0) / 2);
        for (unsigned int v = 0; v <= vmax; ++v) {
            u_max_.at(v) = std::round(std::sqrt(fast_half_patch_size_ * fast_half_patch_size_ - v * v));
        }
        for (unsigned int v = fast_half_patch_size_, v0 = 0; vmin <= v; --v) {
            while (u_max_.at(v0) == u_max_.at(v0 + 1)) {
                ++v0;
            }
            u_max_.at(v) = v0;
            ++v0;
        }
    }

    void debugVisualize(const tracker::Image &trackerImage, cv::Mat &target, VisualizationMode mode) const final {
        assert(mode == VisualizationMode::IMAGE_PYRAMID);
        (void)trackerImage;
        assert(imagePyramid);
        imagePyramid->debugVisualize(target);
    }

private:
    //! parameters for ORB extraction
    const StaticSettings &settings;
    const odometry::ParametersSlam &parameters;

    std::unique_ptr<FeatureDetector> featureDetector;

    std::unique_ptr<ImagePyramid> imagePyramid;

    //! half size of FAST patch
    static constexpr int fast_half_patch_size_ = StaticSettings::ORB_FAST_PATCH_HALF_SIZE;

    //! size of maximum ORB patch radius
    static constexpr unsigned int orb_patch_radius_ = StaticSettings::ORB_PATCH_RADIUS;

    //! A list of the scale factor of each pyramid layer
    std::vector<float> scale_factors_;

    //! Index limitation that used for calculating of keypoint orientation
    std::vector<int> u_max_;

    struct Workspace {
        std::vector<KeyPointVector> keypoints;
    } work;

    unsigned dropInvalidKeypoints(
        const tracker::Camera &camera,
        std::vector<KeyPointVector> &kpsPerLevel) const
    {
        unsigned nLeft = 0;
        for (std::size_t i = 0; i < kpsPerLevel.size(); ++i) {
            float scale = scale_factors_.at(i);
            auto &kps = kpsPerLevel.at(i);
            kps.erase(std::remove_if(kps.begin(), kps.end(),
                [&camera, scale](const KeyPoint &kp) {
                    return !camera.isValidPixel(kp.pt.x * scale, kp.pt.y * scale);
                }
            ), kps.end());
            nLeft += kps.size();
        }
        return nLeft;
    }

    void compute_orientation(const cv::Mat& image, KeyPointVector& keypts) const {
        for (auto& keypt : keypts) {
            keypt.angle = ic_angle(image, cvRound(keypt.pt.x), cvRound(keypt.pt.y));
        }
    }

    float ic_angle(const cv::Mat& image, int x, int y) const {
        int m_01 = 0, m_10 = 0;

        assert(x >= fast_half_patch_size_);
        assert(x < image.cols - fast_half_patch_size_);
        assert(y >= fast_half_patch_size_);
        assert(y < image.rows - fast_half_patch_size_);

        // note: .at(y,x) if using "matrix" indexing, but .at(cv::Point2f(x,y))
        const uchar* const center = &image.at<uchar>(y, x);

        for (int u = -fast_half_patch_size_; u <= fast_half_patch_size_; ++u) {
            m_10 += u * center[u];
        }

        const auto step = static_cast<int>(image.step1());
        for (int v = 1; v <= fast_half_patch_size_; ++v) {
            unsigned int v_sum = 0;
            const int d = u_max_.at(v);
            for (int u = -d; u <= d; ++u) {
                const int val_plus = center[u + v * step];
                const int val_minus = center[u - v * step];
                v_sum += (val_plus - val_minus);
                m_10 += u * (val_plus + val_minus);
            }
            m_01 += v * v_sum;
        }

        // unlike std::atan2, this returns the angle in DEGREES!!
        return cv::fastAtan2(m_01, m_10);
    }

    void compute_orb_descriptors(const cv::Mat& image, KeyPointVector& keypts, int level) const {
        for (auto &kp : keypts) {
            kp.octave = level;
            compute_orb_descriptor(cvRound(kp.pt.x), cvRound(kp.pt.y), kp.angle, image, kp.descriptor);
        }
    }

    void compute_orb_descriptor(int x, int y, float angleDeg, const cv::Mat& image, KeyPoint::Descriptor &descObj) const {
        auto *desc = reinterpret_cast<uchar*>(descObj.data());
        const float angle = angleDeg * M_PI / 180.0;
        const float cos_angle = util::cos(angle);
        const float sin_angle = util::sin(angle);

        // note: at(y,x) if using "matrix" indexing, but .at(cv::Point2f(x,y))
        const uchar* const center = &image.at<uchar>(y, x);
        const auto step = static_cast<int>(image.step);

        assert(x >= int(orb_patch_radius_));
        assert(x < int(image.cols - orb_patch_radius_));
        assert(y >= int(orb_patch_radius_));
        assert(y < int(image.rows - orb_patch_radius_));

    #ifdef USE_SSE_ORB
    #if !((defined _MSC_VER && defined _M_X64)                            \
          || (defined __GNUC__ && defined __x86_64__ && defined __SSE3__) \
          || CV_SSE3)
    #error "The processor is not compatible with SSE. Please configure the CMake with -DUSE_SSE_ORB=OFF."
    #endif

        const __m128 _trig1 = _mm_set_ps(cos_angle, sin_angle, cos_angle, sin_angle);
        const __m128 _trig2 = _mm_set_ps(-sin_angle, cos_angle, -sin_angle, cos_angle);
        __m128 _point_pairs;
        __m128 _mul1;
        __m128 _mul2;
        __m128 _vs;
        __m128i _vi;
        alignas(16) int32_t ii[4];

    #define COMPARE_ORB_POINTS(shift)                          \
        (_point_pairs = _mm_load_ps(orb_point_pairs + shift),  \
         _mul1 = _mm_mul_ps(_point_pairs, _trig1),             \
         _mul2 = _mm_mul_ps(_point_pairs, _trig2),             \
         _vs = _mm_hadd_ps(_mul1, _mul2),                      \
         _vi = _mm_cvtps_epi32(_vs),                           \
         _mm_store_si128(reinterpret_cast<__m128i*>(ii), _vi), \
         center[ii[0] * step + ii[2]] < center[ii[1] * step + ii[3]])

    #else

    #define GET_VALUE(shift)                                                                                        \
        (center[cvRound(*(orb_point_pairs + shift) * sin_angle + *(orb_point_pairs + shift + 1) * cos_angle) * step \
                + cvRound(*(orb_point_pairs + shift) * cos_angle - *(orb_point_pairs + shift + 1) * sin_angle)])

    #define COMPARE_ORB_POINTS(shift) \
        (GET_VALUE(shift) < GET_VALUE(shift + 2))

    #endif

        // interval: (X, Y) x 2 points x 8 pairs = 32
        static constexpr unsigned interval = 32;

        for (unsigned int i = 0; i < orb_point_pairs_size / interval; ++i) {
            int32_t val = COMPARE_ORB_POINTS(i * interval);
            val |= COMPARE_ORB_POINTS(i * interval + 4) << 1;
            val |= COMPARE_ORB_POINTS(i * interval + 8) << 2;
            val |= COMPARE_ORB_POINTS(i * interval + 12) << 3;
            val |= COMPARE_ORB_POINTS(i * interval + 16) << 4;
            val |= COMPARE_ORB_POINTS(i * interval + 20) << 5;
            val |= COMPARE_ORB_POINTS(i * interval + 24) << 6;
            val |= COMPARE_ORB_POINTS(i * interval + 28) << 7;
            desc[i] = static_cast<uchar>(val);
        }

    #undef GET_VALUE
    #undef COMPARE_ORB_POINTS
    }
};
}

std::unique_ptr<OrbExtractor> OrbExtractor::build(const StaticSettings &settings) {
    return std::unique_ptr<OrbExtractor>(new OrbExtractorImplementation(settings));
}
}
