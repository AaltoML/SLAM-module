#include <cassert>
#include <cmath>
#include <accelerated-arrays/image.hpp>
#include <accelerated-arrays/standard_ops.hpp>
#include <accelerated-arrays/cpu/image.hpp>
#include <accelerated-arrays/cpu/operations.hpp>

#include "feature_detector.hpp"
#include "static_settings.hpp"
#include "image_pyramid.hpp"
#include "../tracker/image.hpp"
#include "../odometry/parameters.hpp"
#include "../util/logging.hpp"
#include "../tracker/feature_detector.hpp"

namespace slam {
namespace {
class FeatureDetectorImplementation : public FeatureDetector {
public:
    std::size_t detect(ImagePyramid &imagePyramid, std::vector<KeyPointVector> &keypointsPerLevel) final {
        computeKeypoints(keypointsPerLevel, imagePyramid);

        unsigned int num_keypts = 0;
        for (unsigned int level = 0; level < parameters.orbScaleLevels; ++level) {
            num_keypts += keypointsPerLevel.at(level).size();
        }
        return num_keypts;
    }

    FeatureDetectorImplementation(const StaticSettings &settings, tracker::Image &modelImage) :
        parameters(settings.parameters.slam),
        gpuProcessor(modelImage.getProcessor()),
        gpuImgFactory(modelImage.getImageFactory()),
        gpuOpFactory(modelImage.getOperationsFactory())
    {
        for (auto maxKps : settings.maxNumberOfKeypointsPerLevel()) {
            // copy & change
            odometry::ParametersTracker params = settings.parameters.tracker;
            params.maxTracks = maxKps;
            const auto detector = settings.parameters.slam.slamFeatureDetector;
            if (!detector.empty()) params.featureDetector = detector;
            // subpixel adjustment would happen elsewhere, no need to disable
            featureDetectorParams.push_back(params);
        }

        cpuProcessor = accelerated::Processor::createInstant();
        cpuImgFactory = accelerated::cpu::Image::createFactory();
        cpuOpFactory = accelerated::cpu::operations::createFactory(*cpuProcessor);
    }

private:
    const odometry::ParametersSlam &parameters;
    accelerated::Processor &gpuProcessor;
    accelerated::Image::Factory &gpuImgFactory;
    accelerated::operations::StandardFactory &gpuOpFactory;

    std::unique_ptr<accelerated::Processor> cpuProcessor;
    std::unique_ptr<accelerated::Image::Factory> cpuImgFactory;
    std::unique_ptr<accelerated::operations::StandardFactory> cpuOpFactory;

    std::vector<std::unique_ptr<tracker::FeatureDetector>> featureDetectors;
    std::vector<odometry::ParametersTracker> featureDetectorParams;

    struct Workspace {
        std::vector<std::vector<tracker::Feature::Point>> featurePointsPerLevel;
    } work;

    void computeKeypoints(std::vector<KeyPointVector>& keypoints, ImagePyramid &pyramid) {
        keypoints.resize(parameters.orbScaleLevels);
        work.featurePointsPerLevel.resize(parameters.orbScaleLevels);

        const bool isGpu = pyramid.isGpu();
        for (unsigned int level = 0; level < parameters.orbScaleLevels; ++level) {
            auto &lev = isGpu ? pyramid.getGpuLevel(level) : pyramid.getLevel(level);
            auto &params = featureDetectorParams.at(level);

            if (level >= featureDetectors.size()) {
                // from our GFTT detector
                const auto minDim = std::min(lev.width, lev.height);
                const double su = minDim / 720.0 * 0.8;
                const int minDist = std::floor(params.gfttMinDistance * su + 0.5);
                params.gfttMinDistance = minDist;

                log_debug("Lazy-initializing %s feature detector for ORB level %d (%d x %d), max features: %d, minDist: %d",
                    params.featureDetector.c_str(), int(level), lev.width, lev.height, params.maxTracks, minDist);

                // Note: with a GPU image pyramid, this generates dozens of small textures since each
                // detection level has its own set of workspace textures for the GPU feature detection.
                featureDetectors.push_back(
                    tracker::FeatureDetector::build(lev.width, lev.height,
                        isGpu ? gpuProcessor : *cpuProcessor,
                        isGpu ? gpuImgFactory : *cpuImgFactory,
                        isGpu ? gpuOpFactory : *cpuOpFactory,
                        params));
            }

            auto fut = featureDetectors.at(level)->detect(lev,
                work.featurePointsPerLevel.at(level), {}, params.gfttMinDistance);

            if (level == parameters.orbScaleLevels - 1) fut.wait();
        }

        for (unsigned level = 0; level < parameters.orbScaleLevels; ++level) {
            // log_debug("detected %zu points at level %d", work.featurePoints.size(), level);

            constexpr unsigned MARGIN = StaticSettings::ORB_PATCH_RADIUS;
            constexpr int min_x = MARGIN;
            constexpr int min_y = MARGIN;
            const auto &lev = isGpu ? pyramid.getGpuLevel(level) : pyramid.getLevel(level);
            const int max_x = lev.width - MARGIN;
            const int max_y = lev.height - MARGIN;

            auto &keyptsAtLev = keypoints.at(level);
            const auto &points = work.featurePointsPerLevel.at(level);
            keyptsAtLev.clear();
            keyptsAtLev.reserve(points.size());
            for (const auto& keypt : points) {
                const int x = std::round(keypt.x);
                const int y = std::round(keypt.y);
                // TODO: could support these margins directly in the feature detector
                if (x < min_x || y < min_y || x >= max_x || y >= max_y) {
                    continue;
                }
                keyptsAtLev.push_back({
                    .pt = {
                        .x = keypt.x,
                        .y = keypt.y
                    },
                    .angle = 0, // computed elsewhere
                    .octave = int(level),
                });
            }
        }
    }
};
}

std::unique_ptr<FeatureDetector> FeatureDetector::build(const StaticSettings &settings, tracker::Image &modelImage) {
    return std::unique_ptr<FeatureDetector>(new FeatureDetectorImplementation(settings, modelImage));
}

FeatureDetector::~FeatureDetector() = default;

}
