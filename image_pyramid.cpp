#include "image_pyramid.hpp"
#include "static_settings.hpp"
#include "../odometry/parameters.hpp"
#include "../util/logging.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <accelerated-arrays/opencv_adapter.hpp>

namespace slam {
namespace {
// note: as long as we compute anything from these on the CPU, it makes sense
// to store at least a copy of the results on the CPU side
struct CpuStoredImagePyramid : ImagePyramid {
    std::vector<cv::Mat> pyramid, blurredPyramid;
    std::vector<std::unique_ptr<accelerated::Image>> pyramidImgCpu, blurredPyramidImgCpu;

    CpuStoredImagePyramid(int levels) {
        pyramid.resize(levels);
        blurredPyramid.resize(levels);
        pyramidImgCpu.resize(levels);
        blurredPyramidImgCpu.resize(levels);
    }

    std::size_t numberOfLevels() const final {
        return pyramid.size();
    }

    accelerated::Image &getLevel(std::size_t level) final {
        if (!pyramidImgCpu.at(level)) {
            auto &cvImg = pyramid.at(level);
            assert(!cvImg.empty());
            pyramidImgCpu[level] = accelerated::opencv::ref(cvImg);
        }
        return *pyramidImgCpu.at(level);
    }

    accelerated::Image &getBlurredLevel(std::size_t level) final {
        if (!blurredPyramidImgCpu.at(level)) {
            auto &cvImg = blurredPyramid.at(level);
            assert(!cvImg.empty());
            blurredPyramidImgCpu[level] = accelerated::opencv::ref(cvImg);
        }
        return *blurredPyramidImgCpu.at(level);
    }

    void debugVisualize(cv::Mat &target) final {
        cv::Mat lev0 = pyramid.at(0);
        cv::vconcat(lev0, lev0, target);
        cv::Mat targetOtherSide(target, cv::Rect(0, lev0.rows, lev0.cols, lev0.rows));
        for (std::size_t i = 0; i < pyramid.size(); ++i) {
            auto &level = pyramid.at(i);
            level.copyTo(cv::Mat(target, cv::Rect(0, 0, level.cols, level.rows)));
            cv::Mat blurTarget(targetOtherSide, cv::Rect(0, 0, level.cols, level.rows));
            blurredPyramid.at(i).copyTo(blurTarget);
        }
    }
};

struct CpuImagePyramid : CpuStoredImagePyramid {
    const StaticSettings &settings;

    CpuImagePyramid(const StaticSettings &settings) :
        CpuStoredImagePyramid(settings.parameters.slam.orbScaleLevels),
        settings(settings)
    {}

    void update(tracker::Image &trackerImage) final {
        auto image = reinterpret_cast<tracker::CpuImage&>(trackerImage).getOpenCvMat();
        const auto &parameters = settings.parameters.slam;

        assert(pyramid.size() == parameters.orbScaleLevels);
        assert(blurredPyramid.size() == parameters.orbScaleLevels);

        image.copyTo(pyramid.at(0));
        for (unsigned int level = 1; level < parameters.orbScaleLevels; ++level) {
            const double scale = settings.scaleFactors.at(level);
            const cv::Size size(std::round(image.cols * 1.0 / scale), std::round(image.rows * 1.0 / scale));
            cv::resize(pyramid.at(level - 1), pyramid.at(level), size, 0, 0, cv::INTER_LINEAR);
        }

        for (unsigned level = 0; level < parameters.orbScaleLevels; ++level) {
            cv::Mat &blurredImage = blurredPyramid.at(level);
            cv::GaussianBlur(pyramid.at(level), blurredImage, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);
        }
    }

    accelerated::Image &getGpuLevel(std::size_t level) final {
        assert(false && "not GPU");
        return getLevel(level);
    }

    bool isGpu() const final {
        return false;
    }
};

struct GpuImagePyramid : CpuStoredImagePyramid {
    const StaticSettings &settings;
    std::vector<std::unique_ptr<accelerated::Image>> pyramidGpu, blurredPyramidGpu, blurTmpPyramidGpu;
    accelerated::operations::Function resizeOp, blurX, blurY;

    static std::vector<double> gaussianKernel1D(int width, double stdev) {
        std::vector<double> kernel;
        double sum = 0.0;
        for (int i = 0; i < width; ++i) {
            double x = i - (width - 1) * 0.5;
            double v = std::exp(-0.5 * x * x / (stdev * stdev));
            sum += v;
            kernel.push_back(v);
        }
        for (auto &v : kernel) v /= sum;
        return kernel;
    }

    static std::vector<std::vector<double>> gaussianKernel(bool yDir) {
        constexpr double stdev = 2.0;
        const int width = 7;
        auto kernel1D = gaussianKernel1D(width, stdev);
        std::vector<std::vector<double>> kernel;
        if (yDir) {
            for (const auto &el : kernel1D) kernel.push_back({ el });
        } else {
            kernel.push_back(kernel1D);
        }
        return kernel;
    }

    GpuImagePyramid(const StaticSettings &settings,
            const accelerated::Image &accImage,
            accelerated::Image::Factory &imgFactory,
            accelerated::operations::StandardFactory &opFactory) :
        CpuStoredImagePyramid(settings.parameters.slam.orbScaleLevels),
        settings(settings)
    {
        log_debug("initializing GPU ORB image pyramid");
        const auto &parameters = settings.parameters.slam;

        assert(pyramid.size() == parameters.orbScaleLevels);
        assert(blurredPyramid.size() == parameters.orbScaleLevels);

        pyramidGpu.push_back(imgFactory.createLike(accImage));
        blurredPyramidGpu.push_back(imgFactory.createLike(accImage));
        blurTmpPyramidGpu.push_back(imgFactory.createLike(accImage));

        for (unsigned int level = 1; level < parameters.orbScaleLevels; ++level) {
            const double scale = settings.scaleFactors.at(level);
            const int w = std::round(accImage.width * 1.0 / scale);
            const int h = std::round(accImage.height * 1.0 / scale);

            pyramidGpu.push_back(imgFactory.create(w, h, 1, accImage.dataType));
            blurredPyramidGpu.push_back(imgFactory.createLike(*pyramidGpu.back()));
            blurTmpPyramidGpu.push_back(imgFactory.createLike(*pyramidGpu.back()));
        }

        for (unsigned int level = 0; level < parameters.orbScaleLevels; ++level) {
            auto &cur = *pyramidGpu.at(level);
            pyramid.at(level) = accelerated::opencv::emptyLike(cur);
            blurredPyramid.at(level) = accelerated::opencv::emptyLike(cur);
        }

        resizeOp = opFactory.rescale()
            .setInterpolation(accelerated::Image::Interpolation::LINEAR)
            .build(accImage);

        const auto BORDER_TYPE = accelerated::Image::Border::MIRROR;
        blurX = opFactory.fixedConvolution2D(gaussianKernel(false))
            .setBorder(BORDER_TYPE)
            .build(accImage);
        blurY = opFactory.fixedConvolution2D(gaussianKernel(true))
            .setBorder(BORDER_TYPE)
            .build(accImage);
    }

    void update(tracker::Image &trackerImage) final {
        auto &accImage = trackerImage.getAccImage();
        const int n = pyramid.size();
        accelerated::operations::callUnary(resizeOp, accImage, *pyramidGpu.at(0));
        accelerated::operations::callUnary(blurX, accImage, *blurTmpPyramidGpu.at(0));
        accelerated::operations::callUnary(blurY, *blurTmpPyramidGpu.at(0), *blurredPyramidGpu.at(0));
        accelerated::opencv::copy(accImage, pyramid.at(0));
        accelerated::opencv::copy(*blurredPyramidGpu.at(0), blurredPyramid.at(0));

        for (int level = 1; level < n; ++level) {
            auto &cur = *pyramidGpu.at(level);
            accelerated::operations::callUnary(resizeOp, *pyramidGpu.at(level - 1), cur);
            auto &curTmp = *blurTmpPyramidGpu.at(level);
            accelerated::operations::callUnary(blurX, cur, curTmp);
            auto &curBlur = *blurredPyramidGpu.at(level);
            accelerated::operations::callUnary(blurY, curTmp, curBlur);

            accelerated::opencv::copy(cur, pyramid.at(level));
            auto fut = accelerated::opencv::copy(curBlur, blurredPyramid.at(level));

            if (level == n - 1) fut.wait();
        }
    }

    accelerated::Image &getGpuLevel(std::size_t level) final {
        return *pyramidGpu.at(level);
    }

    bool isGpu() const final {
        return true;
    }
};
}

std::unique_ptr<ImagePyramid> ImagePyramid::build(const StaticSettings &s, tracker::Image &img) {
    auto &accImg = img.getAccImage();
    if (accImg.storageType == accelerated::Image::StorageType::CPU || !s.parameters.slam.useGpuImagePyramid) {
        return std::unique_ptr<ImagePyramid>(new CpuImagePyramid(s));
    } else {
        return std::unique_ptr<ImagePyramid>(new GpuImagePyramid(s,
            accImg,
            img.getImageFactory(),
            img.getOperationsFactory()));
    }
}

ImagePyramid::~ImagePyramid() = default;
}
