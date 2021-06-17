#ifndef SLAM_OPENCV_DATA_PUBLISHER_HPP
#define SLAM_OPENCV_DATA_PUBLISHER_HPP

#include "../slam/viewer_data_publisher.hpp"
#include <opencv2/core.hpp>

namespace slam {

class OpenCVViewerDataPublisher : public ViewerDataPublisher {
public:
    OpenCVViewerDataPublisher(const cmd::ParametersSlam &parameters);

    void visualizeKeyframe(
        const MapDB &mapDB,
        const cv::Mat &frame,
        const Keyframe &kf
    ) final;

    void visualizeOrbs(
        const cv::Mat &frame,
        const Keyframe &kf
    ) final;

    void visualizeMapPointSearch(
        const cv::Mat &frame,
        const SearchedMapPointVector &searched,
        const Vector2dVector &mps,
        const Vector2dVector &kps
    ) final;

    void showMatches(
        const Keyframe &kf1,
        const Keyframe &kf2,
        const std::vector<std::pair<KpId, KpId>> &matches,
        MatchType matchType
    ) final;

    virtual void visualizeOther(const cv::Mat &mat) final;

    std::map<std::string, cv::Mat> pollVisualizations() final;
    void setFrameRotation(int rotation) final;

private:
    cv::Mat tmpWindow, tmpWindow2, tmpWindow3;
    cv::Mat keyframeWindow, matchWindow, matchWindowLoop, searchWindow, otherWindow;
    // Number of CW 90 degree rotations for frame visualizations.
    int rotation = 0;
};
}

#endif
