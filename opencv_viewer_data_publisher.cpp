#include <opencv2/opencv.hpp>
#include "opencv_viewer_data_publisher.hpp"
#include "../api/slam.hpp"
#include "../slam/keyframe.hpp"
#include "../slam/map_point.hpp"
#include "../slam/mapdb.hpp"

#include "../tracker/util.hpp"
#include "../util/util.hpp"

namespace slam {

// local helpers
namespace {

const cv::Scalar triangulatedColor{0, 255, 255}; // yellow
const cv::Scalar unsureColor{255,255,0}; // cyan
const cv::Scalar trackedColor{255, 255, 255}; // white
const cv::Scalar noMatchColor{255, 0, 0}; // blue
const cv::Scalar orbColor{0, 128, 255}; // orange
const cv::Scalar white{255, 255, 255};

void scaleImage(cv::Mat &m) {
    constexpr float maxLonger = 1960.0;
    float scale = maxLonger / static_cast<float>(std::max(m.cols, m.rows));
    if (scale < 1.0) {
        cv::resize(m, m, cv::Size(), scale, scale, cv::INTER_CUBIC);
    }
}

void drawText(cv::Mat &img, cv::Mat &imgWithText, std::string text) {
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_PLAIN, 1.5, 1, 0);

    imgWithText.create(img.rows + textSize.height + 10, img.cols, img.type());
    img.copyTo(imgWithText.rowRange(0, img.rows).colRange(0, img.cols));
    imgWithText.rowRange(img.rows, imgWithText.rows) = cv::Scalar(0);

    cv::Point bottomLeft(5, imgWithText.rows - 5);
    cv::putText(imgWithText, text, bottomLeft, cv::FONT_HERSHEY_PLAIN, 1.5, white, 1, 8);
}

class PointRotator {
public:
    PointRotator(double width, double height, int rotation) {
        switch (util::modulo(rotation, 4)) {
            case 0:
                R << 1, 0, 0, 1;
                t << 0, 0;
                break;
            case 1:
                R << 0, -1, 1, 0;
                t << height, 0;
                break;
            case 2:
                R << -1, 0, 0, -1;
                t << width, height;
                break;
            default:
                R << 0, 1, -1, 0;
                t << 0, width;
                break;
        }
    }

    Eigen::Vector2d rotate(const Eigen::Vector2d &p) {
        return R * p + t;
    }

private:
    Eigen::Matrix2d R;
    Eigen::Vector2d t;
};

void copyAsColor(const cv::Mat &frame, cv::Mat &target) {
    if (frame.channels() == 1) {
        cv::cvtColor(frame, target, cv::COLOR_GRAY2BGR);
    } else {
        frame.copyTo(target);
    }
}

void desaturate(cv::Mat &frame, cv::Mat work = {}) {
    //  :)
    cv::cvtColor(frame, work, cv::COLOR_BGR2GRAY);
    cv::cvtColor(work, frame, cv::COLOR_GRAY2BGR);
}

} // namespace

OpenCVViewerDataPublisher::OpenCVViewerDataPublisher(const cmd::ParametersSlam &parameters) :
    ViewerDataPublisher(parameters)
{}

// * Orange dot: ORB keypoint with no associated map point.
// * Blue dot: Associated map point, but it's not triangulated.
// * Cyan square: "Unsure" or "bad" triangulation.
// * Yellow square: Successful triangulation.
// * White line: Difference between keypoint and map point projection.
// * White circle: Reprojection failed.
void OpenCVViewerDataPublisher::visualizeKeyframe(
    const MapDB &mapDB,
    const cv::Mat &frame,
    const Keyframe &kf
) {
    cv::Mat &img = tmpWindow;
    copyAsColor(frame, img);

    PointRotator r(img.cols, img.rows, rotation);
    tracker::util::rotateMatrixCW90(img, img, rotation);
    std::vector<cv::Point2f> points;
    for (const auto &kp : kf.shared->keyPoints) {
        Eigen::Vector2d p = r.rotate(Eigen::Vector2d(kp.pt.x, kp.pt.y));
        points.push_back(cv::Point2f(p(0), p(1)));
    }

    for (unsigned i = 0; i < kf.shared->keyPoints.size(); i++) {
        cv::Point2f center = points[i];

        cv::circle(img, center, 2, orbColor, -1);

        const MpId mapPointId = kf.mapPoints[i];
        if (mapPointId.v == -1)
            continue;

        auto &mapPoint = mapDB.mapPoints.at(mapPointId);
        if (mapPoint.status == MapPointStatus::NOT_TRIANGULATED) {
            cv::circle(img, center, 2, noMatchColor, -1);
        }
        else {
            float squareSize = 5;
            cv::Point2f pt_begin{center.x - squareSize, center.y - squareSize};
            cv::Point2f pt_end{center.x + squareSize, center.y + squareSize};

            cv::Scalar color = mapPoint.status == MapPointStatus::TRIANGULATED ? triangulatedColor : unsureColor;
            cv::rectangle(img, pt_begin, pt_end, color);
            cv::circle(img, center, 2, color, 1);

            Eigen::Vector2f reprojEigen;
            bool reprojectionOk = kf.reproject(mapPoint.position, reprojEigen);
            Eigen::Vector2d d = r.rotate(reprojEigen.cast<double>());

            cv::Point2f reprojection(d.x(), d.y());
            cv::line(img, center, reprojection, trackedColor);
            cv::circle(img, reprojection, 2, trackedColor, 1);

            if (!reprojectionOk) {
                cv::circle(img, center, squareSize, trackedColor, 3);
            }
        }
    }

    std::stringstream ss;
    {
        std::lock_guard<std::mutex> l(m);
        ss << "KFs: " << keyframes.size() << ", "
           << "MPs: " << mapPoints.size();
    }

    drawText(img, tmpWindow2, ss.str());

    scaleImage(tmpWindow2);

    std::lock_guard<std::mutex> l(m);
    tmpWindow2.copyTo(keyframeWindow);
}


void OpenCVViewerDataPublisher::visualizeOrbs(const cv::Mat &frame, const Keyframe &kf) {
    cv::Mat &img = tmpWindow;
    copyAsColor(frame, img);
    desaturate(img, tmpWindow2);

    const cv::Scalar otherColor(0x50, 0x50, 0x50);
    const cv::Scalar mapPointColor(0, 0xa0, 0xa0);
    const cv::Scalar trackColor(0xff, 0, 0xff);
    const cv::Scalar trackMapPointColor(0x80, 0xff, 0xff);

    for (unsigned i = 0; i < kf.shared->keyPoints.size(); ++i) {
        const auto &kp = kf.shared->keyPoints.at(i);
        int radius = 2 + kp.octave*3; // TODO;

        cv::Scalar color;
        bool isTracked = kf.keyPointToTrackId.count(KpId(i));
        bool hasMapPoint = kf.mapPoints.at(i).v >= 0;
        if (hasMapPoint && isTracked) {
            color = trackMapPointColor;
        } else if (hasMapPoint) {
            color = mapPointColor;
        } else if (isTracked) {
            color = trackColor;
        } else {
            color = otherColor;
        }
        cv::Point2f c(kp.pt.x, kp.pt.y);
        cv::circle(img, c, radius, color, 1);
        const double aRad = kp.angle / 180.0 * M_PI;
        cv::line(img, c, c + cv::Point2f(std::cos(aRad), std::sin(aRad))*radius, color, 1);
    }

    tracker::util::rotateMatrixCW90(img, tmpWindow2, rotation);
    scaleImage(tmpWindow2);

    std::lock_guard<std::mutex> l(m);
    tmpWindow2.copyTo(otherWindow);
}

// * Orange dot: ORB keypoint for which no matching map point was found.
// * Cyan circle: Projection of map point for which no matching ORB was found.
// * White: ORB (dot) and its matching map point projection (circle).
void OpenCVViewerDataPublisher::visualizeMapPointSearch(
    const cv::Mat &frame,
    const SearchedMapPointVector &searched,
    const Vector2dVector &mps,
    const Vector2dVector &kps
) {
    cv::Mat &img = tmpWindow;

    if (frame.channels() == 1) {
        cv::cvtColor(frame, img, cv::COLOR_GRAY2BGR);
    } else {
        frame.copyTo(img);
    }
    PointRotator r(img.cols, img.rows, rotation);
    tracker::util::rotateMatrixCW90(img, img, rotation);

    // Map points here contain also the matches, draw first
    // so they are covered by the match drawings.
    for (const Eigen::Vector2d &mp : mps) {
        Eigen::Vector2d d = r.rotate(mp);
        cv::Point2f p(d.x(), d.y());
        cv::circle(img, p, 4, unsureColor);
    }

    for (const Eigen::Vector2d &kp : kps) {
        Eigen::Vector2d d = r.rotate(kp);
        cv::Point2f p(d.x(), d.y());
        cv::circle(img, p, 2, orbColor);
    }

    for (const SearchedMapPoint &s : searched) {
        Eigen::Vector2d d = r.rotate(s.kp);
        cv::Point2f p0(d.x(), d.y());

        d = r.rotate(s.mp);
        cv::Point2f p1(d.x(), d.y());

        cv::line(img, p0, p1, trackedColor);
        cv::circle(img, p0, 2, trackedColor);
        cv::circle(img, p1, 4, trackedColor);
    }

    char s[100];
    double percent = 100.0 * static_cast<double>(searched.size()) / static_cast<double>(mps.size());
    snprintf(s, 100, "MPs: %zu, matches: %zu (%.0f%%), ORBs: %zu", mps.size(), searched.size(), percent, kps.size());
    drawText(img, tmpWindow2, s);

    scaleImage(tmpWindow2);

    std::lock_guard<std::mutex> l(m);
    tmpWindow2.copyTo(searchWindow);
}

void OpenCVViewerDataPublisher::showMatches(
    const Keyframe &kf1,
    const Keyframe &kf2,
    const std::vector<std::pair<KpId, KpId>> &matches,
    MatchType matchType
) {
    std::vector<cv::DMatch> dmatches;
    for (const auto &match : matches) {
        dmatches.emplace_back(match.first.v, match.second.v, 1);
    }

    PointRotator r(kf1.shared->imgDbg.cols, kf1.shared->imgDbg.rows, rotation);
    std::vector<cv::KeyPoint> keyPoints1;
    std::vector<cv::KeyPoint> keyPoints2;
    constexpr int DUMMY = 100;
    for (const auto &kp : kf1.shared->keyPoints) {
        Eigen::Vector2d p = r.rotate(Eigen::Vector2d(kp.pt.x, kp.pt.y));
        cv::KeyPoint nkp(cv::Point2f(kp.pt.x, kp.pt.y), DUMMY);
        nkp.pt = cv::Point2f(p(0), p(1));
        keyPoints1.push_back(nkp);
    }
    for (const auto &kp : kf2.shared->keyPoints) {
        Eigen::Vector2d p = r.rotate(Eigen::Vector2d(kp.pt.x, kp.pt.y));
        cv::KeyPoint nkp(cv::Point2f(kp.pt.x, kp.pt.y), DUMMY);
        nkp.pt = cv::Point2f(p(0), p(1));
        keyPoints2.push_back(nkp);
    }

    tracker::util::rotateMatrixCW90(kf1.shared->imgDbg, tmpWindow, rotation);
    tracker::util::rotateMatrixCW90(kf2.shared->imgDbg, tmpWindow2, rotation);

    cv::Mat &outImg = tmpWindow3;
    cv::drawMatches(
        tmpWindow,
        keyPoints1,
        tmpWindow2,
        keyPoints2,
        dmatches,
        outImg
    );

    scaleImage(outImg);

    std::lock_guard<std::mutex> l(m);

    if (matchType == MatchType::MAPPER)
        outImg.copyTo(matchWindow);
    else
        outImg.copyTo(matchWindowLoop);
}

void OpenCVViewerDataPublisher::visualizeOther(const cv::Mat &mat) {
    tracker::util::rotateMatrixCW90(mat, tmpWindow, rotation);
    scaleImage(tmpWindow);
    tmpWindow.copyTo(otherWindow);
}

std::map<std::string, cv::Mat> OpenCVViewerDataPublisher::pollVisualizations() {
    std::lock_guard<std::mutex> l(m);
    std::map<std::string, cv::Mat> result;
    if (!matchWindow.empty()) {
        result["ORB matches"] = matchWindow.clone();
        matchWindow.resize(0); // clear without reallocating
    }
    if (!matchWindowLoop.empty()) {
        result["Loop ORB matches"] = matchWindowLoop.clone();
        matchWindowLoop.resize(0); // clear without reallocating
    }
    if (!keyframeWindow.empty()) {
        result["SLAM keyframe"] = keyframeWindow.clone();
        keyframeWindow.resize(0);
    }
    if (!searchWindow.empty()) {
        result["Map point search"] = searchWindow.clone();
        searchWindow.resize(0);
    }
    if (!otherWindow.empty()) {
        result["SLAM (other)"] = otherWindow.clone();
        otherWindow.resize(0);
    }
    return result;
}

void OpenCVViewerDataPublisher::setFrameRotation(int rotation) {
    this->rotation = rotation;
}

} // namespace slam
