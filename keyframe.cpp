#include "keyframe.hpp"

#include <opencv2/imgproc.hpp> // gray frame conversion
#include <functional>

#include "../tracker/image.hpp"
#include "../odometry/parameters.hpp"
#include "../util/logging.hpp"
#include "../util/string_utils.hpp"
#include "map_point.hpp"
#include "mapdb.hpp"
#include "../api/slam.hpp"
#include "orb_extractor.hpp"

using Eigen::Matrix4d;
using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector2f;


namespace slam {
namespace {
cv::Vec3b BLACK(0, 0, 0);
std::vector<int> keyPointTrackIds;

cv::Vec3b getPixel(const cv::Mat &img, cv::Point2f pt);
void setStereoPointCloud(const MapperInput &mapperInput, KeyframeShared &kfShared);

bool convertKeypointToBearing(const tracker::Camera &camera, const KeyPoint& kp, Eigen::Vector3d &out) {
    return camera.pixelToRay(Eigen::Vector2d(kp.pt.x, kp.pt.y), out);
}

// shared code for detectFullFeatures &  detectTrackerFeatures
void processKeyPoints(Keyframe &kf, std::vector<cv::Vec3b> &colors, const MapperInput &mapperInput)  {
    // populate colors for debugging and nice visualizations
    if (!mapperInput.colorFrame.empty()) {
        colors.reserve(kf.shared->keyPoints.size());
        cv::Rect rect(cv::Point(), mapperInput.colorFrame.size());
        for (const auto &kp : kf.shared->keyPoints) {
            cv::Point2f p(kp.pt.x, kp.pt.y);
            colors.push_back(
                rect.contains(p)
                ? getPixel(mapperInput.colorFrame, p)
                : BLACK
            );
        }
    }

    kf.mapPoints.assign(kf.shared->keyPoints.size(), MpId(-1));

    std::map<TrackId, size_t> trackIdToIndex;
    for (size_t idx = 0; idx < mapperInput.trackerFeatures.size(); ++idx)
        trackIdToIndex[TrackId(mapperInput.trackerFeatures.at(idx).id)] = idx;

    for (size_t kpIdx = 0; kpIdx < kf.shared->keyPoints.size(); ++kpIdx) {
        auto &kp = kf.shared->keyPoints.at(kpIdx);
        float depth = -1;
        if (kf.keyPointToTrackId.count(KpId(kpIdx))) {
            depth = mapperInput.trackerFeatures.at(trackIdToIndex.at(kf.keyPointToTrackId.at(KpId(kpIdx)))).depth;
        }
        if (depth < 0) {
            depth = mapperInput.frame->getDepth(Eigen::Vector2f(kp.pt.x, kp.pt.y));
        }
        kf.keyPointDepth.push_back(depth);
        // Note: must have filtered out any possible invalid keypoints
        // before this operation
        assert(convertKeypointToBearing(*kf.shared->camera, kp, kp.bearing));
    }
}
} // namespace

// needs to be defined in a file that include tracker/image.hpp
MapperInput::MapperInput() = default;
MapperInput::~MapperInput() = default;

// For cereal.
Keyframe::Keyframe() {}

Keyframe::Keyframe(const MapperInput &mapperInput) :
    shared(std::make_shared<KeyframeShared>()),
    id(KfId(mapperInput.poseTrail.at(0).frameNumber)),
    previousKfId(KfId(-1)),
    nextKfId(KfId(-1)),
    poseCW(Eigen::Matrix4d::Identity()),
    origPoseCW(mapperInput.poseTrail[0].pose),
    uncertainty(mapperInput.poseTrail[0].uncertainty),
    t(mapperInput.poseTrail[0].t),
    hasFullFeatures(false)
{
    assert(id.v >= 0);
    shared->camera = mapperInput.frame->getCamera();
    setStereoPointCloud(mapperInput, *shared);
}

void Keyframe::addFullFeatures(
    const MapperInput &mapperInput,
    OrbExtractor &orb) {
    hasFullFeatures = true;
    orb.detectAndExtract(
        *mapperInput.frame,
        *shared->camera,
        mapperInput.trackerFeatures,
        shared->keyPoints,
        keyPointTrackIds);

    for (unsigned i = 0; i < shared->keyPoints.size(); ++i) {
        int trackId = keyPointTrackIds.at(i);
        if (trackId >= 0) {
            keyPointToTrackId.emplace(KpId(i), trackId);
        }
    }

    processKeyPoints(*this, shared->colors, mapperInput);

    shared->featureSearch = FeatureSearch::create(shared->keyPoints);
}

void Keyframe::addTrackerFeatures(const MapperInput &mapperInput) {
    for (unsigned i = 0; i < mapperInput.trackerFeatures.size(); ++i) {
        const auto &track = mapperInput.trackerFeatures.at(i);
        const auto &pt = track.points[0];
        if (!shared->camera->isValidPixel(pt.x, pt.y)) continue;
        shared->keyPoints.push_back({
            .pt = pt,
            // should not be used
            .angle = 0,
            .octave = 0
        });
        keyPointToTrackId.emplace(KpId(i), track.id);
    }

    processKeyPoints(*this, shared->colors, mapperInput);
}

Keyframe::Keyframe(const Keyframe &kf) :
    shared(kf.shared),
    id(kf.id),
    previousKfId(kf.previousKfId),
    nextKfId(kf.nextKfId),
    keyPointToTrackId(kf.keyPointToTrackId),
    mapPoints(kf.mapPoints),
    keyPointDepth(kf.keyPointDepth),
    poseCW(kf.poseCW),
    origPoseCW(kf.origPoseCW),
    uncertainty(kf.uncertainty),
    t(kf.t)
{}

std::unique_ptr<KeyframeShared> KeyframeShared::clone() const
{
    auto s = std::make_unique<KeyframeShared>();
    s->camera = camera;
    s->imgDbg = imgDbg; // cv::Mat is copied shallowly.
    s->stereoPointCloud = stereoPointCloud; // shallow copy
    s->stereoPointCloudColor = stereoPointCloudColor;

    // everything else should be empty if this method is used
    assert(keyPoints.empty());
    assert(!featureSearch);

    return s;
}

float Keyframe::computeMedianDepth(const MapDB &mapDB, float defaultDepth) {
    std::vector<float> depths;
    depths.reserve(shared->keyPoints.size());
    const Vector3d &rotZRow = poseCW.block<1, 3>(2, 0);
    const float transZ = poseCW(2, 3);

    for (MpId mpId : mapPoints) {
        if (mpId.v == -1) {
            continue;
        }
        const MapPoint &mp = mapDB.mapPoints.at(mpId);
        if (mp.status != MapPointStatus::TRIANGULATED) {
            continue;
        }

        const float posCZ = rotZRow.dot(mp.position) + transZ;
        depths.push_back(posCZ);
    }

    if (depths.empty()) {
        return defaultDepth;
    }

    std::sort(depths.begin(), depths.end());

    return depths.at((depths.size() - 1) / 2);
}

std::vector<KfId> Keyframe::getNeighbors(const MapDB &mapDB, int minCovisibilities, bool triangulatedOnly) const {
    std::map<KfId, int> covisibilities;

    // Previous and next KF are considered neighbors always.
    if (previousKfId.v != -1) {
        covisibilities.emplace(previousKfId, minCovisibilities);
    }
    if (nextKfId.v != -1) {
        covisibilities.emplace(nextKfId, minCovisibilities);
    }

    for (MpId mpId : mapPoints) {
        if (mpId.v == -1)
            continue;

        const auto &mp = mapDB.mapPoints.at(mpId);
        if (triangulatedOnly && mp.status != MapPointStatus::TRIANGULATED)
            continue;

        for (const auto &kfIdKeypointId : mp.observations) {
            KfId kfId = kfIdKeypointId.first;
            if (covisibilities.count(kfId)) {
                covisibilities[kfId]++;
            } else {
                covisibilities[kfId] = 1;
            }
        }
    }

    std::vector<KfId> res;
    for (const auto &kfIdObs : covisibilities) {
        KfId kfId = kfIdObs.first;
        int covis = kfIdObs.second;
        if (kfId != id && covis >= minCovisibilities) {
            res.push_back(kfIdObs.first);
        }
    }
    return res;
}

// Camera position computed from `poseCW`, in world coordinates.
Eigen::Vector3d Keyframe::cameraCenter() const {
    return worldToCameraMatrixCameraCenter(poseCW);
}

Eigen::Vector3d Keyframe::origPoseCameraCenter() const {
    return worldToCameraMatrixCameraCenter(origPoseCW);
}

void Keyframe::getFeaturesAround(const Eigen::Vector2f &point, float r, std::vector<size_t> &indices) {
    assert(shared->featureSearch);
    shared->featureSearch->getFeaturesAround(point.x(), point.y(), r, indices);
}

// Check angle, scale
bool Keyframe::isInFrustum(const MapPoint &mp, float viewAngleLimitCos) const {
    Eigen::Vector2f reprojectionDummy = Eigen::Vector2f::Zero();
    if (!reproject(mp.position, reprojectionDummy))
        return false;

    Eigen::Vector3f mpToKf = (cameraCenter() - mp.position).cast<float>();
    float dist = mpToKf.norm();
    if (dist < mp.minViewingDistance || mp.maxViewingDistance < dist)
        return false;

    float viewingAngleCos = mpToKf.normalized().dot(mp.norm);
    if (viewingAngleCos < viewAngleLimitCos)
        return false;

    return true;
}

bool Keyframe::reproject(const Vector3d &pointW, Vector2f &reprojected) const {
    Matrix3d rotCW = poseCW.topLeftCorner<3,3>();
    Vector3d transCW = poseCW.block<3,1>(0,3);
    float unused = 0;
    Eigen::Vector2d pointD = Eigen::Vector2d::Zero();
    const bool visible = reprojectToImage(*shared->camera, rotCW, transCW, pointW, pointD, unused);
    reprojected << pointD.x(), pointD.y();
    return visible;
}

void Keyframe::addObservation(MpId mapPointId, KpId keyPointId) {
    assert(mapPoints[keyPointId.v].v == -1);
    mapPoints[keyPointId.v] = mapPointId;
}

void Keyframe::eraseObservation(MpId mapPointId) {
    const auto it = std::find(mapPoints.begin(), mapPoints.end(), mapPointId);
    assert(it != mapPoints.end() && "MapPoint not observed in keyframe");

    (*it).v = -1;
    KpId keyPointId(std::distance(mapPoints.begin(), it));
    if (keyPointToTrackId.count(keyPointId)) {
        keyPointToTrackId.erase(keyPointId);
    }
}

template <class T> void vectorToStringstream(std::stringstream* ss, std::vector<T> v) {
    for(size_t i = 0; i < v.size(); ++i) {
        if (i != 0) {
            *ss << ",";
        }
        *ss << v[i];
    }
}

void keyPointVectorToStringstream(std::stringstream* ss, KeyPointVector v) {
    for(size_t i = 0; i < v.size(); ++i) {
        if (i != 0) {
            *ss << ",";
        }
        *ss << "(" << v[i].pt.x  << "," << v[i].pt.y << ")";
    }
}

std::string Keyframe::toString() {
    std::stringstream ss;
    ss << "keyframe(" << std::to_string(id.v) << ") [";
    ss << "previousKfId=" << std::to_string(previousKfId.v);
    ss << "nextKfId=" << std::to_string(nextKfId.v);
    ss << ", points=" << std::to_string(shared->keyPoints.size());
    ss << ", poseCW=" << util::eigenToString(poseCW);
    ss << ", origPoseCW=" << util::eigenToString(origPoseCW);
    ss << ", camera=" << shared->camera->serialize();
    ss << ", keyPoints=";
    keyPointVectorToStringstream(&ss, shared->keyPoints);
    ss << ", bowFeatureVec=";
    for (auto feature : shared->bowFeatureVec) {
        ss << "(nodeId=" << std::to_string(feature.first) << ", ";
        vectorToStringstream(&ss, feature.second);
        ss << ")";
    }
    ss << "]";
    // TODO: Fields missing text representation
    // std::unordered_map<int, int> keyPointToTrackId;
    // std::vector<MpId> mapPoints;
    // DBoW2::BowVector bowVec;
    // DBoW2::FeatureVector bowFeatureVec;
    return ss.str();
}

const cv::Vec3b &Keyframe::getKeyPointColor(KpId kpId) const {
    if (shared->colors.empty()) return BLACK;
    return shared->colors.at(kpId.v);
}

// implementations copied from openvslam/camera/perspective.cc
bool reprojectToImage(
    const tracker::Camera &camera,
    const Eigen::Matrix3d& rot_cw,
    const Eigen::Vector3d& trans_cw,
    const Eigen::Vector3d& pos_w,
    Eigen::Vector2d& reproj,
    float& x_right)
{
    const Eigen::Vector3d pos_c = rot_cw * pos_w + trans_cw;
    x_right = 0.0;
    if (!camera.rayToPixel(pos_c, reproj)) return false;
    if (!camera.isValidPixel(reproj)) return false;

    // TODO: stereo stuff, currently unused
    x_right = reproj(0); //reproj(0) - focal_x_baseline_ * z_inv;
    return true;
}

bool reprojectToBearing(
    const tracker::Camera &camera,
    const Eigen::Matrix3d& rot_cw,
    const Eigen::Vector3d& trans_cw,
    const Eigen::Vector3d& pos_w,
    Eigen::Vector3d& reproj)
{
    // convert to camera-coordinates
    reproj = rot_cw * pos_w + trans_cw;

    Eigen::Vector2d pix;
    // check if the point is within camera FOV and reproject
    if (!camera.rayToPixel(rot_cw * pos_w + trans_cw, pix))
        return false;

    // check if the point is visible
    if (!camera.isValidPixel(pix)) return false;

    return camera.pixelToRay(pix, reproj);
}

void asciiKeyframes(const std::function<char(KfId)> status, const MapDB &mapDB, int len) {
    if (mapDB.keyframes.empty()) return;
    KfId lastId = mapDB.keyframes.rbegin()->first;

    std::string kfStatus(len, ' ');
    int lastRev = 0;
    for (int ind = 0, rev = len - 1; lastId.v - ind >= 0 && rev >= 0; ++ind) {
        KfId kfId(lastId.v - ind); // TODO: broken
        // Do not draw empty space for non-existing keyframes.
        if (!mapDB.keyframes.count(kfId)) continue;

        kfStatus[rev] = status(kfId);
        lastRev = rev;
        rev--;
    }
    // Set first KF like this because the actual KfId(0) KF may be removed.
    if (lastRev > 0 && kfStatus[lastRev] == ' ') kfStatus[lastRev] = '0';
    std::cout << kfStatus << std::endl;
}

namespace {
cv::Vec3b getPixel(const cv::Mat &img, cv::Point2f pt) {
    switch (img.channels()) {
        case 1: {
            const auto b = img.at<std::uint8_t>(pt);
            return cv::Vec3b(b, b, b);
        }
        case 4: { // RGBA
            const auto c = img.at<cv::Vec4b>(pt);
            return cv::Vec3b(c[2], c[1], c[0]); // BGR -> RGB
        }
        case 3: {
            const auto c = img.at<cv::Vec3b>(pt);
            return cv::Vec3b(c[2], c[1], c[0]); // BGR -> RGB
        }
        default:
            assert(false && "invalid number of channels");
    }
    return cv::Vec3b(0, 0, 0);
}

void setStereoPointCloud(const MapperInput &mapperInput, KeyframeShared &kfShared) {
    if (mapperInput.frame->hasStereoPointCloud()) {
        kfShared.stereoPointCloud = std::make_shared<KeyframeShared::StereoPointCloud>();
        *kfShared.stereoPointCloud = mapperInput.frame->getStereoPointCloud();

        const cv::Mat &img = mapperInput.colorFrame;
        if (!img.empty()) {
            kfShared.stereoPointCloudColor = std::make_shared<std::vector<cv::Vec3b>>();
            for (const Eigen::Vector3f &pCam : *kfShared.stereoPointCloud) {
                cv::Vec3b color = slam::BLACK;
                Eigen::Vector2d pix;
                if (kfShared.camera->rayToPixel(pCam.cast<double>(), pix)) {
                    int x = int(pix.x()), y = int(pix.y());
                    if (x >= 0 && y >= 0 && x < img.cols && y < img.rows) {
                        color = getPixel(img, cv::Point2f(x, y));
                    }
                }
                kfShared.stereoPointCloudColor->push_back(color);
            }
        }
    }
}
}

} // namespace slam
