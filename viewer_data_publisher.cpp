#include "viewer_data_publisher.hpp"

#include "keyframe.hpp"
#include "../util/logging.hpp"

#include <algorithm>

namespace slam {

ViewerDataPublisher::ViewerDataPublisher(const cmd::ParametersSlam &parameters) :
    parameters(parameters)
{}

std::vector<ViewerLoopClosure> ViewerDataPublisher::getLoopClosures() {
    std::lock_guard<std::mutex> l(m);
    return loopClosures;
}

void ViewerDataPublisher::addLoopClosure(const ViewerLoopClosure &lc) {
    std::lock_guard<std::mutex> l(m);
    loopClosures.push_back(lc);
}

void ViewerDataPublisher::setMap(
    const ViewerDataPublisher::MapPointVector &mapPoints,
    const ViewerDataPublisher::KeyframeVector &keyframes,
    const std::map<MapKf, LoopStage> &loopStages,
    double age
) {
    std::lock_guard<std::mutex> l(m);
    this->mapPoints = mapPoints;
    this->keyframes = keyframes;
    this->loopStages = loopStages;
    this->age = age;
}

ViewerDataPublisher::MapPointVector ViewerDataPublisher::getMapPoints() {
    std::lock_guard<std::mutex> l(m);
    return mapPoints;
}

void ViewerDataPublisher::getKeyframes(
    KeyframeVector &keyframes,
    std::map<MapKf, LoopStage> &loopStages,
    double &age
) {
    std::lock_guard<std::mutex> l(m);
    keyframes = this->keyframes;
    loopStages = this->loopStages;
    age = this->age;
}

void ViewerDataPublisher::setAtlas(std::unique_ptr<ViewerAtlas> atlas) {
    std::lock_guard<std::mutex> l(m);
    this->atlas = std::move(atlas);
}

std::unique_ptr<ViewerAtlas> ViewerDataPublisher::takeAtlas() {
    std::lock_guard<std::mutex> l(m);
    if (atlas) {
        return std::move(atlas);
    }
    return {};
}

const cmd::ParametersSlam &ViewerDataPublisher::getParameters() const {
    return parameters;
}

}  // namespace slam
