#ifndef SLAM_SLAM_IMPLEMENTATION_HPP
#define SLAM_SLAM_IMPLEMENTATION_HPP

#include <memory>
#include "../api/slam.hpp"
#include "../api/slam_map_point_record.hpp"

#include "../odometry/parameters.hpp"
#include "../commandline/command_queue.hpp"

namespace slam {

class ViewerDataPublisher;

struct DebugAPI {
    ViewerDataPublisher *dataPublisher = nullptr;
    CommandQueue *commandQueue = nullptr;
    std::string mapSavePath;
    std::function<void(const std::vector<MapPointRecord>&)> endDebugCallback = nullptr;
};

} // namespace slam

#endif //SLAM_SLAM_IMPLEMENTATION_HPP
