#ifndef SLAM_MAPPER_HPP
#define SLAM_MAPPER_HPP

#include <memory>
#include <vector>
#include <Eigen/Dense>

#include "../api/slam.hpp"
#include "slam_implementation.hpp"
#include "keyframe.hpp"
#include "map_point.hpp"
#include "id.hpp"

namespace odometry { struct Parameters; }
namespace slam {

class Keyframe;
class ViewerDataPublisher;

class Mapper {
public:
    static std::unique_ptr<Mapper>
    create(const odometry::Parameters &parameters);

    virtual ~Mapper() = default;

    /**
     * @param mapperInput Data for constructing input keyframe
     * @param resultPose output: The SLAM-corrected pose
     * @param pointCloud output: The currently visible map points
     */
    virtual void advance(
        std::shared_ptr<const MapperInput> mapperInput,
        Eigen::Matrix4d &resultPose,
        Slam::Result::PointCloud &pointCloud) = 0;

    virtual void connectDebugAPI(DebugAPI &debug) = 0;

    virtual bool end(const std::string &mapSavePath) = 0;
};

} // namespace slam

#endif //SLAM_MAPPER_HPP
