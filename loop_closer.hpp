#ifndef SLAM_LOOP_CLOSER_HPP
#define SLAM_LOOP_CLOSER_HPP

#include <set>
#include <memory>
#include <functional>
#include <Eigen/Dense>

#include "../commandline/command_queue.hpp" // TODO: bad!!!
#include "id.hpp"
#include "static_settings.hpp"

namespace slam {

class Keyframe;
class MapDB;
class BowIndex;
class ViewerDataPublisher;

using Atlas = std::vector<MapDB>;

// Different stages of loop processing, used for visualizations.
enum class LoopStage {
    BOW_MATCH,
    QUICK_TESTS,
    MAP_POINT_MATCHES,
    ACCEPTED,
    RELOCATION_MAP_POINT_MATCHES,
    RELOCATION_MAP_POINT_RANSAC,
};

struct LoopClosureEdge {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    KfId kfId1;
    KfId kfId2;
    Eigen::Matrix4d poseDiff;

    template<class Archive>
    void serialize(Archive &ar) {
        ar(kfId1, kfId2, poseDiff);
    }
};

class LoopCloser {
public:
    static std::unique_ptr<LoopCloser> create(
        const StaticSettings &settings,
        BowIndex &bowIndex,
        MapDB &mapDB,
        const Atlas &atlas
    );

    virtual ~LoopCloser() = default;

    virtual bool tryLoopClosure(
        Keyframe &currentKf,
        const std::vector<KfId> &adjacentKfIds
    ) = 0;

    virtual void setViewerDataPublisher(ViewerDataPublisher *dataPublisher) = 0;

    virtual void setCommandQueue(CommandQueue *commands) = 0;
};

} // slam

#endif // SLAM_LOOP_CLOSER_HPP
