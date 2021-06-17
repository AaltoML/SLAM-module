#ifndef SLAM_LOOP_CLOSER_STATS_H_
#define SLAM_LOOP_CLOSER_STATS_H_

namespace slam {
namespace {

class LoopCloserStats {
public:
    enum class Loop {
        // Applied loop closures, up to 1 per frame. Computed automatically from OK updates.
        DONE,
        // All accepted loop closures.
        OK,
        // The failure types.
        TOO_CLOSE_TIME,
        UNNECESSARY_EARLY,
        EARLY_MAP_IGNORED,
        TOO_FEW_FEATURE_MATCHES,
        RANSAC_FAILED,
        UNNECESSARY,
        TOO_LARGE_POSITION_DRIFT,
        TOO_LARGE_ANGLE_DRIFT,
        UNKNOWN,
        // Marker for iteration.
        LAST
    };

private:
    const int last = static_cast<int>(Loop::LAST);
    bool enabled;
    bool applyLoopClosures;
    int loopCount = 0;
    int totalLoopCount = 0;
    bool didLoopClosure = false;

    std::map<Loop, int> loops;
    std::map<Loop, int> totalLoops;

public:
    LoopCloserStats(const odometry::ParametersSlam &parameters) :
        enabled(parameters.printLoopCloserStats),
        applyLoopClosures(parameters.applyLoopClosures)
    {
        if (!enabled) return;
        for (int i = 0; i < last; ++i) {
            Loop l = static_cast<Loop>(i);
            loops.emplace(l, 0);
            totalLoops.emplace(l, 0);
        }
    }

    void newLoop() {
        if (!enabled) return;
        ++loopCount;
    }

    void update(Loop loop) {
        if (!enabled) return;
        ++loops.at(loop);
        if (loop == Loop::OK) didLoopClosure = true;
    }

    void finishFrame() {
        if (!enabled) return;

        if (didLoopClosure && applyLoopClosures) {
            loops.at(Loop::DONE) += 1;
        }
        didLoopClosure = false;

        totalLoopCount += loopCount;
        int count = 0;
        for (int i = 0; i < last; ++i) {
            Loop t = static_cast<Loop>(i);
            totalLoops.at(t) += loops.at(t);
            count += loops.at(t);
        }
        // Should be zero, but won't be if the loop over loop closure candidates does `continue;`
        // without telling us about the loop.
        int unknown = loopCount - count > 0 ? loopCount - count : 0;
        loops.at(Loop::UNKNOWN) += unknown;
        totalLoops.at(Loop::UNKNOWN) += unknown;

        const char *names[12] = {
            "done               ",
            "ok                 ",
            "too close time     ",
            "unnecessary early  ",
            "early map ignored  ",
            "too few features   ",
            "ransac failed      ",
            "unnecessary        ",
            "too large pos drift",
            "too large ang drift",
            "unknown            ",
            "TOTAL              ",
        };
        log_info("");
        log_info("TYPE                     \tNUM\tTOTAL");
        int sum = 0;
        int totalSum = 0;
        for (int i = 0; i < last; ++i) {
            Loop t = static_cast<Loop>(i);
            sum += loops.at(t);
            totalSum += totalLoops.at(t);
            log_info("%s\t%d\t%d", names[i], loops.at(t), totalLoops.at(t));
        }
        log_info("%s\t%d\t%d", names[last], sum, totalSum);

        loopCount = 0;
        for (int i = 0; i < last; ++i) {
            Loop t = static_cast<Loop>(i);
            loops.at(t) = 0;
        }
    }
};

} // anonymous namespace
} // namespace slam

#endif // SLAM_LOOP_CLOSER_STATS_H_
