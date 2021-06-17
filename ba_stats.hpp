#ifndef SLAM_BA_STATS_H_
#define SLAM_BA_STATS_H_

#include "../util/logging.hpp"

namespace slam {
namespace {

class BaStats {
public:
    enum class Ba {
        NONE,
        POSE,
        NEIGHBOR,
        LOCAL,
        GLOBAL,
        // Marker for iteration.
        LAST
    };

private:
    const int last = static_cast<int>(Ba::LAST);
    bool enabled;

    std::map<Ba, int> bas;
    std::map<Ba, int> totalBas;

public:
    BaStats(bool enabled) :
        enabled(enabled)
    {
        if (!enabled) return;
        for (int i = 0; i < last; ++i) {
            Ba l = static_cast<Ba>(i);
            bas.emplace(l, 0);
            totalBas.emplace(l, 0);
        }
    }

    void update(Ba ba) {
        if (!enabled) return;
        ++bas.at(ba);
    }

    void finishFrame() {
        if (!enabled) return;

        int count = 0;
        for (int i = 0; i < last; ++i) {
            Ba t = static_cast<Ba>(i);
            totalBas.at(t) += bas.at(t);
            count += bas.at(t);
        }
        if (count == 0) {
            bas.at(Ba::NONE) += 1;
            totalBas.at(Ba::NONE) += 1;
        }

        const char *names[6] = {
            "none     ",
            "pose     ",
            "neighbor ",
            "local    ",
            "global   ",
            "TOTAL    ",
        };
        log_info("");
        log_info("TYPE   \tNUM\tTOTAL");
        int sum = 0;
        int totalSum = 0;
        for (int i = 0; i < last; ++i) {
            Ba t = static_cast<Ba>(i);
            sum += bas.at(t);
            totalSum += totalBas.at(t);
            log_info("%s\t%d\t%d", names[i], bas.at(t), totalBas.at(t));
        }
        log_info("%s\t%d\t%d", names[last], sum, totalSum);

        for (int i = 0; i < last; ++i) {
            Ba t = static_cast<Ba>(i);
            bas.at(t) = 0;
        }
    }
};

} // anonymous namespace
} // namespace slam

#endif // SLAM_BA_STATS_H_
