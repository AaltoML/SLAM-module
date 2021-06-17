#include "feature_search.hpp"

#include <algorithm>
#include <functional>
#include <vector>

namespace slam {
class FeatureSearchImplementation : public FeatureSearch {
private:
    struct Node {
        float x;
        float y;
        std::size_t keypointIndex;
    };

    // first implementation: keypoints in an array sorted by Y coordinate
    // (the Y size of the image is assumed to be larger than X size)
    std::vector<Node> indexByY;
    std::function<bool(const Node&, const Node&)> comparator;

public:
    FeatureSearchImplementation(const KeyPointVector &keypoints) {
        for (std::size_t i = 0; i < keypoints.size(); ++i) {
            const auto &kp = keypoints[i];
            indexByY.push_back({ kp.pt.x, kp.pt.y, i });
        }
        comparator = [](const Node &a, const Node &b) -> bool {
            return a.y < b.y;
        };
        std::sort(indexByY.begin(), indexByY.end(), comparator);
    }

    void getFeaturesAround(float x, float y, float r, std::vector<size_t> &output) const final {
        output.clear();

        // find Y range begin using binary search
        const Node lb { x, y - r, 0 };
        for (auto itr = std::lower_bound(indexByY.begin(), indexByY.end(), lb, comparator);
            itr != indexByY.end() && itr->y <= y + r;
            ++itr)
        {
            // pick points within correct X bounds
            const float dx = x - itr->x;
            const float dy = y - itr->y;
            if (dx*dx + dy*dy < r*r)
                output.push_back(itr->keypointIndex);
        }
    }
};


std::unique_ptr<FeatureSearch> FeatureSearch::create(const KeyPointVector &kps) {
    return std::unique_ptr<FeatureSearch>(new FeatureSearchImplementation(kps));
}

}
