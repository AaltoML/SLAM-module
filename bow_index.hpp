#ifndef SLAM_BOW_INDEX_HPP
#define SLAM_BOW_INDEX_HPP

#include <list>
#include <memory>
#include <vector>

#include <DBoW2/FORB.h>
#include <DBoW2/TemplatedVocabulary.h>

#include "id.hpp"
#include "key_point.hpp"
#include "../odometry/parameters.hpp"

namespace slam {

class Keyframe;
class MapDB;

using BowVocabulary = DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>;
using Atlas = std::vector<MapDB>;

struct MapKf {
    MapId mapId;
    KfId kfId;
};

bool operator == (const MapKf &lhs, const MapKf &rhs);
bool operator < (const MapKf &lhs, const MapKf &rhs);

struct BowSimilar {
    MapKf mapKf;
    float score;
};

class BowIndex {
public:
    BowIndex(const odometry::ParametersSlam &parameter);

    void add(const Keyframe &keyframe, MapId mapId);

    void remove(MapKf mapKf);

    void transform(
        const KeyPointVector &keypoints,
        DBoW2::BowVector &bowVector,
        DBoW2::FeatureVector &bowFeatureVector
    );

    // Get all keyframes similar to a query keyframe.
    std::vector<BowSimilar> getBowSimilar(const MapDB &mapDB, const Atlas &atlas, const Keyframe &kf);

private:
    const odometry::ParametersSlam &parameters;

    // Called inverse index in the DBoW paper.
    std::vector<std::list<MapKf>> index;
    struct Workspace {
        std::vector<cv::Mat> descVector, cvMatStore;
    } tmp;


    BowVocabulary bowVocabulary;
};

}  // namespace slam

#endif  // SLAM_BOW_INDEX_HPP
