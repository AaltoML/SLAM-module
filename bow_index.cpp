#include "bow_index.hpp"

#include "keyframe.hpp"
#include "key_point.hpp"
#include "mapdb.hpp"
#include "../util/util.hpp"
#include "../util/logging.hpp"

namespace slam {
namespace {

BowVocabulary createBowVocabulary(std::string vocabularyPath) {
    BowVocabulary bowVocabulary;
    assert(!vocabularyPath.empty());
    const std::string inputSuffix = vocabularyPath.substr(util::removeFileSuffix(vocabularyPath).size());
    if (inputSuffix == ".txt") {
        log_debug("Loading BoW vocabulary from TEXT file %s", vocabularyPath.c_str());
        bowVocabulary.loadFromTextFile(vocabularyPath);
    }
    else {
        if (inputSuffix != ".dbow2") {
            log_warn("Expected extension `.txt` or `.dbow2` for vocabulary file.");
        }
        log_debug("Loading BoW vocabulary from BINARY file %s", vocabularyPath.c_str());
        bowVocabulary.loadFromBinaryFile(vocabularyPath);
    }
    return bowVocabulary;
}

} // anonymous namespace

BowIndex::BowIndex(const odometry::ParametersSlam &parameters) :
    parameters(parameters),
    index(),
    bowVocabulary(createBowVocabulary(parameters.vocabularyPath))
{
    // log_debug("BoW tree node count %u, levels %u, branching factor %u.",
    //     bowVocabulary.size(),
    //     bowVocabulary.getDepthLevels(),
    //     bowVocabulary.getBranchingFactor());
    index.resize(bowVocabulary.size());
}

void BowIndex::add(const Keyframe &keyframe, MapId mapId) {
    for (const auto& word : keyframe.shared->bowVec) {
        index[word.first].push_back(MapKf { mapId, keyframe.id });
    }
}

void BowIndex::remove(MapKf mapKf) {
    for (auto &l : index) {
        for (auto it = l.begin(); it != l.end(); ) {
            if (*it == mapKf) it = l.erase(it);
            else it++;
        }
    }
}

void BowIndex::transform(
    const KeyPointVector &keypoints,
    DBoW2::BowVector &bowVector,
    DBoW2::FeatureVector &bowFeatureVector
) {
    // convert to descVector for DBoW2
    unsigned idx = 0;
    tmp.descVector.clear();
    for (const auto &kp : keypoints) {
        const auto &desc = kp.descriptor;
        const unsigned sz = desc.size() * sizeof(desc[0]);
        if (idx >= tmp.cvMatStore.size()) {
            cv::Mat newMat(1, sz, CV_8U);
            tmp.cvMatStore.push_back(newMat);
            assert(tmp.cvMatStore.size() == idx + 1);
        }
        cv::Mat m = tmp.cvMatStore.at(idx);
        const auto *descAsUint8 = reinterpret_cast<const uint8_t*>(desc.data());
        for (unsigned i = 0; i < sz; ++i) m.data[i] = descAsUint8[i];
        tmp.descVector.push_back(m);
        idx++;
    }

    // The vocabulary file we typically use has 6 levels with the branching factor of 10.
    // That means the number of words at the bottom level is roughly 10^6. By going 4
    // levels up, we thus reduce number of nodes to about 10^(6-4) = 100 (size() of
    // `bowFeatureVector`).
    const int levelsUp = 4;
    bowVocabulary.transform(
        tmp.descVector,
        bowVector,
        bowFeatureVector,
        levelsUp
    );
}

std::vector<BowSimilar> BowIndex::getBowSimilar(
    const MapDB &mapDB,
    const Atlas &atlas,
    const Keyframe &kf
) {
    MapKf currentMapKf { MapId(CURRENT_MAP_ID), kf.id };
    std::map<MapKf, unsigned int> wordsInCommon;

    const auto& bowVec = kf.shared->bowVec;

    for (const auto& pair : bowVec) {
        DBoW2::WordId wordId = pair.first;

        // If not in the BoW database, continue
        if (index.at(wordId).empty()) {
            continue;
        }
        // Get the keyframes which share the word (node ID) with the query.
        const std::list<MapKf> &inNode = index.at(wordId);
        // For each keyframe, increase shared word number one by one
        for (MapKf mapKf : inNode) {
            if (mapKf == currentMapKf) {
                continue;
            }
            assert(getMapWithId(mapKf.mapId, mapDB, atlas).keyframes.count(mapKf.kfId));

            // Initialize if not in num_common_words
            if (!wordsInCommon.count(mapKf)) {
                wordsInCommon[mapKf] = 0;
            }
            // Count up the number of words
            ++wordsInCommon.at(mapKf);
        }
    }

    if (wordsInCommon.empty()) {
        return {};
    }

    // TODO When searching loop closure candidates, we want to use only the current map,
    // but now the code sets `maxInCommon` and `minScore` based on the best match
    // across all maps.
    unsigned int maxInCommon = 0;
    for (const auto &word : wordsInCommon) {
        if (word.second > maxInCommon) maxInCommon = word.second;
    }

    // Constrain search around the best word count match.
    const auto minInCommon =
        static_cast<unsigned int>(parameters.bowMinInCommonRatio * static_cast<float>(maxInCommon));

    std::vector<BowSimilar> similar;
    for (const auto &word : wordsInCommon) {
        MapId mapId = word.first.mapId;
        KfId kfId = word.first.kfId;

        if (word.second > minInCommon) {
            const MapDB &m = getMapWithId(mapId, mapDB, atlas);
            float score = bowVocabulary.score(kf.shared->bowVec, m.keyframes.at(kfId)->shared->bowVec);
            similar.push_back(BowSimilar {
                .mapKf = MapKf {
                    .mapId = mapId,
                    .kfId = kfId,
                },
                .score = score,
            });
        }
    }

    if (similar.empty()) return {};

    using vt = decltype(similar)::value_type;
    std::sort(similar.begin(), similar.end(), [](const vt &p1, const vt &p2) { return p1.score > p2.score ; });

    // Return all keyframes with score close enough to the best score.
    float minScore = similar[0].score * parameters.bowScoreRatio;
    auto cut = std::find_if(similar.begin(), similar.end(), [&minScore](const vt &p) { return p.score < minScore; });
    if (cut != similar.end()) {
        similar.erase(cut, similar.end());
    }
    return similar;
}

bool operator == (const MapKf &lhs, const MapKf &rhs) {
    return lhs.mapId == rhs.mapId && lhs.kfId == rhs.kfId;
}

// Needed for std::map.
bool operator < (const MapKf &lhs, const MapKf &rhs) {
    if (lhs.mapId == rhs.mapId) {
        return lhs.kfId < rhs.kfId;
    }
    return lhs.mapId < rhs.mapId;
}

}  // namespace slam
