#ifndef SLAM_SLAM_ID_HPP
#define SLAM_SLAM_ID_HPP

#include <functional>
#include <unordered_map>
#include <unordered_set>

namespace slam {

enum class TriangulationMethod {
    TME, MIDPOINT
};

struct Id {
    int v = -1;

    // Constructor with no arguments may make using std::map and std::set easier,
    // and seems required for serialization with cereal.
    Id() : v(-1) {}

    explicit Id(int v) : v(v) {}
};

/*
// Used for `std::unordered_map<Id, VALUE, IdHash, IdEqual>`.
struct IdHash {
    std::size_t operator() (const Id &id) const {
        return std::hash<int>()(id.v);
    }
};

// Used for `std::unordered_map<Id, VALUE, IdHash, IdEqual>`.
struct IdEqual {
    bool operator() (const Id &lhs, const Id &rhs) const {
        return lhs.v == rhs.v;
    }
};

template<class ID, class T>
using id_unordered_map = std::unordered_map<ID, T, IdHash, IdEqual>;

template<class ID>
using id_unordered_set = std::unordered_set<ID, IdHash, IdEqual>;
*/

// Keyframe id.
struct KfId : Id {
    KfId() : Id() {}
    explicit KfId(int v) : Id(v) {}
};

// Map point id.
struct MpId : Id {
    MpId() : Id() {}
    explicit MpId(int v) : Id(v) {}
};

// Keypoint id.
struct KpId : Id {
    KpId() : Id() {}
    explicit KpId(int v) : Id(v) {}
};

// Track id.
struct TrackId : Id {
    TrackId() : Id() {}
    explicit TrackId(int v) : Id(v) {}
};

// Map id.
struct MapId : Id {
    MapId() : Id() {}
    explicit MapId(int v) : Id(v) {}
};

const MapId CURRENT_MAP_ID = MapId(1000);

#define ID_OP(ID, OP) bool operator OP (const ID &lhs, const ID &rhs);

ID_OP(KfId, ==)
ID_OP(KfId, !=)
ID_OP(KfId, <)
ID_OP(KfId, <=)
ID_OP(KfId, >)
ID_OP(KfId, >=)
ID_OP(MpId, ==)
ID_OP(MpId, !=)
ID_OP(MpId, <)
ID_OP(MpId, >)
ID_OP(KpId, <)
ID_OP(TrackId, <)
ID_OP(MapId, ==)
ID_OP(MapId, !=)
ID_OP(MapId, <)

#undef ID_OP

// Helper for managing g2o ids in common BA tasks.
class VertexIdConverter {
public:
    VertexIdConverter(const std::pair<KfId, MpId> &maxIds);
    int keyframe(KfId kfId) const;
    int mapPoint(MpId mpId) const;
    int custom(int i) const;
    KfId invKeyframe(int id) const;
    MpId invMapPoint(int id) const;
    int invCustom(int i) const;
private:
    int mp0;
    int custom0;
};

} // namespace slam

#endif // SLAM_SLAM_ID_HPP
