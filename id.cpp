#include "id.hpp"

#include <cassert>

namespace slam {

#define ID_OP_IMPL(ID, OP) bool operator OP (const ID &lhs, const ID &rhs) { return lhs.v OP rhs.v; }

ID_OP_IMPL(KfId, ==)
ID_OP_IMPL(KfId, !=)
ID_OP_IMPL(KfId, <)
ID_OP_IMPL(KfId, <=)
ID_OP_IMPL(KfId, >)
ID_OP_IMPL(KfId, >=)
ID_OP_IMPL(MpId, ==)
ID_OP_IMPL(MpId, !=)
ID_OP_IMPL(MpId, <)
ID_OP_IMPL(MpId, >)
ID_OP_IMPL(KpId, <)
ID_OP_IMPL(TrackId, <)
ID_OP_IMPL(MapId, ==)
ID_OP_IMPL(MapId, !=)
ID_OP_IMPL(MapId, <)

#undef ID_OP_IMPL

VertexIdConverter::VertexIdConverter(const std::pair<KfId, MpId> &maxIds) {
    // The arguments are maximum used ids, thus need to add one to get length
    // and avoid overlap with next type.
    mp0 = std::get<0>(maxIds).v + 1;
    custom0 = mp0 + std::get<1>(maxIds).v + 1;
}

int VertexIdConverter::keyframe(KfId kfId) const {
    assert(kfId.v >= 0);
    assert(kfId.v < mp0);
    return kfId.v;
}

int VertexIdConverter::mapPoint(MpId mpId) const {
    assert(mpId.v >= 0);
    assert(mp0 + mpId.v < custom0);
    return mp0 + mpId.v;
}

int VertexIdConverter::custom(int i) const {
    assert(i >= 0);
    return custom0 + i;
}

KfId VertexIdConverter::invKeyframe(int id) const {
    assert(id >= 0 && id < mp0);
    return KfId(id);
}

MpId VertexIdConverter::invMapPoint(int id) const {
    assert(id >= mp0 && id < custom0);
    return MpId(id - mp0);
}

int VertexIdConverter::invCustom(int i) const {
    assert(i >= custom0);
    return i - custom0;
}

} // namespace slam
