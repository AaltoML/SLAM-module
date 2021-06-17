#ifndef SLAM_SERIALIZATION_HPP
#define SLAM_SERIALIZATION_HPP

#include "../util/serialization.hpp"
#include "id.hpp"

namespace cereal {

template<class Archive>
void serialize(Archive &ar, slam::Id &id) {
    ar(id.v);
}

template<class Archive>
void serialize(Archive &ar, cv::KeyPoint &kp) {
    ar(
        kp.angle,
        kp.class_id,
        kp.octave,
        kp.pt,
        kp.response,
        kp.size
    );
}

} // namespace cereal

#endif // SLAM_SERIALIZATION_HPP
