#include <cmath>

#include "static_settings.hpp"
#include "../odometry/parameters.hpp"

namespace slam {
namespace {
// Helpers from OpenVSLAM
std::vector<float> calc_scale_factors(const unsigned int num_scale_levels, const float scale_factor) {
    std::vector<float> scale_factors(num_scale_levels, 1.0);
    for (unsigned int level = 1; level < num_scale_levels; ++level) {
        scale_factors.at(level) = scale_factor * scale_factors.at(level - 1);
    }
    return scale_factors;
}
std::vector<float> calc_level_sigma_sq(const unsigned int num_scale_levels, const float scale_factor) {
    float scale_factor_at_level = 1.0;
    std::vector<float> level_sigma_sq(num_scale_levels, 1.0);
    for (unsigned int level = 1; level < num_scale_levels; ++level) {
        scale_factor_at_level = scale_factor * scale_factor_at_level;
        level_sigma_sq.at(level) = scale_factor_at_level * scale_factor_at_level;
    }
    return level_sigma_sq;
}
}

StaticSettings::StaticSettings(const odometry::Parameters &p) :
    parameters(p),
    scaleFactors(
      calc_scale_factors(
          parameters.slam.orbScaleLevels,
          parameters.slam.orbScaleFactor)),
    levelSigmaSq(
      calc_level_sigma_sq(
          parameters.slam.orbScaleLevels,
          parameters.slam.orbScaleFactor))
{}

std::vector<std::size_t> StaticSettings::maxNumberOfKeypointsPerLevel() const {
    const auto &parameters = this->parameters.slam;
    std::vector<std::size_t> num_keypts_per_level;

    // ---- copied from orb_extractor.cc in OpenVSLAM with minor modifications
    num_keypts_per_level.resize(parameters.orbScaleLevels);

    // compute the desired number of keypoints per scale
    double desired_num_keypts_per_scale
        = parameters.maxKeypoints * (1.0 - 1.0 / parameters.orbScaleFactor)
          / (1.0 - std::pow(1.0 / parameters.orbScaleFactor, static_cast<double>(parameters.orbScaleLevels)));
    unsigned int total_num_keypts = 0;
    for (unsigned int level = 0; level < parameters.orbScaleLevels - 1; ++level) {
        num_keypts_per_level.at(level) = std::round(desired_num_keypts_per_scale);
        total_num_keypts += num_keypts_per_level.at(level);
        desired_num_keypts_per_scale *= 1.0 / parameters.orbScaleFactor;
    }
    num_keypts_per_level.at(parameters.orbScaleLevels - 1) = std::max(static_cast<int>(parameters.maxKeypoints) - static_cast<int>(total_num_keypts), 0);
    // ---- end OpenVSLAM code

    return num_keypts_per_level;
}

}
