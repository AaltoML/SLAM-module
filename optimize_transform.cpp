#include "optimize_transform.hpp"

#include "keyframe.hpp"
#include "map_point.hpp"
#include "mapdb.hpp"
#include "../util/logging.hpp"
#include "../util/util.hpp"

#include <g2o/types/sim3/types_seven_dof_expmap.h>
#include <g2o/core/base_vertex.h>
#include <g2o/types/sim3/sim3.h>

#include <g2o/core/solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>

#include "mapper_helpers.hpp"

/*
namespace {

g2o::Sim3 se3ToSim3(const Eigen::Matrix4d &se3) {
    return g2o::Sim3(
        se3.topLeftCorner<3,3>(),
        se3.block<3,1>(0,3),
        1
    );
}

Eigen::Matrix4d sim3ToSe3(const g2o::Sim3 &sim3) {
    Eigen::Matrix4d se3 = Eigen::Matrix4d::Identity();
    se3.topLeftCorner<3,3>() = sim3.rotation().toRotationMatrix();
    se3.block<3,1>(0, 3) = sim3.translation();
    return se3;
}

g2o::SE3Quat sim3ToSE3Quat(const g2o::Sim3 &sim3) {
    return g2o::SE3Quat(sim3.rotation(), sim3.translation());
}

}
*/

namespace slam {
namespace {
std::unique_ptr<g2o::VertexSBAPointXYZ> getMpVertex(const MapPoint &mp, int vertexId, const Keyframe &kf) {
    auto mpVertex = std::make_unique<g2o::VertexSBAPointXYZ>();
    mpVertex->setEstimate(Eigen::Isometry3d(kf.poseCW) * mp.position);
    // TODO This should probably use map point ids, but now g2o it gives warnings
    // about same ids being inserted multiple times, which probably indicates
    // some bug in our logic.
    mpVertex->setId(vertexId);
    mpVertex->setFixed(true);

    return mpVertex;
}
}

unsigned int OptimizeSim3Transform(
        const Keyframe &kf1,
        const Keyframe &kf2,
        const std::vector<std::pair<MpId, MpId>> &matches,
        MapDB &mapDB,
        g2o::Sim3 &transform12,
        const StaticSettings &settings) {

    const auto &parameters = settings.parameters.slam;
    const double inlierThreshold = parameters.loopClosureInlierThreshold;
    const float deltaHuber = std::sqrt(inlierThreshold);
    const bool fixScale = parameters.loopClosureRansacFixScale;

    auto linearSolver = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
    auto blockSolver = std::make_unique<::g2o::BlockSolverX>(std::move(linearSolver));
    auto algorithm = std::make_unique<g2o::OptimizationAlgorithmLevenberg>(std::move(blockSolver));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(algorithm.release());

    auto Sim3Vertex12 = std::make_unique<g2o::VertexSim3Expmap>();
    Sim3Vertex12->setId(0);
    Sim3Vertex12->setEstimate(transform12);
    Sim3Vertex12->setFixed(false);
    Sim3Vertex12->_fix_scale = fixScale;

    Sim3Vertex12->_principle_point1[0] = 0;
    Sim3Vertex12->_principle_point1[1] = 0;
    Sim3Vertex12->_focal_length1[0] = 1;
    Sim3Vertex12->_focal_length1[1] = 1;

    Sim3Vertex12->_principle_point2[0] = 0;
    Sim3Vertex12->_principle_point2[1] = 0;
    Sim3Vertex12->_focal_length2[0] = 1;
    Sim3Vertex12->_focal_length2[1] = 1;
    optimizer.addVertex(Sim3Vertex12.release());

    int idCounter = 1;
    for (const auto &match : matches) {
        const MapPoint &mp1 = mapDB.mapPoints.at(match.first);
        int vertexId1 = idCounter++;
        auto mpVertex1 = getMpVertex(mp1, vertexId1, kf1);

        const MapPoint &mp2 = mapDB.mapPoints.at(match.second);
        int vertexId2 = idCounter++;
        auto mpVertex2 = getMpVertex(mp2, vertexId2, kf2);

        optimizer.addVertex(mpVertex1.release());
        optimizer.addVertex(mpVertex2.release());

        // TODO: DRY
        int kpIdx1 = mp1.observations.at(kf1.id).v;
        const auto &keyPoint = kf1.shared->keyPoints.at(kpIdx1);
        Eigen::Vector2d obs1 = keyPoint.bearing.segment<2>(0) / keyPoint.bearing.z();

        auto e12 = std::make_unique<g2o::EdgeSim3ProjectXYZ>();
        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(vertexId2)));
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e12->setMeasurement(obs1);
        e12->setInformation(Eigen::Matrix2d::Identity()*settings.levelSigmaSq.at(keyPoint.octave));

        auto rk1 = std::make_unique<g2o::RobustKernelHuber>();
        rk1->setDelta(deltaHuber);
        e12->setRobustKernel(rk1.release());
        optimizer.addEdge(e12.release());

        int kpIdx2 = mp2.observations.at(kf2.id).v;
        const auto &keyPoint2 = kf2.shared->keyPoints.at(kpIdx2);
        Eigen::Vector2d obs2 = keyPoint2.bearing.segment<2>(0) / keyPoint2.bearing.z();

        auto e21 = std::make_unique<g2o::EdgeInverseSim3ProjectXYZ>();
        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(vertexId1)));
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e21->setMeasurement(obs2);
        e21->setInformation(Eigen::Matrix2d::Identity()*settings.levelSigmaSq.at(keyPoint2.octave));

        auto rk2 = std::make_unique<g2o::RobustKernelHuber>();
        rk2->setDelta(deltaHuber);
        e21->setRobustKernel(rk2.release());
        optimizer.addEdge(e21.release());
    }

    optimizer.initializeOptimization();
    optimizer.optimize(20);

    // TODO(jhnj): Inlier check and reoptimize

    // Apply optimization results.
    transform12 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0))->estimate();

    // Return all matches as inliers before inlier check (see TODO above)
    return matches.size();
}

} // slam
