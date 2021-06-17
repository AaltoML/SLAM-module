// NOTE g2o frees memory of the objects given to it, which is why we allocate the
// objects inside smart pointers and then call `release()` on them.

#include <unordered_set>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include "bundle_adjuster.hpp"
#include "../util/logging.hpp"
#include "../util/string_utils.hpp"
#include "../odometry/parameters.hpp"

#include "keyframe.hpp"
#include "mapdb.hpp"
#include "mapper_helpers.hpp"
#include "map_point.hpp"

namespace slam {

namespace {

constexpr float CHI2_THRESHOLD = 5.991;

using PoseVertex = g2o::VertexSE3Expmap;
using PoseEdge = g2o::EdgeSE3Expmap;
using MapPointEdge = g2o::EdgeSE3ProjectXYZ;
using MapPointVertex = g2o::VertexSBAPointXYZ;

Eigen::Matrix4d poseToMatrix(const g2o::SE3Quat &estimate) {
    return estimate.to_homogeneous_matrix();
}

g2o::SE3Quat matrixToPose(const Eigen::Matrix4d &poseDiff) {
    return g2o::SE3Quat(poseDiff.topLeftCorner<3, 3>(), poseDiff.block<3, 1>(0, 3));
}

void setMapPointMeasurement(
    const Keyframe &kf,
    KpId kp,
    const StaticSettings &settings,
    MapPointEdge &edge)
{
    const auto &keyPoint = kf.shared->keyPoints.at(kp.v);
    const double focal = kf.shared->camera->getFocalLength();
    const double information = focal*focal / settings.levelSigmaSq.at(keyPoint.octave);
    edge.setMeasurement(keyPoint.bearing.segment<2>(0) / keyPoint.bearing.z());
    edge.setInformation(Eigen::Matrix2d::Identity() * information);

    auto rk = std::make_unique<g2o::RobustKernelHuber>();
    rk->setDelta(std::sqrt(CHI2_THRESHOLD));
    edge.setRobustKernel(rk.release());

    edge.fx = 1;
    edge.fy = 1;
    edge.cx = 0;
    edge.cy = 0;
}

std::unique_ptr<g2o::OptimizableGraph::Edge> makeOdometryEdge(
    KfId kfId,
    KfId prevKfId,
    g2o::SparseOptimizer &optimizer,
    const VertexIdConverter &conv,
    const MapDB &mapDB,
    const odometry::ParametersSlam &parameters
) {
    auto edge = std::make_unique<PoseEdge>();
    assert(optimizer.vertex(conv.keyframe(kfId)));
    assert(optimizer.vertex(conv.keyframe(prevKfId)));
    edge->setVertex(0, optimizer.vertex(conv.keyframe(kfId)));
    edge->setVertex(1, optimizer.vertex(conv.keyframe(prevKfId)));

    Eigen::Matrix4d poseDiff = mapDB.poseDifference(prevKfId, kfId);
    edge->setMeasurement(matrixToPose(poseDiff));
    edge->setInformation(odometryPriorStrengths(prevKfId, kfId, parameters, mapDB));

    // std::move() shouldn't be necessary here, but older Clang seems to crash without.
    return std::move(edge);
}

std::unique_ptr<PoseEdge> makeLoopClosureEdge(
    const LoopClosureEdge &l,
    g2o::SparseOptimizer &optimizer,
    const VertexIdConverter &conv,
    const odometry::ParametersSlam &parameters
) {
    auto edge = std::make_unique<PoseEdge>();

    if (!optimizer.vertex(conv.keyframe(l.kfId1)) || !optimizer.vertex(conv.keyframe(l.kfId2))) {
        return nullptr;
    }

    edge->setVertex(0, optimizer.vertex(conv.keyframe(l.kfId2)));
    edge->setVertex(1, optimizer.vertex(conv.keyframe(l.kfId1)));
    edge->setMeasurement(matrixToPose(l.poseDiff));

    // Do not use `odometryPriorStrengths()` because we don't want the keyframe id distance to matter.
    double p = parameters.odometryPriorStrengthPosition;
    double r = parameters.odometryPriorStrengthRotation;
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
    information.block(0, 0, 3, 3) *= r * r;
    information.block(3, 3, 3, 3) *= p * p;
    edge->setInformation(information);
    return edge;
}

// Extract results from g2o optimizer and apply them to MapDB objects.
void applyBundleAdjustResults(
    const std::set<KfId, std::greater<KfId>> &keyframes,
    const std::set<MpId> &mapPoints,
    MapDB &mapDB,
    g2o::SparseOptimizer &optimizer,
    const VertexIdConverter &conv
) {
    for (MpId mpId : mapPoints) {
        assert(optimizer.vertices().count(conv.mapPoint(mpId)));
        auto point = reinterpret_cast<MapPointVertex*>(optimizer.vertex(conv.mapPoint(mpId)));
        assert(point);
        MapPoint &mp = mapDB.mapPoints.at(mpId);
        mp.position = point->estimate();
        // TODO Should run things like `updateDistanceAndNorm()`?
    }

    for (KfId kfId : keyframes) {
        assert(optimizer.vertices().count(conv.keyframe(kfId)));
        auto point = reinterpret_cast<PoseVertex*>(optimizer.vertex(conv.keyframe(kfId)));
        assert(point);
        Keyframe &kf = *mapDB.keyframes.at(kfId);
        kf.poseCW = poseToMatrix(point->estimate());
    }
}

} // anonymous namespace

std::set<MpId> localBundleAdjust(
    Keyframe &keyframe,
    WorkspaceBA &workspace,
    MapDB &mapDB,
    int problemMaxSize,
    const StaticSettings &settings
) {
    const auto &parameters = settings.parameters.slam;
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    auto linearSolverType = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
    auto blockSolver = std::make_unique<g2o::BlockSolverX>(std::move(linearSolverType));
    // Released by g2o.
    optimizer.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver)));

    int iterations = static_cast<int>(1 + std::sqrt(static_cast<double>(problemMaxSize)));

    // Sorted from the newest to oldest (greatest ID first).
    std::set<KfId, std::greater<KfId>> &localKeyframes = workspace.localKfIds;
    localKeyframes.clear();

    std::set<MpId> &localMapPoints = workspace.localMpIds;
    localMapPoints.clear();

    std::size_t nCurrentFrameMapPoints = 0;

    // Compute adjacent keyframes again, to make sure new connections found during
    // processing of the current keyframe are included.
    constexpr int minCovisibilities = 15;
    std::vector<KfId> adjacent = computeAdjacentKeyframes(
        keyframe,
        minCovisibilities,
        problemMaxSize,
        mapDB,
        settings);

    localKeyframes.insert(keyframe.id);
    for (KfId kfId : adjacent) {
        localKeyframes.insert(kfId);
    }

    // "Island" in the following refers to a continuous block of keyframes.
    // Typically in a situation after a loop closure and looking at familiar
    // map points, the old area of the map will form a separate island.

    // Set a minimum size for the current island to stabilize it in absence of fixed keyframes.
    {
        int i = 0;
        for (auto it = mapDB.keyframes.rbegin(); it != mapDB.keyframes.rend(); ++it, ++i) {
            localKeyframes.insert(it->first);
            if (i > 5) break;
        }
    }

    std::set<KfId> islandStarts;
    KfId prevLocalKfId(-1);
    int consecutiveNonLocalKfs = 0;
    for (auto it = mapDB.keyframes.rbegin(); it != mapDB.keyframes.rend(); ++it) {
        const KfId kfId = it->first;
        const Keyframe &kf = *it->second;

        // After a gap in the keyframes, mark previous keyframe as start
        // of an island.
        if (!localKeyframes.count(kfId)) {
            consecutiveNonLocalKfs++;
            if (consecutiveNonLocalKfs > 3) {
                islandStarts.insert(prevLocalKfId);
            }
            continue;
        }
        prevLocalKfId = kfId;
        consecutiveNonLocalKfs = 0;

        // Collect the map points for the optimization problem.
        for (MpId mpId : kf.mapPoints) {
            if (mpId.v == -1) continue;
            const MapPoint &mapPoint = mapDB.mapPoints.at(mpId);
            if (mapPoint.status == MapPointStatus::TRIANGULATED) {
                if (kfId == keyframe.id) nCurrentFrameMapPoints++;
                localMapPoints.insert(mpId);
            }
        }
    }

    if (parameters.kfAsciiBA) {
        auto status = [&localKeyframes](KfId kfId) {
            if (localKeyframes.count(kfId)) {
                return '.';
            }
            return ' ';
        };
        asciiKeyframes(status, mapDB, parameters.kfAsciiWidth);
    }

    if (localKeyframes.empty() ||
        nCurrentFrameMapPoints < parameters.minVisibleMapPointsInCurrentFrameBA ||
        localKeyframes.size() < parameters.minKeyframesInBA)
    {
        return localMapPoints;
    }

    VertexIdConverter conv(mapDB.maxIds());

    // Keyframes.
    for (KfId kfId : localKeyframes) {
        const Keyframe &kf = *mapDB.keyframes.at(kfId);
        auto vertex = std::make_unique<PoseVertex>();

        vertex->setId(conv.keyframe(kf.id));
        vertex->setEstimate(matrixToPose(kf.poseCW));
        // In the first stage of optimization, fix all but the current KF.
        vertex->setFixed(kf.id != keyframe.id);
        optimizer.addVertex(vertex.release());
    }

    std::vector<MapPointEdge*> mapPointEdges;

    // MapPoint vertices
    for (MpId mpId : localMapPoints) {
        const MapPoint &mapPoint = mapDB.mapPoints.at(mpId);

        // if (mapPoint.observations.size() < 4)
        //     continue;

        auto vertex = std::make_unique<MapPointVertex>();
        vertex->setEstimate(mapPoint.position);
        vertex->setId(conv.mapPoint(mpId));
        vertex->setFixed(false);
        // vertex->setMarginalized(true);
        optimizer.addVertex(vertex.release());

        for (const auto &kfKeyPoint : mapPoint.observations) {
            Keyframe &kf = *mapDB.keyframes.at(kfKeyPoint.first);

            if (!localKeyframes.count(kf.id)) {
                continue;
            }

            auto edge = std::make_unique<MapPointEdge>();

            assert(optimizer.vertex(conv.mapPoint(mpId)));
            assert(optimizer.vertex(conv.keyframe(kf.id)));
            edge->setVertex(0, optimizer.vertex(conv.mapPoint(mpId)));
            edge->setVertex(1, optimizer.vertex(conv.keyframe(kf.id)));

            setMapPointMeasurement(kf, kfKeyPoint.second, settings, *edge);

            mapPointEdges.push_back(edge.release());
            optimizer.addEdge(mapPointEdges.back());
        }
    }

    // Add edges to chain all the keyframes together. Note that when moving in a previously mapped area,
    // it will make very long edges between the islands, but it shouldn't be a problem since the
    // odometry priors are weighted down by distance.
    {
        KfId otherKfId = KfId(-1);
        for (KfId kfId : localKeyframes) {
            if (otherKfId.v != -1) {
                optimizer.addEdge(makeOdometryEdge(
                    otherKfId,
                    kfId,
                    optimizer,
                    conv,
                    mapDB,
                    parameters
                ).release());
            }
            otherKfId = kfId;
        }
    }

    // Add edges from loop closures if both keyframes have been added to the optimization.
    for (const LoopClosureEdge &l : mapDB.loopClosureEdges) {
        auto edge = makeLoopClosureEdge(l, optimizer, conv, parameters);
        if (edge) {
            optimizer.addEdge(edge.release());
        }
    }

    // Perform first optimization stage to refine current KF pose.
    optimizer.initializeOptimization();
    optimizer.optimize(iterations);

    // Skip neighbordhood BA.
    if (nCurrentFrameMapPoints < parameters.minVisibleMapPointsInNeighborhoodBA) {
        std::set<KfId, std::greater<KfId>> nonFixedKfs;
        nonFixedKfs.insert(keyframe.id);
        applyBundleAdjustResults(nonFixedKfs, localMapPoints, mapDB, optimizer, conv);
        workspace.baStats.update(BaStats::Ba::NEIGHBOR);
        return localMapPoints;
    }

    // Unfix all keyframes for the main optimization.
    for (KfId kfId : localKeyframes) {
        optimizer.vertex(conv.keyframe(kfId))->setFixed(false);
    }

    // Fix orientation of the current keyframe (softly).
    // TODO Do this also in global BA?
    {
        // Use the just optimized pose as reference.
        assert(optimizer.vertices().count(conv.keyframe(keyframe.id)));
        auto point = reinterpret_cast<PoseVertex*>(optimizer.vertex(conv.keyframe(keyframe.id)));
        Eigen::Matrix4d poseCW = poseToMatrix(point->estimate());

        auto vertex = std::make_unique<PoseVertex>();

        vertex->setId(conv.custom(0));
        vertex->setEstimate(matrixToPose(poseCW));
        vertex->setFixed(true);
        optimizer.addVertex(vertex.release());

        auto edge = std::make_unique<PoseEdge>();
        edge->setVertex(0, optimizer.vertex(conv.custom(0)));
        edge->setVertex(1, optimizer.vertex(conv.keyframe(keyframe.id)));
        edge->setMeasurement(matrixToPose(Eigen::Matrix4d::Identity()));

        // (Strongly) fix orientation but do not constrain position.
        // NOTE: Using `setFixed(true)` would not work here since that also
        // fixes the position, not just orientation
        // r is an arbitrary large value that is still somewhat proportionate
        // to the other constraints to avoid numerical issues.
        double r = 100 * parameters.odometryPriorStrengthRotation;
        Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
        information.block(0, 0, 3, 3) *= r * r;
        information.block(3, 3, 3, 3) *= 0;
        edge->setInformation(information);
        optimizer.addEdge(edge.release());
    }

    optimizer.initializeOptimization();
    optimizer.optimize(iterations);

    // Remove outlier observations from the map.
    for (size_t i = 0; i < mapPointEdges.size(); ++i) {
        auto *edge = mapPointEdges.at(i);
        if (edge->chi2() > CHI2_THRESHOLD) {
            MpId mpId = conv.invMapPoint(edge->vertex(0)->id());
            KfId kfId = conv.invKeyframe(edge->vertex(1)->id());
            MapPoint &mapPoint = mapDB.mapPoints.at(mpId);
            mapPoint.eraseObservation(kfId);
            mapDB.keyframes.at(kfId)->eraseObservation(mpId);
            if (mapPoint.observations.size() <= 2) {
                mapPoint.status = MapPointStatus::UNSURE;
            }
        }
    }

    applyBundleAdjustResults(localKeyframes, localMapPoints, mapDB, optimizer, conv);

    workspace.baStats.update(BaStats::Ba::LOCAL);
    return localMapPoints;
}

bool poseBundleAdjust(
    Keyframe &keyframe,
    MapDB &mapDB,
    const StaticSettings &settings
) {
    const auto &parameters = settings.parameters.slam;
    size_t triangulatedMapPoints = 0;
    for (MpId mapPointId : keyframe.mapPoints) {
        if (mapPointId.v != -1 && mapDB.mapPoints.at(mapPointId).status == MapPointStatus::TRIANGULATED) {
            triangulatedMapPoints++;
        }
    }

    // Skip if not enough visible map points.
    if (triangulatedMapPoints < parameters.minVisibleMapPointsInCurrentFrameBA) {
        return false;
    }

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    auto linearSolverType = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
    auto blockSolver = std::make_unique<g2o::BlockSolverX>(std::move(linearSolverType));
    // Released by g2o
    optimizer.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver)));
    VertexIdConverter conv(mapDB.maxIds());

    // Keyframes.
    {
        auto vertex0 = std::make_unique<PoseVertex>();
        vertex0->setId(conv.keyframe(keyframe.id));
        vertex0->setEstimate(matrixToPose(keyframe.poseCW));
        // Released by g2o
        optimizer.addVertex(vertex0.release());

        if (keyframe.previousKfId.v < 0) {
            return false;
        }
        const Keyframe &prevKf = *mapDB.keyframes.at(keyframe.previousKfId);

        auto vertex1 = std::make_unique<PoseVertex>();
        vertex1->setId(conv.keyframe(prevKf.id));
        vertex1->setEstimate(matrixToPose(prevKf.poseCW));
        vertex1->setFixed(true);
        // Released by g2o
        optimizer.addVertex(vertex1.release());

        optimizer.addEdge(makeOdometryEdge(
            keyframe.id,
            keyframe.previousKfId,
            optimizer,
            conv,
            mapDB,
            parameters
        ).release());
    }

    // MapPoints.
    for (size_t i = 0; i < keyframe.mapPoints.size(); i++) {
        MpId mpId = keyframe.mapPoints[i];
        if (mpId.v == -1) {
            continue;
        }
        const MapPoint &mapPoint = mapDB.mapPoints.at(mpId);
        if (mapPoint.status != MapPointStatus::TRIANGULATED) {
            continue;
        }
        auto vertex = std::make_unique<MapPointVertex>();
        vertex->setEstimate(mapPoint.position);
        vertex->setId(conv.mapPoint(mpId));
        vertex->setFixed(true);
        // vertex->setMarginalized(true);
        // Released by g2o
        optimizer.addVertex(vertex.release());
        auto edge = std::make_unique<MapPointEdge>();

        assert(optimizer.vertex(conv.mapPoint(mpId)));
        assert(optimizer.vertex(conv.keyframe(keyframe.id)));
        edge->setVertex(0, optimizer.vertex(conv.mapPoint(mpId)));
        edge->setVertex(1, optimizer.vertex(conv.keyframe(keyframe.id)));

        setMapPointMeasurement(keyframe, KpId(i), settings, *edge);

        // ~g2o::HyperGraph wants to release this
        optimizer.addEdge(edge.release());
    }

    optimizer.initializeOptimization();
    optimizer.optimize(parameters.poseBAIterations);

    // Only update current KF pose, everything else was fixed.
    std::set<KfId, std::greater<KfId>> localKeyframes;
    localKeyframes.insert(keyframe.id);
    std::set<MpId> localMapPoints;
    applyBundleAdjustResults(localKeyframes, localMapPoints, mapDB, optimizer, conv);
    return true;
}

void globalBundleAdjust(
    KfId currentKfId,
    MapDB &mapDB,
    const StaticSettings &settings
) {
    const auto &parameters = settings.parameters.slam;
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    auto linearSolverType = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
    auto blockSolver = std::make_unique<g2o::BlockSolverX>(std::move(linearSolverType));
    // Released by g2o
    optimizer.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver)));

    VertexIdConverter conv(mapDB.maxIds());

    for (const auto &p : mapDB.keyframes) {
        const Keyframe &kf = *p.second;
        auto vertex = std::make_unique<PoseVertex>();

        vertex->setId(conv.keyframe(kf.id));
        vertex->setEstimate(matrixToPose(kf.poseCW));
        // TODO Try unfixing also the current KF, like local BA does.
        vertex->setFixed(kf.id == currentKfId);

        // Released by g2o
        optimizer.addVertex(vertex.release());
    }

    std::vector<MapPointEdge*> mapPointEdges;
    for (const auto &p : mapDB.mapPoints) {
        const auto &mapPoint = p.second;

        if (mapPoint.observations.empty()) continue;

        auto vertex = std::make_unique<MapPointVertex>();
        vertex->setEstimate(mapPoint.position);
        vertex->setId(conv.mapPoint(mapPoint.id));
        vertex->setFixed(false);
        // vertex->setMarginalized(true);
        // Released by g2o
        optimizer.addVertex(vertex.release());

        for (const auto &kfKeyPoint : mapPoint.observations) {
            Keyframe &kf = *mapDB.keyframes.at(kfKeyPoint.first);

            auto edge = std::make_unique<MapPointEdge>();

            assert(optimizer.vertex(conv.mapPoint(mapPoint.id)));
            assert(optimizer.vertex(conv.keyframe(kf.id)));
            edge->setVertex(0, optimizer.vertex(conv.mapPoint(mapPoint.id)));
            edge->setVertex(1, optimizer.vertex(conv.keyframe(kf.id)));

            setMapPointMeasurement(kf, kfKeyPoint.second, settings, *edge);

            mapPointEdges.push_back(edge.release());
            optimizer.addEdge(mapPointEdges.back());
        }
    }

    for (const auto &idKfP : mapDB.keyframes) {
        const Keyframe &kf = *idKfP.second;
        if (kf.previousKfId.v < 0) {
            continue;
        }

        // Distance to previous KF may be large if not moving and/or moving in previously mapped area.
        // However, odometry priors are scaled down based on the distance.
        optimizer.addEdge(makeOdometryEdge(
            kf.id,
            kf.previousKfId,
            optimizer,
            conv,
            mapDB,
            parameters
        ).release());
    }

    // Add loop closure edges, which prevent BA from undoing loop closures.
    for (const LoopClosureEdge &l : mapDB.loopClosureEdges) {
        auto edge = makeLoopClosureEdge(l, optimizer, conv, parameters);
        assert(edge);
        optimizer.addEdge(edge.release());
    }

    optimizer.initializeOptimization();
    optimizer.optimize(parameters.globalBAIterations);

    // Remove outlier observations from the map.
    for (size_t i = 0; i < mapPointEdges.size(); ++i) {
        auto *edge = mapPointEdges.at(i);
        if (edge->chi2() > CHI2_THRESHOLD) {
            MpId mpId = conv.invMapPoint(edge->vertex(0)->id());
            KfId kfId = conv.invKeyframe(edge->vertex(1)->id());
            MapPoint &mapPoint = mapDB.mapPoints.at(mpId);
            mapPoint.eraseObservation(kfId);
            mapDB.keyframes.at(kfId)->eraseObservation(mpId);
            if (mapPoint.observations.size() <= 2) {
                mapPoint.status = MapPointStatus::UNSURE;
            }
        }
    }

    std::set<KfId, std::greater<KfId>> localKeyframes;
    for (auto &p : mapDB.keyframes) {
        localKeyframes.insert(p.first);
    }
    std::set<MpId> localMapPoints;
    for (auto &p : mapDB.mapPoints) {
        localMapPoints.insert(p.first);
    }
    applyBundleAdjustResults(localKeyframes, localMapPoints, mapDB, optimizer, conv);
}

} // namespace slam
