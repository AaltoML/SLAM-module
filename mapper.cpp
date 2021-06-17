#include "mapper.hpp"

#include <thread>
#include <random>
#include <unordered_set>
#include <sstream>

#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <theia/sfm/triangulation/triangulation.h>

#include <cereal/archives/binary.hpp>
#include <cereal/types/map.hpp>

#include "../util/logging.hpp"
#include "../util/util.hpp"
#include "../util/timer.hpp"
#include "../odometry/parameters.hpp"
#include "../odometry/util.hpp"
#include "../commandline/command_queue.hpp"

#include "openvslam/match_base.h"
#include "mapdb.hpp"
#include "bundle_adjuster.hpp"
#include "map_point.hpp"
#include "keyframe_matcher.hpp"
#include "viewer_data_publisher.hpp"
#include "../api/slam.hpp"
#include "loop_closer.hpp"
#include "serialization.hpp"
#include "static_settings.hpp"
#include "orb_extractor.hpp"

// Functions used only from this file.
#include "mapper_helpers.hpp"

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Matrix3x4d = Eigen::Matrix<double,3,4>;
using Eigen::Vector4d;
using Eigen::Vector3d;
using Eigen::Vector2d;
using Eigen::Vector2f;

namespace slam {

struct InputFrame {
    std::unique_ptr<Keyframe> keyframe;
    bool keyFrameDecision;
    std::shared_ptr<const MapperInput> mapperInput;
};

template <class T>
class WorkQueue {
public:
    WorkQueue(size_t maxSize, unsigned int delay) : maxSize(maxSize), delay(delay) {}

    void push(std::unique_ptr<T> task) { // Blocks if queue is full
        {
            std::unique_lock<std::mutex> lock(mutex);
            fullCondition.wait(lock, [this] { return queue.size() < maxSize; });
            queue.push_back(std::move(task));
        }
        emptyCondition.notify_one();
    }

    std::unique_ptr<T> waitAndDequeue() {
        std::unique_ptr<T> task;
        {
            std::unique_lock<std::mutex> lock(mutex);
            emptyCondition.wait(lock, [this] { return queue.size() > delay; });
            task = std::move(queue.front());
            queue.pop_front();
        }
        fullCondition.notify_one();
        return task;
    }

    std::vector<T*> all() { // Returns pointer to all elements in queue, fifo
        std::lock_guard<std::mutex> lock(mutex);
        std::vector<T*> taskPtrs;
        for(auto it = queue.begin(); it != queue.end(); it++) {
            T* taskPtr = (*it).get();
            taskPtrs.push_back(taskPtr);
        }
        return taskPtrs;
    }

    T* task(unsigned int index) {
        std::lock_guard<std::mutex> lock(mutex);
        unsigned int i = 0;
        for(auto it = queue.begin(); it != queue.end(); it++) {
            if (index == i) {
                return (*it).get();
            }
            i++;
        }
        return nullptr;
    }

    void setDelay(const unsigned int newDelay) {
        std::lock_guard<std::mutex> lock(mutex);
        delay = newDelay;
        emptyCondition.notify_one();
    }

private:
    size_t maxSize;
    unsigned int delay;
    std::condition_variable emptyCondition;
    std::condition_variable fullCondition;
    std::deque<std::unique_ptr<T>> queue;
    std::mutex mutex;
};

class MapperImplementation : public Mapper {
private:
    int frontendFrameCounter = 0, backendFrameCounter = 0;
    std::atomic<bool> shouldQuit;
    MapDB mapDB;
    std::unique_ptr<MapDB> frontendMapDB;
    std::vector<MapDB> atlas;
    const StaticSettings settings;

    std::mutex frontendMapMutex;
    WorkQueue<InputFrame> backendQueue;

    bool mapCopyRequested = false;
    std::condition_variable mapCopyCondition;
    std::mutex mapCopyMutex;

    std::unique_ptr<std::thread> thread;

    /**
     * Optional reference to a viewer data publisher. Not owned by this object.
     */
    ViewerDataPublisher *dataPublisher = nullptr;
    CommandQueue *commands = nullptr;
    std::function<void(std::vector<MapPointRecord>)> endDebugCallback = nullptr;

    BowIndex bowIndex;
    std::unique_ptr<LoopCloser> loopCloser;
    std::unique_ptr<OrbExtractor> orbExtractor;

    WorkspaceBA workspaceBA;

public:
    MapperImplementation(const odometry::Parameters &parameters) :
        shouldQuit(false),
        mapDB(),
        frontendMapDB(std::make_unique<MapDB>()),
        atlas(),
        settings(parameters),
        backendQueue(std::max(10,
            (int)settings.parameters.slam.backendProcessDelay
            + (int)settings.parameters.slam.copySlamMapEveryNSlamFrames * 2
            + 2),
            settings.parameters.slam.backendProcessDelay),
        thread(
            settings.parameters.slam.useFrontendSlam
                ? std::make_unique<std::thread>(&MapperImplementation::work, this)
                : nullptr
        ),
        bowIndex(parameters.slam),
        loopCloser(LoopCloser::create(settings, bowIndex, mapDB, atlas)),
        orbExtractor(OrbExtractor::build(settings)),
        workspaceBA(parameters.slam.printBaStats)
    {
        int mapInd = 0;
        for (const auto &loadPath : parameters.slam.mapdbLoadPath) {
            if (loadPath.empty()) continue;
            atlas.push_back(loadMapDB(MapId(mapInd), bowIndex, loadPath));
            ++mapInd;
        }
    }

    void stopAndJoin() {
        if (thread) {
            // Empty backend queue without delay, otherwise it would sit waiting forever
            backendQueue.setDelay(0);
            // Wake up backend thread if it's waiting for copy request from frontend
            shouldQuit.store(true);
            mapCopyCondition.notify_all();
            // Queue terminates when it reaches this task
            backendQueue.push(std::unique_ptr<InputFrame>(nullptr));
            // Wait for backend to finish remaining frames so the map can be saved
            thread->join();
            thread.reset();
        }
    }

    ~MapperImplementation() {
        log_debug("Signaling mapper thread to quit");
        stopAndJoin();
    }

    void requestMapCopy() {
        {
            std::unique_lock<std::mutex> lock(mapCopyMutex);
            mapCopyRequested = true;
        }
        mapCopyCondition.notify_all();
    }

    void mapCopyRequestFulfilled() {
        {
            std::unique_lock<std::mutex> lock(mapCopyMutex);
            mapCopyRequested = false;
        }
        mapCopyCondition.notify_all();
    }

    void waitMapCopyToFinish() {
        {
            std::unique_lock<std::mutex> lock(mapCopyMutex);
            mapCopyCondition.wait(lock, [this] { return !mapCopyRequested; });
        }
    }

    void waitMapCopyRequest() {
        {
            std::unique_lock<std::mutex> lock(mapCopyMutex);
            mapCopyCondition.wait(lock, [this] { return mapCopyRequested || shouldQuit.load(); });
        }
    }

    void work() {
        while(true) {
            std::unique_ptr<InputFrame> input = backendQueue.waitAndDequeue();
            if (!input) {
                // Empty input means stop request
                log_debug("SLAM Backend thread received stop request");
                break;
            }
            int currentFrameNumber = backendFrameCounter++;

            // If backend processing is delayed, skip non-keyframes
            const unsigned int delay = settings.parameters.slam.backendProcessDelay;
            if (currentFrameNumber == 0 || delay == 0 || input->keyFrameDecision) {
                if (delay) { // If backend processing is delayed, we can use newer pose trail information
                    auto futureInput = backendQueue.task(delay - 1); // -1 for the current frame we took out of the queue
                    if (futureInput) {
                        auto newMapperInput = std::make_unique<MapperInput>(*input->mapperInput);
                        auto &newPoseTrail = newMapperInput->poseTrail;
                        newPoseTrail.clear();
                        auto &inputPoseTrail = input->mapperInput->poseTrail;
                        const auto &futurePoseTrail = futureInput->mapperInput->poseTrail;
                        for (size_t i = 0; i < inputPoseTrail.size(); i++) {
                            const int frameNumber = inputPoseTrail[i].frameNumber;
                            auto it = std::find_if(futurePoseTrail.begin(), futurePoseTrail.end(),
                                [frameNumber] (const slam::Pose &p) { return p.frameNumber == frameNumber; } );
                            if (it != futurePoseTrail.end()) {
                                // Only add intersection of current & future frames, using future pose
                                newPoseTrail.push_back(*it);
                            } else if (i == 0) {
                                // Ensure current pose is always in the trail, even if it doesn't exist in the future
                                newPoseTrail.push_back(inputPoseTrail.at(i));
                            }
                        }
                        // must replace the whole mapper input since they need
                        // to be immutable for multi-thead acce
                        input->mapperInput = std::move(newMapperInput);
                    }
                }

                Eigen::Matrix4d unusedResultPose;
                processBackendFrame(input, unusedResultPose, nullptr);
            }
            if ((currentFrameNumber + 1) % settings.parameters.slam.copySlamMapEveryNSlamFrames == 0) {
                if (settings.parameters.slam.deterministicSlamMapCopy) waitMapCopyRequest();
                if (!shouldQuit.load()) {
                    copyMap();
                }
                if (settings.parameters.slam.deterministicSlamMapCopy) mapCopyRequestFulfilled();
            }
        }
    }

    void copyMap() {
        // Copy backend map
        std::unique_ptr<slam::MapDB> newMapDB;
        bool partialMapCopy = settings.parameters.slam.copyPartialMapToFrontend;

        Keyframe *latestKeyframe = mapDB.latestKeyframe();
        if (latestKeyframe == nullptr && partialMapCopy) {
            log_warn("last keyframe null -> full map copy");
            partialMapCopy = false;
        }

        if (partialMapCopy) {
            timer(slam::TIME_STATS, "Copying partial map");
            constexpr int minCovisibilities = 5; // TODO: Pulled out of a hat
            std::vector<KfId> adjacentKfIds = computeAdjacentKeyframes(
                *latestKeyframe,
                minCovisibilities,
                settings.parameters.slam.adjacentSpaceSize,
                mapDB,
                settings,
                true
            );
            std::set<KfId> activeKeyframe(adjacentKfIds.begin(), adjacentKfIds.end());
            activeKeyframe.insert(latestKeyframe->id);
            newMapDB = std::make_unique<MapDB>(mapDB, activeKeyframe);
        } else {
            timer(slam::TIME_STATS, "Copying full map");
            newMapDB = std::make_unique<MapDB>(mapDB);
        }

        // Fast forward new keyframes added during backend processing
        // TODO: this can apparently cause some complications / race conditions
        // if keyframes can be removed in the frontend SLAM too
        // fastForward(*newMapDB.get());

        {
            // Prevent frontend processing while frontend map is replaced
            std::lock_guard<std::mutex> lock(frontendMapMutex);

            // New frames might have come in while previous new frames where being processed
            fastForward(*newMapDB.get());

            // Replace old frontend map with the new one
            frontendMapDB = std::move(newMapDB);
        }
    }

    void fastForward(MapDB &newMapDB) {
        auto newKeyframes = backendQueue.all();
        for(auto it = newKeyframes.begin(); it != newKeyframes.end(); it++) {
            InputFrame* input = *it;
            if (!input) {
                // Should only happen when there is a termination request i.e. nullptr in queue.
                continue;
            }
            Keyframe *kf = input->keyframe.get();
            if (!newMapDB.keyframes.count(kf->id)) { // Previous fast forward might have added some of these
                Eigen::Matrix4d unusedResultPose;
                addKeyframeFrontend(newMapDB, std::make_unique<Keyframe>(*kf), input->keyFrameDecision,
                    *input->mapperInput, settings, unusedResultPose, nullptr);
            }
        }
    }

    void advance(
        std::shared_ptr<const MapperInput> mapperInput,
        Eigen::Matrix4d &resultPose,
        Slam::Result::PointCloud &pointCloud
    ) {
        if (slam::TIME_STATS) slam::TIME_STATS->startFrame();
        if (!settings.parameters.slam.useFrontendSlam) {
            backendOnly(mapperInput, resultPose, pointCloud);
            return;
        }

        std::unique_ptr<Keyframe> kf;
        bool keyFrameDecision;
        {
            kf = std::make_unique<Keyframe>(*mapperInput);
            {
                std::lock_guard<std::mutex> lock(frontendMapMutex);
                keyFrameDecision = makeKeyframeDecision(*kf, frontendMapDB->latestKeyframe(), mapperInput->trackerFeatures,
                    settings.parameters.slam);
            }
        }

        auto kfBackend = std::make_unique<Keyframe>(*kf);
        if (dataPublisher) {
            const auto &cmd = dataPublisher->getParameters();
            if (cmd.visualizeOrbMatching || cmd.visualizeLoopOrbMatching || cmd.visualizeMapPointSearch) {
                // may fills odometry FrameBuffer without .clone()
                kfBackend->shared->imgDbg = mapperInput->colorFrame.clone();
            }
        }

        if (settings.parameters.slam.deterministicSlamMapCopy) waitMapCopyToFinish();


        {
            std::lock_guard<std::mutex> lock(frontendMapMutex);

            // TODO: If we would use pose trail to get delta, we wouldn't need to send non-keyframes to backend
            backendQueue.push(std::make_unique<InputFrame>(InputFrame{
                .keyframe = std::move(kfBackend),
                .keyFrameDecision = keyFrameDecision,
                .mapperInput = mapperInput
            }));

            addKeyframeFrontend(*frontendMapDB.get(), std::move(kf), keyFrameDecision,
                *mapperInput, settings, resultPose, &pointCloud);

            workspaceBA.baStats.finishFrame();
        }

        const int currentFrameNumber = frontendFrameCounter++;
        const int backendTotalDelay = (int)settings.parameters.slam.copySlamMapEveryNSlamFrames * 2
            + (int)settings.parameters.slam.backendProcessDelay
            - 1;
        if (settings.parameters.slam.deterministicSlamMapCopy // Without deterministic map copy, backend decides when to copy the map
            && currentFrameNumber >= backendTotalDelay // Only copy if there are enough frames in backend map
            && (currentFrameNumber + 1) % settings.parameters.slam.copySlamMapEveryNSlamFrames == 0) {
            requestMapCopy();
        }
    }

    void backendOnly(std::shared_ptr<const MapperInput> mapperInput,
        Eigen::Matrix4d &resultPose,
        Slam::Result::PointCloud &pointCloud
    ) {
        std::unique_ptr<Keyframe> kf;
        bool keyFrameDecision;
        {
            kf = std::make_unique<Keyframe>(*mapperInput);
            keyFrameDecision = makeKeyframeDecision(*kf, mapDB.latestKeyframe(), mapperInput->trackerFeatures,
                settings.parameters.slam);
        }
        if (dataPublisher) {
            const auto &cmd = dataPublisher->getParameters();
            if (cmd.visualizeOrbMatching || cmd.visualizeLoopOrbMatching || cmd.visualizeMapPointSearch) {
                // may fills odometry FrameBuffer without .clone()
                kf->shared->imgDbg = mapperInput->colorFrame.clone();
            }
        }
        auto inputFrame = std::make_unique<InputFrame>(InputFrame{
            .keyframe = std::move(kf),
            .keyFrameDecision = keyFrameDecision,
            .mapperInput = mapperInput
        });

        processBackendFrame(inputFrame, resultPose, &pointCloud);
        workspaceBA.baStats.finishFrame();
        Keyframe *currentKeyframe = mapDB.latestKeyframe();
        if (currentKeyframe) visualizeMapperInput(currentKeyframe, *mapperInput);
    }

    KfId processBackendFrame(std::unique_ptr<InputFrame> &input,
        Eigen::Matrix4d &resultPose,
        Slam::Result::PointCloud *pointCloud)
    {
        return addKeyframeBackend(
            mapDB,
            std::move(input->keyframe),
            input->keyFrameDecision,
            *input->mapperInput,
            settings,
            workspaceBA,
            *loopCloser,
            *orbExtractor,
            bowIndex,
            commands,
            dataPublisher,
            resultPose,
            pointCloud);
    }

    void visualizeMapperInput(Keyframe *currentKeyframe, const MapperInput &mapperInput) {
        if (!dataPublisher) return;

        const auto &cmd = dataPublisher->getParameters();

        if (cmd.displayKeyframe && !mapperInput.colorFrame.empty()) {
            assert(currentKeyframe);
            dataPublisher->visualizeKeyframe(
                mapDB,
                mapperInput.colorFrame,
                *currentKeyframe);
        } else if (cmd.visualizeOrbPyramid) {
            cv::Mat tmp;
            orbExtractor->debugVisualize(*mapperInput.frame, tmp, OrbExtractor::VisualizationMode::IMAGE_PYRAMID);
            dataPublisher->visualizeOther(tmp);
        } else if (cmd.visualizeOrbs && !mapperInput.colorFrame.empty()) {
            assert(currentKeyframe);
            dataPublisher->visualizeOrbs(mapperInput.colorFrame, *currentKeyframe);
        }
    }

    void connectDebugAPI(DebugAPI &debug) {
        if (debug.dataPublisher) {
            if (dataPublisher) log_warn("Set data publisher multiple times");
            dataPublisher = debug.dataPublisher;
            loopCloser->setViewerDataPublisher(debug.dataPublisher);

            auto viewerAtlas = std::make_unique<ViewerAtlas>();
            for (const MapDB &m : atlas) {
                viewerAtlas->push_back(mapDBtoViewerAtlasMap(m));
            }
            dataPublisher->setAtlas(std::move(viewerAtlas));
        }
        if (debug.commandQueue) {
            commands = debug.commandQueue;
            loopCloser->setCommandQueue(debug.commandQueue);
        }
        if (debug.endDebugCallback) {
            endDebugCallback = debug.endDebugCallback;
        }
    }

    bool end(const std::string &mapPoseSavePath) {
        stopAndJoin();

        checkConsistency(mapDB);

        // Could merge these into `endDebugCallback()` below.
        if (!settings.parameters.slam.mapdbSavePath.empty()) {
            std::ofstream mapStream;
            mapStream.open(settings.parameters.slam.mapdbSavePath, std::ios::out | std::ios::binary);
            {
                cereal::BinaryOutputArchive oarchive(mapStream);
                oarchive(mapDB);
            } // Serialization flushes at end of scope.
            log_debug("Wrote SLAM map: %.2f MB.", 1e-6 * static_cast<double>(mapStream.tellp()));
        }

        if (!mapPoseSavePath.empty()) {
            std::ofstream saveFile(mapPoseSavePath);
            if (!saveFile) {
                log_warn("failed to open %s", mapPoseSavePath.c_str());
                return false;
            }
            saveFile.precision(8);
            saveFile << std::fixed;

            for (auto kfIt = mapDB.keyframes.begin(); kfIt != mapDB.keyframes.end(); kfIt++) {
                const Keyframe &kf = *kfIt->second;
                const Eigen::Matrix4d &pose = kf.poseCW;
                const Eigen::Matrix4d camToWorld = pose.inverse();
                const Eigen::Matrix4d imuToWorld = camToWorld * settings.parameters.imuToCamera;
                const Eigen::Vector3d pos = imuToWorld.block<3, 1>(0, 3);
                const Eigen::Vector4d quat = odometry::util::rmat2quat(imuToWorld.topLeftCorner<3, 3>());

                saveFile << kf.t
                    << "," << pos(0)
                    << "," << pos(1)
                    << "," << pos(2)
                    << "," << quat(0)
                    << "," << quat(1)
                    << "," << quat(2)
                    << "," << quat(3)
                    << std::endl;
            }
        }

        if (endDebugCallback) {
            // Convert to vector because we don't need the MpIds.
            std::vector<MapPointRecord> collection;
            collection.reserve(mapDB.mapPointRecords.size());
            for (const auto &it : mapDB.mapPointRecords) {
                collection.push_back(it.second);
            }
            endDebugCallback(collection);
        }

        return true;
    }
};

std::unique_ptr<Mapper>
Mapper::create(const odometry::Parameters &parameters) {
    return std::unique_ptr<Mapper>(new MapperImplementation(parameters));
}

} // namespace slam
