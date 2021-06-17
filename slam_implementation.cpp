#include <cmath>
#include <deque>
#include <mutex>
#include <thread>
#include <Eigen/Geometry>

#include <opencv2/core.hpp>

#include "slam_implementation.hpp"
#include "mapper.hpp"
#include "viewer_data_publisher.hpp"
#include "keyframe.hpp"

#include "../commandline/command_queue.hpp"
#include "../odometry/parameters.hpp"
#include "../util/logging.hpp"
#include "../util/string_utils.hpp"

#include "../tracker/image.hpp"

namespace slam {
namespace {
class Worker {
public:
    using Result = Slam::Result;
    struct Task {
        struct Mapper {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            std::shared_ptr<const MapperInput> mapperInput;
            // using a pointer here is an optimization that primarily reduces
            // reallocation of PointCloud objects
            std::promise<Result> *result = nullptr;
        };

        struct End {
            std::promise<bool> result;
        };

        std::unique_ptr<Mapper> mapperInput;
        std::unique_ptr<End> end;

        static std::unique_ptr<Task> wrap(std::unique_ptr<Mapper> mapperInput) {
            auto task = std::make_unique<Task>();
            task->mapperInput = std::move(mapperInput);
            return task;
        }

        static std::unique_ptr<Task> wrap(std::unique_ptr<End> mapperInput) {
            auto task = std::make_unique<Task>();
            task->end = std::move(mapperInput);
            return task;
        }
    };

    Worker(const odometry::Parameters &parameters) :
        mapper(Mapper::create(parameters)),
        resultStorage(100), // MAX_QUEUED_RESULTS = 100
        shouldQuit(false),
        mutex(),
        // Make sure thread is initialized last, because it can immediately call work() before this constructor finishes
        thread(parameters.slam.slamThread ?
            std::make_unique<std::thread>(&Worker::work, this) :
            nullptr)
    {}

    ~Worker() {
        if (thread) {
            log_debug("signaling the SLAM thead to quit");
            {
                std::unique_lock<std::mutex> lock(mutex);
                shouldQuit = true;
            }

            log_debug("waiting for the SLAM thead to quit");
            thread->join();
        }
        log_debug("~Worker end");
    }

    std::future<Result> enqueueMapperInput(std::unique_ptr<Task::Mapper> task) {
        std::promise<Result> *result;
        auto wrapped = Task::wrap(std::move(task));
        if (!thread) {
            result = &nextResult();
            wrapped->mapperInput->result = result;
            process(*wrapped);
        } else {
            std::unique_lock<std::mutex> lock(mutex);
            workQueue.push_back(std::move(wrapped));
            result = &nextResult();
            workQueue.back()->mapperInput->result = result;
        }
        return result->get_future();
    }

    std::future<bool> end() {
        std::unique_ptr<Worker::Task::End> endTask(
            new Worker::Task::End {
                .result = std::promise<bool>(),
            });
        std::promise<bool> &result = endTask->result;
        auto wrapped = Task::wrap(std::move(endTask));
        if (!thread) {
            process(*wrapped);
        } else {
            std::unique_lock<std::mutex> lock(mutex);
            workQueue.push_back(std::move(wrapped));
        }
        return result.get_future();
    }

    void connectDebugAPI(DebugAPI &debug) {
        // NOTE: not thread-safe, but called in practice before anything
        // is going on in the worker thread
        mapper->connectDebugAPI(debug);
        mapSavePath = debug.mapSavePath;
    }

private:
    std::promise<Result> &nextResult() {
        auto &result = resultStorage.at(resultCounter);
        result = {};
        resultCounter = (resultCounter + 1) % resultStorage.size();
        return result;
    }

    std::unique_ptr<Mapper> mapper;

    // TODO: not sure if this is really necessary
    std::vector<std::promise<Result>> resultStorage;
    int resultCounter = 0;

    bool shouldQuit;
    std::mutex mutex;
    std::deque<std::unique_ptr<Task>> workQueue;
    std::unique_ptr<std::thread> thread; // Can call work() before constructor finishes, at the very least keep it last in constructor.

    std::string mapSavePath;

    void work() {
        log_debug("starting SLAM thread");
        while (true) {
            {
                std::unique_lock<std::mutex> lock(mutex);
                if (shouldQuit) return;
            }

            std::unique_ptr<Task> task;
            {
                std::unique_lock<std::mutex> lock(mutex);
                if (!workQueue.empty()) {
                    task = std::move(workQueue.front());
                    workQueue.pop_front();
                }
            }

            if (task) {
                process(*task);
            }
            else {
                // TODO: bad, should properly wait for new elements
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
    }

    void process(Task &typedTask) {
        if (typedTask.mapperInput) {
            Result result;
            Eigen::Matrix4d resultPoseMat;
            auto &task = *typedTask.mapperInput;

            // if (parameters.slam.useFullPoseTrail) {
            //     mapper->updatePoseTrail(*task.mapperInput);
            // }

            mapper->advance(task.mapperInput, resultPoseMat, result.pointCloud);

            result.poseMat = resultPoseMat;
            task.result->set_value(result);
        }
        if (typedTask.end) {
            const bool success = mapper->end(mapSavePath);
            typedTask.end->result.set_value(success);
            log_debug("end processed");
        }
    }
};

class SlamImplementation : public Slam {
private:
    std::unique_ptr<Worker> worker;

public:
    SlamImplementation(const odometry::Parameters &parameters) :
        worker(new Worker(parameters))
    {}

    void connectDebugAPI(DebugAPI &debug) final {
        worker->connectDebugAPI(debug);
    }

    std::future<Slam::Result> addFrame(
        std::shared_ptr<tracker::Image> frame,
        const std::vector<slam::Pose> &poseTrail,
        const std::vector<Feature> &features,
        const cv::Mat &colorFrame
    ) final {
        auto mapperInput = std::unique_ptr<MapperInput>(new MapperInput);
        mapperInput->frame = frame;
        mapperInput->trackerFeatures = features;
        mapperInput->poseTrail = poseTrail;
        mapperInput->t = poseTrail[0].t;
        mapperInput->colorFrame = colorFrame;

        return worker->enqueueMapperInput(std::unique_ptr<Worker::Task::Mapper>(
            new Worker::Task::Mapper {
                .mapperInput = std::move(mapperInput)
            }
        ));
    }

    std::future<bool> end() final {
        log_debug("SlamImplementation::end");
        return worker->end();
    }
};
}

std::unique_ptr<Slam> Slam::build(const odometry::Parameters &parameters) {
    return std::make_unique<SlamImplementation>(parameters);
}

} // namespace slam
