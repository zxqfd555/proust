#include <chrono>
#include <iostream>
#include <fstream>
#include <random>
#include <string_view>
#include <span>
#include <limits>

#include "engine/segtree/segtree.h"
#include "engine/operations.h"
#include "serialization/dataset.h"

// CPU affinity helpers
#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#elif defined(__APPLE__)
#include <pthread.h>
#include <mach/thread_policy.h>
#include <mach/mach.h>
#endif

const size_t QUERY_BATCH_SIZE = 1000;

struct TWeightedDurations {
    uint32_t BatchSize;
    uint32_t AvgDuration;

    TWeightedDurations() = default;

    TWeightedDurations(const uint32_t batchSize, const uint32_t avgDuration)
        : BatchSize(batchSize)
        , AvgDuration(avgDuration)
    {
    }
};

uint32_t CalculatePercentile(const std::vector<TWeightedDurations>& durations, const size_t numerator, const size_t denominator) {
    assert(numerator <= denominator);

    size_t totalCount = 0;
    for (auto& element : durations) {
        totalCount += element.BatchSize;
    }

    size_t currentSum = 0;
    for (auto& element : durations) {
        currentSum += element.BatchSize;
        if (currentSum * denominator >= totalCount * numerator) {
            return element.AvgDuration;
        }
    }

    return std::numeric_limits<uint32_t>::max();
}

void PinExecutionThread(int core_id) {
#if defined(_WIN32)
    DWORD_PTR mask = 1ULL << core_id;
    HANDLE hThread = GetCurrentThread();
    DWORD_PTR result = SetThreadAffinityMask(hThread, mask);
    if (result == 0) {
        spdlog::error("Failed to set CPU affinity on Windows");
    }

#elif defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_t current_thread = pthread_self();
    int ret = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    if (ret != 0) {
        spdlog::error("Failed to set CPU affinity on Linux: {}", ret);
    }

#elif defined(__APPLE__)
    thread_port_t thread = pthread_mach_thread_np(pthread_self());
    thread_affinity_policy_data_t policy = { core_id + 1 };
    kern_return_t kr = thread_policy_set(thread, THREAD_AFFINITY_POLICY, (thread_policy_t)&policy, THREAD_AFFINITY_POLICY_COUNT);
    if (kr != KERN_SUCCESS) {
        spdlog::error("Failed to set CPU affinity on Mac OS");
    }
#else
    #warning "CPU affinity not implemented on this platform"
#endif
}

template<class TValue, class TOperationResult, class TOperation>
void RunDataset(NProust::TStreamingDatasetReader<TValue, TOperationResult>& streamer) {
    PinExecutionThread(0);
    std::vector<TWeightedDurations> weightedDurations;

    auto engine = NProust::CreateRobustEngine<TValue, TOperationResult, TOperation>();
    {
        const auto initStartedAt = std::chrono::high_resolution_clock::now();
        engine->Init(streamer.GetLength());
        const auto initDoneAt = std::chrono::high_resolution_clock::now();
        const auto initDuration = std::chrono::duration_cast<std::chrono::milliseconds>(initDoneAt - initStartedAt).count();
        spdlog::info("Structure initialization done in {} ms", initDuration);
    }    
    
    size_t readCount = 0;
    bool isDatasetExhausted = false;
    const auto runStartedAt = std::chrono::high_resolution_clock::now();
    while (!isDatasetExhausted) {
        std::vector<NProust::TTestPlanItem<TValue, TOperationResult>> datasetChunk;
        datasetChunk.reserve(QUERY_BATCH_SIZE);
        while (datasetChunk.size() < QUERY_BATCH_SIZE) {
            auto maybeItem = streamer.NextQuery();
            if (!maybeItem.has_value()) {
                isDatasetExhausted = true;
                break;
            }
            datasetChunk.emplace_back(std::move(*maybeItem));
            ++readCount;
        }
        if (datasetChunk.empty()) {
            break;
        }

        const auto batchRunStartedAt = std::chrono::high_resolution_clock::now();
        for (const auto& planItem : datasetChunk) {
            switch (planItem.Operation) {
                case NProust::EEngineOperation::SetRange: {
                    engine->SetRange(planItem.Left, planItem.Right, *planItem.Value);
                    break;
                }
                case NProust::EEngineOperation::Query: {
                    const auto result = engine->ComputeRange(planItem.Left, planItem.Right);
                    if (result != *planItem.ExpectedResult) {
                        spdlog::critical("The implementation is incorrect: the benchmark doesn't make sense.");
                        std::terminate();
                    }
                    break;
                }
            };
        }
        const auto batchRunFinishedAt = std::chrono::high_resolution_clock::now();
        const auto avgPerQueryInBatch = std::chrono::duration_cast<std::chrono::nanoseconds>(batchRunFinishedAt - batchRunStartedAt).count() / datasetChunk.size();

        weightedDurations.emplace_back(datasetChunk.size(), avgPerQueryInBatch);
    }

    const auto runFinishedAt = std::chrono::high_resolution_clock::now();
    const auto runTimeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(runFinishedAt - runStartedAt);
    const auto durationPerQuery = std::chrono::duration_cast<std::chrono::nanoseconds>(runFinishedAt - runStartedAt).count() / readCount;
    spdlog::info("The run has finished successfully. {} queries processed in {} ms ({} ns on average per query)", readCount, runTimeElapsed.count(), durationPerQuery);

    std::sort(weightedDurations.begin(), weightedDurations.end(), [](const auto& lhs, const auto& rhs) {return lhs.AvgDuration < rhs.AvgDuration;});
    spdlog::info("Approximate percentile [50]: {} ns", CalculatePercentile(weightedDurations, 1, 2));
    spdlog::info("Approximate percentile [85]: {} ns", CalculatePercentile(weightedDurations, 85, 100));
    spdlog::info("Approximate percentile [95]: {} ns", CalculatePercentile(weightedDurations, 95, 100));
    spdlog::info("Approximate percentile [99]: {} ns", CalculatePercentile(weightedDurations, 99, 100));
    spdlog::info("Approximate percentile [99.9]: {} ns", CalculatePercentile(weightedDurations, 999, 1000));
    spdlog::info("Approximate percentile [99.99]: {} ns", CalculatePercentile(weightedDurations, 9999, 10000));
    spdlog::info("Approximate percentile [99.999]: {} ns", CalculatePercentile(weightedDurations, 99999, 100000));
}

int main(int argc, char* argv[]) {
    std::span<char*> args(argv, argc);

    std::string datasetFn;
    std::string datasetType;

    for (auto it = args.begin() + 1; it != args.end(); ++it) {
        std::string_view arg = *it;

        if (arg == "--dataset-file") {
            if (++it == args.end()) {
                std::cerr << "--dataset-file requires a value" << std::endl;
                return 1;
            }
            datasetFn = *it;
        }
        else if (arg == "--dataset-type") {
            if (++it == args.end()) {
                std::cerr << "--dataset-type requires a value" << std::endl;
                return 1;
            }
            datasetType = *it;
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return 1;
        }
    }

    if (datasetType == "sum_i32") {
        auto datasetStreamer = NProust::TStreamingDatasetReader<int32_t, int32_t>(datasetFn);
        RunDataset<int32_t, int32_t, NProust::TSumOperation>(datasetStreamer);
    } else {
        std::cerr << "Unknown dataset type" << std::endl;
        return 1;
    }

    return 0;
}
