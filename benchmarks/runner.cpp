#include <chrono>
#include <iostream>
#include <fstream>
#include <random>
#include <string_view>
#include <span>
#include <limits>

#include "common/dataset.h"
#include "engine/robust/engine.h"
#include "engine/operations.h"
#include "examples/leetcode_maximum_subarray.h"
#include "examples/leetcode_third_distinct_maximum.h"
#include "serialization/dataset.h"

#include "args_parsing.h"

#if defined(_WIN32)
    #define NOMINMAX
    #include <windows.h>
    #include <psapi.h>
#elif defined(__linux__)
    #include <pthread.h>
    #include <sched.h>
    #include <sys/resource.h>
#elif defined(__APPLE__)
    #include <pthread.h>
    #include <sys/resource.h>
    #include <mach/thread_policy.h>
    #include <mach/mach.h>
#endif

#if defined(_MSC_VER)
    #include <intrin.h>
    #pragma intrinsic(__rdtsc)
    #pragma intrinsic(__rdtscp)
    #pragma intrinsic(__cpuid)
#else
    #include <x86intrin.h>
#endif

constexpr int32_t INT32_SENTINEL = std::numeric_limits<int32_t>::max();

const double WARMUP_DATASET_SHARE = 0.05;
const size_t QUERY_BATCH_SIZE = 100;
const size_t MAX_CYCLES_PER_OPERATION = 10000000;

struct TPercentile {
    const size_t Numerator;
    const size_t Denominator;
    const std::string Name;
};

const std::vector<TPercentile> TARGET_PERCENTILES = {
    {50, 100, "50"},
    {85, 100, "85"},
    {95, 100, "95"},
    {99, 100, "99"},
    {999, 1000, "99.9"},
    {9999, 10000, "99.99"},
    {99999, 100000, "99.999"},
};

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

uint32_t CalculatePercentile(const std::vector<size_t>& buckets, const size_t numerator, const size_t denominator) {
    assert(numerator <= denominator);

    size_t totalCount = 0;
    for (auto& element : buckets) {
        totalCount += element;
    }

    size_t currentSum = 0;
    for (size_t bucketIndex = 0; bucketIndex < buckets.size(); ++bucketIndex) {
        currentSum += buckets[bucketIndex];
        if (currentSum * denominator >= totalCount * numerator) {
            return bucketIndex;
        }
    }

    return std::numeric_limits<uint32_t>::max();
}

static inline uint64_t StartRDTSC() {
    _mm_lfence();
    return __rdtsc();
}

static inline uint64_t EndRDTSC() {
    uint32_t aux;
    uint64_t t = __rdtscp(&aux);
    _mm_lfence();
    return t;
}

void PinExecutionThread(int32_t core_id) {
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
        spdlog::error("Failed to set CPU affinity on MacOS");
    }
#else
    #warning "CPU affinity not implemented on this platform"
#endif
}

int64_t GetPeakMemoryUsageKB() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<long>(pmc.PeakWorkingSetSize / 1024);
    }
    spdlog::error("Failed to get peak used memory on Windows");
    return -1;
#elif defined(__linux__)
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return usage.ru_maxrss;
    }
    spdlog::error("Failed to get peak used memory on Linux");
    return -1;
#elif defined(__APPLE__)
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return usage.ru_maxrss / 1024;
    }
    spdlog::error("Failed to get peak used memory on MacOS");
    return -1;
#else
    spdlog::error("Failed detect the platform");
    return -1;
#endif
}

template<class TValue, class TOperationResult, class TOperation, auto Sentinel = std::nullopt>
void RunDatasetWithSettings(
    NProust::TStreamingDatasetReader<TValue, TOperationResult>& streamer,
    const NProust::EDataLayout layout,
    const NProust::ENodeTopology topology,
    const size_t leafSizeLog,
    const NProust::EBaseGeneratorType baseGeneratorType
) {
    switch (layout) {
        case NProust::EDataLayout::AoS: {
            DispatchTopologyAndRun<TValue, TOperationResult, TOperation, NProust::TAoSLayout<TValue, TOperationResult, Sentinel>>(streamer, topology, leafSizeLog, baseGeneratorType);
            break;
        }
        case NProust::EDataLayout::SoA: {
            DispatchTopologyAndRun<TValue, TOperationResult, TOperation, NProust::TSoALayout<TValue, TOperationResult, Sentinel>>(streamer, topology, leafSizeLog, baseGeneratorType);
            break;
        }
    }
}

template<class TValue, class TOperationResult, class TOperation, NProust::DataLayout<TValue, TOperationResult> TDataLayout>
void DispatchTopologyAndRun(
    NProust::TStreamingDatasetReader<TValue, TOperationResult>& streamer,
    NProust::ENodeTopology topology,
    size_t leafSizeLog,
    const NProust::EBaseGeneratorType baseGeneratorType
) {
    switch (leafSizeLog) {
        case 0: RunWithDispatchedTopology<TValue, TOperationResult, TOperation, TDataLayout, 0>(streamer, topology, baseGeneratorType); break;
        case 1: RunWithDispatchedTopology<TValue, TOperationResult, TOperation, TDataLayout, 1>(streamer, topology, baseGeneratorType); break;
        case 2: RunWithDispatchedTopology<TValue, TOperationResult, TOperation, TDataLayout, 2>(streamer, topology, baseGeneratorType); break;
        case 3: RunWithDispatchedTopology<TValue, TOperationResult, TOperation, TDataLayout, 3>(streamer, topology, baseGeneratorType); break;
        case 4: RunWithDispatchedTopology<TValue, TOperationResult, TOperation, TDataLayout, 4>(streamer, topology, baseGeneratorType); break;
        case 5: RunWithDispatchedTopology<TValue, TOperationResult, TOperation, TDataLayout, 5>(streamer, topology, baseGeneratorType); break;
        case 6: RunWithDispatchedTopology<TValue, TOperationResult, TOperation, TDataLayout, 6>(streamer, topology, baseGeneratorType); break;
        case 7: RunWithDispatchedTopology<TValue, TOperationResult, TOperation, TDataLayout, 7>(streamer, topology, baseGeneratorType); break;
        case 8: RunWithDispatchedTopology<TValue, TOperationResult, TOperation, TDataLayout, 8>(streamer, topology, baseGeneratorType); break;
        case 9: RunWithDispatchedTopology<TValue, TOperationResult, TOperation, TDataLayout, 9>(streamer, topology, baseGeneratorType); break;
        case 10: RunWithDispatchedTopology<TValue, TOperationResult, TOperation, TDataLayout, 10>(streamer, topology, baseGeneratorType); break;
        default: {
            spdlog::critical("Unsupported value of {}: {}", LEAF_SIZE_LOG_PARAM, leafSizeLog);
            std::terminate();
        }
    }
}

template<class TValue, class TOperationResult, class TOperation, NProust::DataLayout<TValue, TOperationResult> TDataLayout, size_t LeafSizeLog>
void RunWithDispatchedTopology(
    NProust::TStreamingDatasetReader<TValue, TOperationResult>& streamer,
    const NProust::ENodeTopology topology,
    const NProust::EBaseGeneratorType baseGeneratorType
) {
    switch (topology) {
        case NProust::ENodeTopology::BinaryHeap: {
            RunDataset<TValue, TOperationResult, TOperation, TDataLayout, NProust::TBinaryHeapNodeTopology<LeafSizeLog>, LeafSizeLog>(streamer, baseGeneratorType);
            break;
        }
        case NProust::ENodeTopology::Preorder: {
            RunDataset<TValue, TOperationResult, TOperation, TDataLayout, NProust::TPreorderNodeTopology<LeafSizeLog>, LeafSizeLog>(streamer, baseGeneratorType);
            break;
        }
    }
}

template<class TValue, class TOperationResult, class TOperation, NProust::DataLayout<TValue, TOperationResult> TDataLayout, NProust::NodeTopology TNodeTopology, size_t LeafSizeLog>
void RunDataset(NProust::TStreamingDatasetReader<TValue, TOperationResult>& streamer, NProust::EBaseGeneratorType baseGeneratorType) {
    std::vector<TWeightedDurations> weightedDurations;
    std::vector<size_t> cyclesSetValue(MAX_CYCLES_PER_OPERATION, 0);
    std::vector<size_t> cyclesComputeRange(MAX_CYCLES_PER_OPERATION, 0);

    std::unique_ptr<NProust::IRangeUpdateEngine<TValue, TOperationResult, TOperation>> rangeUpdateEngine;
    std::unique_ptr<NProust::IPointUpdateEngine<TValue, TOperationResult, TOperation>> pointUpdateEngine;
    {
        const auto initStartedAt = std::chrono::high_resolution_clock::now();
        switch (baseGeneratorType) {
            case (NProust::EBaseGeneratorType::Point): {
                spdlog::info("Using a point-update engine");
                pointUpdateEngine = NProust::CreateRobustPointUpdateEngine<TValue, TOperationResult, TOperation, LeafSizeLog>();
                pointUpdateEngine->Init(streamer.GetLength());
                break;
            }
            case (NProust::EBaseGeneratorType::Range): {
                spdlog::info("Using a range-update engine");
                rangeUpdateEngine = NProust::CreateRobustRangeUpdateEngine<TValue, TOperationResult, TOperation, TNodeTopology, TDataLayout>();
                rangeUpdateEngine->Init(streamer.GetLength());
                break;
            }
        }

        const auto initDoneAt = std::chrono::high_resolution_clock::now();
        const auto initDuration = std::chrono::duration_cast<std::chrono::milliseconds>(initDoneAt - initStartedAt).count();
        spdlog::info("Structure initialization done in {} ms", initDuration);
    }    

    size_t readCount = 0;
    bool isDatasetExhausted = false;
    const auto runStartedAt = std::chrono::high_resolution_clock::now();
    const auto warmupUntilEventId = (size_t)(WARMUP_DATASET_SHARE * streamer.GetQueryCount());
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
            if (readCount == warmupUntilEventId) {
                // Make sure that no chunk is divided between the warmup and the benchmarking part.
                break;
            }
        }
        if (datasetChunk.empty()) {
            break;
        }
        const bool warmupIsDone = warmupUntilEventId < readCount;

        const auto batchRunStartedAt = std::chrono::high_resolution_clock::now();
        for (const auto& planItem : datasetChunk) {
            switch (planItem.Operation) {
                case NProust::EEngineOperation::SetRange: {
                    const auto startCycle = StartRDTSC();
                    switch (baseGeneratorType) {
                        case (NProust::EBaseGeneratorType::Point): {
                            pointUpdateEngine->SetPoint(planItem.Left, *planItem.Value);
                            break;
                        }
                        case (NProust::EBaseGeneratorType::Range): {
                            rangeUpdateEngine->SetRange(planItem.Left, planItem.Right, *planItem.Value);
                            break;
                        }
                    }
                    const auto endCycle = EndRDTSC();
                    const auto totalCycles = std::min(endCycle - startCycle, MAX_CYCLES_PER_OPERATION - 1);
                    if (warmupIsDone) {
                        cyclesSetValue[totalCycles] += 1;
                    }
                    break;
                }
                case NProust::EEngineOperation::Query: {
                    const auto startCycle = StartRDTSC();
                    TOperationResult result;
                    switch (baseGeneratorType) {
                        case (NProust::EBaseGeneratorType::Point): {
                            result = pointUpdateEngine->ComputeRange(planItem.Left, planItem.Right);
                            break;
                        }
                        case (NProust::EBaseGeneratorType::Range): {
                            result = rangeUpdateEngine->ComputeRange(planItem.Left, planItem.Right);
                            break;
                        }
                    }
                    const auto endCycle = EndRDTSC();
                    const auto totalCycles = std::min(endCycle - startCycle, MAX_CYCLES_PER_OPERATION - 1);
                    if (warmupIsDone) {
                        cyclesComputeRange[totalCycles] += 1;
                    }
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

        if (warmupIsDone) {
            weightedDurations.emplace_back((uint32_t)datasetChunk.size(), (uint32_t)avgPerQueryInBatch);
        }
    }

    const auto runFinishedAt = std::chrono::high_resolution_clock::now();
    const auto runTimeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(runFinishedAt - runStartedAt);
    const auto durationPerQuery = std::chrono::duration_cast<std::chrono::nanoseconds>(runFinishedAt - runStartedAt).count() / readCount;
    spdlog::info("The run has finished successfully. {} queries processed in {} ms ({} ns on average per query)", readCount, runTimeElapsed.count(), durationPerQuery);

    std::sort(weightedDurations.begin(), weightedDurations.end(), [](const auto& lhs, const auto& rhs) {return lhs.AvgDuration < rhs.AvgDuration;});
    spdlog::info("Peak memory usage: {} Kb", GetPeakMemoryUsageKB());
    for (const auto& percentile : TARGET_PERCENTILES) {
        spdlog::info("Approximate percentile [{}]: {} ns", percentile.Name, CalculatePercentile(weightedDurations, percentile.Numerator, percentile.Denominator));
    }

    // TSC usage is somehow controversial, but I still do that.
    // Initially, in the earlier CPU architectures, it was used to measure the code performance
    // in the CPU cycles. It was however frequently misused, because the people divided the TSC
    // performance by the CPU frequency, which is wrong: the frequency of CPU is variable and
    // depends on many factors (TurboBoost, power saving mode, idle periods, ...).
    // Then, the TSC was adapted to be better used in the time measuring tasks: modern CPUs have
    // the isolated power source for both computing and the TSC units. So now the TSC can also
    // be uncoupled from the main computation flow.
    //
    // I still use it, because it the only was to see how modification and query behave relatively
    // to each other: the overhead of __rdtsc are low nanoseconds, so we can query it on every operation
    // and have reliable per-operation histograms. Not possible with a steady clock that works in
    // high nanoseconds-low microseconds. Of course, you need to pin your execution to a single core
    // to make sure that start and end counts are taken from a single source.
    //
    // So, you need to be aware of your CPU architecture to correctly interpret the results. Use the
    // TSC cycles as a kind of an abstract time/performance units.
    for (const auto& percentile : TARGET_PERCENTILES) {
        spdlog::info("Modification percentile [{}]: {} TSC cycles", percentile.Name, CalculatePercentile(cyclesSetValue, percentile.Numerator, percentile.Denominator));
    }
    for (const auto& percentile : TARGET_PERCENTILES) {
        spdlog::info("Query percentile [{}]: {} TSC cycles", percentile.Name, CalculatePercentile(cyclesComputeRange, percentile.Numerator, percentile.Denominator));
    }
}

int main(int argc, char* argv[]) {
    // The pin is needed for TSC usage that goes next, and also to reduce the number of
    // process migrations between cores, which would be harmful, for example, because of the
    // cold cache.
    PinExecutionThread(0);

//    static_assert(std::chrono::high_resolution_clock::is_steady, "High resolution clock is not steady");
    static_assert(std::chrono::high_resolution_clock::period::den, "High resolution clock doesn't have a nanosecond precision");

    std::span<char*> args(argv, argc);

    std::optional<std::string> datasetFn;
    std::optional<NProust::EDatasetType> datasetType;
    std::optional<NProust::EDataLayout> dataLayout;
    std::optional<NProust::ENodeTopology> nodeTopology;
    std::optional<size_t> leafSizeLog;
    std::optional<NProust::EBaseGeneratorType> baseGeneratorType;
    bool withSentinel = false;

    for (auto it = args.begin() + 1; it != args.end(); ++it) {
        std::string_view arg = *it;
        if (arg == DATASET_FILE_PARAM) {
            ParseStringArgument(it, args.end(), datasetFn);
        } else if (arg == DATASET_TYPE_PARAM) {
            ParseDatasetTypeArgument(it, args.end(), datasetType);
        } else if (arg == DATA_LAYOUT_PARAM) {
            ParseDataLayoutArgument(it, args.end(), dataLayout);
        } else if (arg == NODE_TOPOLOGY_PARAM) {
            ParseNodeTopologyArgument(it, args.end(), nodeTopology);
        } else if (arg == LEAF_SIZE_LOG_PARAM) {
            ParseIntegralArgument(it, args.end(), leafSizeLog);
        } else if (arg == WITH_SENTINEL_PARAM) {
            withSentinel = true;
        } else if (arg == GENERATOR_TYPE_PARAM) {
            ParseBaseGeneratorTypeArgument(it, args.end(), baseGeneratorType);
        } else {
            spdlog::critical("Unknown argument: {}", arg);
            std::terminate();
        }
    }

    CheckFieldPresence(DATASET_TYPE_PARAM, datasetType);
    CheckFieldPresence(DATASET_FILE_PARAM, datasetFn);
    CheckFieldPresence(DATA_LAYOUT_PARAM, dataLayout);
    CheckFieldPresence(NODE_TOPOLOGY_PARAM, nodeTopology);
    CheckFieldPresence(LEAF_SIZE_LOG_PARAM, leafSizeLog);
    CheckFieldPresence(GENERATOR_TYPE_PARAM, baseGeneratorType);

    switch (*datasetType) {
        case NProust::EDatasetType::SumInt32: {
            auto datasetStreamer = NProust::TStreamingDatasetReader<int32_t, int32_t>(*datasetFn);
            if (withSentinel) {
                RunDatasetWithSettings<int32_t, int32_t, NProust::TSumOperation, INT32_SENTINEL>(datasetStreamer, *dataLayout, *nodeTopology, *leafSizeLog, *baseGeneratorType);
            } else {
                RunDatasetWithSettings<int32_t, int32_t, NProust::TSumOperation>(datasetStreamer, *dataLayout, *nodeTopology, *leafSizeLog, *baseGeneratorType);
            }
            break;
        }
        case NProust::EDatasetType::MaxInt32: {
            auto datasetStreamer = NProust::TStreamingDatasetReader<int32_t, int32_t>(*datasetFn);
            if (withSentinel) {
                spdlog::critical("Sentinel values are not supported for this type of the dataset");
                std::terminate();
            } else {
                RunDatasetWithSettings<int32_t, int32_t, NProust::TMaxOperation>(datasetStreamer, *dataLayout, *nodeTopology, *leafSizeLog, *baseGeneratorType);
            }
            break;
        }
        case NProust::EDatasetType::MaxSumInt32: {
            auto datasetStreamer = NProust::TStreamingDatasetReader<int32_t, NProustExamples::TMaxSumState>(*datasetFn);
            if (withSentinel) {
                RunDatasetWithSettings<int32_t, NProustExamples::TMaxSumState, NProustExamples::TMaxSumStateOperation, INT32_SENTINEL>(
                    datasetStreamer,
                    *dataLayout,
                    *nodeTopology,
                    *leafSizeLog,
                    *baseGeneratorType
                );
            } else {
                RunDatasetWithSettings<int32_t, NProustExamples::TMaxSumState, NProustExamples::TMaxSumStateOperation>(
                    datasetStreamer,
                    *dataLayout,
                    *nodeTopology,
                    *leafSizeLog,
                    *baseGeneratorType
                );
            }
            break;
        }
        case NProust::EDatasetType::OrderStatistic16: {
            auto datasetStreamer = NProust::TStreamingDatasetReader<int32_t, std::vector<int32_t>>(*datasetFn);
            if (withSentinel) {
                spdlog::critical("Sentinel values are not supported for this type of the dataset");
                std::terminate();
            } else {
                RunDatasetWithSettings<int32_t, std::vector<int32_t>, NProustExamples::TInverseOrderStatisticOperation<16>>(
                    datasetStreamer,
                    *dataLayout,
                    *nodeTopology,
                    *leafSizeLog,
                    *baseGeneratorType
                );
            }
            break;
        }
    }

    return 0;
}
