#include <chrono>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <random>
#include <string_view>
#include <span>
#include <limits>
#include <utility>
#include <vector>
#include <string>
#include <cstring>
#include <thread>

#include "common/dataset.h"
#include "engine/robust/engine.h"
#include "engine/wide/engine.h"
#include "engine/external/fenwick.h"
#include "engine/external/algorithmica.h"
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
    #include <cpuid.h>
#endif

constexpr int32_t INT32_SENTINEL = std::numeric_limits<int32_t>::max();

// Select an external baseline (see include/engine/external/) instead of the generic robust
// engine. Both are valid only for the SumInt32 point-modification problem.
constexpr std::string_view FENWICK_IMPLEMENTATION_NAME = "fenwick";
constexpr std::string_view ALGORITHMICA_IMPLEMENTATION_NAME = "algorithmica";
constexpr std::string_view WIDE_IMPLEMENTATION_NAME = "wide";

// Two ways to time the replay.
//   PerOperation brackets every operation in RDTSC fences, giving per-operation TSC histograms —
//     the percentiles and the modification-vs-query split. The fence (LFENCE + RDTSCP) is a
//     serializing barrier, so it adds a fixed overhead and prevents overlap between operations.
//     Right when operations are individually expensive (the overhead is then negligible) or when
//     the per-operation distribution is what you care about.
//   Average drops the per-operation instrumentation entirely and only times whole batches with the
//     wall clock, dividing the total by the operation count. No fences, no per-op overhead, so it
//     measures genuine throughput — but it cannot produce the TSC split. Right when operations are
//     cheap enough that the fence overhead would dominate (e.g. against SOTA prefix-sum structures).
enum class EMeasurementMode {
    PerOperation,
    Average,
};

constexpr std::string_view PER_OPERATION_MODE_NAME = "per-operation";
constexpr std::string_view AVERAGE_MODE_NAME = "average";

std::string_view MeasurementModeName(const EMeasurementMode mode) {
    switch (mode) {
        case EMeasurementMode::PerOperation: return PER_OPERATION_MODE_NAME;
        case EMeasurementMode::Average: return AVERAGE_MODE_NAME;
    }
    return "unknown";
}

std::optional<EMeasurementMode> MeasurementModeFromString(const std::string_view s) {
    if (s == PER_OPERATION_MODE_NAME) {
        return EMeasurementMode::PerOperation;
    }
    if (s == AVERAGE_MODE_NAME) {
        return EMeasurementMode::Average;
    }
    return std::nullopt;
}

const double WARMUP_DATASET_SHARE = 0.05;
const size_t QUERY_BATCH_SIZE = 100;
const size_t MAX_CYCLES_PER_OPERATION = 100000;

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

constexpr std::string_view DEFAULT_CSV_FILE = "benchmark_results.csv";

// Everything needed to describe a single run when it is appended to the CSV log.
// The metrics themselves are filled in once the run has finished; the rest is
// known up front, from the command line. A row is written only when Label is set.
struct TCsvContext {
    std::optional<std::string> Label;
    std::string CsvFile = std::string(DEFAULT_CSV_FILE);
    std::string CpuModel = "unknown";
    std::string Implementation = "robust";
    std::string DatasetType;
    std::string GeneratorType;
    std::string DataLayout;
    std::string NodeTopology;
    size_t LeafSizeLog = 0;
    bool WithSentinel = false;
    EMeasurementMode MeasurementMode = EMeasurementMode::PerOperation;
};

struct TRunMetrics {
    int64_t AvgNsPerQuery = 0;
    int64_t PeakMemoryKb = 0;
    std::vector<uint32_t> WallPercentiles;     // wall-clock latency, nanoseconds
    std::vector<uint32_t> ModifyPercentiles;   // modification latency, TSC cycles
    std::vector<uint32_t> QueryPercentiles;    // query latency, TSC cycles
};

std::string_view DatasetTypeName(const NProust::EDatasetType type) {
    switch (type) {
        case NProust::EDatasetType::SumInt32: return "SumInt32";
        case NProust::EDatasetType::MaxInt32: return "MaxInt32";
        case NProust::EDatasetType::MaxSumInt32: return "MaxSumInt32";
        case NProust::EDatasetType::OrderStatistic16: return "OrderStatistic16";
        case NProust::EDatasetType::SumDouble: return "SumDouble";
        case NProust::EDatasetType::MaxDouble: return "MaxDouble";
        case NProust::EDatasetType::XorInt32: return "XorInt32";
        case NProust::EDatasetType::GcdInt32: return "GcdInt32";
        case NProust::EDatasetType::AndInt32: return "AndInt32";
        case NProust::EDatasetType::OrInt32: return "OrInt32";
        case NProust::EDatasetType::SumInt16: return "SumInt16";
        case NProust::EDatasetType::SumInt64: return "SumInt64";
        case NProust::EDatasetType::XorInt16: return "XorInt16";
        case NProust::EDatasetType::XorInt64: return "XorInt64";
    }
    return "unknown";
}

std::string_view BaseGeneratorTypeName(const NProust::EBaseGeneratorType type) {
    switch (type) {
        case NProust::EBaseGeneratorType::Point: return "Point";
        case NProust::EBaseGeneratorType::Range: return "Range";
    }
    return "unknown";
}

std::string_view DataLayoutName(const NProust::EDataLayout layout) {
    switch (layout) {
        case NProust::EDataLayout::AoS: return NProust::AOS_LAYOUT_NAME;
        case NProust::EDataLayout::SoA: return NProust::SOA_LAYOUT_NAME;
    }
    return "unknown";
}

std::string_view NodeTopologyName(const NProust::ENodeTopology topology) {
    switch (topology) {
        case NProust::ENodeTopology::Preorder: return NProust::PREORDER_LAYOUT_NAME;
        case NProust::ENodeTopology::BinaryHeap: return NProust::BINARY_HEAP_LAYOUT_NAME;
    }
    return "unknown";
}

std::string CsvEscape(const std::string& field) {
    if (field.find_first_of(",\"\n") == std::string::npos) {
        return field;
    }
    std::string escaped = "\"";
    for (const char c : field) {
        if (c == '"') {
            escaped += "\"\"";
        } else {
            escaped += c;
        }
    }
    escaped += '"';
    return escaped;
}

// Appends a single run as one CSV row, writing a header first if the file is new
// or empty. Columns are kept as (name, value) pairs so the header and the row can
// never drift apart.
void AppendCsvResult(const TCsvContext& context, const size_t length, const size_t queryCount, const TRunMetrics& metrics) {
    namespace fs = std::filesystem;
    const bool needHeader = !fs::exists(context.CsvFile) || fs::file_size(context.CsvFile) == 0;

    std::vector<std::pair<std::string, std::string>> columns;
    columns.emplace_back("label", context.Label.value_or(""));
    columns.emplace_back("cpu_model", context.CpuModel);
    columns.emplace_back("measurement_mode", std::string(MeasurementModeName(context.MeasurementMode)));
    columns.emplace_back("implementation", context.Implementation);
    columns.emplace_back("dataset_type", context.DatasetType);
    columns.emplace_back("generator_type", context.GeneratorType);
    columns.emplace_back("data_layout", context.DataLayout);
    columns.emplace_back("node_topology", context.NodeTopology);
    columns.emplace_back("leaf_size_log", std::to_string(context.LeafSizeLog));
    columns.emplace_back("with_sentinel", context.WithSentinel ? "true" : "false");
    columns.emplace_back("length", std::to_string(length));
    columns.emplace_back("query_count", std::to_string(queryCount));
    columns.emplace_back("avg_ns_per_query", std::to_string(metrics.AvgNsPerQuery));
    columns.emplace_back("peak_memory_kb", std::to_string(metrics.PeakMemoryKb));
    for (size_t i = 0; i < TARGET_PERCENTILES.size(); ++i) {
        columns.emplace_back("wall_ns_p" + TARGET_PERCENTILES[i].Name, std::to_string(metrics.WallPercentiles[i]));
    }
    for (size_t i = 0; i < TARGET_PERCENTILES.size(); ++i) {
        columns.emplace_back("modify_tsc_p" + TARGET_PERCENTILES[i].Name, std::to_string(metrics.ModifyPercentiles[i]));
    }
    for (size_t i = 0; i < TARGET_PERCENTILES.size(); ++i) {
        columns.emplace_back("query_tsc_p" + TARGET_PERCENTILES[i].Name, std::to_string(metrics.QueryPercentiles[i]));
    }

    std::ofstream out(context.CsvFile, std::ios::app);
    if (!out) {
        spdlog::critical("Failed to open CSV file for appending: {}", context.CsvFile);
        std::terminate();
    }
    if (needHeader) {
        for (size_t i = 0; i < columns.size(); ++i) {
            out << (i ? "," : "") << CsvEscape(columns[i].first);
        }
        out << '\n';
    }
    for (size_t i = 0; i < columns.size(); ++i) {
        out << (i ? "," : "") << CsvEscape(columns[i].second);
    }
    out << '\n';
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

// Reads the processor brand string via the CPUID extended leaves 0x80000002..0x80000004,
// the canonical way to obtain a human-readable CPU model on x86. Returns "unknown" when the
// leaves are unavailable (e.g. on a non-x86 host).
std::string GetCpuBrandString() {
    uint32_t regs[12] = {0};  // three leaves, four 32-bit registers each
#if defined(_MSC_VER)
    int cpuInfo[4];
    __cpuid(cpuInfo, 0x80000000);
    if (static_cast<uint32_t>(cpuInfo[0]) < 0x80000004) {
        return "unknown";
    }
    for (uint32_t leaf = 0; leaf < 3; ++leaf) {
        __cpuid(cpuInfo, 0x80000002 + leaf);
        for (uint32_t reg = 0; reg < 4; ++reg) {
            regs[leaf * 4 + reg] = static_cast<uint32_t>(cpuInfo[reg]);
        }
    }
#elif defined(__i386__) || defined(__x86_64__)
    uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid(0x80000000, &eax, &ebx, &ecx, &edx) == 0 || eax < 0x80000004) {
        return "unknown";
    }
    for (uint32_t leaf = 0; leaf < 3; ++leaf) {
        __get_cpuid(0x80000002 + leaf, &regs[leaf * 4], &regs[leaf * 4 + 1], &regs[leaf * 4 + 2], &regs[leaf * 4 + 3]);
    }
#else
    return "unknown";
#endif

    char brand[49];
    std::memcpy(brand, regs, sizeof(regs));
    brand[sizeof(regs)] = '\0';

    // The brand string is space-padded; trim the surrounding whitespace for cleaner logs.
    std::string result(brand);
    const auto first = result.find_first_not_of(" \t");
    const auto last = result.find_last_not_of(" \t");
    if (first == std::string::npos) {
        return "unknown";
    }
    return result.substr(first, last - first + 1);
}

// Logs the CPU model and a few basic facts about the host so that a benchmark log is
// self-contained: the numbers below only make sense against the processor that produced them.
void LogProcessorInfo(int32_t pinnedCore) {
    spdlog::info("CPU model: {}", GetCpuBrandString());
    spdlog::info("Logical cores: {}", std::thread::hardware_concurrency());
    spdlog::info("Execution pinned to core: {}", pinnedCore);
}

template<NProust::EBaseGeneratorType Kind, class TEngine, class TValue, class TOperationResult>
void InitAndRunLoop(
    std::unique_ptr<TEngine> engine,
    NProust::TStreamingDatasetReader<TValue, TOperationResult>& streamer,
    const TCsvContext& csvContext,
    std::chrono::high_resolution_clock::time_point initStartedAt
);

template<class TValue, class TOperationResult, class TOperation, auto Sentinel = std::nullopt>
void RunDatasetWithSettings(
    NProust::TStreamingDatasetReader<TValue, TOperationResult>& streamer,
    const NProust::EDataLayout layout,
    const NProust::ENodeTopology topology,
    const size_t leafSizeLog,
    const NProust::EBaseGeneratorType baseGeneratorType,
    const TCsvContext& csvContext
) {
    // The wide engine has no layout, topology or leaf-size knobs: they are ignored for it.
    if (csvContext.Implementation == WIDE_IMPLEMENTATION_NAME && baseGeneratorType == NProust::EBaseGeneratorType::Range) {
        const auto initStartedAt = std::chrono::high_resolution_clock::now();
        spdlog::info("Using the wide range-update engine");
        InitAndRunLoop<NProust::EBaseGeneratorType::Range>(std::make_unique<NProust::TWideRangeUpdateEngine<TValue, TOperationResult, TOperation>>(), streamer, csvContext, initStartedAt);
        return;
    }
    switch (layout) {
        case NProust::EDataLayout::AoS: {
            DispatchTopologyAndRun<TValue, TOperationResult, TOperation, NProust::TAoSLayout<TValue, TOperationResult, Sentinel>>(streamer, topology, leafSizeLog, baseGeneratorType, csvContext);
            break;
        }
        case NProust::EDataLayout::SoA: {
            DispatchTopologyAndRun<TValue, TOperationResult, TOperation, NProust::TSoALayout<TValue, TOperationResult, Sentinel>>(streamer, topology, leafSizeLog, baseGeneratorType, csvContext);
            break;
        }
    }
}

template<class TValue, class TOperationResult, class TOperation, NProust::DataLayout<TValue, TOperationResult> TDataLayout>
void DispatchTopologyAndRun(
    NProust::TStreamingDatasetReader<TValue, TOperationResult>& streamer,
    NProust::ENodeTopology topology,
    size_t leafSizeLog,
    const NProust::EBaseGeneratorType baseGeneratorType,
    const TCsvContext& csvContext
) {
    switch (leafSizeLog) {
        case 0: RunWithDispatchedTopology<TValue, TOperationResult, TOperation, TDataLayout, 0>(streamer, topology, baseGeneratorType, csvContext); break;
        case 1: RunWithDispatchedTopology<TValue, TOperationResult, TOperation, TDataLayout, 1>(streamer, topology, baseGeneratorType, csvContext); break;
        case 2: RunWithDispatchedTopology<TValue, TOperationResult, TOperation, TDataLayout, 2>(streamer, topology, baseGeneratorType, csvContext); break;
        case 3: RunWithDispatchedTopology<TValue, TOperationResult, TOperation, TDataLayout, 3>(streamer, topology, baseGeneratorType, csvContext); break;
        case 4: RunWithDispatchedTopology<TValue, TOperationResult, TOperation, TDataLayout, 4>(streamer, topology, baseGeneratorType, csvContext); break;
        case 5: RunWithDispatchedTopology<TValue, TOperationResult, TOperation, TDataLayout, 5>(streamer, topology, baseGeneratorType, csvContext); break;
        case 6: RunWithDispatchedTopology<TValue, TOperationResult, TOperation, TDataLayout, 6>(streamer, topology, baseGeneratorType, csvContext); break;
        case 7: RunWithDispatchedTopology<TValue, TOperationResult, TOperation, TDataLayout, 7>(streamer, topology, baseGeneratorType, csvContext); break;
        case 8: RunWithDispatchedTopology<TValue, TOperationResult, TOperation, TDataLayout, 8>(streamer, topology, baseGeneratorType, csvContext); break;
        case 9: RunWithDispatchedTopology<TValue, TOperationResult, TOperation, TDataLayout, 9>(streamer, topology, baseGeneratorType, csvContext); break;
        case 10: RunWithDispatchedTopology<TValue, TOperationResult, TOperation, TDataLayout, 10>(streamer, topology, baseGeneratorType, csvContext); break;
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
    const NProust::EBaseGeneratorType baseGeneratorType,
    const TCsvContext& csvContext
) {
    switch (topology) {
        case NProust::ENodeTopology::BinaryHeap: {
            RunDataset<TValue, TOperationResult, TOperation, TDataLayout, NProust::TBinaryHeapNodeTopology<LeafSizeLog>, LeafSizeLog>(streamer, baseGeneratorType, csvContext);
            break;
        }
        case NProust::ENodeTopology::Preorder: {
            RunDataset<TValue, TOperationResult, TOperation, TDataLayout, NProust::TPreorderNodeTopology<LeafSizeLog>, LeafSizeLog>(streamer, baseGeneratorType, csvContext);
            break;
        }
    }
}

// The timed replay loop, templated on the *concrete* engine type so SetPoint / SetRange /
// ComputeRange are direct, inlinable calls rather than virtual dispatch through
// IPointUpdateEngine / IRangeUpdateEngine. Kind selects the modification method at compile time;
// Mode selects per-operation RDTSC instrumentation versus batch-only wall-clock timing — both
// resolved with `if constexpr`, so the unused path leaves no trace in the hot loop.
template<NProust::EBaseGeneratorType Kind, EMeasurementMode Mode, class TEngine, class TValue, class TOperationResult>
void RunBenchmarkLoop(
    NProust::TStreamingDatasetReader<TValue, TOperationResult>& streamer,
    TEngine& engine,
    const TCsvContext& csvContext
) {
    constexpr bool PerOp = (Mode == EMeasurementMode::PerOperation);
    std::vector<TWeightedDurations> weightedDurations;
    std::vector<size_t> cyclesSetValue(MAX_CYCLES_PER_OPERATION, 0);
    std::vector<size_t> cyclesComputeRange(MAX_CYCLES_PER_OPERATION, 0);

    size_t readCount = 0;
    std::chrono::nanoseconds runTimeElapsed(0);
    bool isDatasetExhausted = false;
    const auto runStartedAt = std::chrono::high_resolution_clock::now();
    const auto warmupUntilEventId = (size_t)(WARMUP_DATASET_SHARE * streamer.GetQueryCount());
    std::vector<NProust::TTestPlanItem<TValue, TOperationResult>> datasetChunk;
    datasetChunk.reserve(QUERY_BATCH_SIZE);
    while (!isDatasetExhausted) {
        datasetChunk.clear();
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

        // Allocate some of the variables beforehand, so that their repeated allocation in the cycle doesn't
        // affect the batch timing.
        TOperationResult result;
        uint64_t startCycle = 0;
        uint64_t endCycle = 0;
        uint64_t totalCycles = 0;

        // We write into preallocated buffers because it's faster than random jumps over the histogram vectors
        // Therefore we remove a bit of non-related noise from the measurement
        // The data from the buffers is added into histograms outside of the timed section.
        // In Average mode the per-operation cycle buffers are never touched.
        std::vector<uint64_t> batchSetValueCycles;
        std::vector<uint64_t> batchComputeValueCycles;
        if constexpr (PerOp) {
            batchSetValueCycles.reserve(datasetChunk.size());
            batchComputeValueCycles.reserve(datasetChunk.size());
        }

        const auto batchRunStartedAt = std::chrono::high_resolution_clock::now();
        for (const auto& planItem : datasetChunk) {
            switch (planItem.Operation) {
                case NProust::EEngineOperation::SetRange: {
                    // The RDTSC fences exist only in PerOp mode; in Average mode the modification
                    // runs bare, so adjacent operations can overlap.
                    if constexpr (PerOp) {
                        startCycle = StartRDTSC();
                    }
                    if constexpr (Kind == NProust::EBaseGeneratorType::Point) {
                        engine.SetPoint(planItem.Left, *planItem.Value);
                    } else {
                        engine.SetRange(planItem.Left, planItem.Right, *planItem.Value);
                    }
                    if constexpr (PerOp) {
                        endCycle = EndRDTSC();
                        batchSetValueCycles.push_back(endCycle - startCycle);
                    }
                    break;
                }
                case NProust::EEngineOperation::Query: {
                    if constexpr (PerOp) {
                        startCycle = StartRDTSC();
                    }
                    result = engine.ComputeRange(planItem.Left, planItem.Right);
                    if constexpr (PerOp) {
                        endCycle = EndRDTSC();
                        batchComputeValueCycles.push_back(endCycle - startCycle);
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
        const auto batchTimeElapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(batchRunFinishedAt - batchRunStartedAt);
        const auto avgPerQueryInBatch = batchTimeElapsed.count() / datasetChunk.size();

        if (warmupIsDone) {
            weightedDurations.emplace_back((uint32_t)datasetChunk.size(), (uint32_t)avgPerQueryInBatch);
            if constexpr (PerOp) {
                for (const auto element : batchSetValueCycles) {
                    ++cyclesSetValue[std::min(element, MAX_CYCLES_PER_OPERATION - 1)];
                }
                for (const auto element : batchComputeValueCycles) {
                    ++cyclesComputeRange[std::min(element, MAX_CYCLES_PER_OPERATION - 1)];
                }
            }
            runTimeElapsed += batchTimeElapsed;
        }
    }

    const auto durationPerQuery = runTimeElapsed.count() / (readCount - warmupUntilEventId);
    const auto runTimeElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(runTimeElapsed).count();
    spdlog::info("The run has finished successfully. {} queries processed in {} ms ({} ns on average per query)", readCount, runTimeElapsedMs, durationPerQuery);

    std::sort(weightedDurations.begin(), weightedDurations.end(), [](const auto& lhs, const auto& rhs) {return lhs.AvgDuration < rhs.AvgDuration;});

    // Gather all the reported numbers into a single struct, so the same values
    // feed both the human-readable log below and the optional CSV row.
    TRunMetrics metrics;
    metrics.AvgNsPerQuery = durationPerQuery;
    metrics.PeakMemoryKb = GetPeakMemoryUsageKB();
    for (const auto& percentile : TARGET_PERCENTILES) {
        metrics.WallPercentiles.push_back(CalculatePercentile(weightedDurations, percentile.Numerator, percentile.Denominator));
    }
    // The TSC split exists only in PerOp mode; Average mode never measured individual operations,
    // so those columns are reported as zero to keep the CSV schema and the percentile count stable.
    for (const auto& percentile : TARGET_PERCENTILES) {
        if constexpr (PerOp) {
            metrics.ModifyPercentiles.push_back(CalculatePercentile(cyclesSetValue, percentile.Numerator, percentile.Denominator));
            metrics.QueryPercentiles.push_back(CalculatePercentile(cyclesComputeRange, percentile.Numerator, percentile.Denominator));
        } else {
            metrics.ModifyPercentiles.push_back(0);
            metrics.QueryPercentiles.push_back(0);
        }
    }

    spdlog::info("Measurement mode: {}", MeasurementModeName(Mode));
    spdlog::info("Peak memory usage: {} Kb", metrics.PeakMemoryKb);
    for (size_t i = 0; i < TARGET_PERCENTILES.size(); ++i) {
        spdlog::info("Approximate percentile [{}]: {} ns", TARGET_PERCENTILES[i].Name, metrics.WallPercentiles[i]);
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
    //
    // Only PerOp mode measures this; Average mode deliberately trades the split away to avoid the
    // per-operation fence overhead, so there is nothing to print here.
    if constexpr (PerOp) {
        for (size_t i = 0; i < TARGET_PERCENTILES.size(); ++i) {
            spdlog::info("Modification percentile [{}]: {} TSC cycles", TARGET_PERCENTILES[i].Name, metrics.ModifyPercentiles[i]);
        }
        for (size_t i = 0; i < TARGET_PERCENTILES.size(); ++i) {
            spdlog::info("Query percentile [{}]: {} TSC cycles", TARGET_PERCENTILES[i].Name, metrics.QueryPercentiles[i]);
        }
    }

    // The CSV row is written only when a label was supplied; otherwise the run is
    // a one-off and we leave the log untouched.
    if (csvContext.Label.has_value()) {
        AppendCsvResult(csvContext, streamer.GetLength(), streamer.GetQueryCount(), metrics);
        spdlog::info("Appended run results to '{}' under label '{}'", csvContext.CsvFile, *csvContext.Label);
    }
}

// Initializes a freshly-built concrete engine, logs the init time, and runs the timed loop on it.
template<NProust::EBaseGeneratorType Kind, class TEngine, class TValue, class TOperationResult>
void InitAndRunLoop(
    std::unique_ptr<TEngine> engine,
    NProust::TStreamingDatasetReader<TValue, TOperationResult>& streamer,
    const TCsvContext& csvContext,
    std::chrono::high_resolution_clock::time_point initStartedAt
) {
    engine->Init(streamer.GetLength());
    const auto initDoneAt = std::chrono::high_resolution_clock::now();
    const auto initDuration = std::chrono::duration_cast<std::chrono::milliseconds>(initDoneAt - initStartedAt).count();
    spdlog::info("Structure initialization done in {} ms", initDuration);
    // Turn the runtime measurement mode into the compile-time template argument the loop wants.
    switch (csvContext.MeasurementMode) {
        case EMeasurementMode::PerOperation:
            RunBenchmarkLoop<Kind, EMeasurementMode::PerOperation>(streamer, *engine, csvContext);
            break;
        case EMeasurementMode::Average:
            RunBenchmarkLoop<Kind, EMeasurementMode::Average>(streamer, *engine, csvContext);
            break;
    }
}

// Picks the concrete engine for this run and hands it to the loop. The engine is held by its
// concrete type the whole way down — never as an IPointUpdateEngine / IRangeUpdateEngine pointer —
// so the hot-path calls in RunBenchmarkLoop devirtualize. The external baselines stay guarded to
// the SumInt32 point problem, exactly as before.
template<class TValue, class TOperationResult, class TOperation, NProust::DataLayout<TValue, TOperationResult> TDataLayout, NProust::NodeTopology TNodeTopology, size_t LeafSizeLog>
void RunDataset(NProust::TStreamingDatasetReader<TValue, TOperationResult>& streamer, NProust::EBaseGeneratorType baseGeneratorType, const TCsvContext& csvContext) {
    const auto initStartedAt = std::chrono::high_resolution_clock::now();
    switch (baseGeneratorType) {
        case (NProust::EBaseGeneratorType::Point): {
            if (csvContext.Implementation == FENWICK_IMPLEMENTATION_NAME || csvContext.Implementation == ALGORITHMICA_IMPLEMENTATION_NAME) {
                if constexpr (std::is_same_v<TValue, int32_t> && std::is_same_v<TOperationResult, int32_t> && std::is_same_v<TOperation, NProust::TSumOperation>) {
                    if (csvContext.Implementation == FENWICK_IMPLEMENTATION_NAME) {
                        spdlog::info("Using the external Fenwick point-sum engine");
                        InitAndRunLoop<NProust::EBaseGeneratorType::Point>(std::make_unique<NProust::TFenwickPointSumEngine<>>(), streamer, csvContext, initStartedAt);
                    } else {
                        spdlog::info("Using the external Algorithmica wide-segment-tree point-sum engine");
                        InitAndRunLoop<NProust::EBaseGeneratorType::Point>(std::make_unique<NProust::TAlgorithmicaWideSegmentSumEngine<>>(), streamer, csvContext, initStartedAt);
                    }
                } else if constexpr (std::is_same_v<TValue, int32_t> && std::is_same_v<TOperationResult, int32_t> && std::is_same_v<TOperation, NProust::TBitwiseXorOperation>) {
                    if (csvContext.Implementation == ALGORITHMICA_IMPLEMENTATION_NAME) {
                        spdlog::info("Using the external Algorithmica wide-segment-tree point-XOR engine");
                        InitAndRunLoop<NProust::EBaseGeneratorType::Point>(std::make_unique<NProust::TAlgorithmicaWideSegmentXorEngine<>>(), streamer, csvContext, initStartedAt);
                    } else {
                        spdlog::critical("The external Fenwick engine is specialized to the sum problem; XOR has only the Algorithmica baseline");
                        std::terminate();
                    }
                } else {
                    spdlog::critical("The external engines only support the SumInt32 / XorInt32 point problem");
                    std::terminate();
                }
            } else {
                spdlog::info("Using a point-update engine");
                if constexpr (NProust::HasUndo<TOperation, TOperationResult>) {
                    InitAndRunLoop<NProust::EBaseGeneratorType::Point>(std::make_unique<NProust::TFatPointUpdateEngine<TValue, TOperationResult, TOperation, 4>>(), streamer, csvContext, initStartedAt);
                } else {
                    InitAndRunLoop<NProust::EBaseGeneratorType::Point>(std::make_unique<NProust::TRobustPointUpdateEngine<TValue, TOperationResult, TOperation, LeafSizeLog>>(), streamer, csvContext, initStartedAt);
                }
            }
            break;
        }
        case (NProust::EBaseGeneratorType::Range): {
            spdlog::info("Using a range-update engine");
            InitAndRunLoop<NProust::EBaseGeneratorType::Range>(std::make_unique<NProust::TRobustRangeUpdateEngine<TValue, TOperationResult, TOperation, TNodeTopology, TDataLayout>>(), streamer, csvContext, initStartedAt);
            break;
        }
    }
}

int main(int argc, char* argv[]) {
    // The pin is needed for TSC usage that goes next, and also to reduce the number of
    // process migrations between cores, which would be harmful, for example, because of the
    // cold cache.
    constexpr int32_t PINNED_CORE = 0;
    PinExecutionThread(PINNED_CORE);

    // Record what hardware produced the numbers; latency/TSC figures are only meaningful
    // when interpreted against the specific processor that generated them.
    LogProcessorInfo(PINNED_CORE);

    static_assert(std::chrono::high_resolution_clock::period::den, "High resolution clock doesn't have a nanosecond precision");

    std::span<char*> args(argv, argc);

    std::optional<std::string> datasetFn;
    std::optional<NProust::EDatasetType> datasetType;
    std::optional<NProust::EDataLayout> dataLayout;
    std::optional<NProust::ENodeTopology> nodeTopology;
    std::optional<size_t> leafSizeLog;
    std::optional<NProust::EBaseGeneratorType> baseGeneratorType;
    bool withSentinel = false;
    std::optional<std::string> csvFn;
    std::optional<std::string> csvLabel;
    std::optional<std::string> implementation;
    std::optional<std::string> measurementMode;

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
        } else if (arg == CSV_FILE_PARAM) {
            ParseStringArgument(it, args.end(), csvFn);
        } else if (arg == CSV_LABEL_PARAM) {
            ParseStringArgument(it, args.end(), csvLabel);
        } else if (arg == IMPLEMENTATION_PARAM) {
            ParseStringArgument(it, args.end(), implementation);
        } else if (arg == MEASUREMENT_MODE_PARAM) {
            ParseStringArgument(it, args.end(), measurementMode);
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

    // Describe this run for the CSV log. The metrics are filled in later, once the
    // run has finished; everything else is known from the command line right now.
    TCsvContext csvContext;
    csvContext.Label = csvLabel;
    csvContext.CsvFile = csvFn.value_or(std::string(DEFAULT_CSV_FILE));
    csvContext.CpuModel = GetCpuBrandString();
    csvContext.Implementation = implementation.value_or("robust");
    csvContext.DatasetType = std::string(DatasetTypeName(*datasetType));
    csvContext.GeneratorType = std::string(BaseGeneratorTypeName(*baseGeneratorType));
    csvContext.DataLayout = std::string(DataLayoutName(*dataLayout));
    csvContext.NodeTopology = std::string(NodeTopologyName(*nodeTopology));
    csvContext.LeafSizeLog = *leafSizeLog;
    csvContext.WithSentinel = withSentinel;
    if (measurementMode.has_value()) {
        const auto parsedMode = MeasurementModeFromString(*measurementMode);
        if (!parsedMode.has_value()) {
            spdlog::critical("Unknown measurement mode: {}. Expected '{}' or '{}'.", *measurementMode, PER_OPERATION_MODE_NAME, AVERAGE_MODE_NAME);
            std::terminate();
        }
        csvContext.MeasurementMode = *parsedMode;
    }

    switch (*datasetType) {
        case NProust::EDatasetType::SumInt32: {
            auto datasetStreamer = NProust::TStreamingDatasetReader<int32_t, int32_t>(*datasetFn);
            if (withSentinel) {
                RunDatasetWithSettings<int32_t, int32_t, NProust::TSumOperation, INT32_SENTINEL>(datasetStreamer, *dataLayout, *nodeTopology, *leafSizeLog, *baseGeneratorType, csvContext);
            } else {
                RunDatasetWithSettings<int32_t, int32_t, NProust::TSumOperation>(datasetStreamer, *dataLayout, *nodeTopology, *leafSizeLog, *baseGeneratorType, csvContext);
            }
            break;
        }
        case NProust::EDatasetType::MaxInt32: {
            auto datasetStreamer = NProust::TStreamingDatasetReader<int32_t, int32_t>(*datasetFn);
            if (withSentinel) {
                spdlog::critical("Sentinel values are not supported for this type of the dataset");
                std::terminate();
            } else {
                RunDatasetWithSettings<int32_t, int32_t, NProust::TMaxOperation>(datasetStreamer, *dataLayout, *nodeTopology, *leafSizeLog, *baseGeneratorType, csvContext);
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
                    *baseGeneratorType,
                    csvContext
                );
            } else {
                RunDatasetWithSettings<int32_t, NProustExamples::TMaxSumState, NProustExamples::TMaxSumStateOperation>(
                    datasetStreamer,
                    *dataLayout,
                    *nodeTopology,
                    *leafSizeLog,
                    *baseGeneratorType,
                    csvContext
                );
            }
            break;
        }
        case NProust::EDatasetType::OrderStatistic16: {
            auto datasetStreamer = NProust::TStreamingDatasetReader<int32_t, NProustExamples::TTopKState<16>>(*datasetFn);
            if (withSentinel) {
                spdlog::critical("Sentinel values are not supported for this type of the dataset");
                std::terminate();
            } else {
                RunDatasetWithSettings<int32_t, NProustExamples::TTopKState<16>, NProustExamples::TInverseOrderStatisticOperation<16>>(
                    datasetStreamer,
                    *dataLayout,
                    *nodeTopology,
                    *leafSizeLog,
                    *baseGeneratorType,
                    csvContext
                );
            }
            break;
        }
        case NProust::EDatasetType::SumDouble: {
            auto datasetStreamer = NProust::TStreamingDatasetReader<double, double>(*datasetFn);
            if (withSentinel) {
                spdlog::critical("Sentinel values are not supported for this type of the dataset");
                std::terminate();
            } else {
                RunDatasetWithSettings<double, double, NProust::TSumOperation>(datasetStreamer, *dataLayout, *nodeTopology, *leafSizeLog, *baseGeneratorType, csvContext);
            }
            break;
        }
        case NProust::EDatasetType::MaxDouble: {
            auto datasetStreamer = NProust::TStreamingDatasetReader<double, double>(*datasetFn);
            if (withSentinel) {
                spdlog::critical("Sentinel values are not supported for this type of the dataset");
                std::terminate();
            } else {
                RunDatasetWithSettings<double, double, NProust::TMaxOperation>(datasetStreamer, *dataLayout, *nodeTopology, *leafSizeLog, *baseGeneratorType, csvContext);
            }
            break;
        }
        case NProust::EDatasetType::XorInt32: {
            auto datasetStreamer = NProust::TStreamingDatasetReader<int32_t, int32_t>(*datasetFn);
            if (withSentinel) {
                spdlog::critical("Sentinel values are not supported for this type of the dataset");
                std::terminate();
            } else {
                RunDatasetWithSettings<int32_t, int32_t, NProust::TBitwiseXorOperation>(datasetStreamer, *dataLayout, *nodeTopology, *leafSizeLog, *baseGeneratorType, csvContext);
            }
            break;
        }
        case NProust::EDatasetType::GcdInt32: {
            auto datasetStreamer = NProust::TStreamingDatasetReader<int32_t, int32_t>(*datasetFn);
            if (withSentinel) {
                spdlog::critical("Sentinel values are not supported for this type of the dataset");
                std::terminate();
            } else {
                RunDatasetWithSettings<int32_t, int32_t, NProust::TGcdOperation>(datasetStreamer, *dataLayout, *nodeTopology, *leafSizeLog, *baseGeneratorType, csvContext);
            }
            break;
        }
        case NProust::EDatasetType::AndInt32: {
            auto datasetStreamer = NProust::TStreamingDatasetReader<int32_t, int32_t>(*datasetFn);
            if (withSentinel) {
                spdlog::critical("Sentinel values are not supported for this type of the dataset");
                std::terminate();
            } else {
                RunDatasetWithSettings<int32_t, int32_t, NProust::TBitwiseAndOperation>(datasetStreamer, *dataLayout, *nodeTopology, *leafSizeLog, *baseGeneratorType, csvContext);
            }
            break;
        }
        case NProust::EDatasetType::OrInt32: {
            auto datasetStreamer = NProust::TStreamingDatasetReader<int32_t, int32_t>(*datasetFn);
            if (withSentinel) {
                spdlog::critical("Sentinel values are not supported for this type of the dataset");
                std::terminate();
            } else {
                RunDatasetWithSettings<int32_t, int32_t, NProust::TBitwiseOrOperation>(datasetStreamer, *dataLayout, *nodeTopology, *leafSizeLog, *baseGeneratorType, csvContext);
            }
            break;
        }
        case NProust::EDatasetType::SumInt16: {
            auto datasetStreamer = NProust::TStreamingDatasetReader<int16_t, int16_t>(*datasetFn);
            RunDatasetWithSettings<int16_t, int16_t, NProust::TSumOperation>(datasetStreamer, *dataLayout, *nodeTopology, *leafSizeLog, *baseGeneratorType, csvContext);
            break;
        }
        case NProust::EDatasetType::SumInt64: {
            auto datasetStreamer = NProust::TStreamingDatasetReader<int64_t, int64_t>(*datasetFn);
            RunDatasetWithSettings<int64_t, int64_t, NProust::TSumOperation>(datasetStreamer, *dataLayout, *nodeTopology, *leafSizeLog, *baseGeneratorType, csvContext);
            break;
        }
        case NProust::EDatasetType::XorInt16: {
            auto datasetStreamer = NProust::TStreamingDatasetReader<int16_t, int16_t>(*datasetFn);
            RunDatasetWithSettings<int16_t, int16_t, NProust::TBitwiseXorOperation>(datasetStreamer, *dataLayout, *nodeTopology, *leafSizeLog, *baseGeneratorType, csvContext);
            break;
        }
        case NProust::EDatasetType::XorInt64: {
            auto datasetStreamer = NProust::TStreamingDatasetReader<int64_t, int64_t>(*datasetFn);
            RunDatasetWithSettings<int64_t, int64_t, NProust::TBitwiseXorOperation>(datasetStreamer, *dataLayout, *nodeTopology, *leafSizeLog, *baseGeneratorType, csvContext);
            break;
        }
    }

    return 0;
}
