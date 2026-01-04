#include <chrono>
#include <iostream>
#include <random>

#include <gtest/gtest.h>

#include "common/dataset.h"
#include "engine/robust/engine.h"
#include "engine/operations.h"
#include "examples/leetcode_house_robber.h"
#include "examples/leetcode_maximum_subarray.h"
#include "examples/leetcode_third_distinct_maximum.h"

#include "utils.h"

using NProust::CreateRandomIntGenerator;

constexpr size_t N_QUERIES_IN_STRESS_TEST = 10000;

namespace {
    const bool spdlog_disabled = [] {
        spdlog::set_level(spdlog::level::warn);
        return true;
    }();
}

template<typename TValue, typename TOperationResult, typename TOperation>
using TNaiveEngineFactory = std::function<std::unique_ptr<NProust::IRangeUpdateEngine<TValue, TOperationResult, TOperation>>()>;

template<typename TValue, typename TOperationResult, typename TOperation>
std::unique_ptr<NProust::IRangeUpdateEngine<TValue, TOperationResult, TOperation>> DefaultNaiveEngineFactory() {
    return nullptr;
}

template<typename TValue, typename TOperationResult, typename TOperation>
void TestPointUpdateEngineExhaustively(
    std::function<TValue(std::mt19937&)> valueGenerator,
    TNaiveEngineFactory<TValue, TOperationResult, TOperation> naiveEngineFactory = &DefaultNaiveEngineFactory<TValue, TOperationResult, TOperation>
) {
    // The topology and layout aren't used in the stress test for point-update engine,
    // so we can specify anything.
    using TDefaultTopology = NProust::TPreorderNodeTopology<0>;
    using TDefaultLayout = NProust::TSoALayout<TValue, TOperationResult, std::nullopt>;

    // Test small lengths to catch the bugs around the corner cases
    // Test larger length to ensure larger sizes are processed correctly
    std::vector<size_t> testedLengths;
    for (size_t length = 1; length <= 16; ++length) {
        testedLengths.push_back(length);
    }
    testedLengths.push_back(1000);

    for (size_t length : testedLengths) {
        StressTest<TValue, TOperationResult, TOperation, TDefaultTopology, TDefaultLayout, 0>(
            length,
            N_QUERIES_IN_STRESS_TEST,
            valueGenerator,
            ETestedEngineType::PointRobust,
            naiveEngineFactory()
        );

        StressTest<TValue, TOperationResult, TOperation, TDefaultTopology, TDefaultLayout, 1>(
            length,
            N_QUERIES_IN_STRESS_TEST,
            valueGenerator,
            ETestedEngineType::PointRobust,
            naiveEngineFactory()
        );

        StressTest<TValue, TOperationResult, TOperation, TDefaultTopology, TDefaultLayout, 2>(
            length,
            N_QUERIES_IN_STRESS_TEST,
            valueGenerator,
            ETestedEngineType::PointRobust,
            naiveEngineFactory()
        );

        // For smaller ranges, it's already greater than the full range.
        StressTest<TValue, TOperationResult, TOperation, TDefaultTopology, TDefaultLayout, 5>(
            length,
            N_QUERIES_IN_STRESS_TEST,
            valueGenerator,
            ETestedEngineType::PointRobust,
            naiveEngineFactory()
        );
    }
}

TEST(PointUpdateEngine, IntSum) {
    TestPointUpdateEngineExhaustively<int64_t, int64_t, NProust::TSumOperation>(CreateRandomIntGenerator(-100, 100));
}

TEST(PointUpdateEngine, IntBitwiseAnd) {
    TestPointUpdateEngineExhaustively<int64_t, int64_t, NProust::TBitwiseAndOperation>(CreateRandomIntGenerator(0, 65536));
}

TEST(PointUpdateEngine, IntBitwiseOr) {
    TestPointUpdateEngineExhaustively<int64_t, int64_t, NProust::TBitwiseOrOperation>(CreateRandomIntGenerator(0, 65535));
}

TEST(PointUpdateEngine, IntBitwiseXor) {
    TestPointUpdateEngineExhaustively<int64_t, int64_t, NProust::TBitwiseXorOperation>(CreateRandomIntGenerator(0, 65536));
}

TEST(PointUpdateEngine, IntMin) {
    TestPointUpdateEngineExhaustively<int64_t, int64_t, NProust::TMinOperation>(CreateRandomIntGenerator(-100, 100));
}

TEST(PointUpdateEngine, IntMax) {
    TestPointUpdateEngineExhaustively<int64_t, int64_t, NProust::TMaxOperation>(CreateRandomIntGenerator(-100, 100));
}

TEST(PointUpdateEngine, LeetcodeMaximumSubarrayExample) {
    TestPointUpdateEngineExhaustively<int32_t, NProustExamples::TMaxSumState, NProustExamples::TMaxSumStateOperation>(
        CreateRandomIntGenerator(-10000, 10000),
        []() { return NProustExamples::CreateNaiveMaxSumEngine(); }
    );
}

TEST(PointUpdateEngine, LeetcodeHouseRobberExample) {
    TestPointUpdateEngineExhaustively<int32_t, NProustExamples::THouseRobberDPState, NProustExamples::THouseRobberDPStateOperation>(
        CreateRandomIntGenerator(-10000, 10000),
        []() { return NProustExamples::CreateNaiveLeetcodeHouseRobberEngine(); }
    );
}

TEST(PointUpdateEngine, LeetcodeThirdDistinctMaximumExample) {
    TestPointUpdateEngineExhaustively<int32_t, NProustExamples::TInverseOrderStatisticState, NProustExamples::TInverseOrderStatisticOperation<3>>(
        CreateRandomIntGenerator(-1000000000, 1000000000),
        []() { return NProustExamples::CreateNaiveInverseOrderStatisticsEngine<3>(); }
    );
}
