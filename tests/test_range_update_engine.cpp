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

template<typename TValue, typename TOperationResult, typename TOperation, typename TNodeTopology>
void InvokeTestForDifferentLayouts(
    const size_t length,
    const size_t queryCount,
    std::function<TValue(std::mt19937&)> valueGenerator,
    TNaiveEngineFactory<TValue, TOperationResult, TOperation> naiveEngineFactory = &DefaultNaiveEngineFactory<TValue, TOperationResult, TOperation>
) {
    // Test both variants for SoA: with and without sentinel
    StressTest<TValue, TOperationResult, TOperation, TNodeTopology, NProust::TSoALayout<TValue, TOperationResult, std::nullopt>>(
        length,
        queryCount,
        valueGenerator,
        ETestedEngineType::RangeRobust,
        naiveEngineFactory()
    );

    // Test basic variant for AoS, without sentinel
    StressTest<TValue, TOperationResult, TOperation, TNodeTopology, NProust::TAoSLayout<TValue, TOperationResult, std::nullopt>>(
        length,
        queryCount,
        valueGenerator,
        ETestedEngineType::RangeRobust,
        naiveEngineFactory()
    );

    if constexpr (std::is_same_v<TValue, int32_t> || std::is_same_v<TValue, int64_t>) {
        StressTest<TValue, TOperationResult, TOperation, TNodeTopology, NProust::TSoALayout<TValue, TOperationResult, std::numeric_limits<TValue>::max()>>(
            length,
            queryCount,
            valueGenerator,
            ETestedEngineType::RangeRobust,
            naiveEngineFactory()
        );

        StressTest<TValue, TOperationResult, TOperation, TNodeTopology, NProust::TAoSLayout<TValue, TOperationResult, std::numeric_limits<TValue>::max()>>(
            length,
            queryCount,
            valueGenerator,
            ETestedEngineType::RangeRobust,
            naiveEngineFactory()
        );
    }
}

template<typename TValue, typename TOperationResult, typename TOperation>
void TestRangeUpdateEngineExhaustively(
    std::function<TValue(std::mt19937&)> valueGenerator,
    TNaiveEngineFactory<TValue, TOperationResult, TOperation> naiveEngineFactory = &DefaultNaiveEngineFactory<TValue, TOperationResult, TOperation>
) {
    // Test small lengths to catch the bugs around the corner cases
    // Test larger length to ensure larger sizes are processed correctly
    std::vector<size_t> testedLengths;
    for (size_t length = 1; length <= 16; ++length) {
        testedLengths.push_back(length);
    }
    testedLengths.push_back(1000);

    for (size_t length : testedLengths) {
        // Preorder topology cases
        {
            InvokeTestForDifferentLayouts<TValue, TOperationResult, TOperation, NProust::TPreorderNodeTopology<0>>(
                length,
                N_QUERIES_IN_STRESS_TEST,
                valueGenerator,
                naiveEngineFactory
            );
            InvokeTestForDifferentLayouts<TValue, TOperationResult, TOperation, NProust::TPreorderNodeTopology<1>>(
                length,
                N_QUERIES_IN_STRESS_TEST,
                valueGenerator,
                naiveEngineFactory
            );
            InvokeTestForDifferentLayouts<TValue, TOperationResult, TOperation, NProust::TPreorderNodeTopology<2>>(
                length,
                N_QUERIES_IN_STRESS_TEST,
                valueGenerator,
                naiveEngineFactory
            );

            // The leaf size is greater than the structure size itself
            InvokeTestForDifferentLayouts<TValue, TOperationResult, TOperation, NProust::TPreorderNodeTopology<6>>(
                length,
                N_QUERIES_IN_STRESS_TEST,
                valueGenerator,
                naiveEngineFactory
            );
        }

        // Binary heap topology cases
        {
            InvokeTestForDifferentLayouts<TValue, TOperationResult, TOperation, NProust::TBinaryHeapNodeTopology<0>>(
                length,
                N_QUERIES_IN_STRESS_TEST,
                valueGenerator,
                naiveEngineFactory
            );
            InvokeTestForDifferentLayouts<TValue, TOperationResult, TOperation, NProust::TBinaryHeapNodeTopology<1>>(
                length,
                N_QUERIES_IN_STRESS_TEST,
                valueGenerator,
                naiveEngineFactory
            );
            InvokeTestForDifferentLayouts<TValue, TOperationResult, TOperation, NProust::TBinaryHeapNodeTopology<2>>(
                length,
                N_QUERIES_IN_STRESS_TEST,
                valueGenerator,
                naiveEngineFactory
            );

            // The leaf size is greater than the structure size itself
            InvokeTestForDifferentLayouts<TValue, TOperationResult, TOperation, NProust::TBinaryHeapNodeTopology<6>>(
                length,
                N_QUERIES_IN_STRESS_TEST,
                valueGenerator,
                naiveEngineFactory
            );
        }
    }
}

TEST(RangeUpdateEngine, IntSum) {
    TestRangeUpdateEngineExhaustively<int64_t, int64_t, NProust::TSumOperation>(CreateRandomIntGenerator(-100, 100));
}

TEST(RangeUpdateEngine, IntBitwiseAnd) {
    TestRangeUpdateEngineExhaustively<int64_t, int64_t, NProust::TBitwiseAndOperation>(CreateRandomIntGenerator(0, 65536));
}

TEST(RangeUpdateEngine, IntBitwiseOr) {
    TestRangeUpdateEngineExhaustively<int64_t, int64_t, NProust::TBitwiseOrOperation>(CreateRandomIntGenerator(0, 65535));
}

TEST(RangeUpdateEngine, IntBitwiseXor) {
    TestRangeUpdateEngineExhaustively<int64_t, int64_t, NProust::TBitwiseXorOperation>(CreateRandomIntGenerator(0, 65536));
}

TEST(RangeUpdateEngine, IntMin) {
    TestRangeUpdateEngineExhaustively<int64_t, int64_t, NProust::TMinOperation>(CreateRandomIntGenerator(-100, 100));
}

TEST(RangeUpdateEngine, IntMax) {
    TestRangeUpdateEngineExhaustively<int64_t, int64_t, NProust::TMaxOperation>(CreateRandomIntGenerator(-100, 100));
}

TEST(RangeUpdateEngine, LeetcodeMaximumSubarrayExample) {
    TestRangeUpdateEngineExhaustively<int32_t, NProustExamples::TMaxSumState, NProustExamples::TMaxSumStateOperation>(
        CreateRandomIntGenerator(-10000, 10000),
        []() { return NProustExamples::CreateNaiveMaxSumEngine(); }
    );
}

TEST(RangeUpdateEngine, LeetcodeHouseRobberExample) {
    TestRangeUpdateEngineExhaustively<int32_t, NProustExamples::THouseRobberDPState, NProustExamples::THouseRobberDPStateOperation>(
        CreateRandomIntGenerator(-10000, 10000),
        []() { return NProustExamples::CreateNaiveLeetcodeHouseRobberEngine(); }
    );
}

TEST(RangeUpdateEngine, LeetcodeThirdDistinctMaximumExample) {
    TestRangeUpdateEngineExhaustively<int32_t, NProustExamples::TTopKState<3>, NProustExamples::TInverseOrderStatisticOperation<3>>(
        CreateRandomIntGenerator(-1000000000, 1000000000),
        []() { return NProustExamples::CreateNaiveInverseOrderStatisticsEngine<3>(); }
    );
}

TEST(RangeUpdateEngine, AsymmetricIntSum) {
    TestRangeUpdateEngineExhaustively<TAsymmetricIntWrapper, int64_t, TAsymmetricIntWrapperOperation>(CreateAsymmetricIntGenerator(-1000, 1000));
}

TEST(RangeUpdateEngine, AsymmetricIntSumWithUndo) {
    TestRangeUpdateEngineExhaustively<TAsymmetricIntWrapper, int64_t, TAsymmetricIntWrapperOperationWithUndo>(CreateAsymmetricIntGenerator(-1000, 1000));
}

TEST(RangeUpdateEngine, AsymmetricIntSumWithFromMany) {
    TestRangeUpdateEngineExhaustively<TAsymmetricIntWrapper, int64_t, TAsymmetricIntWrapperOperationWithFromMany>(CreateAsymmetricIntGenerator(-1000, 1000));
}

TEST(RangeUpdateEngine, AsymmetricIntSumWithUndoAndFromMany) {
    TestRangeUpdateEngineExhaustively<TAsymmetricIntWrapper, int64_t, TAsymmetricIntWrapperOperationWithUndoAndFromMany>(CreateAsymmetricIntGenerator(-1000, 1000));
}
