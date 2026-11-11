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

template<typename TValue, typename TOperationResult, typename TOperation>
void TestFatEngineExhaustively(std::function<TValue(std::mt19937&)> valueGenerator) {
    // The topology and layout aren't used in the stress test for point-update engine,
    // so we can specify anything.
    using TDefaultTopology = NProust::TPreorderNodeTopology<0>;
    using TDefaultLayout = NProust::TSoALayout<TValue, TOperationResult, std::nullopt>;

    // Test small lengths to catch the bugs around the corner cases
    // Test larger length to ensure larger sizes are processed correctly
    std::vector<size_t> testedLengths;
    for (size_t length = 1; length <= 100; ++length) {
        testedLengths.push_back(length);
    }
    testedLengths.push_back(1000);

    for (size_t length : testedLengths) {
        StressTest<TValue, TOperationResult, TOperation, TDefaultTopology, TDefaultLayout, 0, 1>(
            length,
            N_QUERIES_IN_STRESS_TEST,
            valueGenerator,
            ETestedEngineType::Fat
        );

        StressTest<TValue, TOperationResult, TOperation, TDefaultTopology, TDefaultLayout, 0, 2>(
            length,
            N_QUERIES_IN_STRESS_TEST,
            valueGenerator,
            ETestedEngineType::Fat
        );

        StressTest<TValue, TOperationResult, TOperation, TDefaultTopology, TDefaultLayout, 0, 3>(
            length,
            N_QUERIES_IN_STRESS_TEST,
            valueGenerator,
            ETestedEngineType::Fat
        );

        StressTest<TValue, TOperationResult, TOperation, TDefaultTopology, TDefaultLayout, 0, 4>(
            length,
            N_QUERIES_IN_STRESS_TEST,
            valueGenerator,
            ETestedEngineType::Fat
        );

        // For smaller ranges, it's already greater than the full range.
        StressTest<TValue, TOperationResult, TOperation, TDefaultTopology, TDefaultLayout, 0, 5>(
            length,
            N_QUERIES_IN_STRESS_TEST,
            valueGenerator,
            ETestedEngineType::Fat
        );
    }
}

TEST(FatEngine, IntSum) {
    TestFatEngineExhaustively<int64_t, int64_t, NProust::TSumOperation>(CreateRandomIntGenerator(-100, 100));
}

TEST(FatEngine, BinaryXor) {
    TestFatEngineExhaustively<int64_t, int64_t, NProust::TBitwiseXorOperation>(CreateRandomIntGenerator(0, 65536));
}

TEST(FatEngine, AsymmetricIntSumWithUndo) {
    TestFatEngineExhaustively<TAsymmetricIntWrapper, int64_t, TAsymmetricIntWrapperOperationWithUndo>(CreateAsymmetricIntGenerator(-1000, 1000));
}

TEST(FatEngine, AsymmetricIntSumWithFromMany) {
    TestFatEngineExhaustively<TAsymmetricIntWrapper, int64_t, TAsymmetricIntWrapperOperationWithUndoAndFromMany>(CreateAsymmetricIntGenerator(-1000, 1000));
}

// Composition of affine maps is a group but not commutative: an update has to be conjugated by the
// prefix in front of it on every level.
TEST(FatEngine, NonCommutativeAffineComposition) {
    TestFatEngineExhaustively<TAffineMap, TAffineMap, TAffineCompositionOperation>(CreateAffineMapGenerator());
}

// The control: the same maps through the commutative fast path give wrong answers, so the test above
// does exercise what the conjugation is for.
TEST(FatEngine, AffineCompositionDeclaredCommutativeIsWrong) {
    using TOperation = TAffineCompositionOperationDeclaredCommutative;
    std::mt19937 generator(7);
    const auto valueGenerator = CreateAffineMapGenerator();
    std::vector<TAffineMap> values(200);
    for (auto& value : values) {
        value = valueGenerator(generator);
    }
    NProust::TFatPointUpdateEngine<TAffineMap, TAffineMap, TOperation, 2> engine;
    engine.InitFrom(values);
    size_t mismatches = 0;
    for (size_t step = 0; step < 50; ++step) {
        const size_t position = generator() % values.size();
        values[position] = valueGenerator(generator);
        engine.SetPoint(position, values[position]);
        auto expected = TOperation::FromRepeated<TAffineMap, TAffineMap>(values[0], 1);
        for (size_t index = 1; index < values.size(); ++index) {
            expected = TOperation::Merge(expected, TOperation::FromRepeated<TAffineMap, TAffineMap>(values[index], 1));
        }
        mismatches += !(engine.ComputeRange(0, values.size()) == expected);
    }
    EXPECT_GT(mismatches, 0u);
}
