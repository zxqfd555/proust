#include <chrono>
#include <iostream>
#include <random>

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include "common/dataset.h"
#include "engine/segtree/segtree.h"
#include "engine/operations.h"

using NProust::CreateRandomIntGenerator;

template<typename TValue, typename TOperationResult, typename TOperation>
void StressTest(const size_t length, const size_t queryCount, std::function<TValue(std::mt19937&)> valueGenerator) {
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    NProust::TStreamingDatasetWriter<TValue, TOperationResult> writer(queryCount, length, stream);
    NProust::GenerateTestPlan<TValue, TOperationResult, TOperation>(length, queryCount, valueGenerator, writer);

    stream.seekg(0);
    NProust::TStreamingDatasetReader<TValue, TOperationResult> reader(stream);

    const auto testStartedAt = std::chrono::high_resolution_clock::now();
    auto robustEngine = NProust::CreateRobustEngine<TValue, TOperationResult, TOperation>();
    robustEngine->Init(length);

    size_t queriesProcessed = 0;
    while (auto maybeItem = reader.NextQuery()) {
        if (!maybeItem.has_value()) {
            break;
        }
        ++queriesProcessed;
        const auto planItem = *maybeItem;
        switch (planItem.Operation) {
            case NProust::EEngineOperation::SetRange: {
                EXPECT_TRUE(planItem.Value.has_value());
                robustEngine->SetRange(planItem.Left, planItem.Right, *planItem.Value);
                break;
            }
            case NProust::EEngineOperation::Query: {
                const auto result = robustEngine->ComputeRange(planItem.Left, planItem.Right);
                EXPECT_EQ(result, *planItem.ExpectedResult);
                break;
            }
        }
    }

    EXPECT_EQ(queriesProcessed, queryCount);
    const auto testEndedAt = std::chrono::high_resolution_clock::now();
    const auto testDuration = std::chrono::duration_cast<std::chrono::milliseconds>(testEndedAt - testStartedAt);
    spdlog::info("Stress test passed. Time elapsed: {} ms", testDuration.count());
}

TEST(InMemoryStress, IntSmallRangeSum) {
    StressTest<int64_t, int64_t, NProust::TSumOperation>(16, 10000000, CreateRandomIntGenerator(-100, 100));
}

TEST(InMemoryStress, IntSmallRangeAnd) {
    StressTest<int64_t, int64_t, NProust::TBitwiseAndOperation>(16, 10000000, CreateRandomIntGenerator(0, 65535));
}

TEST(InMemoryStress, IntSmallRangeOr) {
    StressTest<int64_t, int64_t, NProust::TBitwiseOrOperation>(16, 10000000, CreateRandomIntGenerator(0, 65535));
}

TEST(InMemoryStress, IntSmallRangeXor) {
    StressTest<int64_t, int64_t, NProust::TBitwiseXorOperation>(16, 10000000, CreateRandomIntGenerator(0, 65535));
}

TEST(InMemoryStress, IntSmallRangeMin) {
    StressTest<int64_t, int64_t, NProust::TMinOperation>(16, 10000000, CreateRandomIntGenerator(-100, 100));
}

TEST(InMemoryStress, IntSmallRangeMax) {
    StressTest<int64_t, int64_t, NProust::TMaxOperation>(16, 10000000, CreateRandomIntGenerator(-100, 100));
}

TEST(InMemoryStress, IntRangeSum) {
    StressTest<int64_t, int64_t, NProust::TSumOperation>(1000, 1000000, CreateRandomIntGenerator(-100, 100));
}

TEST(InMemoryStress, IntRangeAnd) {
    StressTest<int64_t, int64_t, NProust::TBitwiseAndOperation>(1000, 1000000, CreateRandomIntGenerator(0, 65535));
}

TEST(InMemoryStress, IntRangeOr) {
    StressTest<int64_t, int64_t, NProust::TBitwiseOrOperation>(1000, 1000000, CreateRandomIntGenerator(0, 65535));
}

TEST(InMemoryStress, IntRangeXor) {
    StressTest<int64_t, int64_t, NProust::TBitwiseXorOperation>(1000, 1000000, CreateRandomIntGenerator(0, 65535));
}

TEST(InMemoryStress, IntRangeMin) {
    StressTest<int64_t, int64_t, NProust::TMinOperation>(1000, 1000000, CreateRandomIntGenerator(-100, 100));
}

TEST(InMemoryStress, IntRangeMax) {
    StressTest<int64_t, int64_t, NProust::TMaxOperation>(1000, 1000000, CreateRandomIntGenerator(-100, 100));
}

TEST(InMemoryStress, DifferentLengths) {
    // idempotent operation
    for (size_t length = 1; length <= 100; ++length) {
        StressTest<int64_t, int64_t, NProust::TMaxOperation>(length, 1000, CreateRandomIntGenerator(-100, 100));
    }

    // non-idempotent operation
    for (size_t length = 1; length <= 100; ++length) {
        StressTest<int64_t, int64_t, NProust::TSumOperation>(length, 1000, CreateRandomIntGenerator(-100, 100));
    }
}