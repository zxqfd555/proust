#pragma once

#include <chrono>
#include <iostream>
#include <random>

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include "common/dataset.h"
#include "engine/segtree/segtree.h"
#include "engine/operations.h"

template<typename TValue, typename TOperationResult, typename TOperation, NProust::NodeTopology TNodeTopology = NProust::TPreorderNodeTopology<0>>
void StressTest(
    const size_t length,
    const size_t queryCount,
    std::function<TValue(std::mt19937&)> valueGenerator,
    std::unique_ptr<NProust::IEngine<TValue, TOperationResult, TOperation>> groundTruthEngine = nullptr
) {
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    NProust::TStreamingDatasetWriter<TValue, TOperationResult> writer(queryCount, length, stream);
    NProust::GenerateTestPlan<TValue, TOperationResult, TOperation>(length, queryCount, valueGenerator, writer, std::move(groundTruthEngine));

    stream.seekg(0);
    NProust::TStreamingDatasetReader<TValue, TOperationResult> reader(stream);

    const auto testStartedAt = std::chrono::high_resolution_clock::now();
    auto robustEngine = NProust::CreateRobustEngine<TValue, TOperationResult, TOperation, TNodeTopology>();
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
