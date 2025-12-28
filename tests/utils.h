#pragma once

#include <chrono>
#include <iostream>
#include <random>

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include "common/dataset.h"
#include "engine/robust/engine.h"
#include "engine/operations.h"

template<typename TValue, typename TOperationResult, typename TOperation, NProust::NodeTopology TNodeTopology = NProust::TPreorderNodeTopology<0>>
void StressTest(
    const size_t length,
    const size_t queryCount,
    std::function<TValue(std::mt19937&)> valueGenerator,
    const NProust::EBaseGeneratorType baseGeneratorType = NProust::EBaseGeneratorType::Range,
    std::unique_ptr<NProust::IRangeUpdateEngine<TValue, TOperationResult, TOperation>> groundTruthEngine = nullptr
) {
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    NProust::TStreamingDatasetWriter<TValue, TOperationResult> writer(queryCount, length, stream);

    std::unique_ptr<NProust::ITestPlanBaseGenerator> baseGenerator;
    std::unique_ptr<NProust::IRangeUpdateEngine<TValue, TOperationResult, TOperation>> rangeUpdateEngine;
    std::unique_ptr<NProust::IPointUpdateEngine<TValue, TOperationResult, TOperation>> pointUpdateEngine;

    switch (baseGeneratorType) {
        case (NProust::EBaseGeneratorType::Point): {
            baseGenerator = std::make_unique<NProust::TPointUpdatePlanGenerator>(length);
            pointUpdateEngine = NProust::CreateRobustPointUpdateEngine<TValue, TOperationResult, TOperation>();
            pointUpdateEngine->Init(length);
            break;
        }
        case (NProust::EBaseGeneratorType::Range): {
            baseGenerator = std::make_unique<NProust::TRangeUpdatePlanGenerator>(length);
            rangeUpdateEngine = NProust::CreateRobustRangeUpdateEngine<TValue, TOperationResult, TOperation, TNodeTopology>();
            rangeUpdateEngine->Init(length);
            break;
        }
    };
    assert(!!baseGenerator);
    NProust::GenerateTestPlan<TValue, TOperationResult, TOperation>(length, queryCount, *baseGenerator, valueGenerator, writer, std::move(groundTruthEngine));

    stream.seekg(0);
    NProust::TStreamingDatasetReader<TValue, TOperationResult> reader(stream);

    const auto testStartedAt = std::chrono::high_resolution_clock::now();

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
                switch (baseGeneratorType) {
                    case (NProust::EBaseGeneratorType::Point): {
                        EXPECT_EQ(planItem.Left + 1, planItem.Right);
                        pointUpdateEngine->SetPoint(planItem.Left, *planItem.Value);
                        break;
                    }
                    case (NProust::EBaseGeneratorType::Range): {
                        rangeUpdateEngine->SetRange(planItem.Left, planItem.Right, *planItem.Value);
                        break;
                    }
                };
                break;
            }
            case NProust::EEngineOperation::Query: {
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
                };
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
