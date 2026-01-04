#pragma once

#include <chrono>
#include <iostream>
#include <random>

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include "common/dataset.h"
#include "engine/robust/engine.h"
#include "engine/operations.h"

enum class ETestedEngineType {
    RangeRobust,
    PointRobust,
    Fat
};

template<
    typename TValue,
    typename TOperationResult,
    typename TOperation,
    NProust::NodeTopology TNodeTopology = NProust::TPreorderNodeTopology<0>,
    typename TLayout = NProust::TSoALayout<TValue, TOperationResult, std::nullopt>,
    size_t LeafSizeLog = 0,
    size_t BranchingLog = 1
>
void StressTest(
    const size_t length,
    const size_t queryCount,
    std::function<TValue(std::mt19937&)> valueGenerator,
    const ETestedEngineType testedEngine = ETestedEngineType::RangeRobust,
    std::unique_ptr<NProust::IRangeUpdateEngine<TValue, TOperationResult, TOperation>> groundTruthEngine = nullptr
) {
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    NProust::TStreamingDatasetWriter<TValue, TOperationResult> writer(queryCount, length, stream);

    std::unique_ptr<NProust::ITestPlanBaseGenerator> baseGenerator;
    std::unique_ptr<NProust::IRangeUpdateEngine<TValue, TOperationResult, TOperation>> rangeUpdateEngine;
    std::unique_ptr<NProust::IPointUpdateEngine<TValue, TOperationResult, TOperation>> pointUpdateEngine;

    switch (testedEngine) {
        case (ETestedEngineType::RangeRobust): {
            baseGenerator = std::make_unique<NProust::TRangeUpdatePlanGenerator>(length);
            rangeUpdateEngine = NProust::CreateRobustRangeUpdateEngine<TValue, TOperationResult, TOperation, TNodeTopology>();
            rangeUpdateEngine->Init(length);
            break;
        }
        case (ETestedEngineType::PointRobust): {
            baseGenerator = std::make_unique<NProust::TPointUpdatePlanGenerator>(length);
            pointUpdateEngine = std::make_unique<NProust::TRobustPointUpdateEngine<TValue, TOperationResult, TOperation, LeafSizeLog>>();
            pointUpdateEngine->Init(length);
            break;
        }
        case (ETestedEngineType::Fat): {
            baseGenerator = std::make_unique<NProust::TPointUpdatePlanGenerator>(length);
            if constexpr (NProust::HasUndo<TOperation, TOperationResult>) {
                pointUpdateEngine = std::make_unique<NProust::TFatPointUpdateEngine<TValue, TOperationResult, TOperation, BranchingLog>>();
            } else {
                spdlog::critical("The specified operation doesn't support Undo");
                std::terminate();
            }
            pointUpdateEngine->Init(length);
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
        assert(planItem.Left <= planItem.Right && "Incorrect test plan: Left >= Right");
        assert(planItem.Right <= length && "Incorrect test plan: the Right endpoint is out of bounds");
        switch (planItem.Operation) {
            case NProust::EEngineOperation::SetRange: {
                assert(planItem.Value.has_value() && "SetRange queries must have the value");
                switch (testedEngine) {
                    case (ETestedEngineType::PointRobust): 
                    case (ETestedEngineType::Fat): {
                        assert(planItem.Left + 1 == planItem.Right && "Incorrect modification for a point-update engine");
                        pointUpdateEngine->SetPoint(planItem.Left, *planItem.Value);
                        break;
                    }
                    case (ETestedEngineType::RangeRobust): {
                        rangeUpdateEngine->SetRange(planItem.Left, planItem.Right, *planItem.Value);
                        break;
                    }
                };
                break;
            }
            case NProust::EEngineOperation::Query: {
                TOperationResult result;
                switch (testedEngine) {
                    case (ETestedEngineType::PointRobust): 
                    case (ETestedEngineType::Fat): {
                        result = pointUpdateEngine->ComputeRange(planItem.Left, planItem.Right);
                        break;
                    }
                    case (ETestedEngineType::RangeRobust): {
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
