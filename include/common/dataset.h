#pragma once

#include <chrono>
#include <iostream>
#include <random>

#include <spdlog/spdlog.h>

#include "engine/naive/naive.h"

namespace NProust {
    enum class EEngineOperation { 
        SetRange,
        Query,
    };

    template<typename TValue, typename TOperationResult>
    struct TTestPlanItem {
        EEngineOperation Operation;
        size_t Left;
        size_t Right;
        std::optional<TValue> Value;
        std::optional<TOperationResult> ExpectedResult;

        TTestPlanItem() = default;

        TTestPlanItem(const EEngineOperation operation, const size_t left, const size_t right, const std::optional<TValue> value, const std::optional<TOperationResult> expectedResult)
            : Operation(operation)
            , Left(left)
            , Right(right)
            , Value(value)
            , ExpectedResult(expectedResult)
        {
        }
    };

    template<typename TValue, typename TOperationResult, typename TOperation>
    std::vector<TTestPlanItem<TValue, TOperationResult>> GenerateTestPlan(const size_t length, const size_t queryCount, std::function<TValue(std::mt19937&)> valueGenerator) {
        const auto generationStartedAt = std::chrono::high_resolution_clock::now();

        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> opDist(0, 1);
        std::uniform_int_distribution<int> leftDist(0, length - 1);
        std::uniform_int_distribution<int> rightDist(0, length);

        auto naiveEngine = NProust::CreateNaiveEngine<TValue, TOperationResult, TOperation>();

        naiveEngine->Init(length);
        std::vector<TTestPlanItem<TValue, TOperationResult>> testPlan;
        testPlan.reserve(queryCount);
        for (size_t querySeqIdx = 0; querySeqIdx < queryCount; ++querySeqIdx) {
            size_t left = leftDist(rng);
            size_t right = rightDist(rng);
            if (left > right) {
                std::swap(left, right);
            }

            EEngineOperation operation = static_cast<EEngineOperation>(opDist(rng));
            switch (operation) {
                case EEngineOperation::SetRange: {
                    const auto value = valueGenerator(rng);
                    naiveEngine->SetRange(left, right, value);
                    testPlan.emplace_back(operation, left, right, value, std::nullopt);
                    break;
                }
                case EEngineOperation::Query: {
                    if (left == right) {
                        right += 1;
                    }
                    const auto expectedResult = naiveEngine->ComputeRange(left, right);
                    testPlan.emplace_back(operation, left, right, std::nullopt, expectedResult);
                    break;
                }
            }
        }
        auto generationEndedAt = std::chrono::high_resolution_clock::now();
        auto generationDuration = std::chrono::duration_cast<std::chrono::milliseconds>(generationEndedAt - generationStartedAt);
        spdlog::info("The test plan has been generated. Time elapsed: {} ms", generationDuration.count());

        return testPlan;
    }

} // namespace NProust