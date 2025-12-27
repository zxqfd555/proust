#pragma once

#include <chrono>
#include <iostream>
#include <optional>
#include <random>

#include <spdlog/spdlog.h>

#include "query.h"
#include "engine/naive/naive.h"
#include "serialization/dataset.h"

namespace NProust {
    enum class EDatasetType {
        SumInt32,
        MaxInt32,
        MaxSumInt32,
        OrderStatistic16,
    };

    std::optional<EDatasetType> DatasetTypeFromString(const std::string_view s);
    void GenerateDataset(const EDatasetType type, size_t length, const size_t queryCount, const std::string fname);

    template<class TValue>
    std::function<TValue(std::mt19937&)> CreateRandomIntGenerator(const TValue lower, const TValue upper) {
        return [lower, upper](std::mt19937& generator) {
            static std::uniform_int_distribution<TValue> valueDistribution(lower, upper);
            auto value = valueDistribution(generator);
            return value;
        };
    }

    // How many intermediate generation stages one wants to see in the logs.
    const size_t GENERATION_LOG_FREQUENCY = 20;

    template<typename TValue, typename TOperationResult, typename TOperation>
    void GenerateTestPlan(
        const size_t length,
        const size_t queryCount,
        std::function<TValue(std::mt19937&)> valueGenerator,
        TStreamingDatasetWriter<TValue, TOperationResult>& target,
        std::unique_ptr<NProust::IEngine<TValue, TOperationResult, TOperation>> groundTruthEngine = nullptr
    ) {
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> opDist(0, 1);
        std::uniform_int_distribution<int> leftDist(0, length - 1);
        std::uniform_int_distribution<int> rightDist(0, length);

        if (!groundTruthEngine) {
            groundTruthEngine = NProust::CreateNaiveEngine<TValue, TOperationResult, TOperation>();
        }

        groundTruthEngine->Init(length);
        const size_t intervalForLogging = queryCount / (GENERATION_LOG_FREQUENCY + 1) + 1;
        const auto generationStartedAt = std::chrono::high_resolution_clock::now();
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
                    groundTruthEngine->SetRange(left, right, value);

                    const auto item = TTestPlanItem<TValue, TOperationResult>(operation, left, right, value, std::nullopt);
                    target.SaveQuery(item);
                    break;
                }
                case EEngineOperation::Query: {
                    if (left == right) {
                        right += 1;
                    }
                    const auto expectedResult = groundTruthEngine->ComputeRange(left, right);

                    const auto item = TTestPlanItem<TValue, TOperationResult>(operation, left, right, std::nullopt, expectedResult);
                    target.SaveQuery(item);
                    break;
                }
            }

            if (querySeqIdx > 0 && querySeqIdx % (intervalForLogging + 1) == 0) {
                const auto intermediateTimestamp = std::chrono::high_resolution_clock::now();
                const auto speedPerQuery = std::chrono::duration_cast<std::chrono::nanoseconds>(intermediateTimestamp - generationStartedAt) / (querySeqIdx + 1);
                const auto estimation = speedPerQuery * (queryCount - querySeqIdx - 1);

                spdlog::info(
                    "Dataset generation: {}/{} entries are created (~{}%). Estimated remaining time: {} minutes",
                    (querySeqIdx + 1),
                    queryCount,
                    ((querySeqIdx + 1) * 100 / queryCount),
                    std::chrono::duration_cast<std::chrono::minutes>(estimation).count()
                );
            }
        }
        target.Finalize();

        auto generationEndedAt = std::chrono::high_resolution_clock::now();
        auto generationDuration = std::chrono::duration_cast<std::chrono::milliseconds>(generationEndedAt - generationStartedAt);
        spdlog::info("The test plan has been generated. Time elapsed: {} ms", generationDuration.count());
    }

    template<typename TValue, typename TOperationResult, typename TOperation>
    void GenerateTestPlanIntoFile(
        const size_t length,
        const size_t queryCount,
        std::function<TValue(std::mt19937&)> valueGenerator,
        const std::string& targetPath,
        std::unique_ptr<NProust::IEngine<TValue, TOperationResult, TOperation>> groundTruthEngine = nullptr
    ) {
        auto streamer = TStreamingDatasetWriter<TValue, TOperationResult>(queryCount, length, targetPath);
        GenerateTestPlan<TValue, TOperationResult, TOperation>(length, queryCount, valueGenerator, streamer, std::move(groundTruthEngine));
    }

    template<typename TValue, typename TOperationResult, typename TOperation>
    void GenerateTestPlanIntoOstream(
        const size_t length,
        const size_t queryCount,
        std::function<TValue(std::mt19937&)> valueGenerator,
        std::ostream& target,
        std::unique_ptr<NProust::IEngine<TValue, TOperationResult, TOperation>> groundTruthEngine = nullptr
    ) {
        auto streamer = TStreamingDatasetWriter<TValue, TOperationResult>(queryCount, length, target);
        GenerateTestPlan<TValue, TOperationResult, TOperation>(length, queryCount, valueGenerator, streamer, std::move(groundTruthEngine));
    }
} // namespace NProust
