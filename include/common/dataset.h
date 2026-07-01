#pragma once

#include <chrono>
#include <iostream>
#include <optional>
#include <random>

#include <spdlog/spdlog.h>

#include "query.h"
#include "engine/naive/engine.h"
#include "serialization/dataset.h"

namespace NProust {
    enum class EDatasetType {
        SumInt32,
        MaxInt32,
        MaxSumInt32,
        OrderStatistic16,
        SumDouble,
        MaxDouble,
        XorInt32,
        GcdInt32,
        AndInt32,
        OrInt32,
        SumInt16,
        SumInt64,
        XorInt16,
        XorInt64,
    };

    enum class EBaseGeneratorType {
        Point,
        Range,
    };

    std::optional<EDatasetType> DatasetTypeFromString(const std::string_view s);
    std::optional<EBaseGeneratorType> BaseGeneratorTypeFromString(const std::string_view s);
    void GenerateDataset(const EDatasetType type, size_t length, const size_t queryCount, const std::string fname, ITestPlanBaseGenerator& baseGenerator);

    template<class TValue>
    std::function<TValue(std::mt19937&)> CreateRandomIntGenerator(const TValue lower, const TValue upper) {
        return [lower, upper](std::mt19937& generator) {
            static std::uniform_int_distribution<TValue> valueDistribution(lower, upper);
            auto value = valueDistribution(generator);
            return value;
        };
    }

    // Produces integer-valued doubles in [lower, upper]. Float/double addition is *not* associative,
    // so the segment tree's merge order would disagree with the naive left-to-right ground truth on
    // rounded values, breaking the bit-exact correctness check. Keeping every value an exact integer
    // small enough that all partial sums stay below 2^53 makes double addition exact regardless of
    // order, so the check stays valid while the engine still executes genuine double FP instructions.
    inline std::function<double(std::mt19937&)> CreateExactDoubleSumGenerator(const int32_t lower, const int32_t upper) {
        return [lower, upper](std::mt19937& generator) {
            static std::uniform_int_distribution<int32_t> valueDistribution(lower, upper);
            return static_cast<double>(valueDistribution(generator));
        };
    }

    // Arbitrary fractional doubles. Safe for max (which never rounds), not for sum.
    inline std::function<double(std::mt19937&)> CreateRandomDoubleGenerator(const double lower, const double upper) {
        return [lower, upper](std::mt19937& generator) {
            static std::uniform_real_distribution<double> valueDistribution(lower, upper);
            return valueDistribution(generator);
        };
    }

    // How many intermediate generation stages one wants to see in the logs.
    const size_t GENERATION_LOG_FREQUENCY = 20;

    template<typename TValue, typename TOperationResult, typename TOperation>
    void GenerateTestPlan(
        const size_t length,
        const size_t queryCount,
        ITestPlanBaseGenerator& baseGenerator,
        std::function<TValue(std::mt19937&)> valueGenerator,
        TStreamingDatasetWriter<TValue, TOperationResult>& target,
        std::unique_ptr<IRangeUpdateEngine<TValue, TOperationResult, TOperation>> groundTruthEngine = nullptr
    ) {
        std::mt19937 rng(std::random_device{}());

        if (!groundTruthEngine) {
            groundTruthEngine = NProust::CreateNaiveRangeUpdateEngine<TValue, TOperationResult, TOperation>();
        }

        groundTruthEngine->Init(length);
        const size_t intervalForLogging = queryCount / (GENERATION_LOG_FREQUENCY + 1) + 1;
        const auto generationStartedAt = std::chrono::high_resolution_clock::now();
        for (size_t querySeqIdx = 0; querySeqIdx < queryCount; ++querySeqIdx) {
            const auto base = baseGenerator.Next();
            switch (base.Operation) {
                case EEngineOperation::SetRange: {
                    const auto value = valueGenerator(rng);
                    groundTruthEngine->SetRange(base.Left, base.Right, value);
                    const auto item = TTestPlanItem<TValue, TOperationResult>(base, value, std::nullopt);
                    target.SaveQuery(item);
                    break;
                }
                case EEngineOperation::Query: {
                    const auto expectedResult = groundTruthEngine->ComputeRange(base.Left, base.Right);
                    const auto item = TTestPlanItem<TValue, TOperationResult>(base, std::nullopt, expectedResult);
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
        ITestPlanBaseGenerator& baseGenerator,
        std::function<TValue(std::mt19937&)> valueGenerator,
        const std::string& targetPath,
        std::unique_ptr<NProust::IRangeUpdateEngine<TValue, TOperationResult, TOperation>> groundTruthEngine = nullptr
    ) {
        auto streamer = TStreamingDatasetWriter<TValue, TOperationResult>(queryCount, length, targetPath);
        GenerateTestPlan<TValue, TOperationResult, TOperation>(length, queryCount, baseGenerator, valueGenerator, streamer, std::move(groundTruthEngine));
    }

    template<typename TValue, typename TOperationResult, typename TOperation>
    void GenerateTestPlanIntoOstream(
        const size_t length,
        const size_t queryCount,
        ITestPlanBaseGenerator& baseGenerator,
        std::function<TValue(std::mt19937&)> valueGenerator,
        std::ostream& target,
        std::unique_ptr<NProust::IRangeUpdateEngine<TValue, TOperationResult, TOperation>> groundTruthEngine = nullptr
    ) {
        auto streamer = TStreamingDatasetWriter<TValue, TOperationResult>(queryCount, length, target);
        GenerateTestPlan<TValue, TOperationResult, TOperation>(length, queryCount, baseGenerator, valueGenerator, streamer, std::move(groundTruthEngine));
    }
} // namespace NProust
