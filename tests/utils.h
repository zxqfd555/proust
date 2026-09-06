#pragma once

#include <chrono>
#include <iostream>
#include <random>

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include "common/dataset.h"
#include "engine/robust/engine.h"
#include "engine/external/fenwick.h"
#include "engine/external/algorithmica.h"
#include "engine/operations.h"

namespace {
    const bool LOG_LEVEL_ADJUSTED = [] {
        spdlog::set_level(spdlog::level::warn);
        return true;
    }();
}

// This wrapper is used to test the cases where the default element is not a neutral one.
// A neutral element isn't required for any of the engines, we need to test that it isn't
// assumed in the point engines with undo.
struct TAsymmetricIntWrapper {
    TAsymmetricIntWrapper()
        : Wrapped(1)
    {
    }

    explicit TAsymmetricIntWrapper(const int32_t wrapped)
        : Wrapped(wrapped)
    {
    }

    int32_t Wrapped = 1;
};

inline std::function<TAsymmetricIntWrapper(std::mt19937&)> CreateAsymmetricIntGenerator(const int32_t lower, const int32_t upper) {
    return [lower, upper](std::mt19937& generator) {
        static std::uniform_int_distribution<int32_t> valueDistribution(lower, upper);
        auto value = valueDistribution(generator);
        return TAsymmetricIntWrapper(value);
    };
}

struct TAsymmetricIntWrapperOperation {
    template<typename TValue, typename TOperationResult>
    inline static TOperationResult FromRepeated(const TValue& v, const size_t times) noexcept(noexcept(TOperationResult(v.Wrapped * times))) {
        return TOperationResult(v.Wrapped * times);
    }

    template<typename TOperationResult>
    inline static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) noexcept(noexcept(lhs + rhs)) {
        return lhs + rhs;
    }
};

struct TAsymmetricIntWrapperOperationWithUndo {
    template<typename TValue, typename TOperationResult>
    inline static TOperationResult FromRepeated(const TValue& v, const size_t times) noexcept(noexcept(TOperationResult(v.Wrapped * times))) {
        return TOperationResult(v.Wrapped * times);
    }

    template<typename TOperationResult>
    inline static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) noexcept(noexcept(lhs + rhs)) {
        return lhs + rhs;
    }

    template<typename TOperationResult>
    inline static TOperationResult Undo(const TOperationResult& lhs, const TOperationResult& rhs) noexcept(noexcept(lhs - rhs)) {
        return lhs - rhs;
    }
};

struct TAsymmetricIntWrapperOperationWithFromMany {
    template<typename TValue, typename TOperationResult>
    inline static TOperationResult FromRepeated(const TValue& v, const size_t times) noexcept(noexcept(TOperationResult(v.Wrapped * times))) {
        return TOperationResult(v.Wrapped * times);
    }

    template<typename TOperationResult>
    inline static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) noexcept(noexcept(lhs + rhs)) {
        return lhs + rhs;
    }

    template<typename TOperationResult, typename TValue>
    inline static TOperationResult FromMany(const TValue* elements, const size_t left, const size_t right) noexcept {
        TOperationResult result = elements[left].Wrapped;
        for (size_t index = left + 1; index < right; ++index) {
            result += elements[index].Wrapped;
        }
        return result;
    }
};

struct TAsymmetricIntWrapperOperationWithUndoAndFromMany {
    template<typename TValue, typename TOperationResult>
    inline static TOperationResult FromRepeated(const TValue& v, const size_t times) noexcept {
        return TOperationResult(v.Wrapped * times);
    }

    template<typename TOperationResult>
    inline static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) noexcept {
        return lhs + rhs;
    }

    template<typename TOperationResult>
    inline static TOperationResult Undo(const TOperationResult& lhs, const TOperationResult& rhs) noexcept {
        return lhs - rhs;
    }

    template<typename TOperationResult, typename TValue>
    inline static TOperationResult FromMany(const TValue* elements, const size_t left, const size_t right) noexcept {
        TOperationResult result = elements[left].Wrapped;
        for (size_t index = left + 1; index < right; ++index) {
            result += elements[index].Wrapped;
        }
        return result;
    }
};

constexpr size_t N_QUERIES_IN_STRESS_TEST = 10000;

template<typename TValue, typename TOperationResult, typename TOperation>
using TNaiveEngineFactory = std::function<std::unique_ptr<NProust::IRangeUpdateEngine<TValue, TOperationResult, TOperation>>()>;

template<typename TValue, typename TOperationResult, typename TOperation>
std::unique_ptr<NProust::IRangeUpdateEngine<TValue, TOperationResult, TOperation>> DefaultNaiveEngineFactory() {
    return nullptr;
}

enum class ETestedEngineType {
    RangeRobust,
    PointRobust,
    Fat,
    ExternalFenwick,
    AlgorithmicaWideTree
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
            // Named explicitly rather than built by the factory, which would substitute its own
            // default for a dropped argument instead of failing the assertions below.
            using TEngine = NProust::TRobustRangeUpdateEngine<TValue, TOperationResult, TOperation, TNodeTopology, TLayout>;
            static_assert(std::is_same_v<typename TEngine::TNodeTopologyType, TNodeTopology>);
            static_assert(std::is_same_v<typename TEngine::TLayoutType, TLayout>);
            rangeUpdateEngine = std::make_unique<TEngine>();
            rangeUpdateEngine->Init(length);
            break;
        }
        case (ETestedEngineType::PointRobust): {
            baseGenerator = std::make_unique<NProust::TPointUpdatePlanGenerator>(length);
            using TEngine = NProust::TRobustPointUpdateEngine<TValue, TOperationResult, TOperation, LeafSizeLog>;
            static_assert(TEngine::LeafSizeLogValue == LeafSizeLog);
            pointUpdateEngine = std::make_unique<TEngine>();
            pointUpdateEngine->Init(length);
            break;
        }
        case (ETestedEngineType::Fat): {
            baseGenerator = std::make_unique<NProust::TPointUpdatePlanGenerator>(length);
            if constexpr (NProust::HasUndo<TOperation, TOperationResult>) {
                using TEngine = NProust::TFatPointUpdateEngine<TValue, TOperationResult, TOperation, BranchingLog>;
                static_assert(TEngine::BranchingLogValue == BranchingLog);
                pointUpdateEngine = std::make_unique<TEngine>();
            } else {
                spdlog::critical("The specified operation doesn't support Undo");
                std::terminate();
            }
            pointUpdateEngine->Init(length);
            break;
        }
        case (ETestedEngineType::ExternalFenwick): {
            baseGenerator = std::make_unique<NProust::TPointUpdatePlanGenerator>(length);
            // The external Fenwick baseline only models the int32 sum problem; it has no
            // generic instantiation, so guard its construction to that single specialization.
            if constexpr (std::is_same_v<TValue, int32_t> && std::is_same_v<TOperationResult, int32_t> && std::is_same_v<TOperation, NProust::TSumOperation>) {
                pointUpdateEngine = NProust::CreateFenwickPointSumEngine();
            } else {
                spdlog::critical("The external Fenwick engine only supports the int32 sum problem");
                std::terminate();
            }
            pointUpdateEngine->Init(length);
            break;
        }
        case (ETestedEngineType::AlgorithmicaWideTree): {
            baseGenerator = std::make_unique<NProust::TPointUpdatePlanGenerator>(length);
            // The Algorithmica wide segment tree, like Fenwick, is fixed to the int32 sum problem.
            if constexpr (std::is_same_v<TValue, int32_t> && std::is_same_v<TOperationResult, int32_t> && std::is_same_v<TOperation, NProust::TSumOperation>) {
                pointUpdateEngine = NProust::CreateAlgorithmicaWideSegmentSumEngine();
            } else {
                spdlog::critical("The external Algorithmica engine only supports the int32 sum problem");
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
                    case (ETestedEngineType::Fat):
                    case (ETestedEngineType::ExternalFenwick):
                    case (ETestedEngineType::AlgorithmicaWideTree): {
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
                    case (ETestedEngineType::Fat):
                    case (ETestedEngineType::ExternalFenwick):
                    case (ETestedEngineType::AlgorithmicaWideTree): {
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
