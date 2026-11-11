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

// A non-commutative group: the maps x -> A x + B modulo a prime, A non-zero, merged as `lhs` first,
// then `rhs`. The default element is the identity map, the only one the engines may build on their own.
struct TAffineMap {
    uint32_t A = 1;
    uint32_t B = 0;
};

inline bool operator==(const TAffineMap& lhs, const TAffineMap& rhs) {
    return lhs.A == rhs.A && lhs.B == rhs.B;
}

inline std::ostream& operator<<(std::ostream& stream, const TAffineMap& map) {
    return stream << "(" << map.A << " x + " << map.B << ")";
}

template<bool DeclaredCommutative>
struct TAffineCompositionOperationBase {
    static constexpr uint64_t MOD = 998244353;

    template<typename TOperationResult>
    static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) noexcept {
        return {
            static_cast<uint32_t>(uint64_t(lhs.A) * rhs.A % MOD),
            static_cast<uint32_t>((uint64_t(rhs.A) * lhs.B + rhs.B) % MOD),
        };
    }

    template<typename TValue, typename TOperationResult>
    static TOperationResult FromRepeated(const TValue& value, size_t times) noexcept {
        TOperationResult result;
        TOperationResult power = value;
        for (; times; times >>= 1) {  // the powers of one map commute: the order does not matter
            if (times & 1) {
                result = Merge(result, power);
            }
            power = Merge(power, power);
        }
        return result;
    }

    // rhs^{-1}, then lhs.
    template<typename TOperationResult>
    static TOperationResult Undo(const TOperationResult& lhs, const TOperationResult& rhs) noexcept {
        uint64_t inverse = 1;
        uint64_t base = rhs.A;
        for (uint64_t exponent = MOD - 2; exponent; exponent >>= 1) {
            if (exponent & 1) {
                inverse = inverse * base % MOD;
            }
            base = base * base % MOD;
        }
        const TOperationResult inverted{static_cast<uint32_t>(inverse), static_cast<uint32_t>((MOD - rhs.B) * inverse % MOD)};
        return Merge(inverted, lhs);
    }
};

using TAffineCompositionOperation = TAffineCompositionOperationBase<false>;

// The same operation, wrongly declared commutative: the tests check that this is noticed, i.e. that
// they exercise a non-commutative case at all.
struct TAffineCompositionOperationDeclaredCommutative : TAffineCompositionOperationBase<true> {
    static constexpr bool IsCommutative = true;
};

inline std::function<TAffineMap(std::mt19937&)> CreateAffineMapGenerator() {
    return [](std::mt19937& generator) {
        std::uniform_int_distribution<uint32_t> multiplier(1, TAffineCompositionOperation::MOD - 1);
        std::uniform_int_distribution<uint32_t> shift(0, TAffineCompositionOperation::MOD - 1);
        if (generator() % 8 == 0) {  // the edges too: identity-like and negating maps
            static constexpr uint32_t EDGE_A[] = {1, 2, TAffineCompositionOperation::MOD - 1};
            return TAffineMap{EDGE_A[generator() % 3], static_cast<uint32_t>(generator() % 2)};
        }
        const uint32_t a = multiplier(generator);
        return TAffineMap{a, shift(generator)};
    };
}

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
