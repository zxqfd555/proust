#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "engine/robust/engine.h"
#include "engine/operations.h"
#include "examples/leetcode_maximum_subarray.h"

#include "utils.h"

// InitFrom must leave an engine indistinguishable from Init + assigning every element: the same
// answers right after the build, and the same answers after further modifications (which checks
// the internal state the build leaves behind -- lazy tags, fat-leaf snapshots, prefix groups).
// The ground truth is a plain array folded with FromRepeated(v, 1) and Merge, independent of any engine.

namespace {
    template<typename TValue, typename TOperationResult, typename TOperation>
    TOperationResult FoldNaively(const std::vector<TValue>& values, const size_t left, const size_t right) {
        auto result = TOperation::template FromRepeated<TValue, TOperationResult>(values[left], 1);
        for (size_t index = left + 1; index < right; ++index) {
            result = TOperation::Merge(result, TOperation::template FromRepeated<TValue, TOperationResult>(values[index], 1));
        }
        return result;
    }

    template<typename TValue, typename TOperationResult, typename TOperation, typename TEngine>
    void ExpectSameAnswers(const TEngine& engine, const std::vector<TValue>& values, std::mt19937& generator) {
        const size_t length = values.size();
        if (length <= 40) {
            for (size_t left = 0; left < length; ++left) {
                for (size_t right = left + 1; right <= length; ++right) {
                    ASSERT_EQ(engine.ComputeRange(left, right), (FoldNaively<TValue, TOperationResult, TOperation>(values, left, right)))
                        << "length " << length << ", range [" << left << ", " << right << ")";
                }
            }
            return;
        }

        std::uniform_int_distribution<size_t> position(0, length);
        for (size_t query = 0; query < 2000; ++query) {
            size_t left = position(generator);
            size_t right = position(generator);
            if (left > right) {
                std::swap(left, right);
            }
            if (left == right) {
                continue;
            }
            ASSERT_EQ(engine.ComputeRange(left, right), (FoldNaively<TValue, TOperationResult, TOperation>(values, left, right)))
                << "length " << length << ", range [" << left << ", " << right << ")";
        }
    }

    template<typename TEngine, typename TValue, typename TOperationResult, typename TOperation>
    void TestInitFrom(std::function<TValue(std::mt19937&)> valueGenerator) {
        constexpr bool IsRangeEngine = std::is_base_of_v<NProust::IRangeUpdateEngine<TValue, TOperationResult, TOperation>, TEngine>;

        std::vector<size_t> lengths;
        for (size_t length = 1; length <= 70; ++length) {
            lengths.push_back(length);
        }
        for (size_t length : {127, 128, 129, 1000, 1024, 1025}) {
            lengths.push_back(length);
        }

        std::mt19937 generator(42);
        for (const size_t length : lengths) {
            TEngine engine;

            // Reuse a dirty engine of another size: the build must not depend on what was there.
            {
                const size_t previousLength = length * 2 + 3;
                engine.Init(previousLength);
                for (size_t step = 0; step < 20; ++step) {
                    const size_t position = generator() % previousLength;
                    if constexpr (IsRangeEngine) {
                        engine.SetRange(position / 2, position + 1, valueGenerator(generator));
                    } else {
                        engine.SetPoint(position, valueGenerator(generator));
                    }
                }
            }

            std::vector<TValue> values(length);
            for (auto& value : values) {
                value = valueGenerator(generator);
            }
            engine.InitFrom(values);
            ExpectSameAnswers<TValue, TOperationResult, TOperation>(engine, values, generator);
            if (::testing::Test::HasFatalFailure()) {
                return;
            }

            std::uniform_int_distribution<size_t> position(0, length - 1);
            for (size_t step = 0; step < 30; ++step) {
                const TValue value = valueGenerator(generator);
                if constexpr (IsRangeEngine) {
                    size_t left = position(generator);
                    size_t right = position(generator);
                    if (left > right) {
                        std::swap(left, right);
                    }
                    engine.SetRange(left, right + 1, value);
                    std::fill(values.begin() + left, values.begin() + right + 1, value);
                } else {
                    const size_t index = position(generator);
                    engine.SetPoint(index, value);
                    values[index] = value;
                }
                if (step % 10 == 9) {
                    ExpectSameAnswers<TValue, TOperationResult, TOperation>(engine, values, generator);
                    if (::testing::Test::HasFatalFailure()) {
                        return;
                    }
                }
            }
        }
    }

    using NProust::TPreorderNodeTopology;
    using NProust::TBinaryHeapNodeTopology;

    template<typename TValue, typename TOperationResult, typename TOperation, typename TTopology, template<class, class, auto> class TLayout>
    using TRange = NProust::TRobustRangeUpdateEngine<TValue, TOperationResult, TOperation, TTopology, TLayout<TValue, TOperationResult, std::nullopt>>;

    template<typename TValue, typename TOperationResult, typename TOperation>
    void TestRangeEngineInitFrom(std::function<TValue(std::mt19937&)> valueGenerator) {
        TestInitFrom<TRange<TValue, TOperationResult, TOperation, TPreorderNodeTopology<0>, NProust::TSoALayout>, TValue, TOperationResult, TOperation>(valueGenerator);
        TestInitFrom<TRange<TValue, TOperationResult, TOperation, TPreorderNodeTopology<1>, NProust::TAoSLayout>, TValue, TOperationResult, TOperation>(valueGenerator);
        TestInitFrom<TRange<TValue, TOperationResult, TOperation, TPreorderNodeTopology<3>, NProust::TSoALayout>, TValue, TOperationResult, TOperation>(valueGenerator);
        TestInitFrom<TRange<TValue, TOperationResult, TOperation, TBinaryHeapNodeTopology<0>, NProust::TAoSLayout>, TValue, TOperationResult, TOperation>(valueGenerator);
        TestInitFrom<TRange<TValue, TOperationResult, TOperation, TBinaryHeapNodeTopology<2>, NProust::TSoALayout>, TValue, TOperationResult, TOperation>(valueGenerator);
        TestInitFrom<TRange<TValue, TOperationResult, TOperation, TBinaryHeapNodeTopology<5>, NProust::TSoALayout>, TValue, TOperationResult, TOperation>(valueGenerator);
    }

    template<typename TValue, typename TOperationResult, typename TOperation>
    void TestPointEngineInitFrom(std::function<TValue(std::mt19937&)> valueGenerator) {
        TestInitFrom<NProust::TRobustPointUpdateEngine<TValue, TOperationResult, TOperation, 0>, TValue, TOperationResult, TOperation>(valueGenerator);
        TestInitFrom<NProust::TRobustPointUpdateEngine<TValue, TOperationResult, TOperation, 2>, TValue, TOperationResult, TOperation>(valueGenerator);
        TestInitFrom<NProust::TRobustPointUpdateEngine<TValue, TOperationResult, TOperation, 4>, TValue, TOperationResult, TOperation>(valueGenerator);
    }

    template<typename TValue, typename TOperationResult, typename TOperation>
    void TestFatEngineInitFrom(std::function<TValue(std::mt19937&)> valueGenerator) {
        TestInitFrom<NProust::TFatPointUpdateEngine<TValue, TOperationResult, TOperation, 1>, TValue, TOperationResult, TOperation>(valueGenerator);
        TestInitFrom<NProust::TFatPointUpdateEngine<TValue, TOperationResult, TOperation, 2>, TValue, TOperationResult, TOperation>(valueGenerator);
        TestInitFrom<NProust::TFatPointUpdateEngine<TValue, TOperationResult, TOperation, 3>, TValue, TOperationResult, TOperation>(valueGenerator);
        TestInitFrom<NProust::TFatPointUpdateEngine<TValue, TOperationResult, TOperation, 4>, TValue, TOperationResult, TOperation>(valueGenerator);
        TestInitFrom<NProust::TFatPointUpdateEngine<TValue, TOperationResult, TOperation, 5>, TValue, TOperationResult, TOperation>(valueGenerator);
    }

    using NProustExamples::TMaxSumState;
    using NProustExamples::TMaxSumStateOperation;
} // namespace

TEST(InitFrom, RangeEngineIntSum) {
    TestRangeEngineInitFrom<int32_t, int32_t, NProust::TSumOperation>(NProust::CreateRandomIntGenerator(-1000, 1000));
}

// Non-commutative merge with FromMany: catches a wrong fold order in the build.
TEST(InitFrom, RangeEngineMaximumSubarray) {
    TestRangeEngineInitFrom<int32_t, TMaxSumState, TMaxSumStateOperation>(NProust::CreateRandomIntGenerator(-100, 100));
}

// The default element is not neutral: catches real elements mixed up with default ones.
TEST(InitFrom, RangeEngineAsymmetric) {
    TestRangeEngineInitFrom<TAsymmetricIntWrapper, int32_t, TAsymmetricIntWrapperOperation>(CreateAsymmetricIntGenerator(-100, 100));
    TestRangeEngineInitFrom<TAsymmetricIntWrapper, int32_t, TAsymmetricIntWrapperOperationWithFromMany>(CreateAsymmetricIntGenerator(-100, 100));
}

TEST(InitFrom, PointEngineIntSum) {
    TestPointEngineInitFrom<int32_t, int32_t, NProust::TSumOperation>(NProust::CreateRandomIntGenerator(-1000, 1000));
}

TEST(InitFrom, PointEngineMaximumSubarray) {
    TestPointEngineInitFrom<int32_t, TMaxSumState, TMaxSumStateOperation>(NProust::CreateRandomIntGenerator(-100, 100));
}

TEST(InitFrom, PointEngineAsymmetric) {
    TestPointEngineInitFrom<TAsymmetricIntWrapper, int32_t, TAsymmetricIntWrapperOperation>(CreateAsymmetricIntGenerator(-100, 100));
    TestPointEngineInitFrom<TAsymmetricIntWrapper, int32_t, TAsymmetricIntWrapperOperationWithUndo>(CreateAsymmetricIntGenerator(-100, 100));
    TestPointEngineInitFrom<TAsymmetricIntWrapper, int32_t, TAsymmetricIntWrapperOperationWithFromMany>(CreateAsymmetricIntGenerator(-100, 100));
    TestPointEngineInitFrom<TAsymmetricIntWrapper, int32_t, TAsymmetricIntWrapperOperationWithUndoAndFromMany>(CreateAsymmetricIntGenerator(-100, 100));
}

TEST(InitFrom, FatEngineIntSum) {
    TestFatEngineInitFrom<int32_t, int32_t, NProust::TSumOperation>(NProust::CreateRandomIntGenerator(-1000, 1000));
}

TEST(InitFrom, FatEngineAsymmetric) {
    TestFatEngineInitFrom<TAsymmetricIntWrapper, int32_t, TAsymmetricIntWrapperOperationWithUndo>(CreateAsymmetricIntGenerator(-100, 100));
    TestFatEngineInitFrom<TAsymmetricIntWrapper, int32_t, TAsymmetricIntWrapperOperationWithUndoAndFromMany>(CreateAsymmetricIntGenerator(-100, 100));
}

// The interface's fallback (Init + one assignment per element) for engines without their own build.
TEST(InitFrom, NaiveEngineFallback) {
    TestInitFrom<NProust::TNaiveRangeUpdateEngine<int32_t, TMaxSumState, TMaxSumStateOperation>, int32_t, TMaxSumState, TMaxSumStateOperation>(NProust::CreateRandomIntGenerator(-100, 100));
}
