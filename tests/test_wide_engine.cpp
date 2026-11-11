#include <array>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "engine/default.h"
#include "engine/operations.h"
#include "engine/wide/engine.h"
#include "examples/leetcode_house_robber.h"
#include "examples/leetcode_maximum_subarray.h"
#include "examples/leetcode_third_distinct_maximum.h"

#include "utils.h"

// The engine against a plain array folded with FromRepeated(v, 1) and Merge. The lengths go around
// the powers of 16, where the shape of the tree changes and slots start reaching past the end.

namespace {
    constexpr uint64_t MOD = 998244353;

    // A non-commutative operation: the composition of x -> A x + B, so an order mistake shows.
    struct TAffine {
        uint64_t A = 1;
        uint64_t B = 0;
        bool operator==(const TAffine& other) const = default;
    };

    struct TAffineOperation {
        template<typename TValue, typename TOperationResult>
        static TOperationResult FromRepeated(const TValue& value, size_t times) {
            TAffine result{1, 0};
            TAffine power = value;
            for (; times; times >>= 1) {
                if (times & 1) {
                    result = Merge<TAffine>(result, power);
                }
                power = Merge<TAffine>(power, power);
            }
            return result;
        }

        template<typename TOperationResult>
        static TOperationResult Merge(const TOperationResult& first, const TOperationResult& second) {
            return {first.A * second.A % MOD, (second.A * first.B + second.B) % MOD};
        }
    };

    template<typename TValue, typename TOperationResult, typename TOperation>
    TOperationResult FoldNaively(const std::vector<TValue>& values, const size_t left, const size_t right) {
        auto result = TOperation::template FromRepeated<TValue, TOperationResult>(values[left], 1);
        for (size_t index = left + 1; index < right; ++index) {
            result = TOperation::Merge(result, TOperation::template FromRepeated<TValue, TOperationResult>(values[index], 1));
        }
        return result;
    }

    template<typename TValue, typename TOperationResult, typename TOperation>
    void Stress(const std::vector<size_t>& lengths, const size_t operations, std::function<TValue(std::mt19937&)> valueGenerator) {
        std::mt19937 generator(42);
        for (const size_t length : lengths) {
            for (const bool fromArray : {false, true}) {
                std::vector<TValue> values(length);
                NProust::TWideRangeUpdateEngine<TValue, TOperationResult, TOperation> engine;
                if (fromArray) {
                    for (TValue& value : values) {
                        value = valueGenerator(generator);
                    }
                    engine.InitFrom(values);
                } else {
                    engine.Init(length);
                }

                std::uniform_int_distribution<size_t> position(0, length - 1);
                for (size_t operation = 0; operation < operations; ++operation) {
                    size_t left = position(generator);
                    size_t right = position(generator);
                    if (left > right) {
                        std::swap(left, right);
                    }
                    ++right;
                    if (generator() % 4 == 0) {  // short ranges, inside one leaf or across two
                        right = std::min(length, left + 1 + generator() % 20);
                    }
                    if (generator() % 2) {
                        const TValue value = valueGenerator(generator);
                        engine.SetRange(left, right, value);
                        std::fill(values.begin() + left, values.begin() + right, value);
                    } else {
                        ASSERT_EQ(engine.ComputeRange(left, right), (FoldNaively<TValue, TOperationResult, TOperation>(values, left, right)))
                            << "length " << length << ", range [" << left << ", " << right << "), operation " << operation;
                    }
                }
            }
        }
    }

    const std::vector<size_t> LENGTHS = {1, 2, 15, 16, 17, 31, 32, 33, 255, 256, 257, 1000, 4095, 4096, 4097, 5000};
    constexpr size_t OPERATIONS = 2000;

    int32_t RandomInt(std::mt19937& generator) {
        return std::uniform_int_distribution<int32_t>(-1000, 1000)(generator);
    }
}

TEST(WideRangeUpdateEngine, IntSum) {
    Stress<int32_t, int64_t, NProust::TSumOperation>(LENGTHS, OPERATIONS, RandomInt);
}

TEST(WideRangeUpdateEngine, IntMax) {
    Stress<int32_t, int32_t, NProust::TMaxOperation>(LENGTHS, OPERATIONS, RandomInt);
}

TEST(WideRangeUpdateEngine, IntMin) {
    Stress<int32_t, int32_t, NProust::TMinOperation>(LENGTHS, OPERATIONS, RandomInt);
}

TEST(WideRangeUpdateEngine, IntBitwiseAnd) {
    Stress<int32_t, int32_t, NProust::TBitwiseAndOperation>(LENGTHS, OPERATIONS, RandomInt);
}

TEST(WideRangeUpdateEngine, IntBitwiseXor) {
    Stress<int32_t, int32_t, NProust::TBitwiseXorOperation>(LENGTHS, OPERATIONS, RandomInt);
}

TEST(WideRangeUpdateEngine, MaximumSubarray) {
    Stress<int32_t, NProustExamples::TMaxSumState, NProustExamples::TMaxSumStateOperation>(LENGTHS, OPERATIONS, RandomInt);
}

TEST(WideRangeUpdateEngine, OrderStatistic) {
    Stress<int32_t, NProustExamples::TTopKState<16>, NProustExamples::TInverseOrderStatisticOperation<16>>(LENGTHS, OPERATIONS, RandomInt);
}

TEST(WideRangeUpdateEngine, DefaultElementIsNotNeutral) {
    Stress<TAsymmetricIntWrapper, int32_t, TAsymmetricIntWrapperOperation>(LENGTHS, OPERATIONS, CreateAsymmetricIntGenerator(-100, 100));
}

TEST(WideRangeUpdateEngine, NonCommutativeComposition) {
    Stress<TAffine, TAffine, TAffineOperation>(LENGTHS, OPERATIONS, [](std::mt19937& generator) {
        return TAffine{generator() % (MOD - 1) + 1, generator() % MOD};
    });
}

TEST(DefaultRangeUpdateEngine, ChoosesByResultSize) {
    static_assert(std::is_same_v<
        NProust::TDefaultRangeUpdateEngine<int32_t, int64_t, NProust::TSumOperation>,
        NProust::TWideRangeUpdateEngine<int32_t, int64_t, NProust::TSumOperation>>);
    // A larger result stays binary, unless the operation folds a block in SIMD.
    static_assert(std::is_same_v<
        NProust::TDefaultRangeUpdateEngine<int32_t, NProustExamples::TTopKState<16>, NProustExamples::TInverseOrderStatisticOperation<16>>,
        NProust::TRobustRangeUpdateEngine<int32_t, NProustExamples::TTopKState<16>, NProustExamples::TInverseOrderStatisticOperation<16>>>);
    static_assert(NProust::HasMergeBlock<NProustExamples::TMaxSumStateOperation, NProustExamples::TMaxSumState> == std::is_same_v<
        NProust::TDefaultRangeUpdateEngine<int32_t, NProustExamples::TMaxSumState, NProustExamples::TMaxSumStateOperation>,
        NProust::TWideRangeUpdateEngine<int32_t, NProustExamples::TMaxSumState, NProustExamples::TMaxSumStateOperation>>);

    auto engine = NProust::CreateRangeUpdateEngine<int32_t, int64_t, NProust::TSumOperation>();
    engine->Init(100);
    engine->SetRange(10, 20, 3);
    ASSERT_EQ(engine->ComputeRange(0, 100), 30);
    ASSERT_EQ(engine->ComputeRange(15, 16), 3);
}

#ifdef PROUST_HAS_MAX_SUM_AVX2_BLOCK
TEST(WideRangeUpdateEngine, MaximumSubarrayAvx2BlockEqualsMergeChain) {
    if (!__builtin_cpu_supports("avx2")) {
        GTEST_SKIP();
    }
    using TOperation = NProustExamples::TMaxSumStateOperation;
    using TState = NProustExamples::TMaxSumState;
    std::mt19937 generator(7);
    for (int round = 0; round < 100000; ++round) {
        const int32_t bound = round % 3 == 0 ? 3 : 100000;
        std::uniform_int_distribution<int32_t> value(-bound, bound);
        std::uniform_int_distribution<size_t> length(1, 40);
        std::array<TState, 16> block;
        for (TState& state : block) {
            state = TOperation::FromRepeated<int32_t, TState>(value(generator), length(generator));
        }
        TState chain = block[0];
        for (size_t index = 1; index < block.size(); ++index) {
            chain = TOperation::Merge(chain, block[index]);
        }
        ASSERT_EQ(chain, NProustExamples::NMaxSumDetail::MergeBlockAvx2(block.data()));
    }
}
#endif

#ifdef PROUST_HAS_FOLD_IN_BLOCK_AVX2
TEST(WideRangeUpdateEngine, FoldInBlockEqualsMergeChain) {
    if (!__builtin_cpu_supports("avx2")) {
        GTEST_SKIP();
    }
    std::mt19937 generator(11);
    std::uniform_int_distribution<int32_t> value(std::numeric_limits<int32_t>::min() / 16, std::numeric_limits<int32_t>::max() / 16);
    for (int round = 0; round < 2000; ++round) {
        alignas(64) std::array<int32_t, 16> block;
        for (int32_t& element : block) {
            element = round % 7 == 0 ? std::numeric_limits<int32_t>::min() / 16 : value(generator);
        }
        for (size_t from = 0; from < block.size(); ++from) {
            for (size_t to = from + 1; to <= block.size(); ++to) {
                int32_t sum = block[from];
                int32_t largest = block[from];
                for (size_t index = from + 1; index < to; ++index) {
                    sum += block[index];
                    largest = std::max(largest, block[index]);
                }
                ASSERT_EQ(sum, NProust::NOpDetail::SumInBlockAvx2(block.data(), from, to));
                ASSERT_EQ(largest, NProust::NOpDetail::MaxInBlockAvx2(block.data(), from, to));
            }
        }
    }
}
#endif
