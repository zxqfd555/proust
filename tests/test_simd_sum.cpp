#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "engine/operations.h"
#include "engine/simd/avx2_sum.h"
#include "engine/simd/avx2_minmax.h"
#include "engine/robust/topology.h"
#include "engine/robust/layout.h"

#include "utils.h"

// Directly compares the AVX2 FromMany reduction against the scalar TSumOperation
// reduction over a range of sizes that exercise the 32-wide block, the 8-wide
// remainder, and the scalar tail, including values that overflow int32 so that
// two's-complement wraparound is covered.
TEST(SimdSumTest, FromManyMatchesScalar) {
    std::mt19937 generator(123456);
    // Bounded so the longest reduction (600 elements) cannot overflow int32:
    // 600 * 3,000,000 < 2^31. This keeps the scalar reference free of the signed
    // overflow that TSumOperation would otherwise hit, while still covering the
    // 32-wide block, the 8-wide remainder, and the scalar tail of the AVX2 path.
    std::uniform_int_distribution<int32_t> valueDistribution(-3'000'000, 3'000'000);

    for (size_t length = 0; length <= 600; ++length) {
        std::vector<int32_t> data(length);
        for (auto& value : data) {
            value = valueDistribution(generator);
        }

        for (size_t from = 0; from <= length; from = from < 5 ? from + 1 : from + 37) {
            const size_t to = length;
            const int32_t scalar = NProust::TSumOperation::FromMany<int32_t, int32_t>(data.data(), from, to);
            const int32_t simd = NProust::TSumOperationAvx2::FromMany<int32_t, int32_t>(data.data(), from, to);
            ASSERT_EQ(scalar, simd) << "length=" << length << " from=" << from << " to=" << to;
        }
    }
}

namespace {
    std::function<int32_t(std::mt19937&)> CreateInt32Generator(const int32_t lower, const int32_t upper) {
        return [lower, upper](std::mt19937& gen) {
            std::uniform_int_distribution<int32_t> distribution(lower, upper);
            return distribution(gen);
        };
    }
}

// End-to-end: the AVX2 operation must produce identical query results to the
// scalar engine across fat-leaf topologies (where FromMany is actually invoked).
TEST(SimdSumTest, RangeEngineStressFatLeaves) {
    StressTest<int32_t, int32_t, NProust::TSumOperationAvx2, NProust::TPreorderNodeTopology<4>>(
        1000, N_QUERIES_IN_STRESS_TEST, CreateInt32Generator(-1000, 1000));
    StressTest<int32_t, int32_t, NProust::TSumOperationAvx2, NProust::TBinaryHeapNodeTopology<6>>(
        1000, N_QUERIES_IN_STRESS_TEST, CreateInt32Generator(-1000, 1000));
    StressTest<int32_t, int32_t, NProust::TSumOperationAvx2, NProust::TPreorderNodeTopology<0>>(
        1000, N_QUERIES_IN_STRESS_TEST, CreateInt32Generator(-1000, 1000));
}

// AVX2 max FromMany vs the scalar max reduction, across all block/remainder/tail
// sizes, including full-range values (no overflow concern for max/min).
TEST(SimdMaxTest, FromManyMatchesScalar) {
    std::mt19937 generator(987654);
    std::uniform_int_distribution<int32_t> valueDistribution(
        std::numeric_limits<int32_t>::min(),
        std::numeric_limits<int32_t>::max());

    for (size_t length = 1; length <= 600; ++length) {
        std::vector<int32_t> data(length);
        for (auto& value : data) {
            value = valueDistribution(generator);
        }
        for (size_t from = 0; from < length; from = from < 5 ? from + 1 : from + 37) {
            const int32_t scalar = NProust::TMaxOperationFromMany::FromMany<int32_t, int32_t>(data.data(), from, length);
            const int32_t simd = NProust::TMaxOperationAvx2::FromMany<int32_t, int32_t>(data.data(), from, length);
            ASSERT_EQ(scalar, simd) << "length=" << length << " from=" << from;
        }
    }
}

// End-to-end range-max: both the scalar-FromMany and the AVX2 max operation must
// match the naive ground truth across fat-leaf topologies.
TEST(SimdMaxTest, RangeEngineStressFatLeaves) {
    StressTest<int32_t, int32_t, NProust::TMaxOperationAvx2, NProust::TPreorderNodeTopology<4>>(
        1000, N_QUERIES_IN_STRESS_TEST, CreateInt32Generator(-100000, 100000));
    StressTest<int32_t, int32_t, NProust::TMaxOperationAvx2, NProust::TBinaryHeapNodeTopology<6>>(
        1000, N_QUERIES_IN_STRESS_TEST, CreateInt32Generator(-100000, 100000));
    StressTest<int32_t, int32_t, NProust::TMaxOperationFromMany, NProust::TPreorderNodeTopology<6>>(
        1000, N_QUERIES_IN_STRESS_TEST, CreateInt32Generator(-100000, 100000));
}
