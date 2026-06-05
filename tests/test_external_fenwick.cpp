#include <random>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "engine/external/fenwick.h"
#include "engine/operations.h"

#include "utils.h"

using NProust::CreateRandomIntGenerator;

namespace {
    // The Fenwick baseline is fixed to the int32 sum problem, so unlike the generic engines
    // there is only one type instantiation worth exercising. We still follow the project's
    // strategy: validate it against the naive engine over the corner-case lengths (1..16) and
    // a larger length, replaying ten thousand random point modifications and queries each.
    using TDefaultTopology = NProust::TPreorderNodeTopology<0>;
    using TDefaultLayout = NProust::TSoALayout<int32_t, int32_t, std::nullopt>;

    void StressTestFenwickAgainstNaive(const size_t length) {
        StressTest<int32_t, int32_t, NProust::TSumOperation, TDefaultTopology, TDefaultLayout, 0>(
            length,
            N_QUERIES_IN_STRESS_TEST,
            CreateRandomIntGenerator(-100, 100),
            ETestedEngineType::ExternalFenwick
        );
    }
} // namespace

TEST(ExternalFenwick, IntSumStress) {
    for (size_t length = 1; length <= 16; ++length) {
        StressTestFenwickAgainstNaive(length);
    }
    StressTestFenwickAgainstNaive(1000);
}

// Exercises the engine at its real, compile-time-fixed capacity (10^6, the dataset length),
// including the high indices that the small stress lengths never touch. The reference is a
// plain array summed directly; both accumulate in int32 so the comparison is exact.
TEST(ExternalFenwick, FullCapacityMatchesBruteForce) {
    constexpr size_t length = 1'000'000;
    auto engine = NProust::CreateFenwickPointSumEngine();
    engine->Init(length);

    std::vector<int32_t> reference(length, 0);

    std::mt19937 generator(20260602);
    std::uniform_int_distribution<size_t> positionDistribution(0, length - 1);
    std::uniform_int_distribution<int32_t> valueDistribution(-100, 100);

    // Spread point assignments across the whole array, deliberately including the last index
    // so the tree's top-most nodes are touched.
    for (int round = 0; round < 2000; ++round) {
        const size_t position = positionDistribution(generator);
        const int32_t value = valueDistribution(generator);
        engine->SetPoint(position, value);
        reference[position] = value;
    }
    engine->SetPoint(length - 1, 42);
    reference[length - 1] = 42;

    const auto bruteForceSum = [&](const size_t left, const size_t right) {
        int32_t sum = 0;
        for (size_t i = left; i < right; ++i) {
            sum += reference[i];
        }
        return sum;
    };

    // A mix of ranges: edges, the full array, and a few random windows.
    const std::vector<std::pair<size_t, size_t>> ranges = {
        {0, 1}, {0, length}, {length - 1, length}, {length - 5, length},
        {0, length / 2}, {length / 2, length}, {123456, 654321}, {999000, 1000000},
    };
    for (const auto& [left, right] : ranges) {
        EXPECT_EQ(engine->ComputeRange(left, right), bruteForceSum(left, right))
            << "mismatch on range [" << left << ", " << right << ")";
    }
}
