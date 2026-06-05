#include <memory>
#include <random>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "engine/external/algorithmica.h"
#include "engine/operations.h"

#include "utils.h"

using NProust::CreateRandomIntGenerator;

namespace {
    // The Algorithmica wide segment tree is fixed to the int32 sum problem, so there is a single
    // instantiation worth exercising. We still validate it against the naive engine over the
    // corner-case lengths (1..16) and a larger length, ten thousand random ops each.
    using TDefaultTopology = NProust::TPreorderNodeTopology<0>;
    using TDefaultLayout = NProust::TSoALayout<int32_t, int32_t, std::nullopt>;

    void StressTestAlgorithmicaAgainstNaive(const size_t length) {
        StressTest<int32_t, int32_t, NProust::TSumOperation, TDefaultTopology, TDefaultLayout, 0>(
            length,
            N_QUERIES_IN_STRESS_TEST,
            CreateRandomIntGenerator(-100, 100),
            ETestedEngineType::AlgorithmicaWideTree
        );
    }
} // namespace

TEST(AlgorithmicaWideTree, IntSumStress) {
    for (size_t length = 1; length <= 16; ++length) {
        StressTestAlgorithmicaAgainstNaive(length);
    }
    StressTestAlgorithmicaAgainstNaive(1000);
}

// Exercises the engine at its real, compile-time-fixed capacity (10^6, the dataset length). This
// is also where the padding to a power of B matters: the full-range query Sum(length) must stay
// in bounds. The reference is a plain array summed directly; both accumulate in int32.
TEST(AlgorithmicaWideTree, FullCapacityMatchesBruteForce) {
    constexpr size_t length = 1'000'000;
    auto engine = NProust::CreateAlgorithmicaWideSegmentSumEngine();
    engine->Init(length);

    std::vector<int32_t> reference(length, 0);

    std::mt19937 generator(20260602);
    std::uniform_int_distribution<size_t> positionDistribution(0, length - 1);
    std::uniform_int_distribution<int32_t> valueDistribution(-100, 100);

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

    // Edges, the full array (exercises Sum(length)), and a few interior windows.
    const std::vector<std::pair<size_t, size_t>> ranges = {
        {0, 1}, {0, length}, {length - 1, length}, {length - 5, length},
        {0, length / 2}, {length / 2, length}, {123456, 654321}, {999000, 1000000},
    };
    for (const auto& [left, right] : ranges) {
        EXPECT_EQ(engine->ComputeRange(left, right), bruteForceSum(left, right))
            << "mismatch on range [" << left << ", " << right << ")";
    }
}
