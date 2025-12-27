#include <gtest/gtest.h>

#include "common/dataset.h"
#include "examples/leetcode_house_robber.h"
#include "examples/leetcode_maximum_subarray.h"
#include "examples/leetcode_third_distinct_maximum.h"

#include "utils.h"

using NProust::CreateRandomIntGenerator;

TEST(Examples, MaxSubarraySum) {
    auto engine = NProustExamples::CreateMaxSumEngine();
    auto bruteForceEngine = NProustExamples::CreateNaiveMaxSumEngine();
    StressTest<int32_t, NProustExamples::TMaxSumState, NProustExamples::TMaxSumStateOperation>(
        1000,
        1000000,
        CreateRandomIntGenerator(-10000, 10000),
        std::move(bruteForceEngine)
    );
}

TEST(Examples, ThirdDistinctMaximum) {
    auto engine = NProustExamples::CreateInverseOrderStatisticsEngine();
    auto bruteForceEngine = NProustExamples::CreateNaiveInverseOrderStatisticsEngine();
    StressTest<int32_t, NProustExamples::TInverseOrderStatisticState, NProustExamples::TInverseOrderStatisticOperation<3>>(
        1000,
        1000000,
        CreateRandomIntGenerator(-1'000'000'000, 1'000'000'000),
        std::move(bruteForceEngine)
    );
}

TEST(Examples, LeetcodeHouseRobber) {
    auto engine = NProustExamples::CreateLeetcodeHouseRobberEngine();
    auto bruteForceEngine = NProustExamples::CreateNaiveLeetcodeHouseRobberEngine();
    StressTest<int32_t, NProustExamples::THouseRobberDPState, NProustExamples::THouseRobberDPStateOperation>(
        1000,
        1000000,
        CreateRandomIntGenerator(-10000, 10000),
        std::move(bruteForceEngine)
    );
}
