#include <gtest/gtest.h>

#include "common/dataset.h"
#include "examples/leetcode_house_robber.h"
#include "examples/leetcode_maximum_subarray.h"
#include "examples/leetcode_third_distinct_maximum.h"

#include "utils.h"

using NProust::CreateRandomIntGenerator;

TEST(Examples, MaxSubarraySum) {
    // Range modification version
    {
        auto bruteForceEngine = NProustExamples::CreateNaiveMaxSumEngine();
        StressTest<int32_t, NProustExamples::TMaxSumState, NProustExamples::TMaxSumStateOperation>(
            1000,
            1000000,
            CreateRandomIntGenerator(-10000, 10000),
            NProust::EBaseGeneratorType::Range,
            std::move(bruteForceEngine)
        );
    }

    // Point modification version
    {
        auto bruteForceEngine = NProustExamples::CreateNaiveMaxSumEngine();
        StressTest<int32_t, NProustExamples::TMaxSumState, NProustExamples::TMaxSumStateOperation>(
            1000,
            1000000,
            CreateRandomIntGenerator(-10000, 10000),
            NProust::EBaseGeneratorType::Point,
            std::move(bruteForceEngine)
        );
    }
}

TEST(Examples, ThirdDistinctMaximum) {
    // Range modification version
    {
        auto bruteForceEngine = NProustExamples::CreateNaiveInverseOrderStatisticsEngine();
        StressTest<int32_t, NProustExamples::TInverseOrderStatisticState, NProustExamples::TInverseOrderStatisticOperation<3>>(
            1000,
            1000000,
            CreateRandomIntGenerator(-1'000'000'000, 1'000'000'000),
            NProust::EBaseGeneratorType::Range,
            std::move(bruteForceEngine)
        );
    }

    // Point modification version
    {
        // Check the common error first
        auto robustEngine = NProust::CreateRobustPointUpdateEngine<int32_t, NProustExamples::TInverseOrderStatisticState, NProustExamples::TInverseOrderStatisticOperation<3>>();
        robustEngine->Init(10);
        EXPECT_EQ(robustEngine->ComputeRange(0, 1), std::vector<int32_t>{0});

        auto bruteForceEngine = NProustExamples::CreateNaiveInverseOrderStatisticsEngine();
        StressTest<int32_t, NProustExamples::TInverseOrderStatisticState, NProustExamples::TInverseOrderStatisticOperation<3>>(
            1000,
            1000000,
            CreateRandomIntGenerator(-1'000'000'000, 1'000'000'000),
            NProust::EBaseGeneratorType::Point,
            std::move(bruteForceEngine)
        );
    }
}

TEST(Examples, LeetcodeHouseRobber) {
    // Range modification version
    {
        auto bruteForceEngine = NProustExamples::CreateNaiveLeetcodeHouseRobberEngine();
        StressTest<int32_t, NProustExamples::THouseRobberDPState, NProustExamples::THouseRobberDPStateOperation>(
            1000,
            1000000,
            CreateRandomIntGenerator(-10000, 10000),
            NProust::EBaseGeneratorType::Range,
            std::move(bruteForceEngine)
        );
    }

    // Point modification version
    {
        auto bruteForceEngine = NProustExamples::CreateNaiveLeetcodeHouseRobberEngine();
        StressTest<int32_t, NProustExamples::THouseRobberDPState, NProustExamples::THouseRobberDPStateOperation>(
            1000,
            1000000,
            CreateRandomIntGenerator(-10000, 10000),
            NProust::EBaseGeneratorType::Point,
            std::move(bruteForceEngine)
        );
    }
}
