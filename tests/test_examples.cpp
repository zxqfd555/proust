#include <gtest/gtest.h>

#include "common/dataset.h"
#include "examples/leetcode_house_robber.h"
#include "examples/leetcode_maximum_subarray.h"
#include "examples/leetcode_third_distinct_maximum.h"

#include "utils.h"

using NProust::CreateRandomIntGenerator;

// A brute force solution to the maximum subarray sum problem, with modifications.
// An implementation of IEngine, adapted to be used as a dataset generator.
class TNaiveMaxSumEngine: public NProust::IEngine<int32_t, NProustExamples::TMaxSumState, NProustExamples::TMaxSumStateOperation> {
public:
    using TBase = NProust::IEngine<int32_t, NProustExamples::TMaxSumState, NProustExamples::TMaxSumStateOperation>;

    virtual NProust::TVersion Init(const size_t length) override {
        Elements.resize(length);
        std::fill(Elements.begin(), Elements.end(), 0);
        return 0;  // no recovery is needed, hence ignoring
    }

    virtual NProust::TVersion SetRange(const size_t left, const size_t right, const int32_t& value) override {
        std::fill(Elements.begin() + left, Elements.begin() + right, value);
        return 0;
    }

    virtual NProustExamples::TMaxSumState ComputeRange(const size_t left, const size_t right) const override {
        // In fact, a query response is just a maximum subsegment sum,
        // but we'll calculate other parts of TMaxSumState too, just to be sure
        // it's computed correctly and no hidden bugs with a correct answer but
        // incorrect state take place.
        int32_t fullSum = 0;
        int32_t maxSumLeft = Elements[left];
        int32_t maxSubarraySum = Elements[left];

        {
            int32_t currentSum = 0;
            for (size_t index = left; index < right; ++index) {
                currentSum += Elements[index];
                maxSubarraySum = std::max(maxSubarraySum, currentSum);
                currentSum = std::max(currentSum, 0);

                fullSum += Elements[index];
                maxSumLeft = std::max(maxSumLeft, fullSum);
            }
        }

        // Everything but the maximum suffix sum is computed at this point
        int32_t maxSumRight = fullSum;
        {
            int32_t currentSum = fullSum;
            for (size_t index = left; index + 1 < right; ++index) {
                currentSum -= Elements[index];
                maxSumRight = std::max(maxSumRight, currentSum);
            }
        }

        return {maxSubarraySum, maxSumLeft, maxSumRight, fullSum};
    }

private:
    std::vector<int32_t> Elements;
};

TEST(Examples, MaxSubarraySum) {
    auto engine = NProustExamples::CreateMaxSumEngine();
    std::unique_ptr<TNaiveMaxSumEngine::TBase> bruteForceEngine = std::make_unique<TNaiveMaxSumEngine>(); 
    StressTest<int32_t, NProustExamples::TMaxSumState, NProustExamples::TMaxSumStateOperation>(
        1000,
        1000000,
        CreateRandomIntGenerator(-10000, 10000),
        std::move(bruteForceEngine)
    );
}

// A brute force solution to the inverse order statistic problem, with modifications.
// An implementation of IEngine, adapted to be used as a dataset generator.
class TNaiveInverseOrderStatisticsEngine: public NProust::IEngine<int32_t, NProustExamples::TInverseOrderStatisticState, NProustExamples::TInverseOrderStatisticOperation> {
public:
    using TBase = NProust::IEngine<int32_t, NProustExamples::TInverseOrderStatisticState, NProustExamples::TInverseOrderStatisticOperation>;

    virtual NProust::TVersion Init(const size_t length) override {
        Elements.resize(length);
        std::fill(Elements.begin(), Elements.end(), 0);
        return 0;  // no recovery is needed, hence ignoring
    }

    virtual NProust::TVersion SetRange(const size_t left, const size_t right, const int32_t& value) override {
        std::fill(Elements.begin() + left, Elements.begin() + right, value);
        return 0;
    }

    virtual NProustExamples::TInverseOrderStatisticState ComputeRange(const size_t left, const size_t right) const override {
        using NProustExamples::INVERSE_ORDER_STATISTICS_REQUIRED;

        std::vector<int32_t> maximums;
        maximums.reserve(INVERSE_ORDER_STATISTICS_REQUIRED + 1);

        for (size_t index = left; index < right; ++index) {
            maximums.push_back(Elements[index]);
            size_t maxIndex = maximums.size() - 1;
            while (maxIndex > 0 && maximums[maxIndex] > maximums[maxIndex - 1]) {
                std::swap(maximums[maxIndex], maximums[maxIndex--]);
            }

            if (maxIndex > 0 && maximums[maxIndex] == maximums[maxIndex - 1]) {
                for (size_t index = maxIndex; index + 1 < maximums.size(); ++index) {
                    maximums[index] = maximums[index + 1];
                }
                maximums.pop_back();
            }

            if (maximums.size() > INVERSE_ORDER_STATISTICS_REQUIRED) {
                maximums.resize(INVERSE_ORDER_STATISTICS_REQUIRED);
            }
        }

        return maximums;
    }

private:
    std::vector<int32_t> Elements;
};

TEST(Examples, ThirdDistinctMaximum) {
    auto engine = NProustExamples::CreateInverseOrderStatisticsEngine();
    std::unique_ptr<TNaiveInverseOrderStatisticsEngine::TBase> bruteForceEngine = std::make_unique<TNaiveInverseOrderStatisticsEngine>();
    StressTest<int32_t, NProustExamples::TInverseOrderStatisticState, NProustExamples::TInverseOrderStatisticOperation>(
        1000,
        1000000,
        CreateRandomIntGenerator(-1'000'000'000, 1'000'000'000),
        std::move(bruteForceEngine)
    );
}

// A brute force solution to the Leetcode "House Robber" problem, with modifications.
// An implementation of IEngine, adapted to be used as a dataset generator.
class TLeetcodeHouseRobberEngine: public NProust::IEngine<int32_t, NProustExamples::THouseRobberDPState, NProustExamples::THouseRobberDPStateOperation> {
public:
    using TBase = NProust::IEngine<int32_t, NProustExamples::THouseRobberDPState, NProustExamples::THouseRobberDPStateOperation>;

    virtual NProust::TVersion Init(const size_t length) override {
        Elements.resize(length);
        std::fill(Elements.begin(), Elements.end(), 0);
        return 0;  // no recovery is needed, hence ignoring
    }

    virtual NProust::TVersion SetRange(const size_t left, const size_t right, const int32_t& value) override {
        std::fill(Elements.begin() + left, Elements.begin() + right, value);
        return 0;
    }

    virtual NProustExamples::THouseRobberDPState ComputeRange(const size_t left, const size_t right) const override {
        return {Solve(left, right), Solve(left, right - 1), Solve(left + 1, right), Solve(left + 1, right - 1)};
    }

private:
    int32_t Solve(const size_t left, const size_t right) const {
        if (left >= right) {
            return 0;
        }
        if (left + 1 == right) {
            return Elements[left];
        }

        int32_t currentOptimum = std::max(Elements[left], Elements[left + 1]);
        int32_t prevOptimum = Elements[left];
        for (size_t index = left + 2; index < right; ++index) {
            int32_t newCurrentOptimum = std::max(currentOptimum, prevOptimum + Elements[index]);
            prevOptimum = currentOptimum;
            currentOptimum = newCurrentOptimum;
        }

        return currentOptimum;
    }

    std::vector<int32_t> Elements;
};

TEST(Examples, LeetcodeHouseRobber) {
    auto engine = NProustExamples::CreateLeetcodeHouseRobberEngine();
    std::unique_ptr<TNaiveInverseOrderStatisticsEngine::TBase> bruteForceEngine = std::make_unique<TNaiveInverseOrderStatisticsEngine>();
    StressTest<int32_t, NProustExamples::TInverseOrderStatisticState, NProustExamples::TInverseOrderStatisticOperation>(
        1000,
        1000000,
        CreateRandomIntGenerator(-10000, 10000),
        std::move(bruteForceEngine)
    );
}
