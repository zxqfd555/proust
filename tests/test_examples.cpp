#include <gtest/gtest.h>

#include "common/dataset.h"
#include "leetcode_maximum_subarray.h"

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
        return 0;  // no recovery is needed, hence
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
