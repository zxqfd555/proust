// Based on: https://leetcode.com/problems/house-robber/description/
// but with O(log N) per subarray query time and O(log N) modifications as a bonus ;)

#pragma once

#include "engine/robust/engine.h"
#include "engine/operations.h"

namespace NProustExamples {
    struct THouseRobberDPState {
        int32_t FullResult;
        int32_t ExceptEndResult;
        int32_t ExceptStartResult;
        int32_t ExceptCornersResult;
    };

    struct THouseRobberDPStateOperation {
        template<typename TValue, typename TOperationResult>
        static TOperationResult Repeat(const TValue& v, size_t times) noexcept {
            if constexpr (std::is_same_v<TValue, int32_t> && std::is_same_v<TOperationResult, THouseRobberDPState>) {
                const int32_t fullResult = std::max((int32_t)0, v * (int32_t)((times + 1) >> 1));
                const int32_t exceptOneResult = std::max((int32_t)0, v * (int32_t)(times >> 1));
                const int32_t exceptCornersResult = std::max((int32_t)0, v * (int32_t)((times - 1) >> 1));
                return {fullResult, exceptOneResult, exceptOneResult, exceptCornersResult};
            } else {
                static_assert(sizeof(TValue) == 0, "Unsupported type for THouseRobberDPOperation::Repeat");
            }
        }

        template<typename TOperationResult>
        static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) noexcept {
            if constexpr (std::is_same_v<TOperationResult, THouseRobberDPState>) {
                const int32_t newFullResult = std::max(
                    lhs.FullResult + rhs.ExceptStartResult, lhs.ExceptEndResult + rhs.FullResult
                );
                const int32_t newExceptEndResult = std::max(
                    lhs.FullResult + rhs.ExceptCornersResult, lhs.ExceptEndResult + rhs.ExceptEndResult
                );
                const int32_t newExceptStartResult = std::max(
                    lhs.ExceptCornersResult + rhs.FullResult, lhs.ExceptStartResult + rhs.ExceptStartResult
                );
                const int32_t newExceptCornersResult = std::max(
                    lhs.ExceptCornersResult + rhs.ExceptEndResult, lhs.ExceptStartResult + rhs.ExceptCornersResult
                );
                return {newFullResult, newExceptEndResult, newExceptStartResult, newExceptCornersResult};
             } else {
                static_assert(sizeof(TOperationResult) == 0, "Unsupported type for THouseRobberDPOperation::Merge");
            }
        }
    };

    inline std::unique_ptr<NProust::IRangeUpdateEngine<int32_t, THouseRobberDPState, THouseRobberDPStateOperation>> CreateLeetcodeHouseRobberEngine() {
        return NProust::CreateRobustRangeUpdateEngine<int32_t, THouseRobberDPState, THouseRobberDPStateOperation>();
    };

    // Not really needed for the task, but used it in a unit test when comparing the result against a
    // brute-force solution.
    inline bool operator==(const THouseRobberDPState& lhs, const THouseRobberDPState& rhs) {
        return (
            lhs.FullResult == rhs.FullResult && 
            lhs.ExceptEndResult == rhs.ExceptEndResult &&
            lhs.ExceptStartResult == rhs.ExceptStartResult && 
            lhs.ExceptCornersResult == rhs.ExceptCornersResult
        );
    }

    // A brute force solution to the Leetcode "House Robber" problem, with modifications.
    // An implementation of IEngine, adapted to be used as a dataset generator.
    class TNaiveLeetcodeHouseRobberEngine: public NProust::IRangeUpdateEngine<int32_t, THouseRobberDPState, THouseRobberDPStateOperation> {
    public:
        virtual NProust::TVersion Init(const size_t length) override {
            Elements.resize(length);
            std::fill(Elements.begin(), Elements.end(), 0);
            return 0;  // no recovery is needed, hence ignoring
        }

        virtual NProust::TVersion SetRange(const size_t left, const size_t right, const int32_t& value) override {
            std::fill(Elements.begin() + left, Elements.begin() + right, value);
            return 0;
        }

        virtual THouseRobberDPState ComputeRange(const size_t left, const size_t right) const override {
            return {Solve(left, right), Solve(left, right - 1), Solve(left + 1, right), Solve(left + 1, right - 1)};
        }

    private:
        int32_t Solve(const size_t left, const size_t right) const {
            if (left >= right) {
                return 0;
            }

            int32_t currentOptimum = std::max(Elements[left], 0);
            int32_t prevOptimum = 0;
            for (size_t index = left + 1; index < right; ++index) {
                int32_t newCurrentOptimum = std::max(currentOptimum, prevOptimum + Elements[index]);
                prevOptimum = currentOptimum;
                currentOptimum = newCurrentOptimum;
            }

            return currentOptimum;
        }

        std::vector<int32_t> Elements;
    };

    inline std::unique_ptr<NProust::IRangeUpdateEngine<int32_t, THouseRobberDPState, THouseRobberDPStateOperation>> CreateNaiveLeetcodeHouseRobberEngine() {
        return std::make_unique<TNaiveLeetcodeHouseRobberEngine>();
    };
} // namespace NProustExamples