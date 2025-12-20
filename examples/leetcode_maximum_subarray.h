// Based on: https://leetcode.com/problems/maximum-subarray/description/
// but with O(log N) per query time and O(log N) modifications as a bonus ;)

#pragma once

#include "engine/segtree/segtree.h"
#include "engine/operations.h"

namespace NProustExamples {
    struct TMaxSumState {
        int32_t MaxSubarraySum;
        int32_t MaxSumFromLeft;
        int32_t MaxSumFromRight;
        int32_t FullSum;
    };

    struct TMaxSumStateOperation {
        template<typename TValue, typename TOperationResult>
        static TOperationResult Repeat(const TValue& v, size_t times) {
            if constexpr (std::is_same_v<TValue, int32_t> && std::is_same_v<TOperationResult, TMaxSumState>) {
                int32_t fullSum = v * times;
                if (v > 0) {
                    return {fullSum, fullSum, fullSum, fullSum};
                } else {
                    return {v, v, v, fullSum};
                }
            } else {
                static_assert(sizeof(TValue) == 0, "Unsupported type for TStateOperation::Repeat");
            }
        }

        template<typename TOperationResult>
        static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) {
            if constexpr (std::is_same_v<TOperationResult, TMaxSumState>) {
                const auto newFullSum = lhs.FullSum + rhs.FullSum;

                const auto childrenMaxSum = std::max(lhs.MaxSubarraySum, rhs.MaxSubarraySum);
                const auto newMaxSubarraySum = std::max(childrenMaxSum, lhs.MaxSumFromRight + rhs.MaxSumFromLeft);

                const auto newMaxSumFromLeft = std::max(lhs.MaxSumFromLeft, lhs.FullSum + rhs.MaxSumFromLeft);
                const auto newMaxSumFromRight = std::max(rhs.MaxSumFromRight, rhs.FullSum + lhs.MaxSumFromRight);

                return {newMaxSubarraySum, newMaxSumFromLeft, newMaxSumFromRight, newFullSum};
            } else {
                static_assert(sizeof(TOperationResult) == 0, "Unsupported type for TMaxSumStateOperation::Merge");
            }
        }
    };

    std::unique_ptr<NProust::IEngine<int32_t, TMaxSumState, TMaxSumStateOperation>> CreateMaxSumEngine() {
        return NProust::CreateRobustEngine<int32_t, TMaxSumState, TMaxSumStateOperation>();
    };

    // Not really needed for the task, but used it in a unit test when comparing the result against a
    // brute-force solution.
    bool operator==(const NProustExamples::TMaxSumState& lhs, const NProustExamples::TMaxSumState& rhs) {
        return (
            lhs.MaxSubarraySum == rhs.MaxSubarraySum && 
            lhs.MaxSumFromLeft == rhs.MaxSumFromLeft &&
            lhs.MaxSumFromRight == rhs.MaxSumFromRight && 
            lhs.FullSum == rhs.FullSum
        );
    }
} // namespace NProustExamples
