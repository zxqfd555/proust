// Based on: https://leetcode.com/problems/house-robber/description/
// but with O(log N) per subarray query time and O(log N) modifications as a bonus ;)

#pragma once

#include "engine/segtree/segtree.h"
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
        static TOperationResult Repeat(const TValue& v, size_t times) {
            if constexpr (std::is_same_v<TValue, int32_t> && std::is_same_v<TOperationResult, THouseRobberDPState>) {
                const int32_t fullResult = v * ((times + 1) >> 1);
                const int32_t exceptOneResult = v * (times >> 1);
                const int32_t exceptCornersResult = v * ((times - 1) >> 1);
                return {fullResult, exceptOneResult, exceptOneResult, exceptCornersResult};
            } else {
                static_assert(sizeof(TValue) == 0, "Unsupported type for THouseRobberDPOperation::Repeat");
            }
        }

        template<typename TOperationResult>
        static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) {
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

    std::unique_ptr<NProust::IEngine<int32_t, THouseRobberDPState, THouseRobberDPStateOperation>> CreateLeetcodeHouseRobberEngine() {
        return NProust::CreateRobustEngine<int32_t, THouseRobberDPState, THouseRobberDPStateOperation>();
    };
} // namespace NProustExamples