// Based on: https://leetcode.com/problems/third-maximum-number/description/
// but with O(log N) per subarray query time and O(log N) modifications as a bonus ;)

#pragma once

#include "engine/segtree/segtree.h"
#include "engine/operations.h"

namespace NProustExamples {
    const size_t INVERSE_ORDER_STATISTICS_REQUIRED = 3;
    using TInverseOrderStatisticState = std::vector<int32_t>;

    struct TInverseOrderStatisticOperation {
        template<typename TValue, typename TOperationResult>
        static TOperationResult Repeat(const TValue& v, size_t times) {
            if constexpr (std::is_same_v<TValue, int32_t> && std::is_same_v<TOperationResult, TInverseOrderStatisticState>) {
                // Since each value is counted only once, the result contains only a single element.
                return std::vector<int32_t>(1, v);
            } else {
                static_assert(sizeof(TValue) == 0, "Unsupported type for TOrderStatisticOperation::Repeat");
            }
        }

        template<typename TOperationResult>
        static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) {
            if constexpr (std::is_same_v<TOperationResult, TInverseOrderStatisticState>) {
                std::vector<int32_t> newMaximums;
                newMaximums.reserve(INVERSE_ORDER_STATISTICS_REQUIRED);

                auto leftCandidateIt = lhs.begin();
                auto rightCandidateIt = rhs.begin();
                for (size_t index = 0; index < INVERSE_ORDER_STATISTICS_REQUIRED && (leftCandidateIt != lhs.end() || rightCandidateIt != rhs.end()); ++index) {
                    if (leftCandidateIt == lhs.end() || (rightCandidateIt < rhs.end() && *rightCandidateIt >= *leftCandidateIt)) {
                        if (leftCandidateIt < lhs.end() && *leftCandidateIt == *rightCandidateIt) {
                            // Avoiding duplicates
                            ++leftCandidateIt;
                        }
                        newMaximums.push_back(*(rightCandidateIt++));
                    } else if (rightCandidateIt == rhs.end() || (leftCandidateIt < lhs.end() && *leftCandidateIt > *rightCandidateIt)) {
                        newMaximums.push_back(*(leftCandidateIt++));
                    }
                }

                return newMaximums;
            } else {
                static_assert(sizeof(TOperationResult) == 0, "Unsupported type for TOrderStatisticOperation::Merge");
            }
        }
    };

    inline std::unique_ptr<NProust::IEngine<int32_t, TInverseOrderStatisticState, TInverseOrderStatisticOperation>> CreateInverseOrderStatisticsEngine() {
        return NProust::CreateRobustEngine<int32_t, TInverseOrderStatisticState, TInverseOrderStatisticOperation>();
    };
} // namespace NProustExamples
