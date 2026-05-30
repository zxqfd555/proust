// Based on: https://leetcode.com/problems/maximum-subarray/description/
// but with O(log N) per subarray query time and O(log N) modifications as a bonus ;)

#pragma once

#include "engine/robust/engine.h"
#include "engine/operations.h"

namespace NProustExamples {
    class TNaiveMaxSumEngine;

    struct TMaxSumState {
        int32_t MaxSubarraySum;
        int32_t MaxPrefixSum;
        int32_t MaxSuffixSum;
        int32_t FullSum;
    };

    struct TMaxSumStateOperation {
        template<typename TValue, typename TOperationResult>
        static TOperationResult FromRepeated(const TValue& v, size_t times) noexcept {
            if constexpr (std::is_same_v<TValue, int32_t> && std::is_same_v<TOperationResult, TMaxSumState>) {
                int32_t fullSum = v * times;
                if (v > 0) {
                    return {fullSum, fullSum, fullSum, fullSum};
                } else {
                    return {v, v, v, fullSum};
                }
            } else {
                static_assert(sizeof(TValue) == 0, "Unsupported type for TMaxSumStateOperation::FromRepeated");
            }
        }

        template<typename TOperationResult>
        static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) noexcept {
            if constexpr (std::is_same_v<TOperationResult, TMaxSumState>) {
                const auto newFullSum = lhs.FullSum + rhs.FullSum;

                const auto childrenMaxSum = std::max(lhs.MaxSubarraySum, rhs.MaxSubarraySum);
                const auto newMaxSubarraySum = std::max(childrenMaxSum, lhs.MaxSuffixSum + rhs.MaxPrefixSum);

                const auto newMaxPrefixSum = std::max(lhs.MaxPrefixSum, lhs.FullSum + rhs.MaxPrefixSum);
                const auto newMaxSuffixSum = std::max(rhs.MaxSuffixSum, rhs.FullSum + lhs.MaxSuffixSum);

                return {newMaxSubarraySum, newMaxPrefixSum, newMaxSuffixSum, newFullSum};
            } else {
                static_assert(sizeof(TOperationResult) == 0, "Unsupported type for TMaxSumStateOperation::Merge");
            }
        }

        // Defined out-of-line below, once TNaiveMaxSumEngine is a complete type.
        template<typename TOperationResult, typename TValue>
        static TOperationResult FromMany(const TValue* data, const size_t rangeStart, const size_t rangeEnd) noexcept;
    };

    inline std::unique_ptr<NProust::IRangeUpdateEngine<int32_t, TMaxSumState, TMaxSumStateOperation>> CreateMaxSumEngine() {
        return NProust::CreateRobustRangeUpdateEngine<int32_t, TMaxSumState, TMaxSumStateOperation>();
    };

    // Not really needed for the task, but used it in a unit test when comparing the result against a
    // brute-force solution.
    inline bool operator==(const TMaxSumState& lhs, const TMaxSumState& rhs) {
        return (
            lhs.MaxSubarraySum == rhs.MaxSubarraySum && 
            lhs.MaxPrefixSum == rhs.MaxPrefixSum &&
            lhs.MaxSuffixSum == rhs.MaxSuffixSum && 
            lhs.FullSum == rhs.FullSum
        );
    }

    // A brute force solution to the maximum subarray sum problem, with modifications.
    // An implementation of IEngine, adapted to be used as a dataset generator.
    class TNaiveMaxSumEngine: public NProust::IRangeUpdateEngine<int32_t, TMaxSumState, TMaxSumStateOperation> {
    public:
        virtual void Init(const size_t length) override {
            Elements.resize(length);
            std::fill(Elements.begin(), Elements.end(), 0);
        }

        virtual void SetRange(const size_t left, const size_t right, const int32_t& value) override {
            std::fill(Elements.begin() + left, Elements.begin() + right, value);
        }

        virtual TMaxSumState ComputeRange(const size_t left, const size_t right) const override {
            return TNaiveMaxSumEngine::ComputeStateNaively(Elements.data(), left, right);
        }

        static TMaxSumState ComputeStateNaively(const int32_t* elements, const size_t left, const size_t right) {
            // In fact, a query response is just a maximum subsegment sum,
            // but we'll calculate other parts of TMaxSumState too, just to be sure
            // it's computed correctly and no hidden bugs with a correct answer but
            // incorrect state take place.
            int32_t fullSum = 0;
            int32_t maxSumLeft = elements[left];
            int32_t maxSubarraySum = elements[left];

            {
                int32_t currentSum = 0;
                for (size_t index = left; index < right; ++index) {
                    currentSum += elements[index];
                    maxSubarraySum = std::max(maxSubarraySum, currentSum);
                    currentSum = std::max(currentSum, 0);

                    fullSum += elements[index];
                    maxSumLeft = std::max(maxSumLeft, fullSum);
                }
            }

            // Everything but the maximum suffix sum is computed at this point
            int32_t maxSumRight = fullSum;
            {
                int32_t currentSum = fullSum;
                for (size_t index = left; index + 1 < right; ++index) {
                    currentSum -= elements[index];
                    maxSumRight = std::max(maxSumRight, currentSum);
                }
            }

            return {maxSubarraySum, maxSumLeft, maxSumRight, fullSum};
        }


    private:
        std::vector<int32_t> Elements;
    };

    template<typename TOperationResult, typename TValue>
    TOperationResult TMaxSumStateOperation::FromMany(const TValue* data, const size_t rangeStart, const size_t rangeEnd) noexcept {
        return TNaiveMaxSumEngine::ComputeStateNaively(data, rangeStart, rangeEnd);
    }

    inline std::unique_ptr<NProust::IRangeUpdateEngine<int32_t, TMaxSumState, TMaxSumStateOperation>> CreateNaiveMaxSumEngine() {
        return std::make_unique<TNaiveMaxSumEngine>();
    };
} // namespace NProustExamples
