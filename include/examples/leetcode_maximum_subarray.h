// Based on: https://leetcode.com/problems/maximum-subarray/description/
// but with O(log N) per subarray query time and O(log N) modifications as a bonus ;)

#pragma once

#include <limits>

#include "engine/robust/engine.h"
#include "engine/operations.h"

#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
    #define PROUST_HAS_MAX_SUM_AVX2_BLOCK
    #include <immintrin.h>
#endif

namespace NProustExamples {
    class TNaiveMaxSumEngine;

    struct TMaxSumState {
        int32_t MaxSubarraySum;
        int32_t MaxPrefixSum;
        int32_t MaxSuffixSum;
        int32_t FullSum;
    };

#ifdef PROUST_HAS_MAX_SUM_AVX2_BLOCK
    namespace NMaxSumDetail {
        // Merge for the two states in each 128-bit half of the arguments. A state's fields are
        // {best, prefix, suffix, sum}, and all four results are "the larger of a sum and an old field":
        //   best   = max(max(a.best, c.best), a.suffix + c.prefix)
        //   prefix = max(a.prefix,            a.sum    + c.prefix)
        //   suffix = max(c.suffix,            a.suffix + c.sum)
        //   sum    = max(lowest,              a.sum    + c.sum)
        [[gnu::target("avx2")]] inline __m256i MergePairs(const __m256i a, const __m256i c) noexcept {
            const __m256i sums = _mm256_add_epi32(_mm256_shuffle_epi32(a, _MM_SHUFFLE(3, 2, 3, 2)), _mm256_shuffle_epi32(c, _MM_SHUFFLE(3, 3, 1, 1)));
            __m256i kept = _mm256_max_epi32(a, c);
            kept = _mm256_blend_epi32(kept, a, 0x22);
            kept = _mm256_blend_epi32(kept, c, 0x44);
            kept = _mm256_blend_epi32(kept, _mm256_set1_epi32(std::numeric_limits<int32_t>::min()), 0x88);
            return _mm256_max_epi32(sums, kept);
        }

        // block[first] and block[first + 2] in one vector, block[first + 1] and block[first + 3] in the
        // other: merging them gives the two merged neighbours in one vector.
        [[gnu::target("avx2")]] inline __m256i MergeFour(const TMaxSumState* block) noexcept {
            const auto load = [](const TMaxSumState* state) [[gnu::target("avx2")]] { return _mm_loadu_si128(reinterpret_cast<const __m128i*>(state)); };
            return MergePairs(
                _mm256_inserti128_si256(_mm256_castsi128_si256(load(block)), load(block + 2), 1),
                _mm256_inserti128_si256(_mm256_castsi128_si256(load(block + 1)), load(block + 3), 1)
            );
        }

        // Two vectors of two neighbouring states each (w, x), (y, z) -> the vector (w x, y z).
        [[gnu::target("avx2")]] inline __m256i MergeTwoVectors(const __m256i first, const __m256i second) noexcept {
            return MergePairs(_mm256_permute2x128_si256(first, second, 0x20), _mm256_permute2x128_si256(first, second, 0x31));
        }

        // 16 states in order: 16 -> 8 (in 4 vectors) -> 4 (in 2) -> 2 (in 1) -> 1.
        [[gnu::target("avx2")]] inline TMaxSumState MergeBlockAvx2(const TMaxSumState* block) noexcept {
            const __m256i low = MergeTwoVectors(MergeFour(block), MergeFour(block + 4));
            const __m256i high = MergeTwoVectors(MergeFour(block + 8), MergeFour(block + 12));
            const __m256i two = MergeTwoVectors(low, high);
            const __m256i one = MergePairs(two, _mm256_permute2x128_si256(two, two, 0x01));
            TMaxSumState result;
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&result), _mm256_castsi256_si128(one));
            return result;
        }
    } // namespace NMaxSumDetail
#endif

    struct TMaxSumStateOperation {
#if defined(PROUST_HAS_MAX_SUM_AVX2_BLOCK) && defined(__AVX2__)
        template<typename TOperationResult>
        static TOperationResult MergeBlock(const TOperationResult* block) noexcept {
            return NMaxSumDetail::MergeBlockAvx2(block);
        }
#endif

        template<typename TValue, typename TOperationResult>
        static TOperationResult FromRepeated(const TValue& v, size_t times) noexcept {
            if constexpr (std::is_same_v<TValue, int32_t> && std::is_same_v<TOperationResult, TMaxSumState>) {
                // Branchless on purpose: the sign of `v` is data-random, and this is called on every
                // level of the range engine's descent (AssignWholeNode), where a mispredicted jump
                // costs a pipeline flush. A ternary compiles to cmov.
                const int32_t fullSum = v * times;
                const int32_t bestPartialSum = v > 0 ? fullSum : v;
                return {bestPartialSum, bestPartialSum, bestPartialSum, fullSum};
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

    // The task modifies one element at a time, and for that the point engine is about twice as fast
    // as the range one below; prefer it unless whole intervals get assigned.
    inline std::unique_ptr<NProust::IPointUpdateEngine<int32_t, TMaxSumState, TMaxSumStateOperation>> CreateMaxSumPointEngine() {
        return NProust::CreateRobustPointUpdateEngine<int32_t, TMaxSumState, TMaxSumStateOperation>();
    };

    inline std::unique_ptr<NProust::IRangeUpdateEngine<int32_t, TMaxSumState, TMaxSumStateOperation>> CreateMaxSumEngine() {
        return NProust::CreateRobustRangeUpdateEngine<int32_t, TMaxSumState, TMaxSumStateOperation>();
    };

    // Instantiate these directly to avoid the factories' indirect call per operation.
    using TMaxSumPointEngine = NProust::TRobustPointUpdateEngine<int32_t, TMaxSumState, TMaxSumStateOperation>;
    using TMaxSumRangeEngine = NProust::TRobustRangeUpdateEngine<int32_t, TMaxSumState, TMaxSumStateOperation>;

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
        // A single forward pass, unlike the naive engine's two-pass reference above (kept two-pass
        // on purpose: the ground truth should not share code shapes with the optimized path). The
        // suffix maximum falls out of the forward pass for free: Kadane's "best subarray ending
        // here" evaluated at the last element *is* the best non-empty suffix sum.
        int32_t fullSum = data[rangeStart];
        int32_t maxPrefixSum = fullSum;
        int32_t maxSubarraySum = fullSum;
        int32_t bestEndingHere = fullSum;

        for (size_t index = rangeStart + 1; index < rangeEnd; ++index) {
            const int32_t element = data[index];
            const int32_t extended = bestEndingHere + element;
            bestEndingHere = extended > element ? extended : element;
            maxSubarraySum = maxSubarraySum > bestEndingHere ? maxSubarraySum : bestEndingHere;
            fullSum += element;
            maxPrefixSum = maxPrefixSum > fullSum ? maxPrefixSum : fullSum;
        }

        return {maxSubarraySum, maxPrefixSum, bestEndingHere, fullSum};
    }

    inline std::unique_ptr<NProust::IRangeUpdateEngine<int32_t, TMaxSumState, TMaxSumStateOperation>> CreateNaiveMaxSumEngine() {
        return std::make_unique<TNaiveMaxSumEngine>();
    };
} // namespace NProustExamples
