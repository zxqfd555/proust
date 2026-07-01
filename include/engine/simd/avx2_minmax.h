#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(__AVX2__)
    #include <immintrin.h>
#endif

namespace NProust {
    // Max operations for the int32 range-max problem, in three flavours that let
    // us separate "adding a bulk FromMany path" from "hand-writing SIMD":
    //
    //   TMaxOperation          (engine/operations.h) - baseline, NO FromMany, so
    //                          the engine falls back to its generic per-element
    //                          Merge loop.
    //   TMaxOperationFromMany  - same result, but exposes a scalar FromMany so the
    //                          engine takes its bulk-reduction path; left for the
    //                          compiler to auto-vectorize.
    //   TMaxOperationAvx2      - FromMany hand-written with _mm256_max_epi32 and 4
    //                          independent accumulators (int32 fast path; scalar
    //                          fallback otherwise / when __AVX2__ is off).
    //
    // Max is order-independent, so all three produce identical results to the
    // naive ground truth; the benchmark validates every query.

    struct TMaxOperationFromMany {
        template<typename TValue, typename TOperationResult>
        inline static TOperationResult FromRepeated(const TValue& v, const size_t) noexcept(noexcept(TOperationResult(v))) {
            return TOperationResult(v);
        }

        template<typename TOperationResult>
        inline static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) noexcept(noexcept(lhs > rhs)) {
            return lhs > rhs ? lhs : rhs;
        }

        template<typename TValue, typename TOperationResult>
        inline static TOperationResult FromMany(const TValue* data, size_t fromIndex, size_t toIndex) {
            TOperationResult result = data[fromIndex];
            for (size_t index = fromIndex + 1; index < toIndex; ++index) {
                result = data[index] > result ? data[index] : result;
            }
            return result;
        }
    };

    struct TMaxOperationAvx2 {
        template<typename TValue, typename TOperationResult>
        inline static TOperationResult FromRepeated(const TValue& v, const size_t) noexcept(noexcept(TOperationResult(v))) {
            return TOperationResult(v);
        }

        template<typename TOperationResult>
        inline static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) noexcept(noexcept(lhs > rhs)) {
            return lhs > rhs ? lhs : rhs;
        }

        template<typename TValue, typename TOperationResult>
        inline static TOperationResult FromMany(const TValue* data, size_t fromIndex, size_t toIndex) {
#if defined(__AVX2__)
            if constexpr (std::is_same_v<TValue, int32_t> && std::is_same_v<TOperationResult, int32_t>) {
                return MaxInt32(data + fromIndex, toIndex - fromIndex);
            } else
#endif
            {
                TOperationResult result = data[fromIndex];
                for (size_t index = fromIndex + 1; index < toIndex; ++index) {
                    result = data[index] > result ? data[index] : result;
                }
                return result;
            }
        }

#if defined(__AVX2__)
    private:
        // Horizontal max of `count` (>= 1) int32 values; 4 independent AVX2
        // accumulators to hide the compare latency, then a lane fold + scalar tail.
        inline static int32_t MaxInt32(const int32_t* data, size_t count) noexcept {
            const __m256i kMin = _mm256_set1_epi32(std::numeric_limits<int32_t>::min());
            __m256i acc0 = kMin, acc1 = kMin, acc2 = kMin, acc3 = kMin;

            size_t index = 0;
            for (; index + 32 <= count; index += 32) {
                acc0 = _mm256_max_epi32(acc0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + index +  0)));
                acc1 = _mm256_max_epi32(acc1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + index +  8)));
                acc2 = _mm256_max_epi32(acc2, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + index + 16)));
                acc3 = _mm256_max_epi32(acc3, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + index + 24)));
            }
            for (; index + 8 <= count; index += 8) {
                acc0 = _mm256_max_epi32(acc0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + index)));
            }

            const __m256i acc = _mm256_max_epi32(_mm256_max_epi32(acc0, acc1), _mm256_max_epi32(acc2, acc3));

            __m128i folded = _mm_max_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
            folded = _mm_max_epi32(folded, _mm_shuffle_epi32(folded, _MM_SHUFFLE(1, 0, 3, 2)));
            folded = _mm_max_epi32(folded, _mm_shuffle_epi32(folded, _MM_SHUFFLE(2, 3, 0, 1)));
            int32_t result = _mm_cvtsi128_si32(folded);

            for (; index < count; ++index) {
                result = data[index] > result ? data[index] : result;
            }
            return result;
        }
#endif
    };
} // namespace NProust
