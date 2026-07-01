#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(__AVX2__)
    #include <immintrin.h>
#endif

namespace NProust {
    // AVX2-accelerated variant of TSumOperation.
    //
    // Semantically identical to TSumOperation: FromRepeated/Merge/Undo are the
    // same scalar definitions, so any (TValue, TOperationResult) pair that works
    // with TSumOperation works here too. The only difference is FromMany, the one
    // bulk per-element reduction in the engine (it runs over a leaf's elements in
    // both ComputeRange and SetRange). For the int32->int32 case we replace the
    // scalar accumulation with explicit AVX2 intrinsics.
    //
    // Correctness note: two's-complement addition is associative and commutative
    // modulo 2^N, so summing 8 SIMD lanes independently and then folding them
    // yields exactly the same 32-bit result (including on overflow/wraparound) as
    // the strictly sequential scalar loop. The two implementations are therefore
    // bit-for-bit interchangeable, which is what lets the benchmark validate the
    // AVX2 path against the scalar ground truth on every query.
    struct TSumOperationAvx2 {
        template<typename TValue, typename TOperationResult>
        inline static TOperationResult FromRepeated(const TValue& v, const size_t times) noexcept(noexcept(TOperationResult(v * times))) {
            return TOperationResult(v * times);
        }

        template<typename TOperationResult>
        inline static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) noexcept(noexcept(lhs + rhs)) {
            return lhs + rhs;
        }

        template<typename TOperationResult>
        inline static TOperationResult Undo(const TOperationResult& lhs, const TOperationResult& rhs) noexcept(noexcept(lhs - rhs)) {
            return lhs - rhs;
        }

        template<typename TValue, typename TOperationResult>
        inline static TOperationResult FromMany(const TValue* data, size_t fromIndex, size_t toIndex) {
#if defined(__AVX2__)
            if constexpr (std::is_same_v<TValue, int32_t> && std::is_same_v<TOperationResult, int32_t>) {
                return SumInt32(data + fromIndex, toIndex - fromIndex);
            } else
#endif
            {
                TOperationResult result = 0;
                for (size_t index = fromIndex; index < toIndex; ++index) {
                    result += data[index];
                }
                return result;
            }
        }

#if defined(__AVX2__)
    private:
        // Horizontally sums `count` 32-bit integers using four independent AVX2
        // accumulators (256 bits = 8 lanes each) to hide add latency, then folds
        // the lanes down and handles the scalar tail.
        inline static int32_t SumInt32(const int32_t* data, size_t count) noexcept {
            __m256i acc0 = _mm256_setzero_si256();
            __m256i acc1 = _mm256_setzero_si256();
            __m256i acc2 = _mm256_setzero_si256();
            __m256i acc3 = _mm256_setzero_si256();

            size_t index = 0;
            for (; index + 32 <= count; index += 32) {
                acc0 = _mm256_add_epi32(acc0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + index +  0)));
                acc1 = _mm256_add_epi32(acc1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + index +  8)));
                acc2 = _mm256_add_epi32(acc2, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + index + 16)));
                acc3 = _mm256_add_epi32(acc3, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + index + 24)));
            }
            for (; index + 8 <= count; index += 8) {
                acc0 = _mm256_add_epi32(acc0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + index)));
            }

            const __m256i acc = _mm256_add_epi32(_mm256_add_epi32(acc0, acc1), _mm256_add_epi32(acc2, acc3));

            __m128i folded = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
            folded = _mm_add_epi32(folded, _mm_shuffle_epi32(folded, _MM_SHUFFLE(1, 0, 3, 2)));
            folded = _mm_add_epi32(folded, _mm_shuffle_epi32(folded, _MM_SHUFFLE(2, 3, 0, 1)));
            int32_t result = _mm_cvtsi128_si32(folded);

            for (; index < count; ++index) {
                result += data[index];
            }
            return result;
        }
#endif
    };
} // namespace NProust
