#pragma once

#include "interface.h"


#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <type_traits>
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
    #define PROUST_HAS_FOLD_IN_BLOCK_AVX2
    #include <immintrin.h>
#elif defined(__AVX2__)
    #include <immintrin.h>
#endif

namespace NProust {
    namespace NOpDetail {
#ifdef PROUST_HAS_FOLD_IN_BLOCK_AVX2
        // The int32 lanes [from, to) of a 16-lane block, the others replaced with `neutral`, in two vectors.
        struct TBlockLanes {
            __m256i Low;
            __m256i High;
        };

        [[gnu::target("avx2")]] inline TBlockLanes MaskedBlockLanes(const int32_t* block, const size_t from, const size_t to, const int32_t neutral) noexcept {
            const __m256i lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            const __m256i eight = _mm256_set1_epi32(8);
            const __m256i before = _mm256_set1_epi32(static_cast<int>(from) - 1);
            const __m256i end = _mm256_set1_epi32(static_cast<int>(to));
            const __m256i lowMask = _mm256_and_si256(_mm256_cmpgt_epi32(lanes, before), _mm256_cmpgt_epi32(end, lanes));
            const __m256i highLanes = _mm256_add_epi32(lanes, eight);
            const __m256i highMask = _mm256_and_si256(_mm256_cmpgt_epi32(highLanes, before), _mm256_cmpgt_epi32(end, highLanes));
            const __m256i filler = _mm256_set1_epi32(neutral);
            return {
                _mm256_blendv_epi8(filler, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block)), lowMask),
                _mm256_blendv_epi8(filler, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block + 8)), highMask),
            };
        }

        [[gnu::target("avx2")]] inline int32_t SumInBlockAvx2(const int32_t* block, const size_t from, const size_t to) noexcept {
            const TBlockLanes lanes = MaskedBlockLanes(block, from, to, 0);
            const __m256i sum = _mm256_add_epi32(lanes.Low, lanes.High);
            __m128i folded = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
            folded = _mm_add_epi32(folded, _mm_shuffle_epi32(folded, _MM_SHUFFLE(1, 0, 3, 2)));
            folded = _mm_add_epi32(folded, _mm_shuffle_epi32(folded, _MM_SHUFFLE(2, 3, 0, 1)));
            return _mm_cvtsi128_si32(folded);
        }

        [[gnu::target("avx2")]] inline int32_t MaxInBlockAvx2(const int32_t* block, const size_t from, const size_t to) noexcept {
            const TBlockLanes lanes = MaskedBlockLanes(block, from, to, std::numeric_limits<int32_t>::min());
            const __m256i largest = _mm256_max_epi32(lanes.Low, lanes.High);
            __m128i folded = _mm_max_epi32(_mm256_castsi256_si128(largest), _mm256_extracti128_si256(largest, 1));
            folded = _mm_max_epi32(folded, _mm_shuffle_epi32(folded, _MM_SHUFFLE(1, 0, 3, 2)));
            folded = _mm_max_epi32(folded, _mm_shuffle_epi32(folded, _MM_SHUFFLE(2, 3, 0, 1)));
            return _mm_cvtsi128_si32(folded);
        }
#endif

        // Masked in-place bulk update for ApplyToMany: block[i] = op(block[i], delta) for i in
        // [from, count); block[0..from) is read but left unchanged. Handles 16/32/64-bit integers
        // with one masked AVX2 op per 256-bit chunk (the Algorithmica wide-tree trick), so each
        // sibling-block suffix update is a couple of straight-line vector ops instead of a short
        // scalar loop. `op == '^'` selects XOR, anything else selects integer add. Non-AVX2 builds
        // and unsupported widths fall back to the exact scalar definition.
        template<char Op, class T>
        PROUST_FORCE_INLINE void MaskedSuffix(T* block, const size_t count, const size_t from, const T delta) noexcept {
            static_assert(Op == '+' || Op == '^');
#if defined(__AVX2__)
            if constexpr (std::is_integral_v<T> && (sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8)) {
                constexpr size_t lanes = sizeof(__m256i) / sizeof(T);
                size_t base = 0;
                for (; base + lanes <= count; base += lanes) {
                    __m256i ids, threshold, vdelta;
                    if constexpr (sizeof(T) == 4) {
                        vdelta = _mm256_set1_epi32(static_cast<int32_t>(delta));
                        threshold = _mm256_set1_epi32(static_cast<int32_t>(from) - 1);
                        ids = _mm256_add_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_set1_epi32(static_cast<int32_t>(base)));
                    } else if constexpr (sizeof(T) == 8) {
                        vdelta = _mm256_set1_epi64x(static_cast<int64_t>(delta));
                        threshold = _mm256_set1_epi64x(static_cast<int64_t>(from) - 1);
                        ids = _mm256_set_epi64x(static_cast<int64_t>(base) + 3, static_cast<int64_t>(base) + 2, static_cast<int64_t>(base) + 1, static_cast<int64_t>(base));
                    } else {  // 2-byte
                        vdelta = _mm256_set1_epi16(static_cast<int16_t>(delta));
                        threshold = _mm256_set1_epi16(static_cast<int16_t>(static_cast<int32_t>(from) - 1));
                        ids = _mm256_add_epi16(_mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), _mm256_set1_epi16(static_cast<int16_t>(base)));
                    }
                    __m256i mask;
                    if constexpr (sizeof(T) == 4) mask = _mm256_cmpgt_epi32(ids, threshold);
                    else if constexpr (sizeof(T) == 8) mask = _mm256_cmpgt_epi64(ids, threshold);
                    else mask = _mm256_cmpgt_epi16(ids, threshold);

                    T* p = block + base;
                    const __m256i cur = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
                    const __m256i term = _mm256_and_si256(vdelta, mask);
                    __m256i res;
                    if constexpr (Op == '^') res = _mm256_xor_si256(cur, term);
                    else if constexpr (sizeof(T) == 4) res = _mm256_add_epi32(cur, term);
                    else if constexpr (sizeof(T) == 8) res = _mm256_add_epi64(cur, term);
                    else res = _mm256_add_epi16(cur, term);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), res);
                }
                for (; base < count; ++base) {
                    if (base >= from) {
                        if constexpr (Op == '^') block[base] ^= delta; else block[base] += delta;
                    }
                }
                return;
            }
#endif
            for (size_t i = from; i < count; ++i) {
                if constexpr (Op == '^') block[i] ^= delta; else block[i] += delta;
            }
        }
    } // namespace NOpDetail

    struct TSumOperation {
#if defined(PROUST_HAS_FOLD_IN_BLOCK_AVX2) && defined(__AVX2__)
        template<typename TOperationResult>
        requires std::is_same_v<TOperationResult, int32_t>
        PROUST_FORCE_INLINE static TOperationResult FoldInBlock(const int32_t* block, const size_t from, const size_t to) noexcept {
            return NOpDetail::SumInBlockAvx2(block, from, to);
        }
#endif

        template<typename TValue, typename TOperationResult>
        PROUST_FORCE_INLINE static TOperationResult FromRepeated(const TValue& v, const size_t times) noexcept(noexcept(TOperationResult(v * times))) {
            return TOperationResult(v * times);
        }

        template<typename TOperationResult>
        PROUST_FORCE_INLINE static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) noexcept(noexcept(lhs + rhs)) {
            return lhs + rhs;
        }

        template<typename TOperationResult>
        PROUST_FORCE_INLINE static TOperationResult Undo(const TOperationResult& lhs, const TOperationResult& rhs) noexcept(noexcept(lhs - rhs)) {
            return lhs - rhs;
        }

        template<typename TValue, typename TOperationResult>
        PROUST_FORCE_INLINE static TOperationResult FromMany(const TValue* data, size_t fromIndex, size_t toIndex) {
            TOperationResult result = 0;
            for (size_t index = fromIndex; index < toIndex; ++index) {
                result += data[index];
            }
            return result;
        }

        // Bulk in-place modification (see HasApplyToMany): block[i] += delta for i in [from, count).
        // For int32 this is the Algorithmica wide-tree trick -- a masked vector add over the whole
        // fixed-size block, so the point engine's per-level update becomes a couple of straight-line
        // SIMD ops instead of a short scalar loop. Other element types (and non-AVX2 builds) fall back
        // to the exact scalar definition, keeping the engine generic.
        template<typename TOperationResult>
        PROUST_FORCE_INLINE static void ApplyToMany(TOperationResult* block, const size_t count, const size_t from, const TOperationResult& delta) noexcept {
#if defined(__AVX2__)
            if constexpr (std::is_same_v<TOperationResult, double>) {
                // Floating point gets its own masked vaddpd (the integer helper only covers integers).
                const __m256d vdelta = _mm256_set1_pd(delta);
                const __m256i threshold = _mm256_set1_epi64x(static_cast<int64_t>(from) - 1);
                size_t base = 0;
                for (; base + 4 <= count; base += 4) {
                    const __m256i ids = _mm256_set_epi64x(static_cast<int64_t>(base) + 3, static_cast<int64_t>(base) + 2, static_cast<int64_t>(base) + 1, static_cast<int64_t>(base));
                    const __m256d mask = _mm256_castsi256_pd(_mm256_cmpgt_epi64(ids, threshold));
                    double* p = block + base;
                    const __m256d cur = _mm256_loadu_pd(p);
                    _mm256_storeu_pd(p, _mm256_add_pd(cur, _mm256_and_pd(vdelta, mask)));
                }
                for (; base < count; ++base) {
                    if (base >= from) {
                        block[base] += delta;
                    }
                }
                return;
            }
#endif
            // 16/32/64-bit integers: masked vector add (one op per chunk); scalar fallback otherwise.
            NOpDetail::MaskedSuffix<'+', TOperationResult>(block, count, from, delta);
        }
    };

    struct TBitwiseAndOperation {
        template<typename TValue, typename TOperationResult>
        PROUST_FORCE_INLINE static TOperationResult FromRepeated(const TValue& v, const size_t times) noexcept(noexcept(TOperationResult(v))) {
            return TOperationResult(v);
        }

        template<typename TOperationResult>
        PROUST_FORCE_INLINE static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) noexcept(noexcept(lhs & rhs)) {
            return lhs & rhs;
        }
    };

    struct TBitwiseOrOperation {
        template<typename TValue, typename TOperationResult>
        PROUST_FORCE_INLINE static TOperationResult FromRepeated(const TValue& v, const size_t times) noexcept(noexcept(TOperationResult(v))) {
            return TOperationResult(v);
        }

        template<typename TOperationResult>
        PROUST_FORCE_INLINE static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) noexcept(noexcept(lhs | rhs)) {
            return lhs | rhs;
        }
    };

    struct TBitwiseXorOperation {
        template<typename TValue, typename TOperationResult>
        PROUST_FORCE_INLINE static TOperationResult FromRepeated(const TValue& v, const size_t times) noexcept(noexcept(TOperationResult(v)) && noexcept(TOperationResult(0))) {
            return (times & 1) ? TOperationResult(v) : TOperationResult(0);
        }

        template<typename TOperationResult>
        PROUST_FORCE_INLINE static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) noexcept(noexcept(lhs ^ rhs)) {
            return lhs ^ rhs;
        }

        template<typename TOperationResult>
        PROUST_FORCE_INLINE static TOperationResult Undo(const TOperationResult& lhs, const TOperationResult& rhs) noexcept(noexcept(lhs ^ rhs)) {
            return lhs ^ rhs;
        }

        // Bulk in-place modification (see HasApplyToMany). Same masked-block trick as the sum: XOR with
        // a per-lane-masked delta, where masked-out lanes get 0 and `x ^ 0 == x` leaves them intact.
        template<typename TOperationResult>
        PROUST_FORCE_INLINE static void ApplyToMany(TOperationResult* block, const size_t count, const size_t from, const TOperationResult& delta) noexcept {
            // 16/32/64-bit integers: masked vector XOR (one op per chunk); scalar fallback otherwise.
            NOpDetail::MaskedSuffix<'^', TOperationResult>(block, count, from, delta);
        }
    };

    struct TMaxOperation {
#if defined(PROUST_HAS_FOLD_IN_BLOCK_AVX2) && defined(__AVX2__)
        template<typename TOperationResult>
        requires std::is_same_v<TOperationResult, int32_t>
        PROUST_FORCE_INLINE static TOperationResult FoldInBlock(const int32_t* block, const size_t from, const size_t to) noexcept {
            return NOpDetail::MaxInBlockAvx2(block, from, to);
        }
#endif

        template<typename TValue, typename TOperationResult>
        PROUST_FORCE_INLINE static TOperationResult FromRepeated(const TValue& v, const size_t times) noexcept(noexcept(TOperationResult(v))) {
            return TOperationResult(v);
        }

        template<typename TOperationResult>
        PROUST_FORCE_INLINE static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) noexcept(noexcept(lhs > rhs)) {
            return lhs > rhs ? lhs : rhs;
        }
    };

    struct TMinOperation {
        template<typename TValue, typename TOperationResult>
        PROUST_FORCE_INLINE static TOperationResult FromRepeated(const TValue& v, const size_t times) noexcept(noexcept(TOperationResult(v))) {
            return TOperationResult(v);
        }

        template<typename TOperationResult>
        PROUST_FORCE_INLINE static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) noexcept(noexcept(lhs < rhs)) {
            return lhs < rhs ? lhs : rhs;
        }
    };

    // Range gcd. Associative and commutative, with identity 0 (gcd(x, 0) == |x|), and no inverse, so
    // it exercises the non-Undo point engine. The merge is a Euclid loop -- the per-node work here is
    // a real computation, unlike the single instruction of sum/max/and/or.
    struct TGcdOperation {
        template<typename TValue, typename TOperationResult>
        PROUST_FORCE_INLINE static TOperationResult FromRepeated(const TValue& v, const size_t times) noexcept {
            return TOperationResult(v);
        }

        template<typename TOperationResult>
        PROUST_FORCE_INLINE static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) noexcept {
            return std::gcd(lhs, rhs);
        }
    };
} // namespace NProust