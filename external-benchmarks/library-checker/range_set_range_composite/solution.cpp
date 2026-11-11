// https://judge.yosupo.jp/problem/range_set_range_composite
//
// f_i(x) = a_i x + b_i; queries: assign c x + d on [l, r), and evaluate f_{r-1}(...f_l(x)) mod p.
//
// A masked wide range-assignment tree (TMaskedAssignmentTree below), self-contained: it does not use
// proust's engines.
// Every node is a block of 16 slots stored field by field (multipliers, translations, tags), and
// every node update is a whole-block masked write instead of per-slot branches and variable-length
// fills. Slots sit in bit-reversed order inside a block, so that folding a block is four rounds of
// "low half then high half", lane by lane, with no shuffles of the data. Assignments and queries
// walk their two boundary paths side by side; operations are read ahead to prefetch their paths,
// and the power records of assignments are built 16 at a time in SIMD lanes.

// The judge compiles without -DNDEBUG; asserts are for development only.
#if !defined(NDEBUG) && !defined(KEEP_ASSERTS)
#define NDEBUG
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <vector>

#include <immintrin.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
    constexpr uint32_t MOD = 998244353;

    // Montgomery arithmetic with R = 2^32 and lazy residues: a stored coefficient is some value in
    // [0, 2p) congruent to the real one (the vector code never brings it lower), and answers are
    // reduced once, in Evaluate. A multiplier A is kept in Montgomery form (A R mod p), a
    // translation B in plain form: the reduction of A * B is then B scaled by A, plain again.
    constexpr uint32_t INVERSE = [] {
        uint32_t inverse = MOD;  // Newton's iteration: each step doubles the number of correct bits
        for (int step = 0; step < 4; ++step) {
            inverse *= 2 - MOD * inverse;
        }
        return inverse;
    }();
    static_assert(MOD * INVERSE == 1);
    static_assert(MOD < (1u << 30));  // 4p < 2^32, 4p^2 + 2^32 p < 2^64
    constexpr uint32_t R1 = static_cast<uint32_t>((uint64_t(1) << 32) % MOD);
    constexpr uint32_t R2 = static_cast<uint32_t>((static_cast<unsigned __int128>(1) << 64) % MOD);

    // a b R^{-1} mod p in [0, p), for a, b < 2p. With m = ab p^{-1} mod 2^32, ab - mp is divisible by
    // 2^32 and the low halves of ab and mp coincide, so the quotient is the difference of the high
    // halves: below 4p^2 / 2^32 < p, and above -p.
    inline uint32_t MulMont(const uint32_t a, const uint32_t b) noexcept {
        const uint64_t t = static_cast<uint64_t>(a) * b;
        const uint32_t m = static_cast<uint32_t>(t) * INVERSE;
        const uint32_t r = static_cast<uint32_t>(t >> 32) - static_cast<uint32_t>((static_cast<uint64_t>(m) * MOD) >> 32);
        return std::min(r, r + MOD);  // a negative r wrapped around: r + p is then the small one
    }

    // a + b, for a < p and b < 2p, in [0, 2p).
    inline uint32_t AddMod(const uint32_t a, const uint32_t b) noexcept {
        const uint32_t s = a + b;
        return std::min(s, s - MOD);
    }

    // x -> A x + B, A in Montgomery form.
    struct TAffine {
        uint32_t A;
        uint32_t B;
    };

    inline TAffine Identity() noexcept {
        return {R1, 0};
    }

    inline TAffine FromInput(const uint32_t a, const uint32_t b) noexcept {
        return {MulMont(a, R2), b};
    }

    // Applies `first`, then `second`: second(first(x)).
    inline TAffine Compose(const TAffine first, const TAffine second) noexcept {
        return {MulMont(first.A, second.A), AddMod(MulMont(second.A, first.B), second.B)};
    }

    // f(x) in [0, p), for x < p.
    inline uint32_t Evaluate(const TAffine f, const uint32_t x) noexcept {
        const uint32_t y = AddMod(MulMont(f.A, x), f.B);
        return std::min(y, y - MOD);
    }


    // ---- Eight functions at once: one per 32-bit lane, multipliers and translations apart ----

    struct TLanes {
        __m256i A;
        __m256i B;
    };

    // a b R^{-1} mod p lane by lane, in [0, 2p) for a, b < 2p: the other form of the reduction,
    // (t + m p) / 2^32 with m = -t p^{-1} mod 2^32, which is below (4p^2 + 2^32 p) / 2^32 < 2p and
    // needs no correction. Even and odd lanes go as two sets of 64-bit products (no 32-bit mullo:
    // 10 cycles of latency on Intel cores).
    inline __m256i MulMontLanes(const __m256i x, const __m256i y) noexcept {
        const __m256i negInverse = _mm256_set1_epi32(static_cast<int>(0u - INVERSE));
        const __m256i mod = _mm256_set1_epi32(static_cast<int>(MOD));
        const __m256i evenT = _mm256_mul_epu32(x, y);
        const __m256i oddT = _mm256_mul_epu32(_mm256_srli_epi64(x, 32), _mm256_srli_epi64(y, 32));
        const __m256i even = _mm256_add_epi64(evenT, _mm256_mul_epu32(_mm256_mul_epu32(evenT, negInverse), mod));
        const __m256i odd = _mm256_add_epi64(oddT, _mm256_mul_epu32(_mm256_mul_epu32(oddT, negInverse), mod));
        return _mm256_blend_epi32(_mm256_srli_epi64(even, 32), odd, 0xAA);
    }

    // a + b lane by lane, in [0, 2p) for a, b < 2p (4p < 2^32).
    inline __m256i AddModLanes(const __m256i a, const __m256i b) noexcept {
        const __m256i s = _mm256_add_epi32(a, b);
        return _mm256_min_epu32(s, _mm256_sub_epi32(s, _mm256_set1_epi32(static_cast<int>(2 * MOD))));
    }

    // Compose, lane by lane.
    inline TLanes ComposeLanes(const TLanes first, const TLanes second) noexcept {
        return {MulMontLanes(first.A, second.A), AddModLanes(MulMontLanes(second.A, first.B), second.B)};
    }

    // Compose, for the four functions held as (multiplier, translation) pairs in the 64-bit lanes of
    // `first` and `second`: both words of a pair get multiplied by the second multiplier at once,
    // giving A1 A2 and A2 B1, and B2 is added to the high word.
    inline __m256i ComposePairs(const __m256i first, const __m256i second) noexcept {
        const __m256i factor = _mm256_shuffle_epi32(second, _MM_SHUFFLE(2, 2, 0, 0));
        const __m256i shift = _mm256_blend_epi32(_mm256_setzero_si256(), second, 0xAA);
        return AddModLanes(MulMontLanes(first, factor), shift);
    }

    // ---- Blocks of 16 slots in bit-reversed order ----
    //
    // Slot s lives in lane BITREV[s]. Lanes j and j + 8 then hold slots 2k and 2k + 1 (k = BITREV[j] / 2),
    // and after composing them lane by lane, lanes j and j + 4 hold pairs 2k' and 2k' + 1, and so on:
    // the block folds in slot order by halving, with no data shuffles but moving a half down.

    constexpr std::array<uint8_t, 16> BITREV = [] {
        std::array<uint8_t, 16> table{};
        for (unsigned s = 0; s < 16; ++s) {
            table[s] = static_cast<uint8_t>(((s & 1) << 3) | ((s & 2) << 1) | ((s & 4) >> 1) | ((s & 8) >> 3));
        }
        return table;
    }();

    // MASKS[from][to]: the lanes of slots [from, to).
    constexpr auto MASKS = [] {
        std::array<std::array<uint16_t, 17>, 17> table{};
        for (unsigned from = 0; from <= 16; ++from) {
            for (unsigned to = from; to <= 16; ++to) {
                for (unsigned s = from; s < to; ++s) {
                    table[from][to] |= static_cast<uint16_t>(1u << BITREV[s]);
                }
            }
        }
        return table;
    }();

    // A 16-bit lane mask as two vectors of all-ones / all-zeros lanes (lanes 0..7, 8..15).
    struct TLaneMask {
        __m256i Low;
        __m256i High;
    };

    inline TLaneMask ExpandMask(const uint32_t mask) noexcept {
        const __m256i bitsLow = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
        const __m256i bitsHigh = _mm256_setr_epi32(256, 512, 1024, 2048, 4096, 8192, 16384, 32768);
        const __m256i broadcast = _mm256_set1_epi32(static_cast<int>(mask));
        return {
            _mm256_cmpeq_epi32(_mm256_and_si256(broadcast, bitsLow), bitsLow),
            _mm256_cmpeq_epi32(_mm256_and_si256(broadcast, bitsHigh), bitsHigh),
        };
    }

    inline __m256i Load(const uint32_t* data) noexcept {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(data));
    }

    inline void Store(uint32_t* data, const __m256i value) noexcept {
        _mm256_store_si256(reinterpret_cast<__m256i*>(data), value);
    }

    // Lanes of `mask` take `value`, the rest keep what is stored.
    inline void BlendStore(uint32_t* data, const TLaneMask& mask, const uint32_t value) noexcept {
        const __m256i broadcast = _mm256_set1_epi32(static_cast<int>(value));
        Store(data, _mm256_blendv_epi8(Load(data), broadcast, mask.Low));
        Store(data + 8, _mm256_blendv_epi8(Load(data + 8), broadcast, mask.High));
    }

    // The composition of the 16 functions of a block in slot order; lanes outside `mask` count as
    // identities. Packed: fewer instructions, a longer dependency chain (by a Zen 3 model: 117 uops
    // and 81 cycles against 131 and 69) -- for queries, whose folds are independent of each other;
    // assignments refold along a chain and take the unpacked one.
    template<bool Packed>
    inline TAffine FoldBlock(const uint32_t* a, const uint32_t* b, const TLaneMask& mask) noexcept {
        const __m256i identityA = _mm256_set1_epi32(static_cast<int>(R1));
        const __m256i zero = _mm256_setzero_si256();
        const TLanes low{_mm256_blendv_epi8(identityA, Load(a), mask.Low), _mm256_blendv_epi8(zero, Load(b), mask.Low)};
        const TLanes high{_mm256_blendv_epi8(identityA, Load(a + 8), mask.High), _mm256_blendv_epi8(zero, Load(b + 8), mask.High)};
        // Round 1, eight compositions with multipliers and translations apart: lane j of the result
        // composes slots BITREV[j], BITREV[j] + 1, and lane j + 4 holds the pair right after lane j's.
        const TLanes v = ComposeLanes(low, high);
        if constexpr (!Packed) {
            TLanes w = ComposeLanes(v, {_mm256_permute2x128_si256(v.A, v.A, 0x01), _mm256_permute2x128_si256(v.B, v.B, 0x01)});
            w = ComposeLanes(w, {_mm256_shuffle_epi32(w.A, _MM_SHUFFLE(3, 2, 3, 2)), _mm256_shuffle_epi32(w.B, _MM_SHUFFLE(3, 2, 3, 2))});
            w = ComposeLanes(w, {_mm256_shuffle_epi32(w.A, _MM_SHUFFLE(1, 1, 1, 1)), _mm256_shuffle_epi32(w.B, _MM_SHUFFLE(1, 1, 1, 1))});
            return {static_cast<uint32_t>(_mm256_cvtsi256_si32(w.A)), static_cast<uint32_t>(_mm256_cvtsi256_si32(w.B))};
        } else {
            // Rounds 2 and 3 on (multiplier, translation) pairs in 64-bit lanes, one product per
            // composition instead of two (see ComposePairs); round 4 is a scalar composition.
            const __m256i lowPairs = _mm256_unpacklo_epi32(v.A, v.B);   // functions 0, 1 | 4, 5
            const __m256i highPairs = _mm256_unpackhi_epi32(v.A, v.B);  // functions 2, 3 | 6, 7
            const __m256i first = _mm256_permute2x128_si256(lowPairs, highPairs, 0x20);   // 0, 1, 2, 3
            const __m256i second = _mm256_permute2x128_si256(lowPairs, highPairs, 0x31);  // 4, 5, 6, 7
            const __m256i quads = ComposePairs(first, second);  // pair k: k then k + 4
            const __m256i halves = ComposePairs(quads, _mm256_permute2x128_si256(quads, quads, 0x01));  // pairs 0, 1 valid
            const __m128i last = _mm256_castsi256_si128(halves);
            return Compose(
                {static_cast<uint32_t>(_mm_cvtsi128_si32(last)), static_cast<uint32_t>(_mm_extract_epi32(last, 1))},
                {static_cast<uint32_t>(_mm_extract_epi32(last, 2)), static_cast<uint32_t>(_mm_extract_epi32(last, 3))});
        }
    }

    // MulMontLanes on the even 32-bit lanes only (one product per 64-bit lane), the result in the
    // low half of each 64-bit lane; the odd lanes of the inputs are ignored.
    inline __m256i MulMontEven(const __m256i x, const __m256i y) noexcept {
        const __m256i t = _mm256_mul_epu32(x, y);
        const __m256i m = _mm256_mul_epu32(t, _mm256_set1_epi32(static_cast<int>(0u - INVERSE)));
        return _mm256_srli_epi64(_mm256_add_epi64(t, _mm256_mul_epu32(m, _mm256_set1_epi32(static_cast<int>(MOD)))), 32);
    }

    // Compose on the even lanes (a function per 64-bit lane).
    inline TLanes ComposeEven(const TLanes first, const TLanes second) noexcept {
        return {MulMontEven(first.A, second.A), AddModLanes(MulMontEven(second.A, first.B), second.B)};
    }

    // The composition of the half of a block, slots [0, 8) (Upper = false) or [8, 16), in order;
    // lanes outside `mask` count as identities. Slots [0, 8) sit exactly in the even lanes of the
    // two vectors of a block, slots [8, 16) in the odd ones (shifted down first): lane-wise
    // composition of the two vectors pairs (0, 1), (4, 5), (2, 3), (6, 7) in the 64-bit lanes
    // 0..3; then lanes k and k + 2, then 0 and 1 -- three rounds, one product per composition.
    template<bool Upper>
    inline TAffine FoldHalf(const uint32_t* a, const uint32_t* b, const TLaneMask& mask) noexcept {
        const __m256i identityA = _mm256_set1_epi32(static_cast<int>(R1));
        const __m256i zero = _mm256_setzero_si256();
        TLanes low{_mm256_blendv_epi8(identityA, Load(a), mask.Low), _mm256_blendv_epi8(zero, Load(b), mask.Low)};
        TLanes high{_mm256_blendv_epi8(identityA, Load(a + 8), mask.High), _mm256_blendv_epi8(zero, Load(b + 8), mask.High)};
        if constexpr (Upper) {
            low = {_mm256_srli_epi64(low.A, 32), _mm256_srli_epi64(low.B, 32)};
            high = {_mm256_srli_epi64(high.A, 32), _mm256_srli_epi64(high.B, 32)};
        }
        TLanes v = ComposeEven(low, high);
        v = ComposeEven(v, {_mm256_permute2x128_si256(v.A, v.A, 0x01), _mm256_permute2x128_si256(v.B, v.B, 0x01)});
        v = ComposeEven(v, {_mm256_shuffle_epi32(v.A, _MM_SHUFFLE(3, 2, 3, 2)), _mm256_shuffle_epi32(v.B, _MM_SHUFFLE(3, 2, 3, 2))});
        return {static_cast<uint32_t>(_mm256_cvtsi256_si32(v.A)), static_cast<uint32_t>(_mm256_cvtsi256_si32(v.B))};
    }

    inline TAffine FoldBlock(const uint32_t* a, const uint32_t* b) noexcept {
        const __m256i ones = _mm256_set1_epi32(-1);
        return FoldBlock<false>(a, b, TLaneMask{ones, ones});
    }

    // ---- Powers of assigned functions, kept per assignment ----
    //
    // An assignment of f gets a record f, f^2, f^4, ..., f^(2^(W-1)) with W = bit_width(N), and hands
    // the tree the record's handle as its value: every power the tree can ask for is then a read of
    // the record plus one composition per further set bit. Records have a fixed stride, handles come
    // from a free list; when it runs dry, the handles still referenced by a tag are marked and the
    // rest is freed -- nothing moves. Handle 0 is "no tag".
    //
    // Records are built in batches of up to BATCH functions, one per SIMD lane: W squarings of all
    // of them at once instead of one dependent chain per assignment.
    class TPowerStore {
    public:
        static constexpr size_t BATCH = 16;

        // For lengths up to `length` and at most `slots` tags alive at once.
        static void Reserve(const size_t length, const size_t slots) {
            Width = static_cast<size_t>(std::bit_width(std::max<size_t>(length, 1)));
#ifdef TIGHT_STORE
            const size_t capacity = slots + 1 + BATCH;  // stress tests: the least that works, collect often
#else
            const size_t capacity = 2 * slots + 2 + BATCH;
#endif
            // Not zero-filled: a record is written before it is read, and filling 10 MB up front
            // would fault in every page before the first operation.
            Steps = std::make_unique_for_overwrite<TAffine[]>(capacity * Width);
            Live.assign(capacity, false);
            Free.clear();
            Returning.clear();
            for (size_t handle = capacity; handle-- > 1;) {
                Free.push_back(static_cast<uint32_t>(handle));
            }
        }

        // Records for functions[0, count), count <= BATCH, with powers up to f^(2^(width - 1)) (at
        // most W); their handles go to `handles`. A record that is not `lasting` (its assignment
        // covers no whole block, so it never becomes a tag) is only read while its batch runs: its
        // handle goes back to the free list at the next call, and the next batch takes the same
        // records again -- in cache, instead of fresh memory. `forEachLive(mark)` must mark every
        // handle still in use if the free list runs dry.
        template<typename TForEachLive>
        static void CreateBatch(const TAffine* functions, const bool* lasting, const size_t count, const size_t width, uint32_t* handles, TForEachLive&& forEachLive) {
            for (const uint32_t handle : Returning) {
                Free.push_back(handle);
            }
            Returning.clear();
            if (Free.size() < count) {
                Collect(forEachLive);
            }
            alignas(32) uint32_t a[BATCH];
            alignas(32) uint32_t b[BATCH];
            for (size_t k = 0; k < BATCH; ++k) {
                const TAffine f = k < count ? functions[k] : Identity();
                a[k] = f.A;
                b[k] = f.B;
            }
            TAffine* records[BATCH];
            for (size_t k = 0; k < count; ++k) {
                handles[k] = Free.back();
                Free.pop_back();
                if (!lasting[k]) {
                    Returning.push_back(handles[k]);
                }
                records[k] = Steps.get() + handles[k] * Width;
            }
            TLanes low{Load(a), Load(b)};
            TLanes high{Load(a + 8), Load(b + 8)};
            for (size_t step = 0; step < width; ++step) {
                if (step > 0) {
                    low = ComposeLanes(low, low);
                    high = ComposeLanes(high, high);
                }
                Store(a, low.A);
                Store(a + 8, high.A);
                Store(b, low.B);
                Store(b + 8, high.B);
                for (size_t k = 0; k < count; ++k) {
                    records[k][step] = {a[k], b[k]};
                }
            }
        }

        // f^(2^step).
        static TAffine Step(const uint32_t handle, const unsigned step) noexcept {
            return Steps[handle * Width + step];
        }

        static TAffine Power(const uint32_t handle, const size_t length) noexcept {
            if (length == 0) {
                return Identity();
            }
            const TAffine* steps = Steps.get() + handle * Width;
            // Powers of one function commute, so the set bits can be taken in any order.
            size_t rest = length;
            TAffine power = steps[std::countr_zero(rest)];
            rest &= rest - 1;
            while (rest) {
                power = Compose(power, steps[std::countr_zero(rest)]);
                rest &= rest - 1;
            }
            return power;
        }

    private:
        inline static size_t Width = 0;
        inline static std::unique_ptr<TAffine[]> Steps;
        inline static std::vector<uint32_t> Free;
        inline static std::vector<uint32_t> Returning;  // handles of the last batch that never become tags
        inline static std::vector<bool> Live;

        template<typename TForEachLive>
        static void Collect(TForEachLive&& forEachLive) {
            std::fill(Live.begin(), Live.end(), false);
            forEachLive([](const uint32_t handle) noexcept { Live[handle] = true; });
            Free.clear();
            for (size_t handle = Live.size(); handle-- > 1;) {
                if (!Live[handle]) {
                    Free.push_back(static_cast<uint32_t>(handle));
                }
            }
        }
    };

    // ---- The tree ----
    //
    // Level 1 nodes (leaves) hold one function per element; a level h >= 2 node has 16 slots, each
    // covering a level h - 1 child of span 16^(h - 1): the slot keeps that child's composition and, if
    // the whole child was assigned a function it has not been told about yet, that assignment's
    // power record handle as a tag (0: none). Elements past the end are identities, so every block
    // folds as a whole, and every fully covered slot has a power-of-two span: one ladder step.
    //
    // Assignment writes covered slots with one masked store per field and descends into at most two
    // boundary children per level; pushing a tag down is an unconditional masked store too.
    class TMaskedAssignmentTree {
    public:
        // Builds over `length` elements, the functions coming in order from `next()`: straight into
        // the leaves, with no intermediate array.
        template<typename TNext>
        void Init(const size_t length, TNext&& next) {
            Length = length;
            const size_t leaves = (Length + 15) >> 4;
            Leaves = std::make_unique_for_overwrite<TLeaf[]>(leaves);  // every slot is written below
            for (size_t index = 0; index < leaves * 16; ++index) {
                const TAffine f = index < Length ? next() : Identity();
                TLeaf& leaf = Leaves[index >> 4];
                leaf.A[BITREV[index & 15]] = f.A;
                leaf.B[BITREV[index & 15]] = f.B;
            }

            Inner.clear();
            Height = 1;
            for (size_t below = leaves; below > 1;) {
                const size_t count = (below + 15) >> 4;
                std::vector<TInner> level(count);
                for (size_t child = 0; child < count * 16; ++child) {
                    TInner& node = level[child >> 4];
                    const TAffine f = child < below ? FoldNode(Height, child) : Identity();
                    node.A[BITREV[child & 15]] = f.A;
                    node.B[BITREV[child & 15]] = f.B;
                    node.T[BITREV[child & 15]] = 0;
                }
                Inner.push_back(std::move(level));
                ++Height;
                below = count;
            }
            LeafBase = Leaves.get();
            Level2Base = Inner.empty() ? nullptr : Inner[0].data();
        }

        // Iterative: a shared descent while the range stays inside one slot, then the left and the right
        // boundary paths side by side, then the refolds of both paths, level by level, again side by
        // side -- two independent dependency chains instead of one twice as long.
        void SetRange(const size_t left, const size_t right, const uint32_t handle) noexcept {
            if (Height == 1) {
                WriteLeaf(Leaves[0], left, right, handle, 0);
                return;
            }
            std::array<size_t, MAX_LEVELS> shared;
            std::array<size_t, MAX_LEVELS> leftPath;
            std::array<size_t, MAX_LEVELS> rightPath;
            leftPath.fill(NONE);
            rightPath.fill(NONE);

            unsigned level = Height;
            size_t index = 0;
            size_t low = left;
            size_t high = right;
            while (true) {
                shared[level] = index;
                if (level == 1) {
                    break;  // reached a leaf (WriteLeaf done on the way)
                }
                const unsigned shift = (level - 1) * 4;
                const size_t span = size_t(1) << shift;
                const size_t first = low >> shift;
                const size_t base = first * span;
                if (first != (high - 1) >> shift || (low == base && high == base + span)) {
                    break;
                }
                const uint32_t tag = TakeTag(Inner[level - 2][index], first);
                index = index * 16 + first;
                low -= base;
                high -= base;
                --level;
                if (level == 1) {
                    WriteLeaf(Leaves[index], low, high, handle, tag);
                } else if (tag) {
                    Fill(Inner[level - 2][index], tag, (level - 1) * 4);
                }
            }
            const unsigned split = level;

            if (split >= 2) {
                TInner& node = Inner[split - 2][index];
                const unsigned shift = (split - 1) * 4;
                const size_t span = size_t(1) << shift;
                const size_t coveredFrom = (low + span - 1) >> shift;
                const size_t coveredTo = high >> shift;
                if (coveredFrom < coveredTo) {
                    Cover(node, MASKS[coveredFrom][coveredTo], handle, shift);
                }

                // Each side: the node whose boundary slot is entered next, that slot, the bound inside it.
                bool goLeft = (low & (span - 1)) != 0;
                bool goRight = (high & (span - 1)) != 0;
                size_t leftNode = index, leftSlot = low >> shift, leftLow = low & (span - 1);
                size_t rightNode = index, rightSlot = high >> shift, rightHigh = high & (span - 1);
                for (unsigned below = split - 1; goLeft || goRight; --below) {
                    const unsigned belowShift = (below - 1) * 4;
                    const size_t belowSpan = size_t(1) << belowShift;
                    if (goLeft) {
                        const uint32_t tag = TakeTag(Inner[below - 1][leftNode], leftSlot);
                        const size_t child = leftNode * 16 + leftSlot;
                        leftPath[below] = child;
                        if (below == 1) {
                            WriteLeaf(Leaves[child], leftLow, 16, handle, tag);
                            goLeft = false;
                        } else {
                            TInner& node = Inner[below - 2][child];
                            if (tag) {
                                Fill(node, tag, belowShift);
                            }
                            const size_t coveredFrom = (leftLow + belowSpan - 1) >> belowShift;
                            if (coveredFrom < 16) {
                                Cover(node, MASKS[coveredFrom][16], handle, belowShift);
                            }
                            goLeft = (leftLow & (belowSpan - 1)) != 0;
                            leftNode = child;
                            leftSlot = leftLow >> belowShift;
                            leftLow &= belowSpan - 1;
                        }
                    }
                    if (goRight) {
                        const uint32_t tag = TakeTag(Inner[below - 1][rightNode], rightSlot);
                        const size_t child = rightNode * 16 + rightSlot;
                        rightPath[below] = child;
                        if (below == 1) {
                            WriteLeaf(Leaves[child], 0, rightHigh, handle, tag);
                            goRight = false;
                        } else {
                            TInner& node = Inner[below - 2][child];
                            if (tag) {
                                Fill(node, tag, belowShift);
                            }
                            const size_t coveredTo = rightHigh >> belowShift;
                            if (coveredTo > 0) {
                                Cover(node, MASKS[0][coveredTo], handle, belowShift);
                            }
                            goRight = (rightHigh & (belowSpan - 1)) != 0;
                            rightNode = child;
                            rightSlot = rightHigh >> belowShift;
                            rightHigh &= belowSpan - 1;
                        }
                    }
                }
            }

            for (unsigned at = 1; at < split; ++at) {
                if (leftPath[at] != NONE) {
                    Refold(at, leftPath[at]);
                }
                if (rightPath[at] != NONE) {
                    Refold(at, rightPath[at]);
                }
            }
            for (unsigned at = split; at < Height; ++at) {
                Refold(at, shared[at]);
            }
        }

        // See Query: one code path per level.
        TAffine ComputeRange(const size_t left, const size_t right) const noexcept {
            // A tree of N <= 5 * 10^5 has 5 levels; up to 8 covers any 32-bit length.
            switch (Height) {
                case 1: return Query<1>(0, left, right);
                case 2: return Query<2>(0, left, right);
                case 3: return Query<3>(0, left, right);
                case 4: return Query<4>(0, left, right);
                case 5: return Query<5>(0, left, right);
                case 6: return Query<6>(0, left, right);
                case 7: return Query<7>(0, left, right);
                default: return Query<8>(0, left, right);
            }
        }

        // The query unrolled by level: every level gets its own code, with its spans as constants
        // and its own branches -- which the predictor then learns per level (a single loop over the
        // levels mixes the very different statistics of top and bottom in the same branches).
        template<unsigned Level>
        TAffine Query(const size_t index, const size_t low, const size_t high) const noexcept {
            if constexpr (Level <= 1) {
                return FoldRange(Leaves[index].A, Leaves[index].B, low, high);
            } else {
                const TInner& node = Inner[Level - 2][index];
                constexpr unsigned shift = (Level - 1) * 4;
                constexpr size_t span = size_t(1) << shift;
                // A boundary slot, partly covered: its tag's power, or the child's part.
                const auto boundary = [&](const size_t slot, const size_t from, const size_t to) {
                    if (const uint32_t tag = node.T[BITREV[slot]]) {
                        return TPowerStore::Power(tag, to - from);
                    }
                    return Query<Level - 1>(index * 16 + slot, from, to);
                };
                const size_t first = low >> shift;
                const size_t last = (high - 1) >> shift;
                if (first == last) {
                    const size_t base = first * span;
                    if (low == base && high == base + span) {
                        return {node.A[BITREV[first]], node.B[BITREV[first]]};
                    }
                    return boundary(first, low - base, high - base);
                }
                const size_t coveredFrom = (low + span - 1) >> shift;
                const size_t coveredTo = high >> shift;
                const bool hasMiddle = coveredFrom < coveredTo;
                TAffine result = hasMiddle ? FoldRange(node.A, node.B, coveredFrom, coveredTo) : Identity();
                if (low & (span - 1)) {
                    const TAffine part = boundary(first, low & (span - 1), span);
                    result = hasMiddle ? Compose(part, result) : part;
                }
                if (high & (span - 1)) {
                    result = Compose(result, boundary(last, 0, high & (span - 1)));
                }
                return result;
            }
        }

        // Starts loading the blocks an operation on [left, right) will touch at the two lowest levels
        // (the higher ones are small enough to stay in cache).
        void Prefetch(const size_t left, const size_t right) const noexcept {
            if (left >= right) {
                return;
            }
            // Straight from the two levels' base addresses (a prefetch never faults, so a missing
            // level 2 -- a tiny tree -- needs no check).
            for (const size_t position : {left, right - 1}) {
                const char* leaf = reinterpret_cast<const char*>(LeafBase + (position >> 4));
                _mm_prefetch(leaf, _MM_HINT_T0);
                _mm_prefetch(leaf + 64, _MM_HINT_T0);
                const char* node = reinterpret_cast<const char*>(Level2Base + (position >> 8));
                _mm_prefetch(node, _MM_HINT_T0);
                _mm_prefetch(node + 64, _MM_HINT_T0);
                _mm_prefetch(node + 128, _MM_HINT_T0);
            }
        }

        // The power steps an assignment of [left, right) needs in its record: tags only ever go to
        // whole aligned slots, and the largest one inside the range -- span 16^k, at most a root
        // slot -- bounds every power read later (covered slots, pushed tags, partial tagged slots
        // in queries, all shorter). A range of length >= 16^k holds an aligned 16^(k - 1) block,
        // so if no aligned 16^k one fits, the next level down does.
        size_t RecordWidth(const size_t left, const size_t right) const noexcept {
            const unsigned levels = std::min<unsigned>((static_cast<unsigned>(std::bit_width(right - left)) - 1) / 4, Height - 1);
            unsigned exponent = 4 * levels;
            if (levels > 0 && ((left + (size_t(1) << exponent) - 1) >> exponent) >= (right >> exponent)) {
                exponent -= 4;
            }
            return exponent + 1;
        }

        // Slots of all level >= 2 nodes: a bound on the tags alive at once.
        size_t SlotCount() const noexcept {
            size_t slots = 0;
            for (const auto& level : Inner) {
                slots += level.size() * 16;
            }
            return slots;
        }

        // Calls `visit(tag)` for every non-zero tag slot (stale ones below another tag included).
        template<typename TVisit>
        void ForEachTag(TVisit&& visit) const {
            for (const auto& level : Inner) {
                for (const TInner& node : level) {
                    for (const uint32_t tag : node.T) {
                        if (tag) {
                            visit(tag);
                        }
                    }
                }
            }
        }

    private:
        struct alignas(64) TLeaf {
            uint32_t A[16];
            uint32_t B[16];
        };

        struct alignas(64) TInner {
            uint32_t A[16];
            uint32_t B[16];
            uint32_t T[16];
        };

        std::unique_ptr<TLeaf[]> Leaves;         // level 1
        std::vector<std::vector<TInner>> Inner;  // Inner[h - 2] is level h
        unsigned Height = 1;
        const TLeaf* LeafBase = nullptr;     // Leaves.data(), for Prefetch
        const TInner* Level2Base = nullptr;  // Inner[0].data() if any
        size_t Length = 0;

        static constexpr size_t MAX_LEVELS = 18;  // 16^16 > any length, plus the leaf level and slack
        static constexpr size_t NONE = ~size_t(0);

        static uint32_t TakeTag(TInner& node, const size_t slot) noexcept {
            const unsigned lane = BITREV[slot];
            const uint32_t tag = node.T[lane];
            node.T[lane] = 0;
            return tag;
        }

        // The whole node becomes the function of `tag` (spans 2^shift each): a pushed-down tag.
        static void Fill(TInner& node, const uint32_t tag, const unsigned shift) noexcept {
            const TAffine power = TPowerStore::Step(tag, shift);
            const __m256i a = _mm256_set1_epi32(static_cast<int>(power.A));
            const __m256i b = _mm256_set1_epi32(static_cast<int>(power.B));
            const __m256i t = _mm256_set1_epi32(static_cast<int>(tag));
            Store(node.A, a);
            Store(node.A + 8, a);
            Store(node.B, b);
            Store(node.B + 8, b);
            Store(node.T, t);
            Store(node.T + 8, t);
        }

        // The slots of `mask` (spans 2^shift each) are wholly assigned the function of `handle`.
        static void Cover(TInner& node, const uint32_t mask, const uint32_t handle, const unsigned shift) noexcept {
            const TAffine power = TPowerStore::Step(handle, shift);
            const TLaneMask lanes = ExpandMask(mask);
            BlendStore(node.A, lanes, power.A);
            BlendStore(node.B, lanes, power.B);
            BlendStore(node.T, lanes, handle);
        }

        // Elements [low, high) of the leaf become the function of `handle`; if `tag` is non-zero, the
        // rest becomes the function of `tag` (a pushed-down tag), else keeps its value.
        static void WriteLeaf(TLeaf& leaf, const size_t low, const size_t high, const uint32_t handle, const uint32_t tag) noexcept {
            const TAffine one = TPowerStore::Step(handle, 0);
            const TLaneMask mask = ExpandMask(MASKS[low][high]);
            __m256i a0 = Load(leaf.A), a1 = Load(leaf.A + 8), b0 = Load(leaf.B), b1 = Load(leaf.B + 8);
            if (tag) {
                const TAffine pushed = TPowerStore::Step(tag, 0);
                a0 = a1 = _mm256_set1_epi32(static_cast<int>(pushed.A));
                b0 = b1 = _mm256_set1_epi32(static_cast<int>(pushed.B));
            }
            const __m256i a = _mm256_set1_epi32(static_cast<int>(one.A));
            const __m256i b = _mm256_set1_epi32(static_cast<int>(one.B));
            Store(leaf.A, _mm256_blendv_epi8(a0, a, mask.Low));
            Store(leaf.A + 8, _mm256_blendv_epi8(a1, a, mask.High));
            Store(leaf.B, _mm256_blendv_epi8(b0, b, mask.Low));
            Store(leaf.B + 8, _mm256_blendv_epi8(b1, b, mask.High));
        }

        // The composition of the whole node (level, index), level >= 1.
        TAffine FoldNode(const unsigned level, const size_t index) const noexcept {
            if (level == 1) {
                return FoldBlock(Leaves[index].A, Leaves[index].B);
            }
            const TInner& node = Inner[level - 2][index];
            return FoldBlock(node.A, node.B);
        }

        // Stores the composition of node (level, index) into its parent's slot.
        void Refold(const unsigned level, const size_t index) noexcept {
            const TAffine f = FoldNode(level, index);
            TInner& parent = Inner[level - 1][index >> 4];
            parent.A[BITREV[index & 15]] = f.A;
            parent.B[BITREV[index & 15]] = f.B;
        }

#ifndef SCALAR_RUN
#define SCALAR_RUN 3
#endif
        // Slots [from, to) of a block, in order; a short run is a plain chain of compositions.
        static TAffine FoldRange(const uint32_t* a, const uint32_t* b, const size_t from, const size_t to) noexcept {
            if (to - from <= SCALAR_RUN) {
                TAffine result{a[BITREV[from]], b[BITREV[from]]};
                for (size_t slot = from + 1; slot < to; ++slot) {
                    result = Compose(result, {a[BITREV[slot]], b[BITREV[slot]]});
                }
                return result;
            }
            const TLaneMask mask = ExpandMask(MASKS[from][to]);
            if (to <= 8) {
                return FoldHalf<false>(a, b, mask);
            }
            if (from >= 8) {
                return FoldHalf<true>(a, b, mask);
            }
            return FoldBlock<true>(a, b, mask);
        }
    };


    // ---- I/O: the whole input is read at once, the whole output is written at once ----

    // Up to eight decimal digits, one per byte, the most significant one in the lowest byte, as
    // values 0..9: three multiply-and-shift rounds pair them into 2-, 4- and 8-digit numbers.
    inline uint32_t ParseEightDigits(uint64_t digits) noexcept {
        digits = (digits * 10 + (digits >> 8)) & 0x00FF00FF00FF00FFull;
        digits = (digits * 100 + (digits >> 16)) & 0x0000FFFF0000FFFFull;
        return static_cast<uint32_t>((digits * 10000 + (digits >> 32)) & 0xFFFFFFFFull);
    }

    class TReader {
    public:
        // A regular file is mapped, not copied: its pages are the page cache's own, and the bytes
        // past its end up to the next page boundary read as zeros. The mapping goes over a reserved
        // zero region a page longer, so the zero padding is there even for a page-aligned size.
        // A pipe (or a failed mapping) is read into a buffer that grows as needed.
        TReader() {
            struct stat status;
            if (::fstat(STDIN_FILENO, &status) == 0 && S_ISREG(status.st_mode) && status.st_size > 0) {
                const size_t size = static_cast<size_t>(status.st_size);
                // A page of zeros before the data too: a record parse loads the 8 bytes ending at a
                // number's last digit, which reach before the start for a short first number.
                void* region = ::mmap(nullptr, size + 3 * 4096, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (region != MAP_FAILED) {
#ifdef NO_POPULATE
                    constexpr int populate = 0;
#else
                    constexpr int populate = MAP_POPULATE;
#endif
                    char* data = static_cast<char*>(region) + 4096;
                    if (::mmap(data, size, PROT_READ, MAP_PRIVATE | MAP_FIXED | populate, STDIN_FILENO, 0) != MAP_FAILED) {
                        Position = data;
                        return;
                    }
                    ::munmap(region, size + 3 * 4096);
                }
            }
            size_t size = 0;
            {
                size_t capacity = size_t(1) << 20;
                Buffer = std::make_unique_for_overwrite<char[]>(FRONT + capacity + PADDING);
                while (true) {
                    size += ReadAll(Buffer.get() + FRONT + size, capacity - size);
                    if (size < capacity) {
                        break;
                    }
                    auto grown = std::make_unique_for_overwrite<char[]>(FRONT + 2 * capacity + PADDING);
                    std::memcpy(grown.get() + FRONT, Buffer.get() + FRONT, size);
                    Buffer = std::move(grown);
                    capacity *= 2;
                }
            }
            // Zero padding: every number can be loaded as a whole 8-byte word, a record's 64 bytes
            // can be loaded past the last one, and the parse stops at a zero byte like at any other
            // separator. Zeros before the data, for the 8-byte loads that end at a short number.
            std::memset(Buffer.get(), 0, FRONT);
            std::memset(Buffer.get() + FRONT + size, 0, PADDING);
            Position = Buffer.get() + FRONT;
        }

        // The input holds only digits and whitespace, so a byte is a digit iff its high nibble is 3.
        uint32_t Next() noexcept {
            while (static_cast<unsigned char>(*Position) < '0') {
                ++Position;
            }
            uint64_t word;
            std::memcpy(&word, Position, sizeof(word));
            const uint64_t nonDigits = (word & 0xF0F0F0F0F0F0F0F0ull) ^ 0x3030303030303030ull;
            const uint64_t digits = word - 0x3030303030303030ull;  // garbage above the number is shifted out
            if (nonDigits) {
                const unsigned length = static_cast<unsigned>(std::countr_zero(nonDigits)) >> 3;  // 1..7
                Position += length;
                return ParseEightDigits(digits << (64 - 8 * length));
            }

            uint32_t value = ParseEightDigits(digits);
            Position += 8;
            while (static_cast<unsigned char>(*Position) >= '0') {
                value = value * 10 + static_cast<uint32_t>(*Position - '0');
                ++Position;
            }
            return value;
        }

        // Lines as records, in the judge's format: the fields of a line are separated by exactly one
        // character (blank lines and whitespace around lines are fine; a run of spaces inside a line
        // is not). The separators of the next 64 bytes go in one mask, every field's end comes from
        // it, and the fields are decoded together, one per 64-bit lane: the 8 bytes ending at a field's
        // last digit, cut to its length, turn into a number by 2-, 4- and 8-digit multiply-adds; a
        // ninth digit, where one can be, is added by a table weight. No branch on any length, and
        // one dependency chain per line instead of one per number.
        //
        // Two lines "a b" of the initial functions (coefficients up to nine digits).
        void TwoPairs(uint32_t* values) noexcept {
            SkipSpace();
            const char* from = Position;
            const uint64_t separators = Separators(from);
            const unsigned e0 = static_cast<unsigned>(std::countr_zero(separators));
            const uint64_t s1 = separators & (separators - 1);
            const unsigned e1 = static_cast<unsigned>(std::countr_zero(s1));
            const uint64_t s2 = s1 & (s1 - 1);
            const unsigned e2 = static_cast<unsigned>(std::countr_zero(s2));
            const unsigned e3 = static_cast<unsigned>(std::countr_zero(s2 & (s2 - 1)));
            const __m256i octets = Decode(from, e0, e1, e2, e3, e0, e1 - e0 - 1, e2 - e1 - 1, e3 - e2 - 1);
            values[0] = static_cast<uint32_t>(_mm256_extract_epi32(octets, 0)) + Ninth(from, e0);
            values[1] = static_cast<uint32_t>(_mm256_extract_epi32(octets, 1)) + Ninth(from + e0 + 1, e1 - e0 - 1);
            values[2] = static_cast<uint32_t>(_mm256_extract_epi32(octets, 4)) + Ninth(from + e1 + 1, e2 - e1 - 1);
            values[3] = static_cast<uint32_t>(_mm256_extract_epi32(octets, 5)) + Ninth(from + e2 + 1, e3 - e2 - 1);
            Position = from + e3 + 1;
        }

        // "t l r c d" (t = 0) or "t l r x" (t = 1): a one-digit type, positions of at most six
        // digits, values of at most nine; the end of the line is picked without a branch.
        void Operation(uint32_t& type, uint32_t* values) noexcept {
            SkipSpace();
            type = static_cast<uint32_t>(*Position - '0');
            const char* from = Position + 2;
            const uint64_t separators = Separators(from);
            const unsigned e0 = static_cast<unsigned>(std::countr_zero(separators));
            const uint64_t s1 = separators & (separators - 1);
            const unsigned e1 = static_cast<unsigned>(std::countr_zero(s1));
            const uint64_t s2 = s1 & (s1 - 1);
            const unsigned e2 = static_cast<unsigned>(std::countr_zero(s2));
            const unsigned e3 = static_cast<unsigned>(std::countr_zero(s2 & (s2 - 1)));
            const __m256i octets = Decode(from, e0, e1, e2, e3, e0, e1 - e0 - 1, e2 - e1 - 1, e3 - e2 - 1);
            values[0] = static_cast<uint32_t>(_mm256_extract_epi32(octets, 0));
            values[1] = static_cast<uint32_t>(_mm256_extract_epi32(octets, 1));
            values[2] = static_cast<uint32_t>(_mm256_extract_epi32(octets, 4)) + Ninth(from + e1 + 1, e2 - e1 - 1);
            values[3] = static_cast<uint32_t>(_mm256_extract_epi32(octets, 5)) + Ninth(from + e2 + 1, e3 - e2 - 1);
            Position = from + (type == 0 ? e3 : e2) + 1;
        }

    private:
        void SkipSpace() noexcept {
            while (static_cast<unsigned char>(*Position) < '0') {
                ++Position;
            }
        }

        // Bit i: byte i from `from` is a separator (anything below '0': whitespace, zero padding).
        static uint64_t Separators(const char* from) noexcept {
            const __m256i zeroDigit = _mm256_set1_epi8('0');
            const uint32_t low = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpgt_epi8(zeroDigit, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(from)))));
            const uint32_t high = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpgt_epi8(zeroDigit, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(from + 32)))));
            return low | static_cast<uint64_t>(high) << 32;
        }

        static uint64_t Word(const char* end) noexcept {
            uint64_t word;
            std::memcpy(&word, end - 8, 8);
            return word;
        }

        // The last up-to-eight digits of four fields ending at offsets e0..e3, of lengths n0..n3:
        // in 32-bit lanes 0, 1, 4 and 5.
        static __m256i Decode(const char* from, unsigned e0, unsigned e1, unsigned e2, unsigned e3, unsigned n0, unsigned n1, unsigned n2, unsigned n3) noexcept {
            const __m256i words = _mm256_setr_epi64x(static_cast<long long>(Word(from + e0)), static_cast<long long>(Word(from + e1)),
                                                     static_cast<long long>(Word(from + e2)), static_cast<long long>(Word(from + e3)));
            const __m256i cuts = _mm256_setr_epi64x(static_cast<long long>(DIGIT_NIBBLES[n0]), static_cast<long long>(DIGIT_NIBBLES[n1]),
                                                    static_cast<long long>(DIGIT_NIBBLES[n2]), static_cast<long long>(DIGIT_NIBBLES[n3]));
            const __m256i pairs = _mm256_maddubs_epi16(_mm256_and_si256(words, cuts), _mm256_set1_epi16(0x010A));
            const __m256i quads = _mm256_madd_epi16(pairs, _mm256_set1_epi32(0x00010064));
            return _mm256_madd_epi16(_mm256_packus_epi32(quads, quads), _mm256_set1_epi32(0x00012710));
        }

        // The ninth digit from the end of a field of `length` digits starting at `start`, weighted.
        static uint32_t Ninth(const char* start, const unsigned length) noexcept {
            return NINTH_WEIGHT[length] * static_cast<uint32_t>(start[0] - '0');
        }

        // DIGIT_NIBBLES[length]: keeps the digit values of the last `length` of 8 bytes (all of them
        // from 8 on); NINTH_WEIGHT[length]: the weight of the first digit beyond those 8.
        static constexpr std::array<uint64_t, 65> DIGIT_NIBBLES = [] {
            std::array<uint64_t, 65> table{};
            for (unsigned length = 1; length < table.size(); ++length) {
                table[length] = 0x0F0F0F0F0F0F0F0Full << (length < 8 ? (8 - length) * 8 : 0);
            }
            return table;
        }();
        static constexpr std::array<uint32_t, 65> NINTH_WEIGHT = [] {
            std::array<uint32_t, 65> table{};
            table[9] = 100000000;
            return table;
        }();

        static constexpr size_t PADDING = 128;
        static constexpr size_t FRONT = 16;
        std::unique_ptr<char[]> Buffer;
        const char* Position;

        // Up to `length` bytes of stdin, fewer only at its end.
        static size_t ReadAll(char* destination, const size_t length) noexcept {
            size_t done = 0;
            while (done < length) {
                const ssize_t chunk = ::read(STDIN_FILENO, destination + done, length - done);
                if (chunk <= 0) {
                    break;
                }
                done += static_cast<size_t>(chunk);
            }
            return done;
        }
    };

    // "0000" .. "9999".
    constexpr auto FOUR_DIGITS = [] {
        std::array<std::array<char, 4>, 10000> table{};
        for (unsigned value = 0; value < table.size(); ++value) {
            table[value] = {
                static_cast<char>('0' + value / 1000),
                static_cast<char>('0' + value / 100 % 10),
                static_cast<char>('0' + value / 10 % 10),
                static_cast<char>('0' + value % 10),
            };
        }
        return table;
    }();

    // Output through a small fixed buffer: memory proportional to the answer count would be
    // touched page by page for nothing.
    class TWriter {
    public:
        ~TWriter() {
            Flush();
        }

        // For value < 10^9, which every answer modulo p is.
        void Line(uint32_t value) noexcept {
            if (Position + MAX_LINE > Buffer + CAPACITY) [[unlikely]] {
                Flush();
            }
            if (value >= 100000000) {
                *Position++ = static_cast<char>('0' + value / 100000000);
                value %= 100000000;
                PutFour(value / 10000);
                PutFour(value % 10000);
            } else if (value >= 10000) {
                PutLeading(value / 10000);
                PutFour(value % 10000);
            } else {
                PutLeading(value);
            }
            *Position++ = '\n';
        }

    private:
        static constexpr size_t CAPACITY = size_t(1) << 16;
        static constexpr size_t MAX_LINE = 11;  // nine digits and a newline, with a byte to spare
        char Buffer[CAPACITY];
        char* Position = Buffer;

        void Flush() noexcept {
            const char* from = Buffer;
            while (from < Position) {
                const ssize_t written = ::write(STDOUT_FILENO, from, static_cast<size_t>(Position - from));
                if (written <= 0) {
                    break;
                }
                from += written;
            }
            Position = Buffer;
        }

        void PutFour(const uint32_t group) noexcept {
            std::memcpy(Position, FOUR_DIGITS[group].data(), 4);
            Position += 4;
        }

        // A group without its leading zeros.
        void PutLeading(const uint32_t group) noexcept {
            const unsigned length = 1 + (group >= 10) + (group >= 100) + (group >= 1000);
            std::memcpy(Position, FOUR_DIGITS[group].data() + (4 - length), length);
            Position += length;
        }
    };

} // namespace

int main() {
    TReader in;
    const uint32_t n = in.Next();
    const uint32_t q = in.Next();

    TMaskedAssignmentTree engine;
    // Two lines of the initial functions at a time; an odd last one is read on its own.
    uint32_t pairs[4];
    uint32_t taken = 0;
    uint32_t pending = 0;  // functions left in `pairs`, the earlier one first
    engine.Init(n, [&] {
        if (pending == 0) {
            if (n - taken >= 2) {
                in.TwoPairs(pairs);
                pending = 2;
            } else {
                pairs[2] = in.Next();
                pairs[3] = in.Next();
                pending = 1;
            }
        }
        const uint32_t* pair = pairs + 4 - 2 * pending;
        --pending;
        ++taken;
        return FromInput(pair[0], pair[1]);
    });
    TPowerStore::Reserve(n, engine.SlotCount());

    // Operations go in blocks of BATCH: the power records of a block's assignments are built
    // together. Operations are also read LOOKAHEAD ahead of their execution, so that the blocks they
    // will touch are already on their way to the cache when their turn comes.
#ifndef LOOKAHEAD
#define LOOKAHEAD 8
#endif
    constexpr size_t BATCH = TPowerStore::BATCH;
    struct TOperation {
        uint32_t Type, L, R, C, D;
        uint32_t Handle;
    };
    constexpr size_t RING = std::bit_ceil(BATCH + LOOKAHEAD);
    std::array<TOperation, RING> ring;
    uint32_t parsed = 0;
    const auto parseUpTo = [&](const uint32_t end) {
        for (; parsed < end; ++parsed) {
            TOperation& op = ring[parsed & (RING - 1)];
            uint32_t values[4];
            in.Operation(op.Type, values);
            op.L = values[0];
            op.R = values[1];
            op.C = values[2];
            op.D = values[3];
            engine.Prefetch(op.L, op.R);
        }
    };

    TWriter out;
    for (uint32_t blockStart = 0; blockStart < q; blockStart += BATCH) {
        const uint32_t blockEnd = std::min<uint32_t>(q, blockStart + BATCH);
        parseUpTo(std::min<uint32_t>(q, blockEnd + LOOKAHEAD));

        TAffine functions[BATCH];
        uint32_t handles[BATCH];
        TOperation* owners[BATCH];
        bool lasting[BATCH];
        size_t count = 0;
        size_t width = 0;
        for (uint32_t i = blockStart; i < blockEnd; ++i) {
            TOperation& op = ring[i & (RING - 1)];
            if (op.Type == 0 && op.L < op.R) {
                const size_t needed = engine.RecordWidth(op.L, op.R);
                lasting[count] = needed > 1;  // a tag needs a whole block, hence at least f^16
                width = std::max(width, needed);
                functions[count] = FromInput(op.C, op.D);
                owners[count++] = &op;
            }
        }
        if (count > 0) {
            TPowerStore::CreateBatch(functions, lasting, count, width, handles, [&](auto mark) { engine.ForEachTag(mark); });
            for (size_t k = 0; k < count; ++k) {
                owners[k]->Handle = handles[k];
            }
        }

        for (uint32_t i = blockStart; i < blockEnd; ++i) {
            const TOperation& op = ring[i & (RING - 1)];
            if (op.Type == 0) {
                if (op.L < op.R) {
                    engine.SetRange(op.L, op.R, op.Handle);
                }
            } else {
                out.Line(op.L == op.R ? op.C : Evaluate(engine.ComputeRange(op.L, op.R), op.C));
            }
        }
    }
    return 0;
}
