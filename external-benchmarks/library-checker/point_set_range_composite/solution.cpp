// https://judge.yosupo.jp/problem/point_set_range_composite
//
// f_i(x) = a_i x + b_i; operations: set f_p = c x + d, and evaluate f_{r-1}(...f_l(x)) mod p.
//
// Affine maps with a non-zero multiplier form a group, so a range is a quotient of two prefixes.
// With F_i = f_{i-1} o ... o f_0 and its inverse H_i = F_i^{-1} (x -> I_i x + J_i),
//
//     y = f_{r-1}(...f_l(x)) = F_r(H_l(x))  <=>  H_r(y) = H_l(x)  <=>  y = (H_l(x) - J_r) / I_r.
//
// The division is not done per query: the denominators are inverted together at the end, with one
// exponentiation for all of them.
//
// The inverse prefixes are kept in levels of 16-slot groups, each slot the inverse of the prefix
// from the start of its group (exclusive), so H_i is the composition of one slot per level, the
// lowest level applied first. Two endpoints with the same slots from some level up share the
// remainder of the composition, and it cancels from the equation above: a query walks only the
// levels where its endpoints differ, along three independent chains.
//
// Setting f_p = g turns every later H_i into D o H_i with one D for all of them: the conjugate of
// E = g^{-1} o f_p (x -> s x + t, s = a_p / c, t = (b_p - d) / c) by the prefix before p. Taken
// relative to the start of the group at every level, the conjugate of (s, t) by a slot (I, J) is
// (s, I t + (1 - s) J): the multiplier is the same everywhere, the translation is one step of a
// chain going up the levels. The later slots of the group become (s I, s J + t) -- a masked write
// of whole blocks in AVX2 lanes.
//
// Every inverse is known before the first operation runs: the whole input is read first, and one
// batch inversion covers the multipliers c of all the updates and the products of the initial
// groups of 16 (the inverse prefixes of a group are then a backward walk from the inverse of its
// product; the higher levels compose inverses and need none of their own).

// The judge compiles without -DNDEBUG; asserts are for development only.
#if !defined(NDEBUG) && !defined(KEEP_ASSERTS)
#define NDEBUG
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>

#include <immintrin.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
    constexpr uint32_t MOD = 998244353;

    // Montgomery arithmetic with R = 2^32. A multiplier is kept in Montgomery form (A R mod p), a
    // translation in plain form: the reduction of A * B is then B scaled by A, plain again. Values
    // are lazy, in [0, 2p), and made canonical only for the output.
    constexpr uint32_t INVERSE = [] {
        uint32_t inverse = MOD;  // Newton's iteration: each step doubles the number of correct bits
        for (int step = 0; step < 4; ++step) {
            inverse *= 2 - MOD * inverse;
        }
        return inverse;
    }();
    static_assert(MOD * INVERSE == 1);
    static_assert(MOD < (1u << 30));
    constexpr uint32_t MOD2 = 2 * MOD;
    constexpr uint32_t R1 = static_cast<uint32_t>((uint64_t(1) << 32) % MOD);
    constexpr uint32_t R2 = static_cast<uint32_t>((static_cast<unsigned __int128>(1) << 64) % MOD);

    // a b R^{-1} mod p in (0, 2p), for a, b < 2p. With m = ab p^{-1} mod 2^32, ab - mp is divisible
    // by 2^32 and the low halves of ab and mp coincide, so the quotient is the difference of the high
    // halves; it lies in (-p, 4p^2 / 2^32), within (-p, p).
    inline uint32_t MulMont(const uint32_t a, const uint32_t b) noexcept {
        const uint64_t t = static_cast<uint64_t>(a) * b;
        const uint32_t m = static_cast<uint32_t>(t) * INVERSE;
        return static_cast<uint32_t>(t >> 32) - static_cast<uint32_t>((static_cast<uint64_t>(m) * MOD) >> 32) + MOD;
    }

    // (a b + c d) R^{-1} mod p in [0, 2p), for a, b, c, d < 2p: the sum is below 8p^2 < 2^63, the
    // quotient in (-p, 8p^2 / 2^32), within (-p, 1.86p).
    inline uint32_t MulSumMont(const uint32_t a, const uint32_t b, const uint32_t c, const uint32_t d) noexcept {
        const uint64_t t = static_cast<uint64_t>(a) * b + static_cast<uint64_t>(c) * d;
        const uint32_t m = static_cast<uint32_t>(t) * INVERSE;
        const uint32_t r = static_cast<uint32_t>(t >> 32) - static_cast<uint32_t>((static_cast<uint64_t>(m) * MOD) >> 32) + MOD;
        return std::min(r, r - MOD2);
    }

    inline uint32_t AddMod(const uint32_t a, const uint32_t b) noexcept {
        const uint32_t s = a + b;
        return std::min(s, s - MOD2);
    }

    inline uint32_t SubMod(const uint32_t a, const uint32_t b) noexcept {
        const uint32_t s = a - b + MOD2;
        return std::min(s, s - MOD2);
    }

    inline uint32_t Canonical(const uint32_t a) noexcept {
        return std::min(a, a - MOD);
    }

    inline uint32_t ToMont(const uint32_t x) noexcept {
        return MulMont(x, R2);
    }

    // x^e for x in Montgomery form, the result too.
    inline uint32_t PowMont(uint32_t x, uint32_t e) noexcept {
        uint32_t result = R1;
        for (; e; e >>= 1) {
            if (e & 1) {
                result = MulMont(result, x);
            }
            x = MulMont(x, x);
        }
        return result;
    }

    // All the large arrays, in one mapping with transparent huge pages where the kernel allows them:
    // some 20 megabytes, faulted in 2 MB at a time rather than 4 KB at a time.
    class TArena {
    public:
        explicit TArena(const size_t bytes) {
            constexpr size_t HUGE_PAGE = size_t(1) << 21;
            const size_t mapped = bytes + HUGE_PAGE;
            void* region = mmap(nullptr, mapped, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            Next = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(region) + HUGE_PAGE - 1) & ~(HUGE_PAGE - 1));
#ifndef NO_HUGE_PAGES
            madvise(Next, bytes, MADV_HUGEPAGE);
#endif
        }

        // `count` objects, uninitialized, 64-byte aligned.
        template <class T>
        T* Take(const size_t count) noexcept {
            T* result = reinterpret_cast<T*>(Next);
            Next += (count * sizeof(T) + 63) & ~size_t(63);
            return result;
        }

    private:
        char* Next;
    };

    // Replaces every value (in Montgomery form, non-zero) by its inverse: prefix products, one
    // exponentiation, then back. Eight interleaved chains, so that eight products are in flight;
    // `count` rounded up to whole blocks of eight, the extra values being R1; `prefix` is scratch
    // space for as many values.
    void InvertAll(uint32_t* values, const size_t count, uint32_t* prefix) {
        constexpr size_t CHAINS = 8;
        assert(count % CHAINS == 0);
        std::array<uint32_t, CHAINS> product;
        product.fill(R1);
        for (size_t block = 0; block < count; block += CHAINS) {
#pragma GCC unroll 8
            for (size_t chain = 0; chain < CHAINS; ++chain) {
                prefix[block + chain] = product[chain];
                product[chain] = MulMont(product[chain], values[block + chain]);
            }
        }
        uint32_t total = R1;
        for (const uint32_t chain : product) {
            total = MulMont(total, chain);
        }
        const uint32_t inverse = PowMont(total, MOD - 2);
        std::array<uint32_t, CHAINS> inverses;  // of each chain: the total's times the other chains
        for (size_t chain = 0; chain < CHAINS; ++chain) {
            uint32_t others = inverse;
            for (size_t other = 0; other < CHAINS; ++other) {
                others = other == chain ? others : MulMont(others, product[other]);
            }
            inverses[chain] = others;
        }
        product = inverses;
        for (size_t block = count; block > 0;) {
            block -= CHAINS;
#pragma GCC unroll 8
            for (size_t chain = 0; chain < CHAINS; ++chain) {
                const uint32_t value = values[block + chain];
                values[block + chain] = MulMont(product[chain], prefix[block + chain]);
                product[chain] = MulMont(product[chain], value);
            }
        }
    }

    inline size_t WholeBlocks(const size_t count) noexcept {
        return (count + 7) & ~size_t(7);
    }

    // ---- Eight lanes of Montgomery arithmetic ----

    // x * s for a fixed s: m = x s p^{-1} mod 2^32 is x (s p^{-1}) mod 2^32, a product independent of x s.
    struct TLaneFactor {
        __m256i Value;
        __m256i Scaled;  // s p^{-1} mod 2^32

        explicit TLaneFactor(const uint32_t s) noexcept
            : Value(_mm256_set1_epi32(static_cast<int>(s)))
            , Scaled(_mm256_set1_epi32(static_cast<int>(s * INVERSE)))
        {
        }
    };

    inline __m256i MulMontLanes(const __m256i x, const TLaneFactor& s) noexcept {
        const __m256i mod = _mm256_set1_epi32(static_cast<int>(MOD));
        const __m256i oddX = _mm256_srli_epi64(x, 32);
        const __m256i evenT = _mm256_mul_epu32(x, s.Value);
        const __m256i oddT = _mm256_mul_epu32(oddX, s.Value);
        const __m256i evenP = _mm256_mul_epu32(_mm256_mul_epu32(x, s.Scaled), mod);
        const __m256i oddP = _mm256_mul_epu32(_mm256_mul_epu32(oddX, s.Scaled), mod);
        const __m256i highT = _mm256_blend_epi32(_mm256_srli_epi64(evenT, 32), oddT, 0xAA);
        const __m256i highP = _mm256_blend_epi32(_mm256_srli_epi64(evenP, 32), oddP, 0xAA);
        return _mm256_add_epi32(_mm256_sub_epi32(highT, highP), mod);
    }

    inline __m256i AddModLanes(const __m256i a, const __m256i b) noexcept {
        const __m256i s = _mm256_add_epi32(a, b);
        return _mm256_min_epu32(s, _mm256_sub_epi32(s, _mm256_set1_epi32(static_cast<int>(MOD2))));
    }

    // ---- The inverse prefixes ----

    // x -> A x + B: A in Montgomery form, B plain.
    struct TFunction {
        uint32_t A, B;
    };

    // 16 slots x -> I x + J: I in Montgomery form, J plain. Two cache lines.
    struct alignas(64) TGroup {
        uint32_t I[16];
        uint32_t J[16];
    };

    class TInversePrefixes {
    public:
        // Level 0, first pass: the forward prefixes F_k = (P, B) of every group, kept in the slots,
        // and the multipliers of the whole groups in `totals`, to be inverted.
        size_t Forward(const TFunction* functions, const size_t length, uint32_t* totals, TArena& arena) {
            Length = length;
            Levels = 1;
            while ((size_t(1) << (4 * Levels)) <= length) {
                ++Levels;
            }
            // Level k has slots for the indices up to length >> 4k (the exclusive end of a range).
            size_t total = 0;
            for (size_t level = 0; level < Levels; ++level) {
                Groups[level] = ((length >> (4 * level)) >> 4) + 1;
                total += Groups[level];
            }
            // Whole blocks at level 0: never beyond the slots of level 1, as 8 divides 16.
            total += (BLOCK - Groups[0] % BLOCK) % BLOCK;
            Groups[0] = (Groups[0] + BLOCK - 1) / BLOCK * BLOCK;
            TGroup* next = arena.Take<TGroup>(total);
            for (size_t level = 0; level < Levels; ++level) {
                Level[level] = next;
                next += Groups[level];
            }
            // Eight groups at a time: their chains are independent.
            for (size_t first = 0; first < Groups[0]; first += BLOCK) {
                std::array<uint32_t, BLOCK> product;
                std::array<uint32_t, BLOCK> shift;
                product.fill(R1);
                shift.fill(0);
                for (size_t slot = 0; slot < 16; ++slot) {
                    for (size_t lane = 0; lane < BLOCK; ++lane) {
                        TGroup& slots = Level[0][first + lane];
                        slots.I[slot] = product[lane];
                        slots.J[slot] = shift[lane];
                        const size_t child = ((first + lane) << 4) + slot;
                        const uint32_t a = child < length ? functions[child].A : R1;
                        const uint32_t b = child < length ? functions[child].B : 0;
                        product[lane] = MulMont(a, product[lane]);
                        shift[lane] = AddMod(MulMont(a, shift[lane]), b);
                    }
                }
                for (size_t lane = 0; lane < BLOCK; ++lane) {
                    totals[first + lane] = product[lane];
                }
            }
            return Groups[0];
        }

        // Level 0, second pass, given the inverses of `totals`: H_k = F_k^{-1} = (1 / P, -B / P), with
        // 1 / P_k = a_k / P_{k+1} from the end of the group. Then the higher levels, whose children are
        // the inverse totals of the groups below: no inversions there.
        void Finish(const TFunction* functions, const uint32_t* inverseTotals) {
            std::array<std::unique_ptr<uint32_t[]>, MAX_LEVELS> totalI;
            std::array<std::unique_ptr<uint32_t[]>, MAX_LEVELS> totalJ;
            totalI[0] = std::make_unique_for_overwrite<uint32_t[]>(Groups[0]);
            totalJ[0] = std::make_unique_for_overwrite<uint32_t[]>(Groups[0]);
            for (size_t first = 0; first < Groups[0]; first += BLOCK) {
                std::array<uint32_t, BLOCK> inverse;
                for (size_t lane = 0; lane < BLOCK; ++lane) {
                    const size_t group = first + lane;
                    TGroup& slots = Level[0][group];
                    inverse[lane] = inverseTotals[group];
                    const size_t last = (group << 4) + 15;
                    const uint32_t lastA = last < Length ? functions[last].A : R1;
                    const uint32_t lastB = last < Length ? functions[last].B : 0;
                    const uint32_t totalShift = AddMod(MulMont(lastA, slots.J[15]), lastB);
                    totalI[0][group] = inverse[lane];
                    totalJ[0][group] = MulMont(MOD2 - totalShift, inverse[lane]);
                }
                for (size_t slot = 16; slot-- > 0;) {
                    for (size_t lane = 0; lane < BLOCK; ++lane) {
                        TGroup& slots = Level[0][first + lane];
                        const size_t child = ((first + lane) << 4) + slot;
                        const uint32_t a = child < Length ? functions[child].A : R1;
                        inverse[lane] = MulMont(inverse[lane], a);
                        slots.I[slot] = inverse[lane];
                        slots.J[slot] = MulMont(MOD2 - slots.J[slot], inverse[lane]);
                    }
                }
            }
            for (size_t level = 1; level < Levels; ++level) {
                const size_t children = Groups[level - 1];
                totalI[level] = std::make_unique_for_overwrite<uint32_t[]>(Groups[level]);
                totalJ[level] = std::make_unique_for_overwrite<uint32_t[]>(Groups[level]);
                for (size_t group = 0; group < Groups[level]; ++group) {
                    uint32_t i = R1;
                    uint32_t j = 0;
                    TGroup& slots = Level[level][group];
                    for (size_t slot = 0; slot < 16; ++slot) {
                        slots.I[slot] = i;
                        slots.J[slot] = j;
                        const size_t child = (group << 4) + slot;
                        if (child < children) {
                            // H o child^{-1}: x -> i (childI x + childJ) + j.
                            j = AddMod(MulMont(i, totalJ[level - 1][child]), j);
                            i = MulMont(i, totalI[level - 1][child]);
                        }
                    }
                    totalI[level][group] = i;
                    totalJ[level][group] = j;
                }
            }
        }

        // H_l(x) - J_r and I_r: the answer is their quotient.
        void Query(const uint32_t left, const uint32_t right, const uint32_t x, uint32_t& numerator, uint32_t& denominator) const noexcept {
            if (left == right) {
                numerator = x;
                denominator = R1;
                return;
            }
            const unsigned top = (std::bit_width(left ^ right) - 1) >> 2;
            switch (top) {
                case 0: return QueryLevels<0>(left, right, x, numerator, denominator);
                case 1: return QueryLevels<1>(left, right, x, numerator, denominator);
                case 2: return QueryLevels<2>(left, right, x, numerator, denominator);
                case 3: return QueryLevels<3>(left, right, x, numerator, denominator);
                default: return QueryLevels<4>(left, right, x, numerator, denominator);
            }
        }

        // E = (s, t), s in Montgomery form, t plain.
        void Update(uint32_t position, const uint32_t s, uint32_t t) noexcept {
            const uint32_t complement = SubMod(R1, s);
            const TLaneFactor factor(s);
            for (size_t level = 0; level < Levels; ++level, position >>= 4) {
                TGroup& group = Level[level][position >> 4];
                const unsigned slot = position & 15;
                t = MulSumMont(group.I[slot], t, complement, group.J[slot]);
                ScaleSuffix(group, slot, factor, t);
            }
        }

        void Prefetch(uint32_t position) const noexcept {
            for (size_t level = 0; level < 2 && level < Levels; ++level, position >>= 4) {
                const TGroup* group = Level[level] + (position >> 4);
                _mm_prefetch(reinterpret_cast<const char*>(group->I), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(group->J), _MM_HINT_T0);
            }
        }

    private:
        template <unsigned TOP>
        void QueryLevels(uint32_t left, uint32_t right, const uint32_t x, uint32_t& numerator, uint32_t& denominator) const noexcept {
            const TGroup& leftGroup = Level[0][left >> 4];
            const TGroup& rightGroup = Level[0][right >> 4];
            uint32_t u = AddMod(MulMont(leftGroup.I[left & 15], x), leftGroup.J[left & 15]);
            uint32_t v = rightGroup.J[right & 15];
            uint32_t w = rightGroup.I[right & 15];
            [&]<unsigned... LEVEL>(std::integer_sequence<unsigned, LEVEL...>) {
                ([&] {
                    constexpr unsigned level = LEVEL + 1;
                    const uint32_t l = left >> (4 * level);
                    const uint32_t r = right >> (4 * level);
                    const TGroup& lg = Level[level][l >> 4];
                    const TGroup& rg = Level[level][r >> 4];
                    u = AddMod(MulMont(lg.I[l & 15], u), lg.J[l & 15]);
                    v = AddMod(MulMont(rg.I[r & 15], v), rg.J[r & 15]);
                    w = MulMont(rg.I[r & 15], w);
                }(), ...);
            }(std::make_integer_sequence<unsigned, TOP>{});
            numerator = SubMod(u, v);
            denominator = w;
        }

        // The slots after `slot` become x -> s (I x + J) + t.
        static void ScaleSuffix(TGroup& group, const unsigned slot, const TLaneFactor& factor, const uint32_t t) noexcept {
            const __m256i lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            const __m256i shift = _mm256_set1_epi32(static_cast<int>(t));
            // Both halves always, the first one perhaps with an empty mask: the slot is random, a
            // loop over the halves it touches would mispredict half of the time.
            for (unsigned half = 0; half < 2; ++half) {
                const __m256i mask = _mm256_cmpgt_epi32(lanes, _mm256_set1_epi32(static_cast<int>(slot) - static_cast<int>(8 * half)));
                __m256i* i = reinterpret_cast<__m256i*>(group.I + 8 * half);
                __m256i* j = reinterpret_cast<__m256i*>(group.J + 8 * half);
                const __m256i oldI = _mm256_load_si256(i);
                const __m256i oldJ = _mm256_load_si256(j);
                _mm256_store_si256(i, _mm256_blendv_epi8(oldI, MulMontLanes(oldI, factor), mask));
                _mm256_store_si256(j, _mm256_blendv_epi8(oldJ, AddModLanes(MulMontLanes(oldJ, factor), shift), mask));
            }
        }

        static constexpr size_t MAX_LEVELS = 5;  // 16^5 > 500000
        std::array<TGroup*, MAX_LEVELS> Level{};
        static constexpr size_t BLOCK = 8;
        std::array<size_t, MAX_LEVELS> Groups{};
        size_t Length = 0;
        size_t Levels = 1;
    };

    // An upper bound of the groups of all levels: the levels shrink 16 times each, and each has a
    // group to spare, plus a block's padding at level 0.
    inline size_t MaxGroups(const size_t length) noexcept {
        return length / 15 + 16;
    }

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

        // "t p c d" (t = 0) or "t l r x" (t = 1): a one-digit type, then always three numbers.
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
            values[1] = static_cast<uint32_t>(_mm256_extract_epi32(octets, 1)) + Ninth(from + e0 + 1, e1 - e0 - 1);
            values[2] = static_cast<uint32_t>(_mm256_extract_epi32(octets, 4)) + Ninth(from + e1 + 1, e2 - e1 - 1);
            Position = from + e2 + 1;
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

        // One line per value(index) for index in [0, count), each value below 10^9 (as every
        // answer modulo p is). The position is a local: through a member, every store of a digit
        // would force it to be reloaded, as a char store may alias anything.
        template <class TValueAt>
        void Lines(const size_t count, TValueAt&& valueAt) noexcept {
            char* position = Position;
            for (size_t index = 0; index < count; ++index) {
                if (position + MAX_LINE > Buffer + CAPACITY) [[unlikely]] {
                    Position = position;
                    Flush();
                    position = Position;
                }
                position = Line(position, valueAt(index));
            }
            Position = position;
        }

    private:
        static constexpr size_t CAPACITY = size_t(1) << 16;
        static constexpr size_t MAX_LINE = 10;  // nine digits and a newline
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

        // Nine digits, most of the time: the branches are predictable.
        static char* Line(char* position, uint32_t value) noexcept {
            if (value >= 100000000) {
                *position++ = static_cast<char>('0' + value / 100000000);
                value %= 100000000;
                position = PutFour(position, value / 10000);
                position = PutFour(position, value % 10000);
            } else if (value >= 10000) {
                position = PutLeading(position, value / 10000);
                position = PutFour(position, value % 10000);
            } else {
                position = PutLeading(position, value);
            }
            *position = '\n';
            return position + 1;
        }

        static char* PutFour(char* position, const uint32_t group) noexcept {
            std::memcpy(position, FOUR_DIGITS[group].data(), 4);
            return position + 4;
        }

        // A group without its leading zeros.
        static char* PutLeading(char* position, const uint32_t group) noexcept {
            const unsigned length = 1 + (group >= 10) + (group >= 100) + (group >= 1000);
            std::memcpy(position, FOUR_DIGITS[group].data() + (4 - length), length);
            return position + length;
        }
    };
} // namespace

int main() {
    TReader in;
    const uint32_t n = in.Next();
    const uint32_t q = in.Next();

    const size_t inverted = WholeBlocks(size_t(q) + MaxGroups(n));  // the most values inverted at once
    TArena arena(sizeof(TFunction) * n + 12 * size_t(q) + 4 * (2 * inverted + 2 * size_t(q) + 8) + sizeof(TGroup) * MaxGroups(n) + 64 * 8);
    TFunction* functions = arena.Take<TFunction>(n);
    {
        uint32_t pairs[4];
        uint32_t index = 0;
        for (; index + 2 <= n; index += 2) {
            in.TwoPairs(pairs);
            functions[index] = {ToMont(pairs[0]), pairs[1]};
            functions[index + 1] = {ToMont(pairs[2]), pairs[3]};
        }
        if (index < n) {
            const uint32_t a = in.Next();
            functions[index] = {ToMont(a), in.Next()};
        }
    }
    // An update: position, c (Montgomery), d; a query: l | QUERY, r, x.
    constexpr uint32_t QUERY = 1u << 31;
    struct TOperation {
        uint32_t Position, X, Y;
    };
    static_assert(sizeof(TOperation) == 12);
    TOperation* operations = arena.Take<TOperation>(q);
    // The multipliers to invert: those of the updates, in order, then the group totals of the prefixes.
    uint32_t* multipliers = arena.Take<uint32_t>(inverted);
    uint32_t* scratch = arena.Take<uint32_t>(inverted);
    uint32_t updates = 0;
    for (uint32_t index = 0; index < q; ++index) {
        uint32_t type;
        uint32_t values[3];
        in.Operation(type, values);
        // Without a branch on the type, which is random: a query writes its r (in Montgomery form)
        // to `multipliers` too, where the next update overwrites it.
        const uint32_t c = ToMont(values[1]);
        operations[index] = {values[0] | (type << 31), type ? values[1] : c, values[2]};
        multipliers[updates] = c;
        updates += 1 - type;
    }
    TInversePrefixes prefixes;
    const size_t groups = prefixes.Forward(functions, n, multipliers + updates, arena);
    const size_t count = WholeBlocks(updates + groups);
    std::fill(multipliers + updates + groups, multipliers + count, R1);
    InvertAll(multipliers, count, scratch);
    prefixes.Finish(functions, multipliers + updates);

    // Answers as fractions: numerators plain, denominators in Montgomery form, inverted together at the end.
    const uint32_t queries = q - updates;
    uint32_t* numerators = arena.Take<uint32_t>(queries);
    uint32_t* denominators = arena.Take<uint32_t>(WholeBlocks(queries));
    constexpr uint32_t LOOKAHEAD = 4;  // operations: enough to hide the latency of the two largest levels
    const auto prefetch = [&](const TOperation& operation) {
        const uint32_t position = operation.Position & ~QUERY;
        prefixes.Prefetch(position);
        if (operation.Position & QUERY) {
            prefixes.Prefetch(operation.X);
        } else {
            _mm_prefetch(reinterpret_cast<const char*>(functions + position), _MM_HINT_T0);
        }
    };
    for (uint32_t index = 0; index < std::min<uint32_t>(q, LOOKAHEAD); ++index) {
        prefetch(operations[index]);
    }
    uint32_t answered = 0;
    uint32_t update = 0;
    for (uint32_t index = 0; index < q; ++index) {
        if (index + LOOKAHEAD < q) {
            prefetch(operations[index + LOOKAHEAD]);
        }
        const TOperation& operation = operations[index];
        if (operation.Position & QUERY) {
            prefixes.Query(operation.Position & ~QUERY, operation.X, operation.Y, numerators[answered], denominators[answered]);
            ++answered;
        } else {
            // E = g^{-1} o f_p: x -> (a_p x + b_p - d) / c.
            TFunction& function = functions[operation.Position];
            const uint32_t inverse = multipliers[update++];
            const uint32_t s = MulMont(function.A, inverse);
            const uint32_t t = MulMont(SubMod(function.B, operation.Y), inverse);
            function = {operation.X, operation.Y};
            prefixes.Update(operation.Position, s, t);
        }
    }

    std::fill(denominators + queries, denominators + WholeBlocks(queries), R1);
    InvertAll(denominators, WholeBlocks(queries), scratch);
    TWriter out;
    out.Lines(queries, [&](const size_t index) {
        return Canonical(MulMont(numerators[index], denominators[index]));
    });
    return 0;
}
