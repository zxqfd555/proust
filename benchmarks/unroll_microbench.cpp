// Closing experiment: is the lever unrolling, not intrinsics?
//
// Compares three implementations of the int32 sum/max reduction:
//   * autovec_1acc  - the natural single-accumulator loop (what operations.h
//                     gives the compiler today; auto-vectorizes to ONE ymm).
//   * scalar_unroll4 - a hand-unrolled loop with 4 INDEPENDENT scalar
//                     accumulators and NO intrinsics whatsoever.
//   * avx2_intrin    - the hand-written 4-accumulator AVX2 from the simd headers.
//
// Claim under test: scalar_unroll4 should auto-vectorize to the same multi-ymm,
// 4-accumulator code as avx2_intrin and match its throughput, with zero
// intrinsics -- i.e. the win came from accumulator count (ILP), not from SIMD
// intrinsics per se.
//
// Build (from repo root):
//   g++ -O3 -march=native -ffast-math -std=c++23 -Iinclude \
//       benchmarks/unroll_microbench.cpp -o /tmp/unroll_microbench
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "engine/operations.h"
#include "engine/simd/avx2_sum.h"
#include "engine/simd/avx2_minmax.h"

// --- single accumulator (identical to operations.h bodies) ----------------
static int32_t SumAutovec1(const int32_t* d, size_t n) {
    int32_t r = 0;
    for (size_t i = 0; i < n; ++i) r += d[i];
    return r;
}
static int32_t MaxAutovec1(const int32_t* d, size_t n) {
    int32_t r = d[0];
    for (size_t i = 1; i < n; ++i) r = d[i] > r ? d[i] : r;
    return r;
}

// --- 4 independent accumulators, plain C++ (no intrinsics) ----------------
static int32_t SumScalarUnroll4(const int32_t* d, size_t n) {
    int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    size_t i = 0;
    const size_t n4 = n & ~size_t(3);
    for (; i < n4; i += 4) {
        a0 += d[i]; a1 += d[i + 1]; a2 += d[i + 2]; a3 += d[i + 3];
    }
    int32_t r = (a0 + a1) + (a2 + a3);
    for (; i < n; ++i) r += d[i];
    return r;
}
static int32_t MaxScalarUnroll4(const int32_t* d, size_t n) {
    int32_t m0 = d[0], m1 = d[0], m2 = d[0], m3 = d[0];
    size_t i = 0;
    const size_t n4 = n & ~size_t(3);
    for (; i < n4; i += 4) {
        m0 = d[i]     > m0 ? d[i]     : m0;
        m1 = d[i + 1] > m1 ? d[i + 1] : m1;
        m2 = d[i + 2] > m2 ? d[i + 2] : m2;
        m3 = d[i + 3] > m3 ? d[i + 3] : m3;
    }
    int32_t r = std::max(std::max(m0, m1), std::max(m2, m3));
    for (; i < n; ++i) r = d[i] > r ? d[i] : r;
    return r;
}

static inline const int32_t* Opaque(const int32_t* p) {
    asm volatile("" : "+r"(p) :: "memory");
    return p;
}

template<typename F>
static double TimeNsPerCall(F&& f, size_t calls) {
    const auto start = std::chrono::high_resolution_clock::now();
    int32_t sink = 0;
    for (size_t i = 0; i < calls; ++i) sink += f();
    const auto end = std::chrono::high_resolution_clock::now();
    asm volatile("" :: "r"(sink) : "memory");
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / double(calls);
}

int main() {
    std::mt19937 gen(20260531);
    std::uniform_int_distribution<int32_t> dist(-1000, 1000);
    const std::vector<size_t> sizes = {64, 128, 256, 512, 1024, 4096, 16384};
    const size_t budget = 1u << 26;

    auto bench = [&](const char* title, auto autovec, auto unroll4, auto avx2) {
        std::printf("\n== %s ==\n", title);
        std::printf("%9s | %16s | %16s | %14s | %12s | %12s\n",
                    "size", "autovec_1acc ns", "scalar_unroll4 ns", "avx2 ns",
                    "unroll4/auto", "avx2/unroll4");
        std::printf("%s\n", std::string(92, '-').c_str());
        for (size_t n : sizes) {
            std::vector<int32_t> data(n);
            for (auto& v : data) v = dist(gen);
            const int32_t* p = data.data();
            const size_t calls = budget / n;

            const int32_t a = autovec(p, n), b = unroll4(p, n), c = avx2(p, 0, n);
            if (a != b || a != c) { std::printf("MISMATCH size=%zu: %d %d %d\n", n, a, b, c); return; }

            const double t1 = TimeNsPerCall([&]{ return autovec(Opaque(p), n); }, calls);
            const double t4 = TimeNsPerCall([&]{ return unroll4(Opaque(p), n); }, calls);
            const double tv = TimeNsPerCall([&]{ return avx2(Opaque(p), 0, n); }, calls);
            std::printf("%9zu | %16.1f | %16.1f | %14.1f | %11.2fx | %11.2fx\n",
                        n, t1, t4, tv, t1 / t4, t4 / tv);
        }
    };

    bench("SUM", SumAutovec1, SumScalarUnroll4,
          NProust::TSumOperationAvx2::FromMany<int32_t, int32_t>);
    bench("MAX", MaxAutovec1, MaxScalarUnroll4,
          NProust::TMaxOperationAvx2::FromMany<int32_t, int32_t>);
    return 0;
}
