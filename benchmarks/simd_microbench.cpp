// Isolated microbenchmark for the one SIMD-relevant kernel in the engine:
// TSumOperation::FromMany, the per-leaf int32 reduction. Strips away all the
// segment-tree traversal so we measure only the reduction over a contiguous
// leaf-sized buffer. Three variants are compared:
//   * scalar_autovec  - the exact scalar loop from operations.h, left for the
//                       compiler to auto-vectorize (-O3 -march=native)
//   * scalar_novec     - the same loop with auto-vectorization disabled, i.e.
//                       what "no SIMD at all" actually costs
//   * avx2_intrinsics  - the hand-written AVX2 reduction from avx2_sum.h
//
// Build (from repo root):
//   g++ -O3 -march=native -ffast-math -std=c++23 -Iinclude \
//       benchmarks/simd_microbench.cpp -o /tmp/simd_microbench
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "engine/operations.h"
#include "engine/simd/avx2_sum.h"
#include "engine/simd/avx2_minmax.h"

// Same body as TSumOperation::FromMany, but the optimizer is told not to
// vectorize it, giving a genuine scalar baseline.
__attribute__((optimize("no-tree-vectorize")))
static int32_t SumScalarNoVec(const int32_t* data, size_t count) {
    int32_t result = 0;
    for (size_t i = 0; i < count; ++i) {
        result += data[i];
    }
    return result;
}

__attribute__((optimize("no-tree-vectorize")))
static int32_t MaxScalarNoVec(const int32_t* data, size_t count) {
    int32_t result = data[0];
    for (size_t i = 1; i < count; ++i) {
        result = data[i] > result ? data[i] : result;
    }
    return result;
}

// Returns an opaque copy of `p` and clobbers memory, so the compiler cannot
// hoist or CSE a pure reduction over an unchanging buffer out of the timing
// loop (which would make a "free" kernel look infinitely fast).
static inline const int32_t* Opaque(const int32_t* p) {
    asm volatile("" : "+r"(p) :: "memory");
    return p;
}

template<typename F>
static double TimeNsPerCall(F&& f, size_t calls) {
    const auto start = std::chrono::high_resolution_clock::now();
    int32_t sink = 0;
    for (size_t i = 0; i < calls; ++i) {
        sink += f();
    }
    const auto end = std::chrono::high_resolution_clock::now();
    asm volatile("" :: "r"(sink) : "memory"); // keep `sink` live
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / double(calls);
}

int main() {
    std::mt19937 gen(20260531);
    std::uniform_int_distribution<int32_t> dist(-1000, 1000);

    const std::vector<size_t> leafSizes = {8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384};
    const size_t totalElements = 1u << 26; // fixed work budget per size => fair ns/elem

    std::printf("%9s | %14s | %14s | %14s | %10s | %10s\n",
                "leaf", "novec ns/call", "autovec ns/c", "avx2 ns/call",
                "avx2/novec", "avx2/auto");
    std::printf("%s\n", std::string(86, '-').c_str());

    for (size_t leaf : leafSizes) {
        std::vector<int32_t> data(leaf);
        for (auto& v : data) v = dist(gen);
        const int32_t* p = data.data();
        const size_t calls = totalElements / leaf;

        // warmup + correctness cross-check
        const int32_t a = SumScalarNoVec(p, leaf);
        const int32_t b = NProust::TSumOperation::FromMany<int32_t, int32_t>(p, 0, leaf);
        const int32_t c = NProust::TSumOperationAvx2::FromMany<int32_t, int32_t>(p, 0, leaf);
        if (a != b || a != c) {
            std::printf("MISMATCH at leaf=%zu: %d %d %d\n", leaf, a, b, c);
            return 1;
        }

        const double novec   = TimeNsPerCall([&]{ return SumScalarNoVec(Opaque(p), leaf); }, calls);
        const double autovec = TimeNsPerCall([&]{ return NProust::TSumOperation::FromMany<int32_t, int32_t>(Opaque(p), 0, leaf); }, calls);
        const double avx2    = TimeNsPerCall([&]{ return NProust::TSumOperationAvx2::FromMany<int32_t, int32_t>(Opaque(p), 0, leaf); }, calls);

        std::printf("%9zu | %14.1f | %14.1f | %14.1f | %9.2fx | %9.2fx\n",
                    leaf, novec, autovec, avx2, novec / avx2, autovec / avx2);
    }

    std::printf("\n== MAX reduction ==\n");
    std::printf("%9s | %14s | %14s | %14s | %10s | %10s\n",
                "leaf", "novec ns/call", "autovec ns/c", "avx2 ns/call",
                "avx2/novec", "avx2/auto");
    std::printf("%s\n", std::string(86, '-').c_str());
    for (size_t leaf : leafSizes) {
        std::vector<int32_t> data(leaf);
        for (auto& v : data) v = dist(gen);
        const int32_t* p = data.data();
        const size_t calls = totalElements / leaf;

        const int32_t a = MaxScalarNoVec(p, leaf);
        const int32_t b = NProust::TMaxOperationFromMany::FromMany<int32_t, int32_t>(p, 0, leaf);
        const int32_t c = NProust::TMaxOperationAvx2::FromMany<int32_t, int32_t>(p, 0, leaf);
        if (a != b || a != c) {
            std::printf("MISMATCH at leaf=%zu: %d %d %d\n", leaf, a, b, c);
            return 1;
        }

        const double novec   = TimeNsPerCall([&]{ return MaxScalarNoVec(Opaque(p), leaf); }, calls);
        const double autovec = TimeNsPerCall([&]{ return NProust::TMaxOperationFromMany::FromMany<int32_t, int32_t>(Opaque(p), 0, leaf); }, calls);
        const double avx2    = TimeNsPerCall([&]{ return NProust::TMaxOperationAvx2::FromMany<int32_t, int32_t>(Opaque(p), 0, leaf); }, calls);

        std::printf("%9zu | %14.1f | %14.1f | %14.1f | %9.2fx | %9.2fx\n",
                    leaf, novec, autovec, avx2, novec / avx2, autovec / avx2);
    }
    return 0;
}
