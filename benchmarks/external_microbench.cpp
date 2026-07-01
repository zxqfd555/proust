// A tight-loop microbenchmark of the three SumInt32 point-update engines, with none of the
// dataset-streaming / per-op RDTSC / correctness-check overhead the main runner carries. That
// overhead (tens of nanoseconds per op) sits above the cost of the fastest structures here, so
// the runner cannot resolve their differences — this isolates the engines themselves.
//
// Every engine is driven through its CONCRETE type, so the hot-path calls inline and there is no
// virtual dispatch — exactly how the runner holds engines in production (concrete type, classes
// marked `final`). The polymorphic `IPointUpdateEngine` path is gone: it imposed a per-op indirect
// call that throttled the latency-bound wide tree's memory-level parallelism (~6× on its query),
// while barely touching the work-bound engines — there is no reason to keep it as the default.
// Three workloads are timed separately:
//   update  — SetPoint(position, value)
//   range   — ComputeRange(l, r), a half-open range sum (two prefix walks for the prefix trees)
//   prefix  — ComputeRange(0, r), a single prefix sum (the workload the Algorithmica article times)

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "engine/interface.h"
#include "engine/operations.h"
#include "engine/robust/engine.h"
#include "engine/external/fenwick.h"
#include "engine/external/algorithmica.h"

using namespace NProust;

using Clock = std::chrono::steady_clock;
static double PerOpNs(Clock::time_point a, Clock::time_point b, size_t q) {
    return std::chrono::duration<double, std::nano>(b - a).count() / q;
}

// Time one concrete engine on the update / range / prefix workloads. The static type is the
// concrete `final` engine, so ComputeRange/SetPoint devirtualize and inline.
template <typename TEngine>
static void Run(const char* name, size_t N, size_t Q,
                const std::vector<int>& pos, const std::vector<int>& val,
                const std::vector<int>& ql, const std::vector<int>& qr) {
    auto owned = std::make_unique<TEngine>();
    TEngine* engine = owned.get();
    engine->Init(N);
    for (size_t i = 0; i < Q; ++i) engine->SetPoint(pos[i], val[i]);  // warm up / populate

    const auto u0 = Clock::now();
    for (size_t i = 0; i < Q; ++i) engine->SetPoint(pos[i], val[i]);
    const auto u1 = Clock::now();

    int64_t rcs = 0;
    const auto r0 = Clock::now();
    for (size_t i = 0; i < Q; ++i) rcs ^= engine->ComputeRange(ql[i], qr[i]);
    const auto r1 = Clock::now();

    int64_t pcs = 0;
    const auto p0 = Clock::now();
    for (size_t i = 0; i < Q; ++i) pcs ^= engine->ComputeRange(0, qr[i]);
    const auto p1 = Clock::now();

    std::printf("%-14s %12.2f %12.2f %12.2f   (cs %lld/%lld)\n", name,
                PerOpNs(u0, u1, Q), PerOpNs(r0, r1, Q), PerOpNs(p0, p1, Q),
                static_cast<long long>(rcs), static_cast<long long>(pcs));
}

// The verbatim upstream wide segment tree (sslotin/amh-code segtree/simd.cc): global arrays,
// free functions, no class wrapping and no virtual dispatch. Benched through the SAME precise
// timing loop, it tells us whether any gap is in the algorithm or in how the engine is wrapped.
// `RawSetPoint` adds the delta-on-assign that our problem needs (the upstream `add` only does raw
// deltas), to stay apples-to-apples.
namespace NRaw {
    typedef int TVec __attribute__((vector_size(32)));
    constexpr int b = 4, B = (1 << b);
    constexpr int RawN = 1 << 20;  // >= 10^6, a power of B, exactly the article's N

    constexpr int Round(int k) { return k & ~(B - 1); }
    constexpr int Height(int n) { return (n <= B ? 1 : Height(n / B) + 1); }
    constexpr int Offset(int h) {
        int s = 0, n = RawN;
        while (h--) { n = (n + B - 1) / B; s += n * B; }
        return s;
    }
    constexpr int RawH = Height(RawN);

    alignas(64) int t[Offset(RawH)];
    alignas(64) int mask[B][B];
    int values[RawN];

    void RawInit() {
        std::memset(t, 0, sizeof(t));
        std::memset(values, 0, sizeof(values));
        for (int k = 0; k < B; k++)
            for (int i = 0; i < B; i++)
                mask[k][i] = (i > k ? -1 : 0);
    }
    int RawSum(int k) {
        int res = 0;
        for (int h = 0; h < RawH; h++)
            res += t[Offset(h) + (k >> (h * b))];
        return res;
    }
    void RawAdd(int k, int x) {
        TVec xv = x + TVec{};
        for (int h = 0; h < RawH; h++) {
            auto* l = (TVec*) &t[Offset(h) + Round(k)];
            auto* m = (TVec*) mask[k % B];
            for (int i = 0; i < B / 8; i++)
                l[i] += xv & m[i];
            k >>= b;
        }
    }
    void RawSetPoint(int k, int v) { int d = v - values[k]; values[k] = v; RawAdd(k, d); }
    int RawRange(int l, int r) { return RawSum(r) - RawSum(l); }
} // namespace NRaw

int main() {
    const size_t N = 1'000'000;
    const size_t Q = 5'000'000;

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> positionDist(0, static_cast<int>(N) - 1);
    std::uniform_int_distribution<int> valueDist(0, 9);

    std::vector<int> pos(Q), val(Q), ql(Q), qr(Q);
    for (size_t i = 0; i < Q; ++i) {
        pos[i] = positionDist(rng);
        val[i] = valueDist(rng);
        int a = positionDist(rng), b = positionDist(rng);
        int l = std::min(a, b), r = std::max(a, b);
        if (l == r) {
            ++r;  // keep l < r <= N
        }
        ql[i] = l;
        qr[i] = r;  // always >= 1, so [0, qr) is a valid non-empty prefix
    }

    std::printf("N=%zu, Q=%zu ops per phase. Every engine driven through its concrete type.\n", N, Q);
    std::printf("%-14s %12s %12s %12s\n", "engine", "update ns", "range ns", "prefix ns");

    // The robust factory selects the fat (Fenwick-style) engine for sum + point, since TSumOperation
    // has Undo: TFatPointUpdateEngine with branching 4 (B = 16).
    Run<TFatPointUpdateEngine<int32_t, int32_t, TSumOperation, 4>>("robust fat", N, Q, pos, val, ql, qr);
    Run<TFenwickPointSumEngine<>>("fenwick", N, Q, pos, val, ql, qr);
    Run<TAlgorithmicaWideSegmentSumEngine<>>("algorithmica", N, Q, pos, val, ql, qr);

    // The raw upstream wide tree, timed identically (free functions, no class wrapping).
    {
        NRaw::RawInit();
        for (size_t i = 0; i < Q; ++i) NRaw::RawSetPoint(pos[i], val[i]);

        const auto u0 = Clock::now();
        for (size_t i = 0; i < Q; ++i) NRaw::RawSetPoint(pos[i], val[i]);
        const auto u1 = Clock::now();

        int64_t rcs = 0;
        const auto r0 = Clock::now();
        for (size_t i = 0; i < Q; ++i) rcs ^= NRaw::RawRange(ql[i], qr[i]);
        const auto r1 = Clock::now();

        int64_t pcs = 0;
        const auto p0 = Clock::now();
        for (size_t i = 0; i < Q; ++i) pcs ^= NRaw::RawSum(qr[i]);
        const auto p1 = Clock::now();

        std::printf("%-14s %12.2f %12.2f %12.2f   (cs %lld/%lld)\n", "raw-upstream",
                    PerOpNs(u0, u1, Q), PerOpNs(r0, r1, Q), PerOpNs(p0, p1, Q),
                    static_cast<long long>(rcs), static_cast<long long>(pcs));
    }
    return 0;
}
