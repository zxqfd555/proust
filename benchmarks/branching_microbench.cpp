// Isolated experiment: does giving the robust point-update tree a WIDER branching factor buy it
// the same cross-operation memory-level parallelism the Algorithmica wide tree enjoys?
//
// TFatPointUpdateEngine<..., BranchingLog> is a multi-way Fenwick-style tree: CHILDREN_COUNT =
// 1<<BranchingLog. We sweep BranchingLog = 1 (binary) .. 6 (64-way) on the SumInt32 point problem,
// the SAME workload as external_microbench.cpp, and for each report:
//   - bare throughput  (tight loop, no fences)         -> overlap allowed
//   - fenced latency   (LFENCE + RDTSCP around each op) -> overlap forbidden
// All engines are driven through their CONCRETE type so the calls inline (no virtual dispatch),
// isolating the algorithm. If wider branching lifts throughput while fenced latency stays flat,
// branching is the lever for parallelism.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include <x86intrin.h>

#include "engine/interface.h"
#include "engine/operations.h"
#include "engine/robust/engine.h"
#include "engine/external/algorithmica.h"

using namespace NProust;

static inline uint64_t StartRDTSC() {
    _mm_lfence();
    return __rdtsc();
}
static inline uint64_t EndRDTSC() {
    uint32_t aux;
    uint64_t t = __rdtscp(&aux);
    _mm_lfence();
    return t;
}

struct TWorkload {
    size_t N, Q;
    std::vector<int> pos, val, ql, qr;
};

using Clock = std::chrono::steady_clock;
static double PerOpNs(Clock::time_point a, Clock::time_point b, size_t q) {
    return std::chrono::duration<double, std::nano>(b - a).count() / q;
}

// Time one concrete engine on the workload, both bare and fenced, and print a row.
template <typename TEngine>
static void RunRow(const char* name, const TWorkload& w) {
    TEngine engine;
    engine.Init(w.N);
    for (size_t i = 0; i < w.Q; ++i) engine.SetPoint(w.pos[i], w.val[i]);  // warm up / populate

    // ---- bare throughput ----
    const auto u0 = Clock::now();
    for (size_t i = 0; i < w.Q; ++i) engine.SetPoint(w.pos[i], w.val[i]);
    const auto u1 = Clock::now();

    int64_t rcs = 0;
    const auto r0 = Clock::now();
    for (size_t i = 0; i < w.Q; ++i) rcs ^= engine.ComputeRange(w.ql[i], w.qr[i]);
    const auto r1 = Clock::now();

    int64_t pcs = 0;
    const auto p0 = Clock::now();
    for (size_t i = 0; i < w.Q; ++i) pcs ^= engine.ComputeRange(0, w.qr[i]);
    const auto p1 = Clock::now();

    const double bareU = PerOpNs(u0, u1, w.Q);
    const double bareR = PerOpNs(r0, r1, w.Q);
    const double bareP = PerOpNs(p0, p1, w.Q);

    // ---- fenced latency ----
    uint64_t sink = 0;
    const auto fu0 = Clock::now();
    for (size_t i = 0; i < w.Q; ++i) { uint64_t s = StartRDTSC(); engine.SetPoint(w.pos[i], w.val[i]); sink += EndRDTSC() - s; }
    const auto fu1 = Clock::now();

    int64_t frcs = 0;
    const auto fr0 = Clock::now();
    for (size_t i = 0; i < w.Q; ++i) { uint64_t s = StartRDTSC(); frcs ^= engine.ComputeRange(w.ql[i], w.qr[i]); sink += EndRDTSC() - s; }
    const auto fr1 = Clock::now();

    int64_t fpcs = 0;
    const auto fp0 = Clock::now();
    for (size_t i = 0; i < w.Q; ++i) { uint64_t s = StartRDTSC(); fpcs ^= engine.ComputeRange(0, w.qr[i]); sink += EndRDTSC() - s; }
    const auto fp1 = Clock::now();

    const double fenU = PerOpNs(fu0, fu1, w.Q);
    const double fenR = PerOpNs(fr0, fr1, w.Q);
    const double fenP = PerOpNs(fp0, fp1, w.Q);

    // prefix is the cleanest latency-bound query; report its overlap factor (fenced/bare).
    std::printf("%-18s | %7.1f %7.1f %7.1f | %7.1f %7.1f %7.1f | %6.1fx  (cs %lld/%lld/%llu)\n",
                name, bareU, bareR, bareP, fenU, fenR, fenP,
                bareP > 0 ? fenP / bareP : 0.0,
                (long long)(rcs ^ frcs), (long long)(pcs ^ fpcs), (unsigned long long)sink);
}

int main() {
    TWorkload w;
    w.N = 1'000'000;
    w.Q = 5'000'000;
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> positionDist(0, (int)w.N - 1);
    std::uniform_int_distribution<int> valueDist(0, 9);
    w.pos.resize(w.Q); w.val.resize(w.Q); w.ql.resize(w.Q); w.qr.resize(w.Q);
    for (size_t i = 0; i < w.Q; ++i) {
        w.pos[i] = positionDist(rng);
        w.val[i] = valueDist(rng);
        int a = positionDist(rng), b = positionDist(rng);
        int l = std::min(a, b), r = std::max(a, b);
        if (l == r) ++r;
        w.ql[i] = l; w.qr[i] = r;
    }

    std::printf("N=%zu, Q=%zu per phase. ns/op.\n", w.N, w.Q);
    std::printf("%-18s | %23s | %23s | %s\n", "engine", "bare (upd/rng/pfx)", "fenced (upd/rng/pfx)", "pfx overlap");
    std::printf("%-18s-+-%23s-+-%23s-+-%s\n", "------------------", "-----------------------", "-----------------------", "-----------");

    using TOp = TSumOperation;
    RunRow<TFatPointUpdateEngine<int32_t, int32_t, TOp, 1>>("robust B=2  (log1)", w);
    RunRow<TFatPointUpdateEngine<int32_t, int32_t, TOp, 2>>("robust B=4  (log2)", w);
    RunRow<TFatPointUpdateEngine<int32_t, int32_t, TOp, 3>>("robust B=8  (log3)", w);
    RunRow<TFatPointUpdateEngine<int32_t, int32_t, TOp, 4>>("robust B=16 (log4)", w);
    RunRow<TFatPointUpdateEngine<int32_t, int32_t, TOp, 5>>("robust B=32 (log5)", w);
    RunRow<TFatPointUpdateEngine<int32_t, int32_t, TOp, 6>>("robust B=64 (log6)", w);
    RunRow<TAlgorithmicaWideSegmentSumEngine<>>("algorithmica B=16", w);
    return 0;
}
