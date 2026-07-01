// Why does per-operation timing inflate some engines far more than others, when the RDTSC + fence
// instructions cost the same everywhere? This isolates the mechanism on the prefix-sum query.
//
// For each engine we time the SAME query loop two ways:
//   throughput      — bare loop; the CPU is free to overlap one query's cache misses with the next.
//   latency(fenced) — each query bracketed by LFENCE/RDTSCP, so it must fully retire before the next
//                     begins (this is what the runner's per-operation mode measures).
// The "gap" is latency - throughput. A fixed RDTSC cost would make the gap constant across engines;
// the fence floor below shows what that fixed cost actually is. Anything beyond it is overlap the
// fences destroyed — large for latency-bound structures, near zero for compute-bound ones.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include <x86intrin.h>

#include "engine/operations.h"
#include "engine/robust/engine.h"
#include "engine/external/fenwick.h"
#include "engine/external/algorithmica.h"

using namespace NProust;

static inline uint64_t StartRDTSC() { _mm_lfence(); return __rdtsc(); }
static inline uint64_t EndRDTSC() { uint32_t aux; uint64_t t = __rdtscp(&aux); _mm_lfence(); return t; }

using Clock = std::chrono::steady_clock;
static double nsPer(Clock::time_point a, Clock::time_point b, size_t q) {
    return std::chrono::duration<double, std::nano>(b - a).count() / q;
}

template<class TEngine>
void Bench(const char* name, TEngine* engine, const std::vector<int>& qr, size_t Q) {
    int64_t checksum = 0;

    const auto t0 = Clock::now();
    for (size_t i = 0; i < Q; ++i) {
        checksum ^= engine->ComputeRange(0, qr[i]);
    }
    const auto t1 = Clock::now();
    const double throughput = nsPer(t0, t1, Q);

    uint64_t sink = 0;
    const auto t2 = Clock::now();
    for (size_t i = 0; i < Q; ++i) {
        const uint64_t s = StartRDTSC();
        checksum ^= engine->ComputeRange(0, qr[i]);
        sink += EndRDTSC() - s;
    }
    const auto t3 = Clock::now();
    const double latency = nsPer(t2, t3, Q);

    std::printf("%-14s throughput %6.1f ns/op   latency(fenced) %6.1f ns/op   gap %6.1f ns   (cs %lld, sink %llu)\n",
                name, throughput, latency, latency - throughput,
                static_cast<long long>(checksum), static_cast<unsigned long long>(sink));
}

int main() {
    const size_t N = 1'000'000;
    const size_t Q = 5'000'000;

    std::mt19937 rng(7);
    std::uniform_int_distribution<int> posDist(0, static_cast<int>(N) - 1);
    std::uniform_int_distribution<int> valDist(0, 9);
    std::vector<int> p(Q), v(Q), qr(Q);
    for (size_t i = 0; i < Q; ++i) {
        p[i] = posDist(rng);
        v[i] = valDist(rng);
        const int r = posDist(rng);
        qr[i] = r != 0 ? r : 1;
    }

    {
        volatile uint64_t x = 0;
        uint64_t acc = 0;
        const auto t0 = Clock::now();
        for (size_t i = 0; i < Q; ++i) {
            const uint64_t s = StartRDTSC();
            x = x + 1;
            acc += EndRDTSC() - s;
        }
        const auto t1 = Clock::now();
        std::printf("fence floor (LFENCE/RDTSCP around a no-op): %.1f ns/op\n\n", nsPer(t0, t1, Q));
    }

    auto robust = std::make_unique<TFatPointUpdateEngine<int32_t, int32_t, TSumOperation, 4>>();
    auto fenwick = std::make_unique<TFenwickPointSumEngine<>>();
    auto algorithmica = std::make_unique<TAlgorithmicaWideSegmentSumEngine<>>();
    robust->Init(N);
    fenwick->Init(N);
    algorithmica->Init(N);
    for (size_t i = 0; i < Q; ++i) {
        robust->SetPoint(p[i], v[i]);
        fenwick->SetPoint(p[i], v[i]);
        algorithmica->SetPoint(p[i], v[i]);
    }

    Bench("robust(fat)", robust.get(), qr, Q);
    Bench("fenwick", fenwick.get(), qr, Q);
    Bench("algorithmica", algorithmica.get(), qr, Q);
    return 0;
}
