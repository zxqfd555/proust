// Query-ONLY microbenchmark. The whole earlier debate is about the read path: at equal branching
// factor the wide tree extracts far more memory-level parallelism than the robust fat tree, and
// that path contains no AVX2 in either engine. So here we strip the update entirely — every engine
// is populated ONCE (untimed), then we time nothing but ComputeRange. Pure query vs pure query.
//
// Two query shapes:
//   prefix — ComputeRange(0, r)   : one prefix walk, the cleanest latency-bound read
//   range  — ComputeRange(l, r)   : two prefix walks
//
// Two timing modes per shape:
//   bare   — tight loop, no fences        -> the CPU may overlap consecutive queries (throughput)
//   fenced — LFENCE + RDTSCP around each  -> overlap forbidden (per-op latency)
// overlap = fenced/bare on the prefix query: how many queries the core keeps in flight at once.
//
// Noise control: inputs are pre-touched, one untimed warmup pass primes caches, then each timed
// phase runs REPEATS passes and we keep the BEST (minimum) ns/op — the steady-state estimate.
// Concrete types throughout, so calls inline (no virtual dispatch in the measured region).

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

#include <x86intrin.h>

#include "engine/interface.h"
#include "engine/operations.h"
#include "engine/robust/engine.h"
#include "engine/external/algorithmica.h"
#include "engine/external/fenwick.h"

using namespace NProust;

// Fixed-storage variant of the B=16 int32-sum fat tree. The algorithm and node layout are a
// faithful copy of TFatPointUpdateEngine<int32,int32,TSumOperation,4>; the ONLY difference is that
// the node and value arrays are pre-allocated, 64-byte-aligned fixed buffers (like the wide tree's
// `int32_t Tree[]`) instead of std::vector. Isolates whether the std::vector backing — the data
// pointer indirection, lost alignment/aliasing info — is part of the query gap vs the wide tree.
template<size_t Capacity = 16'000'000>
class TFixedFatSumEngine final : public IPointUpdateEngine<int32_t, int32_t, TSumOperation> {
    static constexpr size_t BranchingLog = 4;
    static constexpr size_t CHILDREN_COUNT = 1 << BranchingLog;  // 16
    static constexpr size_t INDEX_SHIFT = CHILDREN_COUNT - 2;    // 14
    static constexpr size_t LEFTMOST_CHILD_INDEX = 2;
    static constexpr size_t ROOT_NODE_ID = 1;

    static constexpr size_t ComputeTotalNodes(size_t allocatedLength) {
        size_t totalNodes = 1, nodesInCurrentLayer = 1, nodesUnderOneLeaf = allocatedLength;
        while (nodesUnderOneLeaf > CHILDREN_COUNT) {
            nodesInCurrentLayer <<= BranchingLog;
            totalNodes += nodesInCurrentLayer;
            nodesUnderOneLeaf >>= BranchingLog;
        }
        return totalNodes;
    }
    static constexpr size_t MaxAllocatedLength = std::bit_ceil(Capacity + 1);
    static constexpr size_t MaxTotalNodes = ComputeTotalNodes(MaxAllocatedLength);

public:
    void Init(const size_t length) override {
        AllocatedLength = std::bit_ceil(length + 1);
        if (AllocatedLength < CHILDREN_COUNT) AllocatedLength = CHILDREN_COUNT;
        size_t totalNodes = 1, nodesInCurrentLayer = 1, nodesUnderOneLeaf = AllocatedLength;
        while (nodesUnderOneLeaf > CHILDREN_COUNT) {
            nodesInCurrentLayer <<= BranchingLog;
            totalNodes += nodesInCurrentLayer;
            nodesUnderOneLeaf >>= BranchingLog;
        }
        LeafSize = nodesUnderOneLeaf;
        LeafSizeLog = std::countr_zero(LeafSize);
        BottomLayerStart = (totalNodes + 1) - (AllocatedLength >> LeafSizeLog);
        std::memset(NodeValues, 0, (totalNodes + 1) * sizeof(int32_t));   // all-zero init (sum)
        std::memset(PointValues, 0, AllocatedLength * sizeof(int32_t));
    }

    void SetPoint(const size_t position, const int32_t& value) override {
        const int32_t delta = value - PointValues[position];
        PointValues[position] = value;
        size_t currentNode = BottomLayerStart + (position >> LeafSizeLog);
        while (currentNode >= ROOT_NODE_ID) {
            for (size_t index = currentNode + 1; (index & (CHILDREN_COUNT - 1)) != LEFTMOST_CHILD_INDEX; ++index) {
                NodeValues[index] += delta;
            }
            currentNode = (currentNode + INDEX_SHIFT) >> BranchingLog;
        }
    }

    int32_t ComputeRange(const size_t rangeStart, const size_t rangeEnd) const override {
        if (rangeStart + (CHILDREN_COUNT << 1) > rangeEnd) [[unlikely]] {
            return BruteForce(rangeStart, rangeEnd);
        }
        const int32_t full = PrefixBackwards(rangeEnd);
        if (rangeStart == 0) [[unlikely]] return full;
        return full - PrefixBackwards(rangeStart);
    }

private:
    int32_t BruteForce(size_t from, size_t to) const {
        int32_t r = 0;
        for (size_t i = from; i < to; ++i) r += PointValues[i];
        return r;
    }
    int32_t PrefixBackwards(size_t prefix) const {
        int32_t result;
        size_t currentNode = BottomLayerStart + (prefix >> LeafSizeLog);
        if (prefix & (LeafSize - 1)) [[likely]] {
            result = BruteForce(prefix & ~(LeafSize - 1), prefix);
        } else {
            result = NodeValues[currentNode];
            currentNode = (currentNode + INDEX_SHIFT) >> BranchingLog;
        }
        while (currentNode > ROOT_NODE_ID) {
            result = NodeValues[currentNode] + result;
            currentNode = (currentNode + INDEX_SHIFT) >> BranchingLog;
        }
        return result;
    }

    size_t AllocatedLength = 0, LeafSize = 0, LeafSizeLog = 0, BottomLayerStart = 0;
    alignas(64) int32_t NodeValues[MaxTotalNodes + 1];
    alignas(64) int32_t PointValues[MaxAllocatedLength];
};

// Fixed-trip, fully-unrolled B=16 int32-sum fat tree. Templated on the exact Length so the tree
// height (number of node reads per prefix walk) is a COMPILE-TIME constant; the walk is then a
// fold-expression over std::make_index_sequence<Layers>, i.e. fully unrolled, branchless, with
// per-layer addresses that are independent functions of n0 (compile-time constants folded in) and
// a reassociable integer-`+` reduction. This is the wide tree's query shape applied to the fat
// tree's packed layout — the last structural difference left to test. Update stays scalar.
template<size_t Length>
class TUnrolledFatSumEngine final : public IPointUpdateEngine<int32_t, int32_t, TSumOperation> {
    static constexpr size_t BranchingLog = 4;
    static constexpr size_t CHILDREN_COUNT = 16;
    static constexpr size_t INDEX_SHIFT = 14;
    static constexpr size_t LEFTMOST_CHILD_INDEX = 2;
    static constexpr size_t ROOT_NODE_ID = 1;

    static constexpr size_t CeilPow2(size_t x) { size_t p = 1; while (p < x) p <<= 1; return p; }
    static constexpr size_t AllocatedLength =
        CeilPow2(Length + 1) < CHILDREN_COUNT ? CHILDREN_COUNT : CeilPow2(Length + 1);

    struct TLayout { size_t totalNodes, leafSize, leafSizeLog, bottomStart, layers; };
    static constexpr TLayout ComputeLayout() {
        size_t totalNodes = 1, nodesInLayer = 1, nUnderLeaf = AllocatedLength;
        while (nUnderLeaf > CHILDREN_COUNT) { nodesInLayer <<= BranchingLog; totalNodes += nodesInLayer; nUnderLeaf >>= BranchingLog; }
        size_t leafSize = nUnderLeaf, leafSizeLog = 0;
        for (size_t t = leafSize; t > 1; t >>= 1) ++leafSizeLog;
        size_t bottomStart = (totalNodes + 1) - (AllocatedLength >> leafSizeLog);
        size_t layers = 1, node = bottomStart;  // layer 0 = the bottom node itself
        for (size_t parent = (node + INDEX_SHIFT) >> BranchingLog; parent > ROOT_NODE_ID;
             parent = (node + INDEX_SHIFT) >> BranchingLog) { ++layers; node = parent; }
        return {totalNodes, leafSize, leafSizeLog, bottomStart, layers};
    }
    static constexpr TLayout L = ComputeLayout();
    static constexpr size_t TotalNodes = L.totalNodes, LeafSize = L.leafSize,
                            LeafSizeLog = L.leafSizeLog, BottomLayerStart = L.bottomStart, Layers = L.layers;

    static constexpr std::array<size_t, Layers> MakeParentConstants() {
        std::array<size_t, Layers> c{};
        size_t acc = 0, powB = 1;
        for (size_t i = 1; i < Layers; ++i) { acc += (CHILDREN_COUNT - 2) * powB; c[i] = acc; powB <<= BranchingLog; }
        return c;
    }
    static constexpr std::array<size_t, Layers> ParentConstants = MakeParentConstants();

public:
    void Init(const size_t /*length*/) override {
        std::memset(NodeValues, 0, (TotalNodes + 1) * sizeof(int32_t));
        std::memset(PointValues, 0, AllocatedLength * sizeof(int32_t));
    }
    void SetPoint(const size_t position, const int32_t& value) override {
        const int32_t delta = value - PointValues[position];
        PointValues[position] = value;
        size_t currentNode = BottomLayerStart + (position >> LeafSizeLog);
        while (currentNode >= ROOT_NODE_ID) {
            for (size_t index = currentNode + 1; (index & (CHILDREN_COUNT - 1)) != LEFTMOST_CHILD_INDEX; ++index) {
                NodeValues[index] += delta;
            }
            currentNode = (currentNode + INDEX_SHIFT) >> BranchingLog;
        }
    }
    int32_t ComputeRange(const size_t l, const size_t r) const override {
        if (l + (CHILDREN_COUNT << 1) > r) [[unlikely]] return BruteForce(l, r);
        const int32_t full = PrefixBackwards(r);
        if (l == 0) [[unlikely]] return full;
        return full - PrefixBackwards(l);
    }

private:
    int32_t BruteForce(size_t from, size_t to) const { int32_t s = 0; for (size_t i = from; i < to; ++i) s += PointValues[i]; return s; }

    template<size_t I> inline int32_t Layer(size_t n0) const {
        constexpr size_t shift = I * BranchingLog;
        const size_t node = (n0 + ParentConstants[I]) >> shift;
        return node > ROOT_NODE_ID ? NodeValues[node] : 0;  // cmov, not a branch
    }
    template<size_t... I> inline int32_t SumLayers(size_t n0, std::index_sequence<I...>) const {
        return (Layer<I>(n0) + ... + 0);  // fully unrolled, reassociable
    }
    int32_t PrefixBackwards(size_t prefix) const {
        const size_t n0 = BottomLayerStart + (prefix >> LeafSizeLog);
        const int32_t base = (prefix & (LeafSize - 1)) ? BruteForce(prefix & ~(LeafSize - 1), prefix) : 0;
        return base + SumLayers(n0, std::make_index_sequence<Layers>{});
    }

    alignas(64) int32_t NodeValues[TotalNodes + 1];
    alignas(64) int32_t PointValues[AllocatedLength];
};

// EVERYTHING at once: the unrolled engine PLUS the leaf-tail removal. A parallel LeafPrefix[] holds
// the intra-block running prefix sums (LeafPrefix[j] = sum of the leaf block up to and including j),
// maintained on update, so the partial-leaf contribution of a query is ONE read instead of a
// brute-force scan of up to LeafSize-1 raw values. That is the last structural difference from the
// wide tree (which precomputes the same thing via its masked update). Combined with fixed storage,
// closed-form independent addresses, and the fixed-trip unrolled reassociable walk, this is the
// union of every optimization we have tried. Update stays scalar.
template<size_t Length>
class TAllOptFatSumEngine final : public IPointUpdateEngine<int32_t, int32_t, TSumOperation> {
    static constexpr size_t BranchingLog = 4;
    static constexpr size_t CHILDREN_COUNT = 16;
    static constexpr size_t INDEX_SHIFT = 14;
    static constexpr size_t LEFTMOST_CHILD_INDEX = 2;
    static constexpr size_t ROOT_NODE_ID = 1;

    static constexpr size_t CeilPow2(size_t x) { size_t p = 1; while (p < x) p <<= 1; return p; }
    static constexpr size_t AllocatedLength =
        CeilPow2(Length + 1) < CHILDREN_COUNT ? CHILDREN_COUNT : CeilPow2(Length + 1);

    struct TLayout { size_t totalNodes, leafSize, leafSizeLog, bottomStart, layers; };
    static constexpr TLayout ComputeLayout() {
        size_t totalNodes = 1, nodesInLayer = 1, nUnderLeaf = AllocatedLength;
        while (nUnderLeaf > CHILDREN_COUNT) { nodesInLayer <<= BranchingLog; totalNodes += nodesInLayer; nUnderLeaf >>= BranchingLog; }
        size_t leafSize = nUnderLeaf, leafSizeLog = 0;
        for (size_t t = leafSize; t > 1; t >>= 1) ++leafSizeLog;
        size_t bottomStart = (totalNodes + 1) - (AllocatedLength >> leafSizeLog);
        size_t layers = 1, node = bottomStart;
        for (size_t parent = (node + INDEX_SHIFT) >> BranchingLog; parent > ROOT_NODE_ID;
             parent = (node + INDEX_SHIFT) >> BranchingLog) { ++layers; node = parent; }
        return {totalNodes, leafSize, leafSizeLog, bottomStart, layers};
    }
    static constexpr TLayout L = ComputeLayout();
    static constexpr size_t TotalNodes = L.totalNodes, LeafSize = L.leafSize,
                            LeafSizeLog = L.leafSizeLog, BottomLayerStart = L.bottomStart, Layers = L.layers;

    static constexpr std::array<size_t, Layers> MakeParentConstants() {
        std::array<size_t, Layers> c{};
        size_t acc = 0, powB = 1;
        for (size_t i = 1; i < Layers; ++i) { acc += (CHILDREN_COUNT - 2) * powB; c[i] = acc; powB <<= BranchingLog; }
        return c;
    }
    static constexpr std::array<size_t, Layers> ParentConstants = MakeParentConstants();

public:
    void Init(const size_t /*length*/) override {
        std::memset(NodeValues, 0, (TotalNodes + 1) * sizeof(int32_t));
        std::memset(PointValues, 0, AllocatedLength * sizeof(int32_t));
        std::memset(LeafPrefix, 0, AllocatedLength * sizeof(int32_t));
    }
    void SetPoint(const size_t position, const int32_t& value) override {
        const int32_t delta = value - PointValues[position];
        PointValues[position] = value;
        // maintain intra-block running prefix sums: every LeafPrefix[j] with j >= position in the
        // same leaf block gains delta (a suffix-of-block update, the scalar analog of the wide
        // tree's masked vector add).
        const size_t blockEnd = (position | (LeafSize - 1)) + 1;
        for (size_t j = position; j < blockEnd; ++j) LeafPrefix[j] += delta;
        // inter-block node walk (unchanged)
        size_t currentNode = BottomLayerStart + (position >> LeafSizeLog);
        while (currentNode >= ROOT_NODE_ID) {
            for (size_t index = currentNode + 1; (index & (CHILDREN_COUNT - 1)) != LEFTMOST_CHILD_INDEX; ++index) {
                NodeValues[index] += delta;
            }
            currentNode = (currentNode + INDEX_SHIFT) >> BranchingLog;
        }
    }
    int32_t ComputeRange(const size_t l, const size_t r) const override {
        if (l + (CHILDREN_COUNT << 1) > r) [[unlikely]] {
            int32_t s = 0; for (size_t i = l; i < r; ++i) s += PointValues[i]; return s;
        }
        const int32_t full = PrefixBackwards(r);
        if (l == 0) [[unlikely]] return full;
        return full - PrefixBackwards(l);
    }

private:
    template<size_t I> inline int32_t Layer(size_t n0) const {
        constexpr size_t shift = I * BranchingLog;
        const size_t node = (n0 + ParentConstants[I]) >> shift;
        return node > ROOT_NODE_ID ? NodeValues[node] : 0;
    }
    template<size_t... I> inline int32_t SumLayers(size_t n0, std::index_sequence<I...>) const {
        return (Layer<I>(n0) + ... + 0);
    }
    int32_t PrefixBackwards(size_t prefix) const {
        const size_t n0 = BottomLayerStart + (prefix >> LeafSizeLog);
        // single read instead of a brute-force scan: LeafPrefix[prefix-1] = sum of the partial leaf.
        const int32_t base = (prefix & (LeafSize - 1)) ? LeafPrefix[prefix - 1] : 0;
        return base + SumLayers(n0, std::make_index_sequence<Layers>{});
    }

    alignas(64) int32_t NodeValues[TotalNodes + 1];
    alignas(64) int32_t PointValues[AllocatedLength];
    alignas(64) int32_t LeafPrefix[AllocatedLength];
};

// The production fat engine (B=16 int32 sum) with EXACTLY ONE change: leaf-tail removal. Same
// std::vector storage, same recurrence walk, same runtime Init(length). The query's partial-leaf
// contribution is one LeafPrefix read instead of a brute-force scan, and SetPoint maintains
// LeafPrefix. No unroll, no fixed storage, no closed-form addresses — this isolates whether
// leaf-tail removal ALONE captures the win, with no compile-time machinery and full runtime sizing.
class TFatLeafPrefixSumEngine final : public IPointUpdateEngine<int32_t, int32_t, TSumOperation> {
    static constexpr size_t BranchingLog = 4;
    static constexpr size_t CHILDREN_COUNT = 16;
    static constexpr size_t INDEX_SHIFT = 14;
    static constexpr size_t LEFTMOST_CHILD_INDEX = 2;
    static constexpr size_t ROOT_NODE_ID = 1;

public:
    void Init(const size_t length) override {
        AllocatedLength = std::bit_ceil(length + 1);
        if (AllocatedLength < CHILDREN_COUNT) AllocatedLength = CHILDREN_COUNT;
        size_t totalNodes = 1, nodesInLayer = 1, nUnderLeaf = AllocatedLength;
        while (nUnderLeaf > CHILDREN_COUNT) { nodesInLayer <<= BranchingLog; totalNodes += nodesInLayer; nUnderLeaf >>= BranchingLog; }
        LeafSize = nUnderLeaf;
        LeafSizeLog = std::countr_zero(LeafSize);
        BottomLayerStart = (totalNodes + 1) - (AllocatedLength >> LeafSizeLog);
        NodeValues.assign(totalNodes + 1, 0);
        PointValues.assign(AllocatedLength, 0);
        LeafPrefix.assign(AllocatedLength, 0);
    }
    void SetPoint(const size_t position, const int32_t& value) override {
        const int32_t delta = value - PointValues[position];
        PointValues[position] = value;
        const size_t blockEnd = (position | (LeafSize - 1)) + 1;   // suffix-of-block prefix maintenance
        for (size_t j = position; j < blockEnd; ++j) LeafPrefix[j] += delta;
        size_t currentNode = BottomLayerStart + (position >> LeafSizeLog);
        while (currentNode >= ROOT_NODE_ID) {
            for (size_t index = currentNode + 1; (index & (CHILDREN_COUNT - 1)) != LEFTMOST_CHILD_INDEX; ++index) {
                NodeValues[index] += delta;
            }
            currentNode = (currentNode + INDEX_SHIFT) >> BranchingLog;
        }
    }
    int32_t ComputeRange(const size_t l, const size_t r) const override {
        if (l + (CHILDREN_COUNT << 1) > r) [[unlikely]] {
            int32_t s = 0; for (size_t i = l; i < r; ++i) s += PointValues[i]; return s;
        }
        const int32_t full = Prefix(r);
        if (l == 0) [[unlikely]] return full;
        return full - Prefix(l);
    }

private:
    int32_t Prefix(size_t prefix) const {
        size_t currentNode = BottomLayerStart + (prefix >> LeafSizeLog);
        int32_t result;
        if (prefix & (LeafSize - 1)) [[likely]] {
            result = LeafPrefix[prefix - 1];   // <-- the ONLY change vs production (was a brute-force scan)
        } else {
            result = NodeValues[currentNode];
            currentNode = (currentNode + INDEX_SHIFT) >> BranchingLog;
        }
        while (currentNode > ROOT_NODE_ID) {
            result = NodeValues[currentNode] + result;
            currentNode = (currentNode + INDEX_SHIFT) >> BranchingLog;
        }
        return result;
    }

    std::vector<int32_t> NodeValues, PointValues, LeafPrefix;
    size_t AllocatedLength = 0, LeafSize = 0, LeafSizeLog = 0, BottomLayerStart = 0;
};

static inline uint64_t StartRDTSC() { _mm_lfence(); return __rdtsc(); }
static inline uint64_t EndRDTSC()   { uint32_t a; uint64_t t = __rdtscp(&a); _mm_lfence(); return t; }

using Clock = std::chrono::steady_clock;
static double NsPerOp(Clock::time_point a, Clock::time_point b, size_t q) {
    return std::chrono::duration<double, std::nano>(b - a).count() / q;
}

struct TWorkload {
    size_t N, Q;
    std::vector<int> pos, val, ql, qr;
};

static constexpr int BARE_REPEATS = 5;
static constexpr int FENCED_REPEATS = 2;

// --- bare update throughput: Q SetPoints, best of 3 passes ---
template <typename TEngine>
static double UpdateNs(TEngine& e, const TWorkload& w) {
    double best = 1e18;
    for (int rep = 0; rep < 3; ++rep) {
        const auto t0 = Clock::now();
        for (size_t i = 0; i < w.Q; ++i) e.SetPoint(w.pos[i], w.val[i]);
        const auto t1 = Clock::now();
        best = std::min(best, NsPerOp(t0, t1, w.Q));
    }
    return best;
}

// --- bare throughput over Q queries, best of BARE_REPEATS passes ---
template <typename TEngine, bool Prefix>
static double BareNs(const TEngine& e, const TWorkload& w, int64_t& checksum) {
    double best = 1e18;
    int64_t cs = 0;
    for (int rep = 0; rep < BARE_REPEATS; ++rep) {
        const auto t0 = Clock::now();
        for (size_t i = 0; i < w.Q; ++i) {
            cs ^= Prefix ? e.ComputeRange(0, w.qr[i]) : e.ComputeRange(w.ql[i], w.qr[i]);
        }
        const auto t1 = Clock::now();
        best = std::min(best, NsPerOp(t0, t1, w.Q));
    }
    checksum ^= cs;
    return best;
}

// --- fenced per-op latency over Q queries, best of FENCED_REPEATS passes ---
template <typename TEngine, bool Prefix>
static double FencedNs(const TEngine& e, const TWorkload& w, int64_t& checksum) {
    double best = 1e18;
    int64_t cs = 0;
    uint64_t sink = 0;
    for (int rep = 0; rep < FENCED_REPEATS; ++rep) {
        const auto t0 = Clock::now();
        for (size_t i = 0; i < w.Q; ++i) {
            const uint64_t s = StartRDTSC();
            cs ^= Prefix ? e.ComputeRange(0, w.qr[i]) : e.ComputeRange(w.ql[i], w.qr[i]);
            sink += EndRDTSC() - s;
        }
        const auto t1 = Clock::now();
        best = std::min(best, NsPerOp(t0, t1, w.Q));
    }
    checksum ^= cs ^ (int64_t)sink;
    return best;
}

template <typename TEngine>
static void RunRow(const char* name, const TWorkload& w) {
    // Heap-allocate: some engines (the wide tree at large capacity) embed a multi-MB array that
    // would overflow the stack. The static type stays the concrete `final` type, so calls through
    // the reference still devirtualize and inline.
    auto owned = std::make_unique<TEngine>();
    TEngine& engine = *owned;
    engine.Init(w.N);
    for (size_t i = 0; i < w.Q; ++i) engine.SetPoint(w.pos[i], w.val[i]);  // populate, UNTIMED

    const double updNs = UpdateNs<TEngine>(engine, w);  // SetPoint throughput

    int64_t checksum = 0;
    // warmup pass (untimed) so the first timed pass isn't paying first-touch costs
    for (size_t i = 0; i < w.Q; ++i) checksum ^= engine.ComputeRange(w.ql[i], w.qr[i]);

    // RANGE queries only, with random ql (rarely 0): this avoids the always-rangeStart==0 prefix
    // calls that consistently mispredict the [[unlikely]] branch in the fat tree's ComputeRange.
    const double bareR = BareNs<TEngine, false>(engine, w, checksum);
    const double fenR  = FencedNs<TEngine, false>(engine, w, checksum);

    std::printf("%-18s | %9.2f | %9.2f | %9.2f | %6.1fx  (cs %lld)\n",
                name, updNs, bareR, fenR,
                bareR > 0 ? fenR / bareR : 0.0, (long long)checksum);
}

int main(int argc, char** argv) {
    TWorkload w;
    // Working-set sweep: pass N on the command line (<= 1'000'000, the engines' compile-time cap)
    // to shrink the touched footprint into a given cache level. Queries are regenerated within
    // [0, N), so a smaller N means a smaller hot set, not just a smaller Init.
    w.N = (argc > 1) ? static_cast<size_t>(std::stoul(argv[1])) : 1'000'000;
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
    // Pre-touch every input array so the timed region measures the structure's access cost,
    // not the query-input pages faulting in.
    volatile int touch = 0;
    for (size_t i = 0; i < w.Q; ++i) touch ^= w.pos[i] ^ w.val[i] ^ w.ql[i] ^ w.qr[i];
    (void)touch;

    std::printf("RANGE-QUERY-ONLY. N=%zu, Q=%zu queries, best of %d bare / %d fenced passes. ns/op.\n",
                w.N, w.Q, BARE_REPEATS, FENCED_REPEATS);
    std::printf("%-18s | %9s | %9s | %9s | %s\n", "engine", "update", "bare rng", "fenced rng", "rng overlap");
    std::printf("%-18s-+-%9s-+-%9s-+-%9s-+-%s\n", "------------------", "---------", "---------", "---------", "-----------");

    // Fenwick needs a compile-time capacity >= the largest N swept (we go up to 20M). The wide tree
    // is capped at 16M: it sits just under 16^6, so its power-of-16 padding rounds to 16^6 = ~16.77M
    // slots (~72 MB) rather than 16^7 (~1.1 GB), and it is only run for N <= that cap.
    constexpr size_t FenwickCap = 24'000'000;
    constexpr size_t WideCap = 16'000'000;
    using TOp = TSumOperation;

    // Correctness gate: the fixed-storage port must produce identical query results to the real
    // std::vector B=16 fat tree. Compare over a sample of the workload queries before timing.
    {
        auto real = std::make_unique<TFatPointUpdateEngine<int32_t, int32_t, TOp, 4>>();
        auto leaf = std::make_unique<TFatLeafPrefixSumEngine>();
        real->Init(w.N); leaf->Init(w.N);
        for (size_t i = 0; i < w.Q; ++i) { real->SetPoint(w.pos[i], w.val[i]); leaf->SetPoint(w.pos[i], w.val[i]); }
        size_t mismatches = 0;
        const size_t checks = std::min<size_t>(w.Q, 200000);
        for (size_t i = 0; i < checks; ++i) {
            if (real->ComputeRange(w.ql[i], w.qr[i]) != leaf->ComputeRange(w.ql[i], w.qr[i]) ||
                real->ComputeRange(0, w.qr[i]) != leaf->ComputeRange(0, w.qr[i])) {
                ++mismatches;
            }
        }
        std::printf("correctness: +leafpfx vs real B=16 over %zu queries -> %zu mismatches%s\n",
                    checks, mismatches, mismatches ? "  *** FAIL ***" : "  (ok)");
        if (mismatches) return 1;
    }

    RunRow<TFatPointUpdateEngine<int32_t, int32_t, TOp, 4>>("robust orig (vec) ", w);
    RunRow<TFatLeafPrefixSumEngine>                        ("robust +leafpfx   ", w);
    RunRow<TFenwickPointSumEngine<FenwickCap>>             ("fenwick", w);
    if (w.N <= WideCap) RunRow<TAlgorithmicaWideSegmentSumEngine<WideCap>>("algorithmica B=16", w);
    return 0;
}
