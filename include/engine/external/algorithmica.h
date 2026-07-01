#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "../interface.h"
#include "../operations.h"

namespace NProust {
    namespace NAlgorithmicaDetail {
        // 256-bit vector of eight int32 lanes, via GCC/Clang vector extensions — the exact
        // type the reference uses. AVX2 lowers `&` and `+` to vpand / vpaddd.
        typedef int32_t TVec __attribute__((vector_size(32)));
    } // namespace NAlgorithmicaDetail

    // A port of the "wide segment tree", the fastest structure in Sergey Slotin's
    // Algorithmica/HPC segment-tree chapter (https://en.algorithmica.org/hpc/data-structures/
    // segment-trees/, file segtree/simd.cc in github.com/sslotin/amh-code). Like the Fenwick
    // baseline it is an *external* reference for one partial case only: point modifications and
    // range sums over a fixed-size array of int32.
    //
    // The idea: a B-ary tree (B = 16) laid out as a flat array of contiguous, cache-line-sized
    // nodes. Each node stores the B running prefix sums of its children, so a prefix-sum query
    // is just one scalar read per level (H reads total, no branches). An update adds the delta
    // to the suffix of one node per level, done branchlessly with a precomputed mask and a
    // single AVX2 fused load-and-add — hence the speed.
    //
    // `Sum(k)` returns the sum of the half-open prefix [0, k); a range sum [l, r) is
    // `Sum(r) - Sum(l)`. The reference uses compile-time global arrays sized to a power of two;
    // here the storage is owned and the slot count is padded up to the next power of B, so that
    // the whole-array query `Sum(length)` (the right endpoint of a full-range query) stays in
    // bounds. Accumulation is in int32, matching the dataset's ground-truth wrap semantics.
    template<size_t Capacity = 1'000'000>
    class TAlgorithmicaWideSegmentSumEngine final : public IPointUpdateEngine<int32_t, int32_t, TSumOperation> {
        static constexpr int BLog = 4;
        static constexpr int B = 1 << BLog;  // 16 children per node

        // The smallest power of B strictly greater than Capacity. "Strictly" matters: a
        // full-range query reads the prefix up to index `length`, so that index must be a valid
        // slot, which requires the padded size to exceed the capacity.
        static constexpr size_t ComputeSlots() {
            size_t slots = B;
            while (slots <= Capacity) {
                slots *= B;
            }
            return slots;
        }
        static constexpr size_t Slots = ComputeSlots();

        // Number of layers, i.e. log_B(Slots).
        static constexpr int ComputeHeight() {
            int height = 0;
            for (size_t slots = Slots; slots > 1; slots /= B) {
                ++height;
            }
            return height;
        }
        static constexpr int H = ComputeHeight();

        // Start of layer `targetLayer` in the flat array. Layer 0 is the bottom (finest) layer;
        // every layer is rounded up to a whole number of B-sized nodes.
        static constexpr size_t LayerOffset(const int targetLayer) {
            size_t start = 0;
            size_t nodes = Slots;
            for (int layer = 0; layer < targetLayer; ++layer) {
                nodes = (nodes + B - 1) / B;
                start += nodes * B;
            }
            return start;
        }

        // Precompute every layer's offset so the hot loops index a constant table rather than
        // re-deriving offsets each iteration — matching how the reference's `offset(h)` folds to
        // a constant when the loop is unrolled.
        static constexpr std::array<size_t, H + 1> ComputeOffsets() {
            std::array<size_t, H + 1> offsets{};
            for (int layer = 0; layer <= H; ++layer) {
                offsets[layer] = LayerOffset(layer);
            }
            return offsets;
        }
        static constexpr std::array<size_t, H + 1> Offsets = ComputeOffsets();
        static constexpr size_t TreeSize = Offsets[H];

        static constexpr int RoundDownToBlock(const int k) {
            return k & ~(B - 1);  // = k / B * B
        }

    public:
        TAlgorithmicaWideSegmentSumEngine() = default;

        virtual void Init(const size_t length) override {
            assert(length <= Capacity && "The wide segment tree is fixed to its compile-time capacity");
            Length = length;
            std::memset(Tree, 0, sizeof(Tree));
            Values.assign(Capacity, 0);
            // The suffix mask: within a node, an update at local position p must bump every
            // prefix sum strictly past p, so lane i is all-ones exactly when i > p.
            for (int position = 0; position < B; ++position) {
                for (int lane = 0; lane < B; ++lane) {
                    Mask[position][lane] = (lane > position) ? -1 : 0;
                }
            }
        }

        virtual void SetPoint(const size_t position, const int32_t& value) override {
            assert(position < Length && "The point modification goes beyond the active length");
            // The tree speaks deltas; Values remembers the current element so a "set" becomes one.
            const int32_t delta = value - Values[position];
            Values[position] = value;
            Add(static_cast<int>(position), delta);
        }

        virtual int32_t ComputeRange(const size_t left, const size_t right) const override {
            assert(left < right && "The operation result for an empty range is undefined");
            assert(right <= Length && "The range goes beyond the active length");
            return Sum(static_cast<int>(right)) - Sum(static_cast<int>(left));
        }

    private:
        // Sum of the half-open prefix [0, k): one read per layer, no branches.
        int32_t Sum(const int k) const {
            int32_t result = 0;
            for (int layer = 0; layer < H; ++layer) {
                result += Tree[Offsets[layer] + (k >> (layer * BLog))];
            }
            return result;
        }

        // Add `x` to element `k`: at each layer, fold the delta into the suffix of the one node
        // that covers k, using the precomputed mask and an AVX2 masked add.
        void Add(int k, const int32_t x) {
            using NAlgorithmicaDetail::TVec;
            const TVec broadcast = x + TVec{};  // splat x across all eight lanes
            for (int layer = 0; layer < H; ++layer) {
                auto* node = reinterpret_cast<TVec*>(&Tree[Offsets[layer] + RoundDownToBlock(k)]);
                auto* mask = reinterpret_cast<TVec*>(&Mask[k & (B - 1)][0]);
                for (int i = 0; i < B / 8; ++i) {
                    node[i] += broadcast & mask[i];
                }
                k >>= BLog;
            }
        }

        size_t Length = 0;
        alignas(64) int32_t Tree[TreeSize];  // flat layered storage; 64-byte aligned for the vector loads
        alignas(64) int32_t Mask[B][B];      // suffix masks, one row per intra-node position
        std::vector<int32_t> Values;         // current element values, for delta-on-assign
    };

    inline std::unique_ptr<IPointUpdateEngine<int32_t, int32_t, TSumOperation>> CreateAlgorithmicaWideSegmentSumEngine() {
        // The engine is over-aligned and large (a few MB of Tree); make_unique routes through the
        // aligned operator new, and the base class's virtual destructor frees it correctly.
        return std::make_unique<TAlgorithmicaWideSegmentSumEngine<>>();
    }

    // XOR specialization of the same wide segment tree. XOR is a commutative group whose inverse is
    // itself (a ^ a == 0), so the structure carries over verbatim with `+`/`-` replaced by `^`: each
    // node stores running prefix-XORs of its children, a prefix query XORs one read per layer, a
    // range XOR is `prefix(r) ^ prefix(l)`, and a point update folds `old ^ new` into the suffix with
    // a masked vector XOR. This is exactly the contrast with the generic engine: the upstream code is
    // hard-wired to addition, so supporting XOR means *re-deriving and re-writing the whole structure*
    // here, whereas the generic engine needed only the one `ApplyToMany` hook on `TBitwiseXorOperation`.
    template<size_t Capacity = 1'000'000>
    class TAlgorithmicaWideSegmentXorEngine final : public IPointUpdateEngine<int32_t, int32_t, TBitwiseXorOperation> {
        static constexpr int BLog = 4;
        static constexpr int B = 1 << BLog;

        static constexpr size_t ComputeSlots() {
            size_t slots = B;
            while (slots <= Capacity) {
                slots *= B;
            }
            return slots;
        }
        static constexpr size_t Slots = ComputeSlots();

        static constexpr int ComputeHeight() {
            int height = 0;
            for (size_t slots = Slots; slots > 1; slots /= B) {
                ++height;
            }
            return height;
        }
        static constexpr int H = ComputeHeight();

        static constexpr size_t LayerOffset(const int targetLayer) {
            size_t start = 0;
            size_t nodes = Slots;
            for (int layer = 0; layer < targetLayer; ++layer) {
                nodes = (nodes + B - 1) / B;
                start += nodes * B;
            }
            return start;
        }

        static constexpr std::array<size_t, H + 1> ComputeOffsets() {
            std::array<size_t, H + 1> offsets{};
            for (int layer = 0; layer <= H; ++layer) {
                offsets[layer] = LayerOffset(layer);
            }
            return offsets;
        }
        static constexpr std::array<size_t, H + 1> Offsets = ComputeOffsets();
        static constexpr size_t TreeSize = Offsets[H];

        static constexpr int RoundDownToBlock(const int k) {
            return k & ~(B - 1);
        }

    public:
        TAlgorithmicaWideSegmentXorEngine() = default;

        virtual void Init(const size_t length) override {
            assert(length <= Capacity && "The wide segment tree is fixed to its compile-time capacity");
            Length = length;
            std::memset(Tree, 0, sizeof(Tree));   // 0 is the XOR identity
            Values.assign(Capacity, 0);
            for (int position = 0; position < B; ++position) {
                for (int lane = 0; lane < B; ++lane) {
                    Mask[position][lane] = (lane > position) ? -1 : 0;
                }
            }
        }

        virtual void SetPoint(const size_t position, const int32_t& value) override {
            assert(position < Length && "The point modification goes beyond the active length");
            // Flipping the element from old to new is one XOR by (old ^ new).
            const int32_t delta = value ^ Values[position];
            Values[position] = value;
            Apply(static_cast<int>(position), delta);
        }

        virtual int32_t ComputeRange(const size_t left, const size_t right) const override {
            assert(left < right && "The operation result for an empty range is undefined");
            assert(right <= Length && "The range goes beyond the active length");
            return Prefix(static_cast<int>(right)) ^ Prefix(static_cast<int>(left));
        }

    private:
        // XOR of the half-open prefix [0, k): one read per layer, no branches.
        int32_t Prefix(const int k) const {
            int32_t result = 0;
            for (int layer = 0; layer < H; ++layer) {
                result ^= Tree[Offsets[layer] + (k >> (layer * BLog))];
            }
            return result;
        }

        // Fold `x` into element `k`: masked vector XOR over the suffix of one node per layer.
        void Apply(int k, const int32_t x) {
            using NAlgorithmicaDetail::TVec;
            const TVec broadcast = x + TVec{};  // splat x across all eight lanes (bit pattern only)
            for (int layer = 0; layer < H; ++layer) {
                auto* node = reinterpret_cast<TVec*>(&Tree[Offsets[layer] + RoundDownToBlock(k)]);
                auto* mask = reinterpret_cast<TVec*>(&Mask[k & (B - 1)][0]);
                for (int i = 0; i < B / 8; ++i) {
                    node[i] ^= broadcast & mask[i];
                }
                k >>= BLog;
            }
        }

        size_t Length = 0;
        alignas(64) int32_t Tree[TreeSize];
        alignas(64) int32_t Mask[B][B];
        std::vector<int32_t> Values;
    };

    inline std::unique_ptr<IPointUpdateEngine<int32_t, int32_t, TBitwiseXorOperation>> CreateAlgorithmicaWideSegmentXorEngine() {
        return std::make_unique<TAlgorithmicaWideSegmentXorEngine<>>();
    }
} // namespace NProust
