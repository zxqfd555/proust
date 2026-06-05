#pragma once

#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

#include "../interface.h"
#include "../operations.h"

namespace NProust {
    // A Fenwick tree (binary indexed tree) specialized for the single partial case that
    // matters for the benchmarks: point assignments and half-open range-sum queries over a
    // fixed-size array of `int32_t`. It is an *external* reference implementation — it does
    // not plug into the generic engine machinery (no configurable value/result types, no
    // layouts or topologies); it exists only to give the generic engines a well-understood,
    // minimal-constant baseline to be measured against.
    //
    // The structure is sized once, at compile time, to the dataset's fixed length (10^6).
    // A Fenwick tree gains nothing algorithmically from a static size, but pinning it makes
    // the intent explicit: this is the "IntSum32, point modification, length 10^6" engine and
    // nothing else. Both the prefix-sum query and the point update run in O(log N).
    //
    // Accumulation is deliberately kept in `int32_t` so the result matches the ground truth
    // bit-for-bit: the dataset's reference answers are produced by `TSumOperation::Merge`,
    // which also adds in `int32_t`. The summation-task generator bounds element magnitudes to
    // `INT32_MAX / length`, so no individual prefix sum can overflow the type.
    template<size_t Capacity = 1'000'000>
    class TFenwickPointSumEngine final : public IPointUpdateEngine<int32_t, int32_t, TSumOperation> {
    public:
        TFenwickPointSumEngine() = default;

        virtual void Init(const size_t length) override {
            assert(length <= Capacity && "The Fenwick engine is fixed to its compile-time capacity");
            Length = length;
            // The tree is 1-indexed, hence the extra slot. Values keeps the current element
            // so a "set" can be expressed as the "add a delta" the tree natively supports.
            Tree.assign(Capacity + 1, 0);
            Values.assign(Capacity, 0);
        }

        virtual void SetPoint(const size_t position, const int32_t& value) override {
            assert(position < Length && "The point modification goes beyond the active length");
            const int32_t delta = value - Values[position];
            Values[position] = value;
            // The element at 0-based `position` lives at 1-based index `position + 1`.
            for (size_t index = position + 1; index <= Length; index += LowestSetBit(index)) {
                Tree[index] += delta;
            }
        }

        virtual int32_t ComputeRange(const size_t left, const size_t right) const override {
            assert(left < right && "The operation result for an empty range is undefined");
            assert(right <= Length && "The range goes beyond the active length");
            // Half-open [left, right) = prefix(right) - prefix(left).
            return PrefixSum(right) - PrefixSum(left);
        }

    private:
        // Sum of the first `prefix` elements, i.e. of the 0-based half-open range [0, prefix).
        int32_t PrefixSum(const size_t prefix) const {
            int32_t result = 0;
            for (size_t index = prefix; index > 0; index -= LowestSetBit(index)) {
                result += Tree[index];
            }
            return result;
        }

        static constexpr size_t LowestSetBit(const size_t value) {
            return value & (~value + 1);
        }

        size_t Length = 0;
        std::vector<int32_t> Tree;    // 1-indexed binary indexed tree
        std::vector<int32_t> Values;  // current element values, for delta-on-assign
    };

    inline std::unique_ptr<IPointUpdateEngine<int32_t, int32_t, TSumOperation>> CreateFenwickPointSumEngine() {
        return std::make_unique<TFenwickPointSumEngine<>>();
    }
} // namespace NProust
