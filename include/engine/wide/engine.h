#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

#include "../interface.h"

namespace NProust {
    // A range-assignment engine over a 16-ary tree, for any ValidOperation: nothing beyond FromRepeated
    // and Merge is asked of the operation (HasFromMany is used on the leaves when present).
    //
    // Level 0 nodes (leaves) hold 16 raw values. A node of level h > 0 has 16 slots, each covering
    // a level h - 1 child of span 16^h elements: the slot keeps the child's result and, if the whole
    // child was assigned a value it has not been told about yet, that value as a tag.
    //
    // The tree spans a multiple of 16^h elements, past the length as needed. A slot reaching past the
    // length holds a valid but meaningless result (the extra elements are TValue()): it is never read,
    // since every operation stays inside the length, so no slot that holds one is ever fully covered.
    //
    // An assignment walks its two boundary paths side by side, and then refolds them side by side:
    // two independent dependency chains rather than one twice as long. A query never pushes tags, it
    // answers a tagged boundary slot with FromRepeated.
    template<typename TValue, typename TOperationResult, typename TOperation>
    requires ValidOperation<TOperation, TValue, TOperationResult>
    class TWideRangeUpdateEngine final : public IRangeUpdateEngine<TValue, TOperationResult, TOperation> {
    public:
        virtual void Init(const size_t length) override {
            AllocateGeometry(length);
            for (TLeaf& leaf : Leaves) {
                leaf.Values.fill(TValue());
            }
            for (size_t level = 1; level < Height; ++level) {
                const TOperationResult whole = FromRepeated(TValue(), SpanOfSlot(level));
                for (TInner& node : Inner[level]) {
                    node.Results.fill(whole);
                    node.TagMask = 0;
                }
            }
        }

        virtual void InitFrom(std::span<const TValue> values) override {
            AllocateGeometry(values.size());
            for (size_t leaf = 0; leaf < Leaves.size(); ++leaf) {
                for (size_t slot = 0; slot < FANOUT; ++slot) {
                    const size_t position = (leaf << LOG_FANOUT) + slot;
                    Leaves[leaf].Values[slot] = position < values.size() ? values[position] : TValue();
                }
            }
            for (size_t level = 1; level < Height; ++level) {
                for (size_t child = 0; child < Inner[level].size() << LOG_FANOUT; ++child) {
                    TInner& node = Inner[level][child >> LOG_FANOUT];
                    node.Results[child & (FANOUT - 1)] = child < NodesAt(level - 1)
                        ? FoldNode(level - 1, child)
                        : FromRepeated(TValue(), SpanOfSlot(level));
                    node.TagMask = 0;
                }
            }
        }

        virtual void SetRange(const size_t rangeStart, const size_t rangeEnd, const TValue& value) override {
            assert(rangeStart <= rangeEnd && "Range start can't be bigger than its end");
            assert(rangeEnd <= Length && "Range goes beyond the allocated length");
            if (rangeStart >= rangeEnd) [[unlikely]] {
                return;
            }

            std::array<size_t, MAX_HEIGHT> spine;
            std::array<size_t, MAX_HEIGHT> leftPath;
            std::array<size_t, MAX_HEIGHT> rightPath;

            // The shared descent: while the range stays strictly inside one slot.
            size_t level = Height - 1;
            size_t index = 0;
            size_t low = rangeStart;
            size_t high = rangeEnd;
            while (level > 0) {
                spine[level] = index;
                const size_t shift = level * LOG_FANOUT;
                const size_t slot = low >> shift;
                const size_t base = slot << shift;
                if (slot != (high - 1) >> shift || (low == base && high == base + (size_t(1) << shift))) {
                    break;
                }
                const size_t child = (index << LOG_FANOUT) + slot;
                PushInto(Inner[level][index], slot, level - 1, child);
                index = child;
                low -= base;
                high -= base;
                --level;
            }
            spine[level] = index;
            const size_t split = level;

            size_t leftDeepest = split;
            size_t rightDeepest = split;
            if (split == 0) {
                AssignInLeaf(Leaves[index], low, high, value);
            } else {
                const size_t shift = split * LOG_FANOUT;
                const size_t span = size_t(1) << shift;
                const size_t coveredFrom = (low + span - 1) >> shift;
                const size_t coveredTo = high >> shift;
                if (coveredFrom < coveredTo) {
                    Cover(Inner[split][index], coveredFrom, coveredTo, split, value);
                }

                // Each side: the slot of the node entered next, and the bound inside that slot.
                bool goLeft = (low & (span - 1)) != 0;
                bool goRight = (high & (span - 1)) != 0;
                size_t leftNode = index;
                size_t rightNode = index;
                size_t leftSlot = low >> shift;
                size_t rightSlot = high >> shift;
                size_t leftLow = low & (span - 1);
                size_t rightHigh = high & (span - 1);

                for (size_t below = split - 1; goLeft || goRight; --below) {
                    const size_t belowShift = below * LOG_FANOUT;
                    const size_t belowMask = (size_t(1) << belowShift) - 1;
                    if (goLeft) {
                        const size_t child = (leftNode << LOG_FANOUT) + leftSlot;
                        PushInto(Inner[below + 1][leftNode], leftSlot, below, child);
                        leftPath[below] = child;
                        leftDeepest = below;
                        if (below == 0) {
                            AssignInLeaf(Leaves[child], leftLow, FANOUT, value);
                            goLeft = false;
                        } else {
                            const size_t coveredFromBelow = (leftLow + belowMask) >> belowShift;
                            if (coveredFromBelow < FANOUT) {
                                Cover(Inner[below][child], coveredFromBelow, FANOUT, below, value);
                            }
                            goLeft = (leftLow & belowMask) != 0;
                            leftNode = child;
                            leftSlot = leftLow >> belowShift;
                            leftLow &= belowMask;
                        }
                    }
                    if (goRight) {
                        const size_t child = (rightNode << LOG_FANOUT) + rightSlot;
                        PushInto(Inner[below + 1][rightNode], rightSlot, below, child);
                        rightPath[below] = child;
                        rightDeepest = below;
                        if (below == 0) {
                            AssignInLeaf(Leaves[child], 0, rightHigh, value);
                            goRight = false;
                        } else {
                            const size_t coveredToBelow = rightHigh >> belowShift;
                            if (coveredToBelow > 0) {
                                Cover(Inner[below][child], 0, coveredToBelow, below, value);
                            }
                            goRight = (rightHigh & belowMask) != 0;
                            rightNode = child;
                            rightSlot = rightHigh >> belowShift;
                            rightHigh &= belowMask;
                        }
                    }
                }
            }

            for (size_t at = std::min(leftDeepest, rightDeepest); at < split; ++at) {
                if (at >= leftDeepest) {
                    Refold(at, leftPath[at]);
                }
                if (at >= rightDeepest) {
                    Refold(at, rightPath[at]);
                }
            }
            for (size_t at = split; at + 1 < Height; ++at) {
                Refold(at, spine[at]);
            }
        }

        virtual TOperationResult ComputeRange(const size_t rangeStart, const size_t rangeEnd) const override {
            assert(rangeStart < rangeEnd && "The operation result for an empty set is undefined");
            assert(rangeEnd <= Length && "Range goes beyond the allocated length");
            switch (Height) {
                case 1: return Query<0>(0, rangeStart, rangeEnd);
                case 2: return Query<1>(0, rangeStart, rangeEnd);
                case 3: return Query<2>(0, rangeStart, rangeEnd);
                case 4: return Query<3>(0, rangeStart, rangeEnd);
                case 5: return Query<4>(0, rangeStart, rangeEnd);
                case 6: return Query<5>(0, rangeStart, rangeEnd);
                case 7: return Query<6>(0, rangeStart, rangeEnd);
                default: return Query<7>(0, rangeStart, rangeEnd);
            }
        }

    private:
        static constexpr size_t LOG_FANOUT = 4;
        static constexpr size_t FANOUT = size_t(1) << LOG_FANOUT;
        static constexpr size_t MAX_HEIGHT = 8;  // 16^8 = 2^32 elements
        static constexpr size_t CACHE_LINE = 64;

        struct alignas(CACHE_LINE) TLeaf {
            std::array<TValue, FANOUT> Values;
        };

        struct alignas(CACHE_LINE) TInner {
            std::array<TOperationResult, FANOUT> Results;
            std::array<TValue, FANOUT> Tags;
            uint32_t TagMask = 0;  // bit i: Tags[i] is pending for the child in slot i
        };

        std::vector<TLeaf> Leaves;
        std::array<std::vector<TInner>, MAX_HEIGHT> Inner;  // Inner[h] is level h; Inner[0] is unused
        size_t Height = 1;
        size_t Length = 0;

        static TOperationResult FromRepeated(const TValue& value, const size_t times) {
            return TOperation::template FromRepeated<TValue, TOperationResult>(value, times);
        }

        // The elements under one slot of a node of `level`.
        static size_t SpanOfSlot(const size_t level) noexcept {
            return size_t(1) << (level * LOG_FANOUT);
        }

        size_t NodesAt(const size_t level) const noexcept {
            return level == 0 ? Leaves.size() : Inner[level].size();
        }

        void AllocateGeometry(const size_t length) {
            Length = length;
            Leaves.resize(std::max<size_t>((length + FANOUT - 1) >> LOG_FANOUT, 1));
            Height = 1;
            for (size_t below = Leaves.size(); below > 1; below = Inner[Height - 1].size()) {
                assert(Height < MAX_HEIGHT);
                Inner[Height].resize((below + FANOUT - 1) >> LOG_FANOUT);
                ++Height;
            }
        }

        // Slots [from, to): a fixed-shape select for small types, which the compiler turns into a
        // blend with no loop exit to mispredict; the larger ones are not worth rewriting whole.
        template<typename TSlot>
        static void FillSlots(std::array<TSlot, FANOUT>& slots, const size_t from, const size_t to, const TSlot& value) {
            if constexpr (sizeof(TSlot) <= sizeof(uint64_t) && std::is_trivially_copyable_v<TSlot>) {
                for (size_t slot = 0; slot < FANOUT; ++slot) {
                    slots[slot] = (slot >= from && slot < to) ? value : slots[slot];
                }
            } else {
                std::fill(slots.begin() + from, slots.begin() + to, value);
            }
        }

        static uint32_t SlotsMask(const size_t from, const size_t to) noexcept {
            return ((uint32_t(1) << to) - 1) & ~((uint32_t(1) << from) - 1);
        }

        void Cover(TInner& node, const size_t from, const size_t to, const size_t level, const TValue& value) {
            FillSlots(node.Results, from, to, FromRepeated(value, SpanOfSlot(level)));
            FillSlots(node.Tags, from, to, value);
            node.TagMask |= SlotsMask(from, to);
        }

        // If the slot of `parent` has a pending tag, the child (level, index) takes it over.
        void PushInto(TInner& parent, const size_t slot, const size_t level, const size_t index) {
            const uint32_t bit = uint32_t(1) << slot;
            if (!(parent.TagMask & bit)) {
                return;
            }
            parent.TagMask &= ~bit;
            const TValue tag = parent.Tags[slot];
            if (level == 0) {
                Leaves[index].Values.fill(tag);
            } else {
                TInner& child = Inner[level][index];
                child.Results.fill(FromRepeated(tag, SpanOfSlot(level)));
                child.Tags.fill(tag);
                child.TagMask = (uint32_t(1) << FANOUT) - 1;
            }
        }

        static void AssignInLeaf(TLeaf& leaf, const size_t from, const size_t to, const TValue& value) {
            FillSlots(leaf.Values, from, to, value);
        }

        TOperationResult FoldLeaf(const TLeaf& leaf, const size_t from, const size_t to) const {
            if constexpr (HasFoldInBlock<TOperation, TOperationResult, TValue>) {
                return TOperation::template FoldInBlock<TOperationResult>(leaf.Values.data(), from, to);
            } else if constexpr (HasFromMany<TOperation, TValue, TOperationResult>) {
                return TOperation::template FromMany<TOperationResult>(leaf.Values.data(), from, to);
            } else {
                TOperationResult result = FromRepeated(leaf.Values[from], 1);
                for (size_t slot = from + 1; slot < to; ++slot) {
                    result = TOperation::Merge(result, FromRepeated(leaf.Values[slot], 1));
                }
                return result;
            }
        }

        static TOperationResult FoldSlots(const TInner& node, const size_t from, const size_t to) {
            if constexpr (HasFoldInBlock<TOperation, TOperationResult, TOperationResult>) {
                return TOperation::template FoldInBlock<TOperationResult>(node.Results.data(), from, to);
            } else {
                TOperationResult result = node.Results[from];
                for (size_t slot = from + 1; slot < to; ++slot) {
                    result = TOperation::Merge(result, node.Results[slot]);
                }
                return result;
            }
        }

        // The result of a whole node of `level`.
        TOperationResult FoldNode(const size_t level, const size_t index) const {
            // A whole node is a loop of fixed length, which needs no mask: FoldInBlock would only slow
            // it down, as the compiler vectorizes the chain by itself when Merge is simple enough.
            if (level == 0) {
                if constexpr (HasFromMany<TOperation, TValue, TOperationResult>) {
                    return TOperation::template FromMany<TOperationResult>(Leaves[index].Values.data(), 0, FANOUT);
                } else {
                    TOperationResult result = FromRepeated(Leaves[index].Values[0], 1);
                    for (size_t slot = 1; slot < FANOUT; ++slot) {
                        result = TOperation::Merge(result, FromRepeated(Leaves[index].Values[slot], 1));
                    }
                    return result;
                }
            } else if constexpr (HasMergeBlock<TOperation, TOperationResult>) {
                return TOperation::template MergeBlock<TOperationResult>(Inner[level][index].Results.data());
            } else {
                const TInner& node = Inner[level][index];
                TOperationResult result = node.Results[0];
                for (size_t slot = 1; slot < FANOUT; ++slot) {
                    result = TOperation::Merge(result, node.Results[slot]);
                }
                return result;
            }
        }

        // Refreshes the slot of the parent that covers the node (level, index).
        void Refold(const size_t level, const size_t index) {
            Inner[level + 1][index >> LOG_FANOUT].Results[index & (FANOUT - 1)] = FoldNode(level, index);
        }

        // The query unrolled by level: spans are constants, and each level has its own branches, which
        // the predictor then learns per level.
        template<size_t Level>
        TOperationResult Query(const size_t index, const size_t low, const size_t high) const {
            if constexpr (Level == 0) {
                return FoldLeaf(Leaves[index], low, high);
            } else {
                const TInner& node = Inner[Level][index];
                constexpr size_t shift = Level * LOG_FANOUT;
                constexpr size_t span = size_t(1) << shift;
                // A boundary slot, partly covered: the power of its tag, or the child's part.
                const auto boundary = [&](const size_t slot, const size_t from, const size_t to) {
                    if (node.TagMask & (uint32_t(1) << slot)) {
                        return FromRepeated(node.Tags[slot], to - from);
                    }
                    return Query<Level - 1>((index << LOG_FANOUT) + slot, from, to);
                };

                const size_t first = low >> shift;
                const size_t last = (high - 1) >> shift;
                if (first == last) {
                    const size_t base = first << shift;
                    if (low == base && high == base + span) {
                        return node.Results[first];
                    }
                    return boundary(first, low - base, high - base);
                }

                const size_t coveredFrom = (low + span - 1) >> shift;
                const size_t coveredTo = high >> shift;
                const bool hasLeft = (low & (span - 1)) != 0;
                const bool hasMiddle = coveredFrom < coveredTo;
                const bool hasRight = (high & (span - 1)) != 0;
                if (hasLeft) {
                    TOperationResult result = boundary(first, low & (span - 1), span);
                    if (hasMiddle) {
                        result = TOperation::Merge(result, FoldSlots(node, coveredFrom, coveredTo));
                    }
                    return hasRight ? TOperation::Merge(result, boundary(last, 0, high & (span - 1))) : result;
                }
                // A range with no left part has a middle, or would have stayed inside one slot.
                TOperationResult result = FoldSlots(node, coveredFrom, coveredTo);
                return hasRight ? TOperation::Merge(result, boundary(last, 0, high & (span - 1))) : result;
            }
        }
    };
} // namespace NProust
