#pragma once

#include <bit>
#include <cassert>
#include <vector>
#include <optional>

#include <spdlog/spdlog.h>

#include "../interface.h"
#include "layout.h"
#include "topology.h"

namespace NProust {
    template<
        typename TValue,
        typename TOperationResult,
        typename TOperation,
        // Heap topology keeps siblings adjacent (rightChild = leftChild + 1), so the merge-up on
        // SetRange reads one cache line instead of a far-away one -- ~25% faster on the hot path
        // than preorder (measured). This also matches the default of CreateRobustRangeUpdateEngine.
        NodeTopology TNodeTopology = TBinaryHeapNodeTopology<0>,
        typename TLayout = TSoALayout<TValue, TOperationResult, std::nullopt>
    >
    requires DataLayout<TLayout, TValue, TOperationResult>
    class TRobustRangeUpdateEngine final : public IRangeUpdateEngine<TValue, TOperationResult, TOperation> {
    private:
        const size_t ROOT_NODE_ID = 0;

    public:
        virtual void Init(const size_t length) override {
            Nodes.ReserveCapacities(TNodeTopology::CapacityToReserve(length));
            if (TNodeTopology::IsLeaf(2)) {
                // If the leaf size is one, there is no need for an extra allocation.
                LazySnapshot.resize(length);
                std::fill(LazySnapshot.begin(), LazySnapshot.end(), TValue());
            }
            AllocatedLength = length;
            InitNodes(ROOT_NODE_ID, 0, length);
        }

        virtual void SetRange(const size_t rangeStart, const size_t rangeEnd, const TValue& value) override {
            assert(rangeStart <= rangeEnd && "Range start can't be bigger than its end");
            assert(rangeEnd <= AllocatedLength && "Range goes beyond the allocated length");

            if (rangeStart < rangeEnd) [[likely]] {
                ModifyNodes(ROOT_NODE_ID, rangeStart, rangeEnd, 0, AllocatedLength, value);
            }
        }

        virtual TOperationResult ComputeRange(const size_t rangeStart, const size_t rangeEnd) const override {
            assert(rangeStart <= rangeEnd && "Range start can't be bigger than its end");
            assert(rangeStart != rangeEnd && "The operation result for an empty set is undefined");
            assert(rangeEnd <= AllocatedLength && "Range goes beyond the allocated length");
            return QueryNodes(ROOT_NODE_ID, rangeStart, rangeEnd, 0, AllocatedLength);
        }

    private:
        TLayout Nodes;
        std::vector<TValue> LazySnapshot;
        size_t AllocatedLength = 0;

        void InitNodes(const size_t currentNodeId, const size_t nodeStart, const size_t nodeEnd) {
            Nodes.SetSubtreeValue(currentNodeId, TOperation::template FromRepeated<TValue, TOperationResult>(TValue(), nodeEnd - nodeStart));
            Nodes.DropLazilyPropagated(currentNodeId);

            if (!TNodeTopology::IsLeaf(nodeEnd - nodeStart)) {
                const size_t rangeLength = nodeEnd - nodeStart;
                const size_t midpoint = nodeStart + (rangeLength >> 1);
                InitNodes(TNodeTopology::LeftChild(currentNodeId, rangeLength), nodeStart, midpoint);
                InitNodes(TNodeTopology::RightChild(currentNodeId, rangeLength), midpoint, nodeEnd);
            }
        }

        // Out-of-line on purpose: see the call site in ModifyNodes. Rebuilds a (fat) leaf's subtree
        // value after assigning `value` to [rangeStart, rangeEnd) within [nodeStart, nodeEnd).
        [[gnu::noinline]] void ModifyLeaf(const size_t nodeId, const size_t rangeStart, const size_t rangeEnd, const size_t nodeStart, const size_t nodeEnd, const TValue& value) {
            const size_t segmentStart = std::max(rangeStart, nodeStart);
            const size_t segmentEnd = std::min(rangeEnd, nodeEnd);
            if (Nodes.HasLazilyPropagated(nodeId)) {
                const auto lazilyPropagated = Nodes.GetLazilyPropagated(nodeId);
                std::fill(LazySnapshot.begin() + nodeStart, LazySnapshot.begin() + segmentStart, lazilyPropagated);
                std::fill(LazySnapshot.begin() + segmentStart, LazySnapshot.begin() + segmentEnd, value);
                std::fill(LazySnapshot.begin() + segmentEnd, LazySnapshot.begin() + nodeEnd, lazilyPropagated);
                Nodes.DropLazilyPropagated(nodeId);

                const auto segmentValue = TOperation::template FromRepeated<TValue, TOperationResult>(value, segmentEnd - segmentStart);

                const auto toSegmentEnd =
                    segmentStart == nodeStart ? segmentValue :
                    TOperation::Merge(TOperation::template FromRepeated<TValue, TOperationResult>(lazilyPropagated, segmentStart - nodeStart), segmentValue);

                auto nodeValue =
                    segmentEnd == nodeEnd ? toSegmentEnd :
                    TOperation::Merge(toSegmentEnd, TOperation::template FromRepeated<TValue, TOperationResult>(lazilyPropagated, nodeEnd - segmentEnd));

                Nodes.SetSubtreeValue(nodeId, std::move(nodeValue));
            } else {
                std::fill(LazySnapshot.begin() + segmentStart, LazySnapshot.begin() + segmentEnd, value);

                if constexpr (HasFromMany<TOperation, TValue, TOperationResult>) {
                    // With no pending lazy tag, the snapshot now holds the leaf's full content
                    // over [nodeStart, nodeEnd) (the [segmentStart, segmentEnd) slice was just
                    // overwritten with `value`), so the subtree value is a single bulk build.
                    Nodes.SetSubtreeValue(nodeId, TOperation::template FromMany<TOperationResult>(LazySnapshot.data(), nodeStart, nodeEnd));
                } else {
                    TOperationResult subtreeValue = TOperation::template FromRepeated<TValue, TOperationResult>(LazySnapshot[nodeStart], 1);
                    if (segmentStart > nodeStart) {
                        for (size_t index = nodeStart + 1; index < segmentStart; ++index) {
                            subtreeValue = TOperation::Merge(subtreeValue, TOperation::template FromRepeated<TValue, TOperationResult>(LazySnapshot[index], 1));
                        }
                        subtreeValue = TOperation::Merge(subtreeValue, TOperation::template FromRepeated<TValue, TOperationResult>(value, segmentEnd - segmentStart));
                    } else {
                        subtreeValue = TOperation::template FromRepeated<TValue, TOperationResult>(value, segmentEnd - segmentStart);
                    }

                    for (size_t index = segmentEnd; index < nodeEnd; ++index) {
                        subtreeValue = TOperation::Merge(subtreeValue, TOperation::template FromRepeated<TValue, TOperationResult>(LazySnapshot[index], 1));
                    }

                    Nodes.SetSubtreeValue(nodeId, std::move(subtreeValue));
                }
            }
        }

        // Assigns the value to the whole node: record the lazy tag and the recomputed subtree value.
        inline void AssignWholeNode(const size_t nodeId, const size_t nodeLength, const TValue& value) {
            Nodes.SetLazilyPropagated(nodeId, value);
            Nodes.SetSubtreeValue(nodeId, TOperation::template FromRepeated<TValue, TOperationResult>(value, nodeLength));
        }

        // One-sided descent: assigns the *suffix* [rangeStart, nodeEnd) of the node. Once the generic
        // descent splits at the highest node straddling the range, the left fringe only ever sees
        // suffix assignments, so it needs a single boundary test per level instead of two -- and the
        // fully covered right child is handled with two straight-line writes instead of a recursive
        // call re-deriving that coverage from both bounds.
        inline void ModifySuffix(const size_t nodeId, const size_t rangeStart, const size_t nodeStart, const size_t nodeEnd, const TValue& value) {
            if (rangeStart <= nodeStart) {
                AssignWholeNode(nodeId, nodeEnd - nodeStart, value);
                return;
            }

            if (TNodeTopology::IsLeaf(nodeEnd - nodeStart)) [[unlikely]] {
                ModifyLeaf(nodeId, rangeStart, nodeEnd, nodeStart, nodeEnd, value);
                return;
            }

            const size_t midpoint = (nodeStart + nodeEnd) >> 1;
            const size_t leftChildId = TNodeTopology::LeftChild(nodeId, nodeEnd - nodeStart);
            const size_t rightChildId = TNodeTopology::RightChild(nodeId, nodeEnd - nodeStart);
            const bool hasLazilyPropagated = Nodes.HasLazilyPropagated(nodeId);

            if (rangeStart < midpoint) {
                // The right child is fully covered by the assignment, so the parent's pending tag
                // never needs to reach it: it is overwritten wholesale right here.
                if (hasLazilyPropagated) {
                    Nodes.SetLazilyPropagated(leftChildId, Nodes.GetLazilyPropagated(nodeId));
                    Nodes.DropLazilyPropagated(nodeId);
                }
                ModifySuffix(leftChildId, rangeStart, nodeStart, midpoint, value);
                AssignWholeNode(rightChildId, nodeEnd - midpoint, value);
            } else {
                // The assignment lives entirely in the right child; the left one only receives the
                // pending tag (with its subtree value refreshed, as it is not descended into).
                if (hasLazilyPropagated) {
                    const TValue toPropagate = Nodes.GetLazilyPropagated(nodeId);
                    Nodes.SetLazilyPropagated(leftChildId, toPropagate);
                    Nodes.SetSubtreeValue(leftChildId, TOperation::template FromRepeated<TValue, TOperationResult>(toPropagate, midpoint - nodeStart));
                    Nodes.SetLazilyPropagated(rightChildId, toPropagate);
                    Nodes.DropLazilyPropagated(nodeId);
                }
                ModifySuffix(rightChildId, rangeStart, midpoint, nodeEnd, value);
            }

            Nodes.SetSubtreeValue(nodeId, TOperation::Merge(Nodes.GetSubtreeValue(leftChildId), Nodes.GetSubtreeValue(rightChildId)));
        }

        // Mirror of ModifySuffix: assigns the *prefix* [nodeStart, rangeEnd) of the node.
        inline void ModifyPrefix(const size_t nodeId, const size_t rangeEnd, const size_t nodeStart, const size_t nodeEnd, const TValue& value) {
            if (rangeEnd >= nodeEnd) {
                AssignWholeNode(nodeId, nodeEnd - nodeStart, value);
                return;
            }

            if (TNodeTopology::IsLeaf(nodeEnd - nodeStart)) [[unlikely]] {
                ModifyLeaf(nodeId, nodeStart, rangeEnd, nodeStart, nodeEnd, value);
                return;
            }

            const size_t midpoint = (nodeStart + nodeEnd) >> 1;
            const size_t leftChildId = TNodeTopology::LeftChild(nodeId, nodeEnd - nodeStart);
            const size_t rightChildId = TNodeTopology::RightChild(nodeId, nodeEnd - nodeStart);
            const bool hasLazilyPropagated = Nodes.HasLazilyPropagated(nodeId);

            if (rangeEnd > midpoint) {
                if (hasLazilyPropagated) {
                    Nodes.SetLazilyPropagated(rightChildId, Nodes.GetLazilyPropagated(nodeId));
                    Nodes.DropLazilyPropagated(nodeId);
                }
                AssignWholeNode(leftChildId, midpoint - nodeStart, value);
                ModifyPrefix(rightChildId, rangeEnd, midpoint, nodeEnd, value);
            } else {
                if (hasLazilyPropagated) {
                    const TValue toPropagate = Nodes.GetLazilyPropagated(nodeId);
                    Nodes.SetLazilyPropagated(rightChildId, toPropagate);
                    Nodes.SetSubtreeValue(rightChildId, TOperation::template FromRepeated<TValue, TOperationResult>(toPropagate, nodeEnd - midpoint));
                    Nodes.SetLazilyPropagated(leftChildId, toPropagate);
                    Nodes.DropLazilyPropagated(nodeId);
                }
                ModifyPrefix(leftChildId, rangeEnd, nodeStart, midpoint, value);
            }

            Nodes.SetSubtreeValue(nodeId, TOperation::Merge(Nodes.GetSubtreeValue(leftChildId), Nodes.GetSubtreeValue(rightChildId)));
        }

        // The generic descent is now only the *spine*: it walks down while the range is strictly
        // interior to the node, and hands the two fringes to the one-sided specializations at the
        // split node. Ranges touching a node boundary are handed over immediately.
        inline void ModifyNodes(const size_t nodeId, const size_t rangeStart, const size_t rangeEnd, const size_t nodeStart, const size_t nodeEnd, const TValue& value) {
            if (rangeStart <= nodeStart) {
                ModifyPrefix(nodeId, rangeEnd, nodeStart, nodeEnd, value);
                return;
            }
            if (rangeEnd >= nodeEnd) {
                ModifySuffix(nodeId, rangeStart, nodeStart, nodeEnd, value);
                return;
            }

            if (TNodeTopology::IsLeaf(nodeEnd - nodeStart)) [[unlikely]] {
                // The leaf rebuild carries the (auto-vectorized) FromMany / Merge-fold kernel. Keeping
                // it out of line means the recursive descent body stays free of AVX, so its prologue
                // skips the 32-byte stack realignment and vzeroupper that the inlined kernel forces.
                ModifyLeaf(nodeId, rangeStart, rangeEnd, nodeStart, nodeEnd, value);
                return;
            }

            const size_t midpoint = (nodeStart + nodeEnd) >> 1;
            const size_t leftChildId = TNodeTopology::LeftChild(nodeId, nodeEnd - nodeStart);
            const size_t rightChildId = TNodeTopology::RightChild(nodeId, nodeEnd - nodeStart);

            const bool hasLazilyPropagated = Nodes.HasLazilyPropagated(nodeId);
            TValue toPropagate = hasLazilyPropagated ? Nodes.GetLazilyPropagated(nodeId) : TValue();

            if (hasLazilyPropagated) {
                Nodes.SetLazilyPropagated(leftChildId, toPropagate);
                Nodes.SetLazilyPropagated(rightChildId, toPropagate);
                Nodes.DropLazilyPropagated(nodeId);
            }

            if (rangeEnd <= midpoint) {
                ModifyNodes(leftChildId, rangeStart, rangeEnd, nodeStart, midpoint, value);
                if (hasLazilyPropagated) {
                    Nodes.SetSubtreeValue(rightChildId, TOperation::template FromRepeated<TValue, TOperationResult>(toPropagate, nodeEnd - midpoint));
                }
            } else if (rangeStart >= midpoint) {
                ModifyNodes(rightChildId, rangeStart, rangeEnd, midpoint, nodeEnd, value);
                if (hasLazilyPropagated) {
                    Nodes.SetSubtreeValue(leftChildId, TOperation::template FromRepeated<TValue, TOperationResult>(toPropagate, midpoint - nodeStart));
                }
            } else {
                ModifySuffix(leftChildId, rangeStart, nodeStart, midpoint, value);
                ModifyPrefix(rightChildId, rangeEnd, midpoint, nodeEnd, value);
            }

            Nodes.SetSubtreeValue(nodeId, TOperation::Merge(Nodes.GetSubtreeValue(leftChildId), Nodes.GetSubtreeValue(rightChildId)));
        }

        // Out-of-line leaf reduction for the query path (see the call site in QueryNodes).
        [[gnu::noinline]] TOperationResult QueryLeaf(const size_t segmentStart, const size_t segmentEnd) const {
            if constexpr (HasFromMany<TOperation, TValue, TOperationResult>) {
                return TOperation::template FromMany<TOperationResult>(LazySnapshot.data(), segmentStart, segmentEnd);
            } else {
                TOperationResult result = TOperation::template FromRepeated<TValue, TOperationResult>(LazySnapshot[segmentStart], 1);
                for (size_t index = segmentStart + 1; index < segmentEnd; ++index) {
                    result = TOperation::Merge(result, TOperation::template FromRepeated<TValue, TOperationResult>(LazySnapshot[index], 1));
                }
                return result;
            }
        }

        // One-sided descent for the query's left fringe: computes the *suffix* [rangeStart, nodeEnd).
        // One boundary test per level, and the fully covered right sibling contributes with a direct
        // array read instead of a recursive call that would re-derive its coverage from both bounds.
        inline TOperationResult QuerySuffix(const size_t nodeId, const size_t rangeStart, const size_t nodeStart, const size_t nodeEnd) const {
            if (rangeStart <= nodeStart) {
                return Nodes.GetSubtreeValue(nodeId);
            }

            if (Nodes.HasLazilyPropagated(nodeId)) {
                return TOperation::template FromRepeated<TValue, TOperationResult>(Nodes.GetLazilyPropagated(nodeId), nodeEnd - rangeStart);
            }

            if (TNodeTopology::IsLeaf(nodeEnd - nodeStart)) {
                return QueryLeaf(rangeStart, nodeEnd);
            }

            const size_t midpoint = (nodeStart + nodeEnd) >> 1;
            if (rangeStart < midpoint) {
                const size_t leftChildId = TNodeTopology::LeftChild(nodeId, nodeEnd - nodeStart);
                const size_t rightChildId = TNodeTopology::RightChild(nodeId, nodeEnd - nodeStart);
                return TOperation::Merge(QuerySuffix(leftChildId, rangeStart, nodeStart, midpoint), Nodes.GetSubtreeValue(rightChildId));
            }
            return QuerySuffix(TNodeTopology::RightChild(nodeId, nodeEnd - nodeStart), rangeStart, midpoint, nodeEnd);
        }

        // Mirror of QuerySuffix: computes the *prefix* [nodeStart, rangeEnd).
        inline TOperationResult QueryPrefix(const size_t nodeId, const size_t rangeEnd, const size_t nodeStart, const size_t nodeEnd) const {
            if (rangeEnd >= nodeEnd) {
                return Nodes.GetSubtreeValue(nodeId);
            }

            if (Nodes.HasLazilyPropagated(nodeId)) {
                return TOperation::template FromRepeated<TValue, TOperationResult>(Nodes.GetLazilyPropagated(nodeId), rangeEnd - nodeStart);
            }

            if (TNodeTopology::IsLeaf(nodeEnd - nodeStart)) {
                return QueryLeaf(nodeStart, rangeEnd);
            }

            const size_t midpoint = (nodeStart + nodeEnd) >> 1;
            if (rangeEnd > midpoint) {
                const size_t leftChildId = TNodeTopology::LeftChild(nodeId, nodeEnd - nodeStart);
                const size_t rightChildId = TNodeTopology::RightChild(nodeId, nodeEnd - nodeStart);
                return TOperation::Merge(Nodes.GetSubtreeValue(leftChildId), QueryPrefix(rightChildId, rangeEnd, midpoint, nodeEnd));
            }
            return QueryPrefix(TNodeTopology::LeftChild(nodeId, nodeEnd - nodeStart), rangeEnd, nodeStart, midpoint);
        }

        // The generic query is now only the *spine* (see ModifyNodes): it descends while the range is
        // strictly interior and splits into the two one-sided fringe walks at the straddling node.
        inline TOperationResult QueryNodes(const size_t nodeId, const size_t rangeStart, const size_t rangeEnd, const size_t nodeStart, const size_t nodeEnd) const {
            if (rangeStart <= nodeStart) {
                return QueryPrefix(nodeId, rangeEnd, nodeStart, nodeEnd);
            }
            if (rangeEnd >= nodeEnd) {
                return QuerySuffix(nodeId, rangeStart, nodeStart, nodeEnd);
            }

            if (Nodes.HasLazilyPropagated(nodeId)) {
                return TOperation::template FromRepeated<TValue, TOperationResult>(Nodes.GetLazilyPropagated(nodeId), rangeEnd - rangeStart);
            }

            if (TNodeTopology::IsLeaf(nodeEnd - nodeStart)) {
                // Out of line: keeps the AVX leaf reduction (and its stack realignment) out of the
                // recursive descent. See ModifyLeaf for the same reasoning.
                return QueryLeaf(rangeStart, rangeEnd);
            }

            const size_t midpoint = (nodeStart + nodeEnd) >> 1;
            if (rangeEnd <= midpoint) {
                return QueryNodes(TNodeTopology::LeftChild(nodeId, nodeEnd - nodeStart), rangeStart, rangeEnd, nodeStart, midpoint);
            }
            if (rangeStart >= midpoint) {
                return QueryNodes(TNodeTopology::RightChild(nodeId, nodeEnd - nodeStart), rangeStart, rangeEnd, midpoint, nodeEnd);
            }

            const size_t leftChildId = TNodeTopology::LeftChild(nodeId, nodeEnd - nodeStart);
            const size_t rightChildId = TNodeTopology::RightChild(nodeId, nodeEnd - nodeStart);
            return TOperation::Merge(
                QuerySuffix(leftChildId, rangeStart, nodeStart, midpoint),
                QueryPrefix(rightChildId, rangeEnd, midpoint, nodeEnd));
        }
    };

    template<
        typename TValue,
        typename TOperationResult,
        typename TOperation,
        NodeTopology TNodeTopology = TBinaryHeapNodeTopology<0>,
        typename TLayout = TSoALayout<TValue, TOperationResult>
    >
    requires DataLayout<TLayout, TValue, TOperationResult>
    std::unique_ptr<IRangeUpdateEngine<TValue, TOperationResult, TOperation>> CreateRobustRangeUpdateEngine() {
        return std::make_unique<TRobustRangeUpdateEngine<TValue, TOperationResult, TOperation, TNodeTopology, TLayout>>();
    }

    template<typename TValue, typename TOperationResult, typename TOperation, size_t BranchingLog = 1>
    requires ValidOperation<TOperation, TValue, TOperationResult> && HasUndo<TOperation, TOperationResult>
    class TFatPointUpdateEngine final : public IPointUpdateEngine<TValue, TOperationResult, TOperation> {
    private:
        static constexpr size_t ROOT_NODE_ID = 1;
        static constexpr size_t CHILDREN_COUNT = 1 << BranchingLog;
        static constexpr size_t INDEX_SHIFT_FOR_PARENT_CALCULATION = CHILDREN_COUNT - 2;
        static constexpr size_t LEFTMOST_CHILD_INDEX = BranchingLog == 1 ? 0 : 2;

    public:
        static_assert(BranchingLog > 0, "BranchingLog must be strictly positive");

        virtual void Init(const size_t length) override {
            // Overbook capacities up to the closest power of two
            // Here, length + 1 is used, in order to avoid branching in ComputePrefixBackwards
            AllocatedLength = std::bit_ceil((length + 1));
            if (AllocatedLength < CHILDREN_COUNT) {
                AllocatedLength = CHILDREN_COUNT;
            }

            size_t totalNodes = 1;
            size_t nodesInCurrentLayer = 1;
            size_t nodesUnderOneLeaf = AllocatedLength;
            while (nodesUnderOneLeaf > CHILDREN_COUNT) {
                totalNodes += nodesInCurrentLayer <<= BranchingLog;
                nodesUnderOneLeaf >>= BranchingLog;
            }

            LeafSize = nodesUnderOneLeaf;
            LeafSizeLog = std::countr_zero(LeafSize);
            assert((1 << LeafSizeLog) == LeafSize);
            {
                PointValues.resize(AllocatedLength);
                std::fill(PointValues.begin(), PointValues.end(), TValue());
            }

            NodeValues.resize(totalNodes + 1);  // +1 for 1-indexing
            BottomLayerStart = NodeValues.size() - (AllocatedLength >> LeafSizeLog);

            size_t currentLayerStart = BottomLayerStart;
            size_t layerNodes = (AllocatedLength >> LeafSizeLog);
            size_t nodeCoverage = LeafSize;
            const auto neutral = TOperation::Undo(
                TOperation::template FromRepeated<TValue, TOperationResult>(TValue(), 1),
                TOperation::template FromRepeated<TValue, TOperationResult>(TValue(), 1)
            );
            while (layerNodes > 1) {
                for (size_t nodeIndex = currentLayerStart; nodeIndex < currentLayerStart + layerNodes; ++nodeIndex) {
                    const bool isLeftmost = (nodeIndex & (CHILDREN_COUNT - 1)) == LEFTMOST_CHILD_INDEX;
                    if (isLeftmost) {
                        NodeValues[nodeIndex] = neutral;
                    } else {
                        const auto defaultChunk = TOperation::template FromRepeated<TValue, TOperationResult>(TValue(), nodeCoverage);
                        NodeValues[nodeIndex] = TOperation::Merge(NodeValues[nodeIndex - 1], defaultChunk);
                    }
                }
                layerNodes >>= BranchingLog;
                nodeCoverage <<= BranchingLog;
                currentLayerStart -= layerNodes;
            }
        }

        // no-tree-vectorize is deliberate: the suffix loop below trips at most CHILDREN_COUNT - 1
        // times, and auto-vectorizing such a short map loop only pays head/tail overhead (measured
        // ~4% slower). The attribute keeps it the tight scalar pointer loop the bound below enables.
#if defined(__GNUC__) && !defined(__clang__)
        [[gnu::optimize("no-tree-vectorize")]]
#endif
        virtual void SetPoint(const size_t position, const TValue& value) override {
            const auto delta = TOperation::Undo(
                TOperation::template FromRepeated<TValue, TOperationResult>(value, 1),
                TOperation::template FromRepeated<TValue, TOperationResult>(PointValues[position], 1)
            );
            PointValues[position] = value;

            // Walk up to (but not through) the root: the root node has no right-siblings, so its
            // suffix update is always empty, and the root's own value is never read by a query
            // (ComputePrefixBackwards stops at `> ROOT_NODE_ID`). Stopping here also keeps the block
            // index below from underflowing when currentNode would be the root.
            auto currentNode = BottomLayerStart + (position >> LeafSizeLog);
            while (currentNode > ROOT_NODE_ID) {
                // Fold the delta into the suffix of `currentNode`'s sibling group.
                const size_t groupStart = (currentNode - LEFTMOST_CHILD_INDEX) & ~(CHILDREN_COUNT - 1);
                const size_t blockStart = groupStart + LEFTMOST_CHILD_INDEX;
                if constexpr (HasApplyToMany<TOperation, TOperationResult>) {
                    // Whole fixed-size block, vectorized (Algorithmica-style masked update).
                    TOperation::template ApplyToMany<TOperationResult>(
                        NodeValues.data() + blockStart, CHILDREN_COUNT, (currentNode + 1) - blockStart, delta);
                } else {
                    // Scalar fallback: a countable pointer loop (4 instr/iter) over the suffix.
                    const size_t groupEnd = blockStart + CHILDREN_COUNT;
                    for (size_t index = currentNode + 1; index < groupEnd; ++index) {
                        NodeValues[index] = TOperation::Merge(NodeValues[index], delta);
                    }
                }
                currentNode = (currentNode + INDEX_SHIFT_FOR_PARENT_CALCULATION) >> BranchingLog;
            }
        }

        virtual TOperationResult ComputeRange(const size_t rangeStart, const size_t rangeEnd) const {
            if (rangeStart + (CHILDREN_COUNT << 1) > rangeEnd) [[unlikely]] {
                return ComputeRangeBruteForce(rangeStart, rangeEnd);
            }

            const auto fullPrefix = ComputePrefixBackwards(rangeEnd);
            if (rangeStart == 0) [[unlikely]] {
                return fullPrefix;
            }
            const auto partialPrefix = ComputePrefixBackwards(rangeStart);
            return TOperation::Undo(fullPrefix, partialPrefix);
        }

    private:
        std::vector<TOperationResult> NodeValues;
        std::vector<TValue> PointValues;
        size_t AllocatedLength = 0;
        size_t LeafSize = 0;
        size_t LeafSizeLog = 0;
        size_t BottomLayerStart = 0;

        inline TOperationResult ComputeRangeBruteForce(const size_t rangeStart, const size_t rangeEnd) const {
            if constexpr (HasFromMany<TOperation, TValue, TOperationResult>) {
                return TOperation::template FromMany<TOperationResult>(PointValues.data(), rangeStart, rangeEnd);
            } else {
                auto result = TOperation::template FromRepeated<TValue, TOperationResult>(PointValues[rangeStart], 1);
                for (size_t index = rangeStart + 1; index < rangeEnd; ++index) {
                    result = TOperation::Merge(result, TOperation::template FromRepeated<TValue, TOperationResult>(PointValues[index], 1));
                }
                return result;
            }
        }

        inline TOperationResult ComputePrefixBackwards(const size_t prefix) const {
            TOperationResult result;            

            size_t currentNode = BottomLayerStart + (prefix >> LeafSizeLog);
            if (prefix & (LeafSize - 1)) [[likely]] {
                result = ComputeRangeBruteForce(prefix & ~(LeafSize - 1), prefix);
            } else {
                result = NodeValues[currentNode];
                currentNode = (currentNode + INDEX_SHIFT_FOR_PARENT_CALCULATION) >> BranchingLog;
            }

            while (currentNode > ROOT_NODE_ID) {
                result = TOperation::Merge(NodeValues[currentNode], result);
                currentNode = (currentNode + INDEX_SHIFT_FOR_PARENT_CALCULATION) >> BranchingLog;
            }
            return result;
        }
    };

    template<typename TValue, typename TOperationResult, typename TOperation, size_t LeafSizeLog = 1>
    class TRobustPointUpdateEngine final : public IPointUpdateEngine<TValue, TOperationResult, TOperation> {
    private:
        // We use 1-indexing here: although not as nice, it allows us a bit cheaper topology operations
        // as compared to 0-indexing.
        static constexpr size_t ROOT_NODE_ID = 1;
        static constexpr size_t LEAF_SIZE = 1 << LeafSizeLog;
    
    public:
        virtual void Init(const size_t length) override {
            // Overbook capacities up to the closest power of two
            AllocatedLength = std::bit_ceil(length);
            if (AllocatedLength < LEAF_SIZE) {
                AllocatedLength = LEAF_SIZE;
            }

            // One leaf-group node per LEAF_SIZE elements. Like the fat engine, we materialize only
            // the leaf-group tree (GroupCount leaves + their ancestors = 2*GroupCount nodes) of
            // TOperationResult, and keep the raw values in a separate TValue array. A leaf's own
            // result is never stored: it is rebuilt from PointValues on demand (FromMany, or a
            // per-element FromRepeated fold). For LeafSizeLog == 0 this degenerates to the classic
            // 2*AllocatedLength per-element tree.
            GroupCount = AllocatedLength >> LeafSizeLog;
            NodeValues.resize(GroupCount << 1);

            PointValues.resize(AllocatedLength);
            std::fill(PointValues.begin(), PointValues.end(), TValue());

            // Initialize the leaf-group nodes with the state of LEAF_SIZE default values, then build
            // the tree above them.
            const auto defaultLeaf = TOperation::template FromRepeated<TValue, TOperationResult>(TValue(), LEAF_SIZE);
            std::fill(NodeValues.begin() + GroupCount, NodeValues.end(), defaultLeaf);
            for (size_t index = GroupCount - 1; index >= 1; --index) {
                NodeValues[index] = TOperation::Merge(NodeValues[(index << 1)], NodeValues[(index << 1) + 1]);
            }
        }

        virtual void SetPoint(const size_t position, const TValue& value) override {
            assert(position < AllocatedLength && "The position is beyond the allocated size");

            // Capture the old raw value before overwriting: the Undo fast path folds it back out.
            // The FromMany leaf rebuild reads PointValues, so it must already reflect this update.
            [[maybe_unused]] const TValue oldValue = PointValues[position];
            PointValues[position] = value;

            size_t node = GroupCount + (position >> LeafSizeLog);

            if constexpr (LeafSizeLog == 0) {
                // The leaf-group is a single element; just overwrite its result.
                NodeValues[node] = TOperation::template FromRepeated<TValue, TOperationResult>(value, 1);
            } else if constexpr (HasUndo<TOperation, TOperationResult>) {
                // Fold the old element out of the group and the new one in, in O(1).
                const auto undone = TOperation::Undo(NodeValues[node], TOperation::template FromRepeated<TValue, TOperationResult>(oldValue, 1));
                NodeValues[node] = TOperation::Merge(undone, TOperation::template FromRepeated<TValue, TOperationResult>(value, 1));
            } else {
                // No inverse: rebuild the leaf-group's result from its LEAF_SIZE raw values.
                const size_t leafStart = position & ~(LEAF_SIZE - 1);
                NodeValues[node] = ComputeRangeBruteForce(leafStart, leafStart + LEAF_SIZE);
            }

            while (node > 1) {
                NodeValues[node >> 1] = TOperation::Merge(NodeValues[node & ~1], NodeValues[node | 1]);
                node >>= 1;
            }
        }

        virtual TOperationResult ComputeRange(const size_t rangeStart, const size_t rangeEnd) const override {
            assert(rangeStart <= rangeEnd && "Range start can't be bigger than its end");
            assert(rangeStart != rangeEnd && "The operation result for an empty set is undefined");
            assert(rangeEnd <= AllocatedLength && "Range goes beyond the allocated length");

            if constexpr (HasUndo<TOperation, TOperationResult>) {
                const auto fullPrefixResult = ComputePrefixUnaligned(rangeEnd);
                if (rangeStart == 0) [[unlikely]] {
                    return fullPrefixResult;
                }
                const auto irrelevantPrefixResult = ComputePrefixUnaligned(rangeStart);
                return TOperation::Undo(fullPrefixResult, irrelevantPrefixResult);
            } else {
                const auto alignedStart = ((rangeStart + LEAF_SIZE - 1) >> LeafSizeLog) << LeafSizeLog;
                const auto alignedEnd = (rangeEnd >> LeafSizeLog) << LeafSizeLog;
                if (alignedStart >= alignedEnd) [[unlikely]] {
                    return ComputeRangeBruteForce(rangeStart, rangeEnd);
                }

                const auto centralResult = ComputeAlignedRange(alignedStart, alignedEnd);
                if (alignedStart > rangeStart) [[likely]] {
                    const auto leftOffcutResult = ComputeRangeBruteForce(rangeStart, alignedStart);
                    const auto leftCenterResult = TOperation::Merge(leftOffcutResult, centralResult);

                    if (alignedEnd < rangeEnd) [[likely]] {
                        const auto rightOffcutResult = ComputeRangeBruteForce(alignedEnd, rangeEnd);
                        return TOperation::Merge(leftCenterResult, rightOffcutResult);
                    } else {
                        return leftCenterResult;
                    }
                } else {
                    if (alignedEnd < rangeEnd) [[likely]] {
                        const auto rightOffcutResult = ComputeRangeBruteForce(alignedEnd, rangeEnd);
                        return TOperation::Merge(centralResult, rightOffcutResult);
                    } else {
                        return centralResult;
                    }
                }
            }
        }

    private:
        std::vector<TOperationResult> NodeValues;
        std::vector<TValue> PointValues;
        size_t AllocatedLength;
        size_t GroupCount;  // = AllocatedLength >> LeafSizeLog; number of leaf-group nodes

        inline TOperationResult ComputeRangeBruteForce(const size_t rangeStart, const size_t rangeEnd) const {
            assert(rangeStart < rangeEnd);
            if constexpr (HasFromMany<TOperation, TValue, TOperationResult>) {
                return TOperation::template FromMany<TOperationResult>(PointValues.data(), rangeStart, rangeEnd);
            } else {
                // Per-element leaf results are no longer stored, so rebuild from the raw values.
                auto result = TOperation::template FromRepeated<TValue, TOperationResult>(PointValues[rangeStart], 1);
                for (size_t index = rangeStart + 1; index < rangeEnd; ++index) {
                    result = TOperation::Merge(result, TOperation::template FromRepeated<TValue, TOperationResult>(PointValues[index], 1));
                }
                return result;
            }
        }

        inline TOperationResult ComputePrefixUnaligned(const size_t prefix) const {
            const auto alignedBorder = (prefix >> LeafSizeLog) << LeafSizeLog;
            if (alignedBorder == 0) [[unlikely]] {
                return ComputeRangeBruteForce(0, prefix);
            }

            const auto mainResult = ComputeRangeBackwards(GroupCount, alignedBorder >> LeafSizeLog);
            if constexpr (LeafSizeLog == 0) {
                return mainResult;
            }

            if (alignedBorder == prefix) [[unlikely]] {
                return mainResult;
            }
            return TOperation::Merge(mainResult, ComputeRangeBruteForce(alignedBorder, prefix));
        }

        inline TOperationResult ComputeAlignedRange(const size_t rangeStart, const size_t rangeEnd) const {
            // Here we construct the value of the target function by composing the related nodes
            // from the segment tree. The base of this method is recursive (see range modification engine for
            // reference), but since we don't have lazy tags, we can construct the exact set of nodes that
            // composes the answer without recursion, moreover, with the usage of cheap binary operations.
            //
            // As a reward, there's a 4x speedup (even before SIMD and large leaf optimization) for being
            // efficient.
            //
            // We use a binary heap layout over the leaf-group nodes. `rangeStart`/`rangeEnd` arrive
            // leaf-aligned (multiples of LEAF_SIZE), so we convert them to group coordinates and
            // walk the group heap: the leaf-group node for group g is at index GroupCount + g.
            auto currentNode = GroupCount + (rangeStart >> LeafSizeLog);
            auto queryLength = (rangeEnd - rangeStart) >> LeafSizeLog;

            // Due to heap topology, each node has a contiguous range of power-2 segments starting there,
            // the cardinality of this set is same as the number of zeros at the end of node representation.
            // We refer to it in the code as `maxJump`.
            // maxJump is the lowest set bit of currentNode, always a power of two, so the index
            // `currentNode / maxJump` is exactly `currentNode >> ctz(currentNode)` -- a shift, not the
            // ~20-40 cycle hardware division the compiler otherwise emits (it cannot prove the divisor
            // is a power of two at compile time). Same trick in ComputeRangeBackwards below.
            size_t maxShift = std::countr_zero(currentNode);
            size_t maxJump = size_t(1) << maxShift;
            if (maxJump <= queryLength) {
                auto result = NodeValues[currentNode >> maxShift];
                queryLength -= maxJump;
                currentNode += maxJump;
                maxShift = std::countr_zero(currentNode);
                maxJump = size_t(1) << maxShift;

                // While maxJump is smaller than the queried range, the things are simple:
                // we just take the largest node possible, reuse its result, and proceed.
                // The query length decreases, and the leftmost matching node index increases.
                while (maxJump <= queryLength) {
                    result = TOperation::Merge(result, NodeValues[currentNode >> maxShift]);
                    queryLength -= maxJump;
                    currentNode += maxJump;
                    maxShift = std::countr_zero(currentNode);
                    maxJump = size_t(1) << maxShift;
                }

                if (queryLength > 0) [[likely]] {
                    // At some point, we're in a major node that allows us longer jumps that we need.
                    // It means, we can no longer act in the same way, and we need to dismantle
                    // the remaining length by its' powers of two, and perform the same jumps backwards.
                    // Since `maxJump` is bigger that the whole interval, we can assume that those nodes
                    // will always exist.

                    return TOperation::Merge(result, ComputeRangeBackwards(currentNode, queryLength));
                } else {
                    return result;
                }
            }

            return ComputeRangeBackwards(currentNode, queryLength);
        }

        inline TOperationResult ComputeRangeBackwards(size_t currentNode, size_t queryLength) const {
            // We move to the end of the desired interval and go backwards.
            size_t lastNode = currentNode + queryLength;

            // We pick the length of the last node that will participate in our greedy movement.
            // lastStep is the lowest set bit of queryLength (a power of two), so `prevNode / lastStep`
            // is the shift `prevNode >> ctz(queryLength)` -- avoiding the hardware division (see
            // ComputeAlignedRange).
            size_t lastShift = std::countr_zero(queryLength);
            size_t lastStep = size_t(1) << lastShift;
            size_t prevNode = lastNode - lastStep;

            // And use the result from that node.
            auto result = NodeValues[prevNode >> lastShift];
            lastNode = prevNode;

            // Basically subtract the length of the taken interval, but faster.
            queryLength &= queryLength - 1;

            // Repeat the process when needed.
            while (queryLength > 0) {
                lastShift = std::countr_zero(queryLength);
                lastStep = size_t(1) << lastShift;
                prevNode = lastNode - lastStep;
                result = TOperation::Merge(NodeValues[prevNode >> lastShift], result);  // Note the unusual operand order: the operation is non-commutative!
                lastNode = prevNode;
                queryLength &= queryLength - 1;
            }
            return result;
        }
    };

    template<typename TValue, typename TOperationResult, typename TOperation, size_t LeafSizeLog = 0>
    std::unique_ptr<IPointUpdateEngine<TValue, TOperationResult, TOperation>> CreateRobustPointUpdateEngine() {
        if constexpr (HasUndo<TOperation, TOperationResult>) {
            return std::make_unique<TFatPointUpdateEngine<TValue, TOperationResult, TOperation, 4>>();
        } else {
            return std::make_unique<TRobustPointUpdateEngine<TValue, TOperationResult, TOperation, LeafSizeLog>>();
        }
    }
} // namespace NProust
