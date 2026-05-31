#pragma once

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
        NodeTopology TNodeTopology = TPreorderNodeTopology<0>,
        typename TLayout = TSoALayout<TValue, TOperationResult, std::nullopt>
    >
    requires DataLayout<TLayout, TValue, TOperationResult>
    class TRobustRangeUpdateEngine: public IRangeUpdateEngine<TValue, TOperationResult, TOperation> {
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

        inline void ModifyNodes(const size_t nodeId, const size_t rangeStart, const size_t rangeEnd, const size_t nodeStart, const size_t nodeEnd, const TValue& value) {
            if (rangeStart <= nodeStart && rangeEnd >= nodeEnd) {
                Nodes.SetLazilyPropagated(nodeId, value);
                Nodes.SetSubtreeValue(nodeId, TOperation::template FromRepeated<TValue, TOperationResult>(value, nodeEnd - nodeStart));
                return;
            }

            if (TNodeTopology::IsLeaf(nodeEnd - nodeStart)) [[unlikely]] {
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
            } else {
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

                if (midpoint > rangeStart) {
                    ModifyNodes(leftChildId, rangeStart, rangeEnd, nodeStart, midpoint, value);
                } else if (hasLazilyPropagated) {
                    Nodes.SetSubtreeValue(leftChildId, TOperation::template FromRepeated<TValue, TOperationResult>(toPropagate, midpoint - nodeStart));
                }

                if (midpoint < rangeEnd) {
                    ModifyNodes(rightChildId, rangeStart, rangeEnd, midpoint, nodeEnd, value);
                } else if (hasLazilyPropagated) {
                    Nodes.SetSubtreeValue(rightChildId, TOperation::template FromRepeated<TValue, TOperationResult>(toPropagate, nodeEnd - midpoint));
                }

                Nodes.SetSubtreeValue(nodeId, TOperation::Merge(Nodes.GetSubtreeValue(leftChildId), Nodes.GetSubtreeValue(rightChildId)));
            }
        }

        inline TOperationResult QueryNodes(const size_t nodeId, const size_t rangeStart, const size_t rangeEnd, const size_t nodeStart, const size_t nodeEnd) const {
            if (rangeStart <= nodeStart && rangeEnd >= nodeEnd) {
                return Nodes.GetSubtreeValue(nodeId);
            }

            if (Nodes.HasLazilyPropagated(nodeId)) {
                const auto lazilyPropagated = Nodes.GetLazilyPropagated(nodeId);
                const size_t segmentStart = std::max(rangeStart, nodeStart);
                const size_t segmentEnd = std::min(rangeEnd, nodeEnd);
                return TOperation::template FromRepeated<TValue, TOperationResult>(lazilyPropagated, segmentEnd - segmentStart);
            }

            if (TNodeTopology::IsLeaf(nodeEnd - nodeStart)) {
                const size_t segmentStart = std::max(rangeStart, nodeStart);
                const size_t segmentEnd = std::min(rangeEnd, nodeEnd);
                if constexpr (HasFromMany<TOperation, TValue, TOperationResult>) {
                    return TOperation::template FromMany<TOperationResult>(LazySnapshot.data(), segmentStart, segmentEnd);
                } else {
                    TOperationResult result = TOperation::template FromRepeated<TValue, TOperationResult>(LazySnapshot[segmentStart], 1);
                    for (size_t index = segmentStart + 1; index < segmentEnd; ++index) {
                        result = TOperation::Merge(result, TOperation::template FromRepeated<TValue, TOperationResult>(LazySnapshot[index], 1));
                    }
                    return result;
                }
            } else {
                const size_t midpoint = (nodeStart + nodeEnd) >> 1;
                if (midpoint > rangeStart) {
                    const size_t leftChildId = TNodeTopology::LeftChild(nodeId, nodeEnd - nodeStart);
                    const TOperationResult leftResult = QueryNodes(leftChildId, rangeStart, rangeEnd, nodeStart, midpoint);
                    if (midpoint < rangeEnd) {
                        const size_t rightChildId = TNodeTopology::RightChild(nodeId, nodeEnd - nodeStart);
                        const TOperationResult rightResult = QueryNodes(rightChildId, rangeStart, rangeEnd, midpoint, nodeEnd);
                        return TOperation::Merge(leftResult, rightResult);
                    } else {
                        return leftResult;
                    }
                }

                const size_t rightChildId = TNodeTopology::RightChild(nodeId, nodeEnd - nodeStart);
                return QueryNodes(rightChildId, rangeStart, rangeEnd, midpoint, nodeEnd);
            }
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
    class TFatPointUpdateEngine: public IPointUpdateEngine<TValue, TOperationResult, TOperation> {
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

        virtual void SetPoint(const size_t position, const TValue& value) override {
            const auto delta = TOperation::Undo(
                TOperation::template FromRepeated<TValue, TOperationResult>(value, 1),
                TOperation::template FromRepeated<TValue, TOperationResult>(PointValues[position], 1)
            );
            PointValues[position] = value;

            auto currentNode = BottomLayerStart + (position >> LeafSizeLog);
            while (currentNode >= ROOT_NODE_ID) {
                for (size_t index = currentNode + 1; (index & (CHILDREN_COUNT - 1)) != LEFTMOST_CHILD_INDEX; ++index) {
                    NodeValues[index] = TOperation::Merge(NodeValues[index], delta);
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
    class TRobustPointUpdateEngine: public IPointUpdateEngine<TValue, TOperationResult, TOperation> {
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

            // Being a power of two, the tree takes 2x nodes from the length
            NodeValues.resize(AllocatedLength << 1);

            // The raw point values are kept alongside the tree so the FromMany fast path
            // can rebuild a leaf's result from the original values (not from merged results).
            PointValues.resize(AllocatedLength);
            std::fill(PointValues.begin(), PointValues.end(), TValue());

            // Initialize the tree with the default values starting from the leafs
            const auto leafValue = TOperation::template FromRepeated<TValue, TOperationResult>(TValue(), 1);
            std::fill(NodeValues.begin(), NodeValues.end(), leafValue);
            for (size_t index = AllocatedLength - 1; index >= 1; --index) {
                NodeValues[index] = TOperation::Merge(NodeValues[(index << 1)], NodeValues[(index << 1) + 1]);
            }
        }

        virtual void SetPoint(const size_t position, const TValue& value) override {
            assert(position < AllocatedLength && "The position is beyond the allocated size");
            size_t nodePosition = AllocatedLength + position;

            // Keep the raw value current before any leaf rebuild: the FromMany path in
            // ComputeRangeBruteForce reads PointValues, so it must already reflect this update.
            PointValues[position] = value;

            const size_t largeLeafStart = position & ~(LEAF_SIZE - 1);

            if constexpr (HasUndo<TOperation, TOperationResult>) {
                const auto oldValue = NodeValues[nodePosition];
                NodeValues[nodePosition] = TOperation::template FromRepeated<TValue, TOperationResult>(value, 1);
                if constexpr (LeafSizeLog > 0) {
                    nodePosition >>= LeafSizeLog;
                    const auto oldValueUndone = TOperation::Undo(NodeValues[nodePosition], oldValue);
                    NodeValues[nodePosition] = TOperation::Merge(oldValueUndone, TOperation::template FromRepeated<TValue, TOperationResult>(value, 1));
                }
            } else {
                NodeValues[nodePosition] = TOperation::template FromRepeated<TValue, TOperationResult>(value, 1);
                if constexpr (LeafSizeLog > 0) {
                    NodeValues[nodePosition >>= LeafSizeLog] = ComputeRangeBruteForce(largeLeafStart, largeLeafStart + LEAF_SIZE);
                }
            }

            while (nodePosition > 1) {
                NodeValues[nodePosition >> 1] = TOperation::Merge(NodeValues[nodePosition & ~1], NodeValues[nodePosition | 1]);
                nodePosition >>= 1;
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

        inline TOperationResult ComputeRangeBruteForce(const size_t rangeStart, const size_t rangeEnd) const {
            assert(rangeStart < rangeEnd);
            if constexpr (HasFromMany<TOperation, TValue, TOperationResult>) {
                return TOperation::template FromMany<TOperationResult>(PointValues.data(), rangeStart, rangeEnd);
            } else {
                auto result = NodeValues[AllocatedLength + rangeStart];
                for (size_t index = AllocatedLength + rangeStart + 1; index < AllocatedLength + rangeEnd; ++index) {
                    result = TOperation::Merge(result, NodeValues[index]);
                }
                return result;
            }
        }

        inline TOperationResult ComputePrefixUnaligned(const size_t prefix) const {
            const auto alignedBorder = (prefix >> LeafSizeLog) << LeafSizeLog;
            if (alignedBorder == 0) [[unlikely]] {
                return ComputeRangeBruteForce(0, prefix);
            }

            const auto mainResult = ComputeRangeBackwards(AllocatedLength, alignedBorder);
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
            // No matter the structure of the tree, we use a binary heap layout. We need to take
            // any node starting at `rangeStart` coordinate, and we basically want to take the longest
            // one, not exceeding the query range. At least the node indexed as `AllocatedLength + rangeStart`,
            // an associate leaf, will do for us.
            auto currentNode = AllocatedLength + rangeStart;
            auto queryLength = rangeEnd - rangeStart;

            // Due to heap topology, each node has a contiguous range of power-2 segments starting there,
            // the cardinality of this set is same as the number of zeros at the end of node representation.
            // We refer to it in the code as `maxJump`.
            size_t maxJump = currentNode - (currentNode & (currentNode - 1));
            if (maxJump <= queryLength) {
                auto result = NodeValues[currentNode / maxJump];
                queryLength -= maxJump;
                currentNode += maxJump;
                maxJump = currentNode - (currentNode & (currentNode - 1));

                // While maxJump is smaller than the queried range, the things are simple:
                // we just take the largest node possible, reuse its result, and proceed.
                // The query length decreases, and the leftmost matching node index increases.
                while (maxJump <= queryLength) {
                    result = TOperation::Merge(result, NodeValues[currentNode / maxJump]);
                    queryLength -= maxJump;
                    currentNode += maxJump;
                    maxJump = currentNode - (currentNode & (currentNode - 1));
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
            size_t lastStep = queryLength - (queryLength & (queryLength - 1));
            size_t prevNode = lastNode - lastStep;

            // And use the result from that node.
            auto result = NodeValues[prevNode / lastStep];
            lastNode = prevNode;

            // Basically subtract the length of the taken interval, but faster.
            queryLength &= queryLength - 1;

            // Repeat the process when needed.
            while (queryLength > 0) {
                lastStep = queryLength - (queryLength & (queryLength - 1));
                prevNode = lastNode - lastStep;
                result = TOperation::Merge(NodeValues[prevNode / lastStep], result);  // Note the unusual operand order: the operation is non-commutative!
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
