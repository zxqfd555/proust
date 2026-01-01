#pragma once

#include <cassert>
#include <vector>
#include <optional>

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
        virtual TVersion Init(const size_t length) override {
            Nodes.ReserveCapacities(TNodeTopology::CapacityToReserve(length));
            if (TNodeTopology::IsLeaf(2)) {
                // If the leaf size is one, there is no need for an extra allocation.
                LazySnapshot.resize(length);
                TOperationResult defaultSnapshotValue = TOperation::template Repeat<TValue, TOperationResult>(TValue(), 1);
                std::fill(LazySnapshot.begin(), LazySnapshot.end(), defaultSnapshotValue);
            }
            InitNodes(ROOT_NODE_ID, 0, length);
            AllocatedLength = length;
            return CurrentVersion++;
        }

        virtual TVersion SetRange(const size_t rangeStart, const size_t rangeEnd, const TValue& value) override {
            assert(rangeStart <= rangeEnd && "Range start can't be bigger than its end");
            assert(rangeEnd <= AllocatedLength && "Range goes beyond the allocated length");

            if (rangeStart < rangeEnd) [[likely]] {
                ModifyNodes(ROOT_NODE_ID, rangeStart, rangeEnd, 0, AllocatedLength, value);
            }

            return CurrentVersion++;
        }

        virtual TOperationResult ComputeRange(const size_t rangeStart, const size_t rangeEnd) const override {
            assert(rangeStart <= rangeEnd && "Range start can't be bigger than its end");
            assert(rangeStart != rangeEnd && "The operation result for an empty set is undefined");
            assert(rangeEnd <= AllocatedLength && "Range goes beyond the allocated length");
            return QueryNodes(ROOT_NODE_ID, rangeStart, rangeEnd, 0, AllocatedLength);
        }

    private:
        TLayout Nodes;
        std::vector<TOperationResult> LazySnapshot;
        size_t AllocatedLength = 0;
        TVersion CurrentVersion = 0;

        void InitNodes(const size_t currentNodeId, const size_t nodeStart, const size_t nodeEnd) {
            Nodes.SetSubtreeValue(currentNodeId, TOperation::template Repeat<TValue, TOperationResult>(TValue(), nodeEnd - nodeStart));
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
                Nodes.SetSubtreeValue(nodeId, TOperation::template Repeat<TValue, TOperationResult>(value, nodeEnd - nodeStart));
                return;
            }

            if (TNodeTopology::IsLeaf(nodeEnd - nodeStart)) [[unlikely]] {
                const TOperationResult modifyValue = TOperation::template Repeat<TValue, TOperationResult>(value, 1);
                const size_t segmentStart = std::max(rangeStart, nodeStart);
                const size_t segmentEnd = std::min(rangeEnd, nodeEnd);
                if (Nodes.HasLazilyPropagated(nodeId)) {
                    const auto lazilyPropagated = Nodes.GetLazilyPropagated(nodeId);
                    const TOperationResult offcutValue = TOperation::template Repeat<TValue, TOperationResult>(lazilyPropagated, 1);
                    std::fill(LazySnapshot.begin() + nodeStart, LazySnapshot.begin() + segmentStart, offcutValue);
                    std::fill(LazySnapshot.begin() + segmentStart, LazySnapshot.begin() + segmentEnd, modifyValue);
                    std::fill(LazySnapshot.begin() + segmentEnd, LazySnapshot.begin() + nodeEnd, offcutValue);
                    Nodes.DropLazilyPropagated(nodeId);

                    const auto segmentValue = TOperation::template Repeat<TValue, TOperationResult>(value, segmentEnd - segmentStart);

                    const auto toSegmentEnd = 
                        segmentStart == nodeStart ? segmentValue :
                        TOperation::Merge(TOperation::template Repeat<TValue, TOperationResult>(lazilyPropagated, segmentStart - nodeStart), segmentValue);

                    auto nodeValue =
                        segmentEnd == nodeEnd ? toSegmentEnd :
                        TOperation::Merge(toSegmentEnd, TOperation::template Repeat<TValue, TOperationResult>(lazilyPropagated, nodeEnd - segmentEnd));

                    Nodes.SetSubtreeValue(nodeId, std::move(nodeValue));
                } else {
                    std::fill(LazySnapshot.begin() + segmentStart, LazySnapshot.begin() + segmentEnd, modifyValue);

                    TOperationResult subtreeValue = LazySnapshot[nodeStart];
                    if (segmentStart > nodeStart) {
                        for (size_t index = nodeStart + 1; index < segmentStart; ++index) {
                            subtreeValue = TOperation::Merge(subtreeValue, LazySnapshot[index]);
                        }
                        subtreeValue = TOperation::Merge(subtreeValue, TOperation::template Repeat<TValue, TOperationResult>(value, segmentEnd - segmentStart));
                    } else {
                        subtreeValue = TOperation::template Repeat<TValue, TOperationResult>(value, segmentEnd - segmentStart);
                    }

                    for (size_t index = segmentEnd; index < nodeEnd; ++index) {
                        subtreeValue = TOperation::Merge(subtreeValue, LazySnapshot[index]);
                    }

                    Nodes.SetSubtreeValue(nodeId, std::move(subtreeValue));
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
                    Nodes.SetSubtreeValue(leftChildId, TOperation::template Repeat<TValue, TOperationResult>(toPropagate, midpoint - nodeStart));
                }

                if (midpoint < rangeEnd) {
                    ModifyNodes(rightChildId, rangeStart, rangeEnd, midpoint, nodeEnd, value);
                } else if (hasLazilyPropagated) {
                    Nodes.SetSubtreeValue(rightChildId, TOperation::template Repeat<TValue, TOperationResult>(toPropagate, nodeEnd - midpoint));
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
                return TOperation::template Repeat<TValue, TOperationResult>(lazilyPropagated, segmentEnd - segmentStart);
            }

            if (TNodeTopology::IsLeaf(nodeEnd - nodeStart)) {
                const size_t segmentStart = std::max(rangeStart, nodeStart);
                const size_t segmentEnd = std::min(rangeEnd, nodeEnd);
                auto result = LazySnapshot[segmentStart];
                for (size_t index = segmentStart + 1; index < segmentEnd; ++index) {
                    result = TOperation::Merge(result, LazySnapshot[index]);
                }
                return result;
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

    template<typename TValue, typename TOperationResult, typename TOperation, size_t LeafSizeLog = 0>
    class TRobustPointUpdateEngine: public IPointUpdateEngine<TValue, TOperationResult, TOperation> {
    private:
        // We use 1-indexing here: although not as nice, it allows us a bit cheaper topology operations
        // as compared to 0-indexing.
        static constexpr size_t ROOT_NODE_ID = 1;
        static constexpr size_t LEAF_SIZE = 1 << LeafSizeLog;
    
    public:
        virtual TVersion Init(const size_t length) override {
            // Overbook capacities up to the closest power of two
            AllocatedLength = std::bit_ceil(length);

            // Being a power of two, the tree takes 2x nodes from the length
            NodeValues.resize(AllocatedLength << 1);

            // Initialize the tree with the default values starting from the leafs
            const auto leafValue = TOperation::template Repeat<TValue, TOperationResult>(TValue(), 1);
            std::fill(NodeValues.begin(), NodeValues.end(), leafValue);
            for (size_t index = AllocatedLength - 1; index >= 1; --index) {
                NodeValues[index] = TOperation::Merge(NodeValues[(index << 1)], NodeValues[(index << 1) + 1]);
            }
            return CurrentVersion++;
        }

        virtual TVersion SetPoint(const size_t position, const TValue& value) override {
            assert(position < AllocatedLength && "The position is beyond the allocated size");
            size_t nodePosition = AllocatedLength + position;
            NodeValues[nodePosition] = TOperation::template Repeat<TValue, TOperationResult>(value, 1);

            const size_t largeLeafStart = position & ~(LEAF_SIZE - 1);
            NodeValues[nodePosition >>= LeafSizeLog] = ComputeRangeBruteForce(largeLeafStart, largeLeafStart + LEAF_SIZE);
            
            while (nodePosition > 1) {
                NodeValues[nodePosition >> 1] = TOperation::Merge(NodeValues[nodePosition & ~1], NodeValues[nodePosition | 1]);
                nodePosition >>= 1;
            }
            return CurrentVersion++;
        }

        virtual TOperationResult ComputeRange(const size_t rangeStart, const size_t rangeEnd) const override {
            assert(rangeStart <= rangeEnd && "Range start can't be bigger than its end");
            assert(rangeStart != rangeEnd && "The operation result for an empty set is undefined");
            assert(rangeEnd <= AllocatedLength && "Range goes beyond the allocated length");

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

    private:
        std::vector<TOperationResult> NodeValues;
        size_t AllocatedLength;
        TVersion CurrentVersion = 0;

        inline TOperationResult ComputeRangeBruteForce(const size_t rangeStart, const size_t rangeEnd) const {
            assert(rangeStart < rangeEnd);
            auto result = NodeValues[AllocatedLength + rangeStart];
            for (size_t index = AllocatedLength + rangeStart + 1; index < AllocatedLength + rangeEnd; ++index) {
                result = TOperation::Merge(result, NodeValues[index]);
            }
            return result;
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

        inline TOperationResult ComputeRangeBackwards(size_t& currentNode, size_t& queryLength) const {
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
        return std::make_unique<TRobustPointUpdateEngine<TValue, TOperationResult, TOperation, LeafSizeLog>>();
    }
} // namespace NProust
