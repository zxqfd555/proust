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
                const size_t midpoint = nodeStart + ((nodeEnd - nodeStart) >> 1);
                const size_t leftChildId = TNodeTopology::LeftChild(nodeId, nodeEnd - nodeStart);
                const size_t rightChildId = TNodeTopology::RightChild(nodeId, nodeEnd - nodeStart);

                const bool hasLazilyPropagated = Nodes.HasLazilyPropagated(nodeId);
                TValue toPropagate = Nodes.GetLazilyPropagated(nodeId);

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
                const size_t midpoint = nodeStart + ((nodeEnd - nodeStart) >> 1);
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

    template<typename TValue, typename TOperationResult, typename TOperation>
    class TRobustPointUpdateEngine: public IPointUpdateEngine<TValue, TOperationResult, TOperation> {
    private:
        const size_t ROOT_NODE_ID = 0;
    
    public:
        virtual TVersion Init(const size_t length) override {
            // Overbook capacities up to the closest power of two
            AllocatedLength = std::bit_ceil(length);

            // Being a power of two, the tree takes 2x nodes from the length
            NodeValues.resize(AllocatedLength << 1);

            // Initialize the tree with the default values starting from the leafs
            const auto leafValue = TOperation::template Repeat<TValue, TOperationResult>(TValue(), AllocatedLength);
            std::fill(NodeValues.begin() + AllocatedLength, NodeValues.begin() + (AllocatedLength << 1), leafValue);
            for (size_t index = 0; index < AllocatedLength; ++index) {
                const size_t actualIndex = AllocatedLength - index - 1;
                NodeValues[actualIndex] = TOperation::Merge(NodeValues[(actualIndex << 1) + 1], NodeValues[(actualIndex << 1) + 2]);
            }
            return CurrentVersion++;
        }

        virtual TVersion SetPoint(const size_t position, const TValue& value) override {
            size_t nodePosition = AllocatedLength + position - 1;
            NodeValues[nodePosition] = TOperation::template Repeat<TValue, TOperationResult>(value, (size_t)1);
            while (nodePosition > 0) {
                nodePosition = (nodePosition - 1) >> 1;
                NodeValues[nodePosition] = TOperation::Merge(NodeValues[(nodePosition << 1) + 1], NodeValues[(nodePosition << 1) + 2]);
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
        std::vector<TOperationResult> NodeValues;
        size_t AllocatedLength;
        TVersion CurrentVersion = 0;

        TOperationResult QueryNodes(const size_t nodeId, const size_t rangeStart, const size_t rangeEnd, const size_t nodeStart, const size_t nodeEnd) const {
            if (rangeStart <= nodeStart && nodeEnd <= rangeEnd) {
                return NodeValues[nodeId];
            }

            const size_t midpoint = nodeStart + ((nodeEnd - nodeStart) >> 1);
            if (rangeStart < midpoint) {
                const auto resultLeft = QueryNodes((nodeId << 1) + 1, rangeStart, rangeEnd, nodeStart, midpoint);
                if (midpoint < rangeEnd) {
                    const auto resultRight = QueryNodes((nodeId << 1) + 2, rangeStart, rangeEnd, midpoint, nodeEnd);
                    return TOperation::Merge(resultLeft, resultRight);
                }
                return resultLeft;
            }

            return QueryNodes((nodeId << 1) + 2, rangeStart, rangeEnd, midpoint, nodeEnd);
        }
    };

    template<typename TValue, typename TOperationResult, typename TOperation>
    std::unique_ptr<IPointUpdateEngine<TValue, TOperationResult, TOperation>> CreateRobustPointUpdateEngine() {
        return std::make_unique<TRobustPointUpdateEngine<TValue, TOperationResult, TOperation>>();
    }
} // namespace NProust
