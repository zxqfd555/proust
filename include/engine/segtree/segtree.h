#pragma once

#include <cassert>
#include <vector>
#include <optional>

#include "../interface.h"

namespace NProust {
    // TODO: it doesn't make any sense to limit the size of the node index. Maybe there is
    // some sense in using 4-byte ranges, not the 64-bit ones.
    using TNodeIndex = size_t;

    template<typename TValue, typename TOperationResult, typename TOperation>
    class TSegmentTreeEngine: public IEngine<TValue, TOperationResult, TOperation> {
    public:
        virtual TVersion Init(const size_t length) override {
            Nodes.resize(length << 1);
            NodeCount = 0;
            AllocatedLength = length;

            assert(InitNodes(0, length) == 0 && "The root node isn't the first one");

            return CurrentVersion++;
        }

        virtual TVersion SetRange(const size_t rangeStart, const size_t rangeEnd, const TValue& value) override {
            assert(rangeStart <= rangeEnd && "Range start can't be bigger than its end");
            assert(rangeEnd <= AllocatedLength && "Range goes beyond the allocated length");

            if (rangeStart < rangeEnd) {
                ModifyNodes(0, rangeStart, rangeEnd, 0, AllocatedLength, value);
            }

            return CurrentVersion++;
        }

        virtual TOperationResult ComputeRange(const size_t rangeStart, const size_t rangeEnd) override {
            assert(rangeStart <= rangeEnd && "Range start can't be bigger than its end");
            assert(rangeStart != rangeEnd && "The operation result for an empty set is undefined");
            assert(rangeEnd <= AllocatedLength && "Range goes beyond the allocated length");
            return QueryNodes(0, rangeStart, rangeEnd, 0, AllocatedLength);
        }

    private:
        struct TNode {
            TOperationResult SubtreeValue;
            std::optional<TValue> LazilyPropagated;

            TNode() = default;

            TNode(const size_t rangeLength)
                : SubtreeValue(TOperation::template Repeat<TValue, TOperationResult>(TValue(), rangeLength))
                , LazilyPropagated(std::nullopt)
            {
            }
        };

        size_t AllocatedLength{0};
        std::vector<TNode> Nodes;
        size_t NodeCount{0};
        TVersion CurrentVersion{0};

        TNodeIndex InitNodes(const size_t nodeStart, const size_t nodeEnd) {
            const TNodeIndex currentNodeId = NodeCount++;
            assert(currentNodeId < Nodes.size() && "There aren't enough pre-allocated nodes");
            Nodes[currentNodeId] = TNode(nodeEnd - nodeStart);

            if (nodeStart + 1 < nodeEnd) {
                const size_t midpoint = nodeStart + ((nodeEnd - nodeStart) >> 1);
                InitNodes(nodeStart, midpoint);
                InitNodes(midpoint, nodeEnd);
            }

            return currentNodeId;
        }

        void ModifyNodes(const TNodeIndex nodeId, const size_t rangeStart, const size_t rangeEnd, const size_t nodeStart, const size_t nodeEnd, const TValue& value) {
            TNode& currentNode = Nodes[nodeId];
            if (rangeStart <= nodeStart && rangeEnd >= nodeEnd) {
                currentNode.LazilyPropagated = value;
                currentNode.SubtreeValue = TOperation::template Repeat<TValue, TOperationResult>(value, nodeEnd - nodeStart);
                return;
            }

            if (Nodes[nodeId].LazilyPropagated.has_value()) {
                Propagate(nodeId, nodeStart, nodeEnd);
            }

            const size_t midpoint = nodeStart + ((nodeEnd - nodeStart) >> 1);
            const TNodeIndex leftSubtreeSize = ((midpoint - nodeStart) << 1) - 1;
            const TNodeIndex leftChildId = nodeId + 1;
            const TNodeIndex rightChildId = nodeId + leftSubtreeSize + 1;
            if (midpoint > rangeStart) {
                ModifyNodes(leftChildId, rangeStart, rangeEnd, nodeStart, midpoint, value);
            }
            if (midpoint < rangeEnd) {
                ModifyNodes(rightChildId, rangeStart, rangeEnd, midpoint, nodeEnd, value);
            }
            currentNode.SubtreeValue = TOperation::Merge(Nodes[leftChildId].SubtreeValue, Nodes[rightChildId].SubtreeValue);
        }

        void Propagate(const TNodeIndex nodeId, const size_t nodeStart, const size_t nodeEnd) {
            TNode& currentNode = Nodes[nodeId];

            // Asserting, but it must also be checked upstream, before calling propagation.
            assert(currentNode.LazilyPropagated.has_value());
            const TValue toPropagate = *currentNode.LazilyPropagated;

            currentNode.LazilyPropagated = std::nullopt;
            currentNode.SubtreeValue = TOperation::template Repeat<TValue, TOperationResult>(toPropagate, nodeEnd - nodeStart);
            const size_t midpoint = nodeStart + ((nodeEnd - nodeStart) >> 1);

            Nodes[nodeId + 1].LazilyPropagated = toPropagate;
            Nodes[nodeId + 1].SubtreeValue = TOperation::template Repeat<TValue, TOperationResult>(toPropagate, midpoint - nodeStart);
            Nodes[nodeId + ((midpoint - nodeStart) << 1)].LazilyPropagated = toPropagate;
            Nodes[nodeId + ((midpoint - nodeStart) << 1)].SubtreeValue = TOperation::template Repeat<TValue, TOperationResult>(toPropagate, nodeEnd - midpoint);
        }

        TOperationResult QueryNodes(const TNodeIndex nodeId, const size_t rangeStart, const size_t rangeEnd, const size_t nodeStart, const size_t nodeEnd) {
            const TNode& currentNode = Nodes[nodeId];
            if (currentNode.LazilyPropagated.has_value()) {
                const size_t segmentStart = std::max(rangeStart, nodeStart);
                const size_t segmentEnd = std::min(rangeEnd, nodeEnd);
                return TOperation::template Repeat<TValue, TOperationResult>(*currentNode.LazilyPropagated, segmentEnd - segmentStart);
            }

            if (rangeStart <= nodeStart && rangeEnd >= nodeEnd) {
                return currentNode.SubtreeValue;
            }

            const size_t midpoint = nodeStart + ((nodeEnd - nodeStart) >> 1);
            if (midpoint > rangeStart) {
                const TNodeIndex leftChildId = nodeId + 1;
                const TOperationResult leftResult = QueryNodes(leftChildId, rangeStart, rangeEnd, nodeStart, midpoint);
                if (midpoint < rangeEnd) {
                    const TNodeIndex leftSubtreeSize = ((midpoint - nodeStart) << 1) - 1;
                    const TNodeIndex rightChildId = nodeId + leftSubtreeSize + 1;
                    const TOperationResult rightResult = QueryNodes(rightChildId, rangeStart, rangeEnd, midpoint, nodeEnd);
                    return TOperation::Merge(leftResult, rightResult);
                } else {
                    return leftResult;
                }
            }

            const TNodeIndex leftSubtreeSize = ((midpoint - nodeStart) << 1) - 1;
            const TNodeIndex rightChildId = nodeId + leftSubtreeSize + 1;
            return QueryNodes(rightChildId, rangeStart, rangeEnd, midpoint, nodeEnd);
        }
    };

    template<typename TValue, typename TOperationResult, typename TOperation>
    std::unique_ptr<IEngine<TValue, TOperationResult, TOperation>> CreateRobustEngine() {
        return std::make_unique<TSegmentTreeEngine<TValue, TOperationResult, TOperation>>();
    }
} // namespace NProust
