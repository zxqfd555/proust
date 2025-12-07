#pragma once

#include <cassert>
#include <vector>
#include <optional>

#include "../interface.h"

namespace NProust {
    template <typename TLayout>
    concept NodeLayout = requires(const TLayout layout, size_t nodeId, size_t rangeLength)
    {
        { layout.CapacityToReserve(rangeLength) } -> std::convertible_to<size_t>;
        { layout.LeftChild(nodeId, rangeLength) } -> std::convertible_to<size_t>;
        { layout.RightChild(nodeId, rangeLength) } -> std::convertible_to<size_t>;
    };

    struct TPreorderNodeLayout {
        static size_t CapacityToReserve(const size_t rangeLength) {
            return (rangeLength << 1);
        }

        static size_t LeftChild(const size_t nodeId, const size_t rangeLength) {
            return nodeId + 1;
        }

        static size_t RightChild(const size_t nodeId, const size_t rangeLength) {
            // The node is the root of a tree for `rangeLength` elements.
            // The left subtree of the node represents `L = rangeLength / 2` elements.
            // The number of elements in a segment tree for a size `S` is always `2 * S - 1`,
            // (can be proven by induction).
            //
            // Then, the last node in the left subtree is `nodeId + [rangeLength / 2] * 2 - 1`.
            // And the first node in the right subtree - the right son in this type of traversal,
            // is `nodeId + [rangeLength / 2] * 2`.
            //
            // For even `rangeLength`, it's just `nodeId + rangeLength`.
            // If the length is odd, `nodeId + rangeLength - 1`.
            //
            // To avoid branching and be CPU-efficient, we can express it with binary operations:
            // `nodeId + rangeLength - (rangeLength % 2)`, where base-2 arithmetic
            // is just shift and AND.
            return nodeId + (rangeLength & ~size_t(1));
        }
    };

    struct TBinaryHeapNodeLayout {
        static size_t CapacityToReserve(const size_t rangeLength) {
            return (rangeLength << 2);
        }

        static size_t LeftChild(const size_t nodeId, const size_t rangeLength) {
            return (nodeId << 1) + 1;
        }

        static size_t RightChild(const size_t nodeId, const size_t rangeLength) {
            return (nodeId << 1) + 2;
        }
    };

    template<typename TValue, typename TOperationResult, typename TOperation, NodeLayout TNodeLayout = TBinaryHeapNodeLayout>
    class TSegmentTreeEngine: public IEngine<TValue, TOperationResult, TOperation> {
    public:
        virtual TVersion Init(const size_t length) override {
            Nodes.resize(TNodeLayout::CapacityToReserve(length));
            NodeCount = 0;
            AllocatedLength = length;
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

        size_t AllocatedLength = 0;
        std::vector<TNode> Nodes;
        size_t NodeCount = 0;
        TVersion CurrentVersion = 0;

        void InitNodes(const size_t currentNodeId, const size_t nodeStart, const size_t nodeEnd) {
            assert(currentNodeId < Nodes.size() && "There aren't enough pre-allocated nodes");
            Nodes[currentNodeId] = TNode(nodeEnd - nodeStart);

            if (nodeStart + 1 < nodeEnd) {
                const size_t rangeLength = nodeEnd - nodeStart;
                const size_t midpoint = nodeStart + (rangeLength >> 1);
                InitNodes(TNodeLayout::LeftChild(currentNodeId, rangeLength), nodeStart, midpoint);
                InitNodes(TNodeLayout::RightChild(currentNodeId, rangeLength), midpoint, nodeEnd);
            }
        }

        void ModifyNodes(const size_t nodeId, const size_t rangeStart, const size_t rangeEnd, const size_t nodeStart, const size_t nodeEnd, const TValue& value) {
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
            const size_t leftChildId = TNodeLayout::LeftChild(nodeId, nodeEnd - nodeStart);
            const size_t rightChildId = TNodeLayout::RightChild(nodeId, nodeEnd - nodeStart);
            if (midpoint > rangeStart) {
                ModifyNodes(leftChildId, rangeStart, rangeEnd, nodeStart, midpoint, value);
            }
            if (midpoint < rangeEnd) {
                ModifyNodes(rightChildId, rangeStart, rangeEnd, midpoint, nodeEnd, value);
            }
            currentNode.SubtreeValue = TOperation::Merge(Nodes[leftChildId].SubtreeValue, Nodes[rightChildId].SubtreeValue);
        }

        void Propagate(const size_t nodeId, const size_t nodeStart, const size_t nodeEnd) {
            TNode& currentNode = Nodes[nodeId];

            // Asserting, but it must also be checked upstream, before calling propagation.
            assert(currentNode.LazilyPropagated.has_value());
            const TValue toPropagate = *currentNode.LazilyPropagated;

            currentNode.LazilyPropagated = std::nullopt;
            currentNode.SubtreeValue = TOperation::template Repeat<TValue, TOperationResult>(toPropagate, nodeEnd - nodeStart);
            const size_t midpoint = nodeStart + ((nodeEnd - nodeStart) >> 1);

            const size_t leftChildId = TNodeLayout::LeftChild(nodeId, nodeEnd - nodeStart);
            const size_t rightChildId = TNodeLayout::RightChild(nodeId, nodeEnd - nodeStart);
            Nodes[leftChildId].LazilyPropagated = toPropagate;
            Nodes[leftChildId].SubtreeValue = TOperation::template Repeat<TValue, TOperationResult>(toPropagate, midpoint - nodeStart);
            Nodes[rightChildId].LazilyPropagated = toPropagate;
            Nodes[rightChildId].SubtreeValue = TOperation::template Repeat<TValue, TOperationResult>(toPropagate, nodeEnd - midpoint);
        }

        TOperationResult QueryNodes(const size_t nodeId, const size_t rangeStart, const size_t rangeEnd, const size_t nodeStart, const size_t nodeEnd) {
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
                const size_t leftChildId = TNodeLayout::LeftChild(nodeId, nodeEnd - nodeStart);
                const TOperationResult leftResult = QueryNodes(leftChildId, rangeStart, rangeEnd, nodeStart, midpoint);
                if (midpoint < rangeEnd) {
                    const size_t rightChildId = TNodeLayout::RightChild(nodeId, nodeEnd - nodeStart);
                    const TOperationResult rightResult = QueryNodes(rightChildId, rangeStart, rangeEnd, midpoint, nodeEnd);
                    return TOperation::Merge(leftResult, rightResult);
                } else {
                    return leftResult;
                }
            }

            const size_t rightChildId = TNodeLayout::RightChild(nodeId, nodeEnd - nodeStart);
            return QueryNodes(rightChildId, rangeStart, rangeEnd, midpoint, nodeEnd);
        }
    };

    template<typename TValue, typename TOperationResult, typename TOperation, NodeLayout TNodeLayout = TPreorderNodeLayout>
    std::unique_ptr<IEngine<TValue, TOperationResult, TOperation>> CreateRobustEngine() {
        return std::make_unique<TSegmentTreeEngine<TValue, TOperationResult, TOperation, TNodeLayout>>();
    }
} // namespace NProust
