#pragma once

namespace NProust {
    template <typename TTopology>
    concept NodeTopology = requires(const TTopology layout, size_t nodeId, size_t rangeLength) {
        { layout.CapacityToReserve(rangeLength) } -> std::convertible_to<size_t>;
        { layout.LeftChild(nodeId, rangeLength) } -> std::convertible_to<size_t>;
        { layout.RightChild(nodeId, rangeLength) } -> std::convertible_to<size_t>;
        { layout.IsLeaf(rangeLength) } -> std::convertible_to<bool>;
    };

    template<size_t LeafSizeLog = 0>
    struct TPreorderNodeTopology {
        static constexpr size_t LeafSize = size_t(1) << LeafSizeLog;

        inline static size_t CapacityToReserve(size_t rangeLength) {
            if constexpr (LeafSizeLog == 0) {
                return rangeLength << 1;
            } else {
                return rangeLength;
            }
        }

        inline static size_t LeftChild(size_t nodeId, size_t rangeLength) {
            return nodeId + 1;
        }

        inline static size_t RightChild(size_t nodeId, size_t rangeLength) {
            if constexpr (LeafSizeLog == 0) {
                return nodeId + (rangeLength & ~size_t(1));
            } else {
                return nodeId + (rangeLength >> 1) - 1;
            }
        }

        inline static bool IsLeaf(size_t rangeLength) {
            if constexpr (LeafSizeLog == 0) {
                return rangeLength == 1;
            } else {
                return rangeLength <= LeafSize;
            }
        }
    };

    template<size_t LeafSizeLog = 0>
    struct TBinaryHeapNodeTopology {
        static constexpr size_t LeafSize = size_t(1) << LeafSizeLog;

        static size_t CapacityToReserve(const size_t rangeLength) {
            // Greedily compute the biggest index of a node to have
            // a talored mem estimation
            size_t currentNodeIndex = 0;
            size_t nodeStart = 0;
            size_t nodeEnd = rangeLength;
            while (!IsLeaf(nodeEnd - nodeStart)) {
                const auto midpoint = nodeStart + ((nodeEnd - nodeStart) >> 1);
                const size_t leftChildSize = midpoint - nodeStart;
                const size_t rightChildSize = nodeEnd - midpoint;
                if (!IsLeaf(leftChildSize) && IsLeaf(rightChildSize)) {
                    nodeEnd = midpoint;
                    currentNodeIndex = LeftChild(currentNodeIndex, nodeEnd - nodeStart);
                } else {
                    nodeStart = midpoint;
                    currentNodeIndex = RightChild(currentNodeIndex, nodeEnd - nodeStart);
                }
            }
            return currentNodeIndex + 1;
        }

        static size_t LeftChild(const size_t nodeId, const size_t rangeLength) {
            return (nodeId << 1) + 1;
        }

        static size_t RightChild(const size_t nodeId, const size_t rangeLength) {
            return (nodeId << 1) + 2;
        }

        static size_t IsLeaf(const size_t rangeLength) {
            if constexpr (LeafSizeLog == 0) {
                return rangeLength == 1;
            } else {
                return rangeLength <= LeafSize;
            }
        }
    };
}
