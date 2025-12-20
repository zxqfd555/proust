#pragma once

#include <optional>
#include <vector>
#include <concepts>
#include <cstddef>

namespace NProust {
    template <typename TLayout, typename TValue, typename TOperationResult>
    concept DataLayout = requires(TLayout layout, size_t nodeId, std::optional<TValue> lazilyPropagated, TOperationResult subtreeValue, size_t size) {
        { layout.ReserveCapacities(size) } -> std::same_as<void>;

        { layout.GetLazilyPropagated(nodeId) } -> std::same_as<std::optional<TValue>>;
        { layout.SetLazilyPropagated(nodeId, lazilyPropagated) } -> std::same_as<void>;

        { layout.GetSubtreeValue(nodeId) } -> std::same_as<TOperationResult>;
        { layout.SetSubtreeValue(nodeId, subtreeValue) } -> std::same_as<void>;
    };

    template<class TValue, class TOperationResult>
    class TAoSLayout {
    public:
        void ReserveCapacities(const size_t size) {
            Nodes.resize(size);
        }

        std::optional<TValue> GetLazilyPropagated(const size_t nodeId) const {
            return Nodes[nodeId].LazilyPropagated;
        }

        void SetLazilyPropagated(const size_t nodeId, const std::optional<TValue>& lazilyPropagated) {  // TODO: move semantics maybe
            Nodes[nodeId].LazilyPropagated = lazilyPropagated;
        }

        TOperationResult GetSubtreeValue(const size_t nodeId) const {
            return Nodes[nodeId].SubtreeValue;
        }

        void SetSubtreeValue(const size_t nodeId, const TOperationResult& value) {
            Nodes[nodeId].SubtreeValue = value;
        }

    private:
        struct TNode {
            TOperationResult SubtreeValue;
            std::optional<TValue> LazilyPropagated;
        };

        std::vector<TNode> Nodes;
    };

    template<class TValue, class TOperationResult>
    class TSoALayout {
    public:
        void ReserveCapacities(const size_t size) {
            SubtreeValue.resize(size);
            LazilyPropagated.resize(size);
        }

        std::optional<TValue> GetLazilyPropagated(const size_t nodeId) const {
            return LazilyPropagated[nodeId];
        }

        void SetLazilyPropagated(const size_t nodeId, const std::optional<TValue>& lazilyPropagated) {  // TODO: move semantics maybe
            LazilyPropagated[nodeId] = lazilyPropagated;
        }

        TOperationResult GetSubtreeValue(const size_t nodeId) const {
            return SubtreeValue[nodeId];
        }

        void SetSubtreeValue(const size_t nodeId, const TOperationResult& value) {
            SubtreeValue[nodeId] = value;
        }

    private:
        std::vector<TOperationResult> SubtreeValue;
        std::vector<std::optional<TValue>> LazilyPropagated;
    };
} // namespace NProust
