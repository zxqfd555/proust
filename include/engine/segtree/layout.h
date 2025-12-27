#pragma once

#include <optional>
#include <vector>
#include <concepts>
#include <cstddef>

namespace NProust {
    constexpr std::string_view AOS_LAYOUT_NAME = "AoS";
    constexpr std::string_view SOA_LAYOUT_NAME = "SoA";

    enum class EDataLayout {
        AoS,
        SoA,
    };

    [[nodiscard]] inline std::optional<EDataLayout> DataLayoutFromString(const std::string_view s) {
        if (s == AOS_LAYOUT_NAME) {
            return EDataLayout::AoS;
        }
        if (s == SOA_LAYOUT_NAME) {
            return EDataLayout::SoA;
        }
        return std::nullopt;
    }

    template<class T, class TValue>
    concept LazyTag = requires(T tag, TValue v) {
        { tag.Defined() } -> std::same_as<bool>;
        { tag.GetValue() } -> std::same_as<const TValue&>;
        tag.SetValue(v);
        tag.Drop();
    };

    template<class TValue, class TSentinel = void>
    class TLazyTag;

    template<class TValue>
    class TLazyTag<TValue, void> {
    public:
        TLazyTag() = default;

        template<typename T>
        TLazyTag(T&& value) noexcept(noexcept(Value = std::forward<T>(value)))
            : Value(std::forward<T>(value))
            , IsDefined(true)
        {
        }

        [[nodiscard]] bool Defined() const noexcept {
            return IsDefined;
        }

        template<typename T>
        void SetValue(T&& value) noexcept(noexcept(Value = std::forward<T>(value))) {
            Value = std::forward<T>(value);
            IsDefined = true;
        }

        [[nodiscard]] const TValue& GetValue() const noexcept {
            assert(IsDefined);
            return Value;
        }

        void Drop() noexcept {
            IsDefined = false;
        }

    private:
        TValue Value;
        [[no_unique_address]] bool IsDefined = false;
    };

    template<class TValue, TValue Sentinel>
    class TLazyTag<TValue, std::integral_constant<TValue, Sentinel>> {
    public:
        TLazyTag() = default;

        template<typename T>
        TLazyTag(T&& value) noexcept(noexcept(Value = std::forward<T>(value)))
            : Value(std::forward<T>(value))
        {
        }

        [[nodiscard]] bool Defined() const noexcept {
            return Value != Sentinel;
        }

        template<typename T>
        void SetValue(T&& value) noexcept(noexcept(Value = std::forward<T>(value))) {
            Value = std::forward<T>(value);
        }

        [[nodiscard]] const TValue& GetValue() const noexcept {
            return Value;
        }

        void Drop() noexcept {
            Value = Sentinel;
        }
    
    private:
        TValue Value = Sentinel;
    };

    template<class TValue, class TOperationResult, auto Sentinel = std::nullopt>
    class TAoSLayout;

    template<class TValue, class TOperationResult, auto Sentinel = std::nullopt>
    class TSoALayout;

    namespace NDetail {
        template<class TValue, auto Sentinel>
        struct TLazyTagSelectorImpl;

        template<class TValue>
        struct TLazyTagSelectorImpl<TValue, std::nullopt> {
            using TLazyTagImpl = TLazyTag<TValue>;
        };

        template<class TValue, auto Sentinel>
        struct TLazyTagSelectorImpl {
            static_assert(std::is_same_v<decltype(Sentinel), TValue>, "Sentinel must have the same type as TValue");

            using TLazyTagImpl = TLazyTag<TValue, std::integral_constant<TValue, Sentinel>>;
        };

        template<class TValue, auto Sentinel>
        using TLazyTagSelector = typename TLazyTagSelectorImpl<TValue, Sentinel>::TLazyTagImpl;
    } // namespace NDetail

    template<class TValue, class TOperationResult, auto Sentinel>
    class TAoSLayout {
    private:
        using TLazyTag = NDetail::TLazyTagSelector<TValue, Sentinel>;

        struct TNode {
            TOperationResult SubtreeValue;
            TLazyTag LazilyPropagated;
        };

        std::vector<TNode> Nodes;

    public:
        void ReserveCapacities(const size_t size) {
            Nodes.resize(size);
        }

        [[nodiscard]] bool HasLazilyPropagated(const size_t nodeId) const noexcept {
            return Nodes[nodeId].LazilyPropagated.Defined();
        }

        [[nodiscard]] const TValue& GetLazilyPropagated(const size_t nodeId) const noexcept {
            return Nodes[nodeId].LazilyPropagated.GetValue();
        }

        template<typename T>
        void SetLazilyPropagated(const size_t nodeId, T&& lazilyPropagated)
        noexcept(noexcept(Nodes[nodeId].LazilyPropagated.SetValue(std::forward<T>(lazilyPropagated)))) {
            Nodes[nodeId].LazilyPropagated.SetValue(std::forward<T>(lazilyPropagated));
        }

        void DropLazilyPropagated(const size_t nodeId) noexcept {
            Nodes[nodeId].LazilyPropagated.Drop();
        }
    
        [[nodiscard]] const TOperationResult& GetSubtreeValue(const size_t nodeId) const noexcept {
            return Nodes[nodeId].SubtreeValue;
        }

        template<typename T>
        void SetSubtreeValue(const size_t nodeId, T&& value) noexcept(noexcept(Nodes[nodeId].SubtreeValue = std::forward<T>(value))) {
            Nodes[nodeId].SubtreeValue = std::forward<T>(value);
        }
    };

    template<class TValue, class TOperationResult, auto Sentinel>
    class TSoALayout {
    private:
        using TLazyTag = NDetail::TLazyTagSelector<TValue, Sentinel>;

        std::vector<TOperationResult> SubtreeValue;
        std::vector<TLazyTag> LazilyPropagated;

    public:
        void ReserveCapacities(const size_t size) {
            SubtreeValue.resize(size);
            LazilyPropagated.resize(size);
        }

        [[nodiscard]] bool HasLazilyPropagated(const size_t nodeId) const noexcept {
            return LazilyPropagated[nodeId].Defined();
        }

        [[nodiscard]] const TValue& GetLazilyPropagated(const size_t nodeId) const noexcept {
            return LazilyPropagated[nodeId].GetValue();
        }

        template<typename T>
        void SetLazilyPropagated(const size_t nodeId, T&& lazilyPropagated) noexcept(noexcept(LazilyPropagated[nodeId].SetValue(std::forward<T>(lazilyPropagated)))) {
            LazilyPropagated[nodeId].SetValue(std::forward<T>(lazilyPropagated));
        }

        void DropLazilyPropagated(const size_t nodeId) noexcept {
            LazilyPropagated[nodeId].Drop();
        }

        [[nodiscard]] const TOperationResult& GetSubtreeValue(const size_t nodeId) const noexcept {
            return SubtreeValue[nodeId];
        }

        template<typename T>
        void SetSubtreeValue(const size_t nodeId, T&& value) noexcept(noexcept(SubtreeValue[nodeId] = std::forward<T>(value))) {
            SubtreeValue[nodeId] = std::forward<T>(value);
        }
    };
} // namespace NProust
