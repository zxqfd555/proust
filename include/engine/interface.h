#pragma once

#include <cstdint>
#include <functional>
#include <memory>

namespace NProust {
    using TVersion = uint64_t;

    template<typename Ops, typename TValue, typename TOperationResult>
    concept ValidOperation = requires(const TValue& value, const TOperationResult& lhs, const TOperationResult& rhs, const size_t times) {
        {
            Ops::template Repeat<TValue, TOperationResult>(value, times)
        } -> std::same_as<TOperationResult>;

        {
            Ops::template Merge<TOperationResult>(lhs, rhs)
        } -> std::same_as<TOperationResult>;
    };

    template<typename TValue, typename TOperationResult, typename TOperation>
    requires ValidOperation<TOperation, TValue, TOperationResult>
    class IEngine {
    public:
        virtual TVersion Init(const size_t length) = 0;
        virtual TVersion SetRange(const size_t left, const size_t right, const TValue& value) = 0;
        virtual TOperationResult ComputeRange(const size_t left, const size_t right) const = 0;

        virtual ~IEngine() = default;
    };

} // namespace NProust
