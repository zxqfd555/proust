#pragma once

#include <cstdint>
#include <functional>
#include <memory>

namespace NProust {
    template<typename TOperation, typename TValue, typename TOperationResult>
    concept ValidOperation = requires(const TValue& value, const TOperationResult& lhs, const TOperationResult& rhs, const size_t times) {
        { TOperation::template FromRepeated<TValue, TOperationResult>(value, times) } -> std::same_as<TOperationResult>;
        { TOperation::template Merge<TOperationResult>(lhs, rhs) } -> std::same_as<TOperationResult>;
    };

    template<typename TOperation, typename TValue, typename TOperationResult>
    concept HasFromMany = requires(const TValue* data, size_t fromIndex, size_t toIndex) {
        { TOperation::template FromMany<TOperationResult>(data, fromIndex, toIndex) } -> std::same_as<TOperationResult>;
    };

    template<typename TOperation, typename TOperationResult>
    concept HasUndo = requires(const TOperationResult& lhs, const TOperationResult& rhs) {
        { TOperation::template Undo<TOperationResult>(lhs, rhs) } -> std::same_as<TOperationResult>;
    };

    // Optional bulk modification hook. The engine sometimes needs to merge one `delta` into a whole
    // contiguous run of accumulators in place; by default it does so with a scalar loop. An operation
    // may instead expose `ApplyToMany` and do the run however it likes -- e.g. in a few SIMD
    // instructions. The operation is told the full `block` of `count` accumulators and which prefix
    // to leave untouched (`from`), so it is free to process the whole fixed-size block branchlessly
    // with a lane mask; the engine never reveals *why* the run is shaped this way.
    //
    // Contract: ApplyToMany(block, count, from, delta) must leave it observably equal to
    //   for (i = from; i < count; ++i) block[i] = Merge(block[i], delta);
    // and may read (but must not modify) block[0 .. from), which is always valid memory.
    template<typename TOperation, typename TOperationResult>
    concept HasApplyToMany = requires(TOperationResult* block, size_t count, size_t from, const TOperationResult& delta) {
        { TOperation::template ApplyToMany<TOperationResult>(block, count, from, delta) };
    };

    template<class TOperationResult>
    class IQueryableEngine {
    public:
        virtual void Init(const size_t length) = 0;
        virtual TOperationResult ComputeRange(const size_t left, const size_t right) const = 0;

        virtual ~IQueryableEngine() = default;
    };

    template<typename TValue, typename TOperationResult, typename TOperation>
    requires ValidOperation<TOperation, TValue, TOperationResult>
    class IRangeUpdateEngine: public IQueryableEngine<TOperationResult> {
    public:
        virtual void SetRange(const size_t left, const size_t right, const TValue& value) = 0;
        virtual ~IRangeUpdateEngine() = default;
    };

    template<typename TValue, typename TOperationResult, typename TOperation>
    requires ValidOperation<TOperation, TValue, TOperationResult>
    class IPointUpdateEngine: public IQueryableEngine<TOperationResult> {
    public:
        virtual void SetPoint(const size_t position, const TValue& value) = 0;
        virtual ~IPointUpdateEngine() = default;
    };
} // namespace NProust
