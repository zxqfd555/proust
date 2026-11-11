#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <span>

// These operation primitives are one or two machine instructions each; they exist to be inlined
// into the engine's hot loops. At -O3 GCC inlines the integer instantiations but, for reasons of
// its cost model, leaves the floating-point ones out of line -- emitting a real `call` per element
// in the point engine's update loop and the leaf reductions. Forcing the inline removes that call
// overhead and lets the surrounding loop keep the value in a register. Integers are unaffected
// (already inlined); the attribute simply makes the decision uniform across value types.
#if defined(__GNUC__) || defined(__clang__)
    #define PROUST_FORCE_INLINE [[gnu::always_inline]] inline
#else
    #define PROUST_FORCE_INLINE inline
#endif

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

    // Optional hook of the wide engine: the merge of exactly 16 consecutive results, in order. That is
    // the bulk of the work of an assignment there, and a fixed shape can be done in SIMD lanes; with no
    // hook it is a chain of Merge. `block` is aligned to 64 bytes if the type allows.
    //
    // Contract: MergeBlock(block) must equal Merge(... Merge(block[0], block[1]) ..., block[15]).
    template<typename TOperation, typename TOperationResult>
    concept HasMergeBlock = requires(const TOperationResult* block) {
        { TOperation::template MergeBlock<TOperationResult>(block) } -> std::same_as<TOperationResult>;
    };

    // Optional hook of the wide engine: the merge of the elements [from, to) of a 16-element block, for
    // from < to; `TElement` is what the block holds, TOperationResult or TValue. A chain of Merge has a
    // loop of data-dependent length, whose exit mispredicts on nearly every call, and such a flush also
    // throws away the work in flight; an operation with a known neutral element can process the whole
    // block in SIMD lanes under a mask instead, with no branch at all.
    //
    // Contract: FoldInBlock(block, from, to) must equal the chain of Merge over FromRepeated(block[i], 1)
    // if TElement is TValue, or over block[i] itself if it is TOperationResult. All 16 elements may be read.
    template<typename TOperation, typename TOperationResult, typename TElement>
    concept HasFoldInBlock = requires(const TElement* block, size_t from, size_t to) {
        { TOperation::template FoldInBlock<TOperationResult>(block, from, to) } -> std::same_as<TOperationResult>;
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

        // Init(values.size()) followed by assigning every element, but engines override it with an
        // O(N) bottom-up build. The default is the O(N log N) fallback for those that do not.
        virtual void InitFrom(std::span<const TValue> values) {
            this->Init(values.size());
            for (size_t index = 0; index < values.size(); ++index) {
                SetRange(index, index + 1, values[index]);
            }
        }

        virtual ~IRangeUpdateEngine() = default;
    };

    template<typename TValue, typename TOperationResult, typename TOperation>
    requires ValidOperation<TOperation, TValue, TOperationResult>
    class IPointUpdateEngine: public IQueryableEngine<TOperationResult> {
    public:
        virtual void SetPoint(const size_t position, const TValue& value) = 0;

        // Same contract as IRangeUpdateEngine::InitFrom.
        virtual void InitFrom(std::span<const TValue> values) {
            this->Init(values.size());
            for (size_t index = 0; index < values.size(); ++index) {
                SetPoint(index, values[index]);
            }
        }

        virtual ~IPointUpdateEngine() = default;
    };
} // namespace NProust
