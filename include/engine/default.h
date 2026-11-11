#pragma once

#include <cstdint>
#include <memory>
#include <type_traits>

#include "interface.h"
#include "robust/engine.h"
#include "wide/engine.h"

namespace NProust {
    // Which range-assignment engine to use when nothing is known about the task but its types.
    //
    // A wide node refolds 16 results where a binary one merges two, which pays off when a merge is cheap
    // and does not when it is not. The types cannot tell a merge's cost, but a result of up to 8 bytes
    // has little room to be expensive: there the wide engine measured 32-38% faster than the best
    // binary configuration on every operation tried. At 16 bytes it already depends on the operation
    // (18% faster with a plain sum and max, 35% slower with the maximum subarray), so anything larger
    // stays binary -- unless the operation folds a block in SIMD (HasMergeBlock), which changes the
    // picture: the maximum subarray goes from 35% slower to 21% faster. A task that knows better just
    // names the engine it wants.
    template<typename TValue, typename TOperationResult, typename TOperation>
    inline constexpr bool PREFERS_WIDE_RANGE_ENGINE =
        HasMergeBlock<TOperation, TOperationResult> || (
            sizeof(TOperationResult) <= sizeof(uint64_t) && std::is_trivially_copyable_v<TOperationResult> &&
            sizeof(TValue) <= sizeof(uint64_t) && std::is_trivially_copyable_v<TValue>
        );

    template<typename TValue, typename TOperationResult, typename TOperation>
    requires ValidOperation<TOperation, TValue, TOperationResult>
    using TDefaultRangeUpdateEngine = std::conditional_t<
        PREFERS_WIDE_RANGE_ENGINE<TValue, TOperationResult, TOperation>,
        TWideRangeUpdateEngine<TValue, TOperationResult, TOperation>,
        TRobustRangeUpdateEngine<TValue, TOperationResult, TOperation>
    >;

    template<typename TValue, typename TOperationResult, typename TOperation>
    requires ValidOperation<TOperation, TValue, TOperationResult>
    std::unique_ptr<IRangeUpdateEngine<TValue, TOperationResult, TOperation>> CreateRangeUpdateEngine() {
        return std::make_unique<TDefaultRangeUpdateEngine<TValue, TOperationResult, TOperation>>();
    }
} // namespace NProust
