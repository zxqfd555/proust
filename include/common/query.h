#pragma once

namespace NProust {
    enum class EEngineOperation { 
        SetRange,
        Query,
    };

    template<typename TValue, typename TOperationResult>
    struct TTestPlanItem {
        EEngineOperation Operation;
        size_t Left;
        size_t Right;
        std::optional<TValue> Value;
        std::optional<TOperationResult> ExpectedResult;

        TTestPlanItem() = default;

        TTestPlanItem(const EEngineOperation operation, const size_t left, const size_t right, const std::optional<TValue> value, const std::optional<TOperationResult> expectedResult)
            : Operation(operation)
            , Left(left)
            , Right(right)
            , Value(value)
            , ExpectedResult(expectedResult)
        {
        }

        bool operator==(const TTestPlanItem& other) const {
            return Operation == other.Operation && Left == other.Left && Right == other.Right && Value == other.Value && ExpectedResult == other.ExpectedResult;
        }
    };
} // namespace NProust
