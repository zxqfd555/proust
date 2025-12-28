#pragma once

#include <random>

namespace NProust {
    enum class EEngineOperation { 
        SetRange,
        Query,
    };

    struct TTestPlanBase {
        EEngineOperation Operation;
        size_t Left;
        size_t Right;
    };

    class ITestPlanBaseGenerator {
    public:
        ITestPlanBaseGenerator(const size_t length, const double pModification = 0.5)
            : Length(length)
            , Rng(std::random_device{}())
            , IsModification(pModification)
            , RangeStartDist(0, length - 1)
            , RangeEndDist(0, length)
        {
        }

        virtual TTestPlanBase Next() = 0;
        virtual ~ITestPlanBaseGenerator() = default;

    protected:
        const size_t Length;
        std::mt19937 Rng;
        std::bernoulli_distribution IsModification;
        std::uniform_int_distribution<int32_t> RangeStartDist;
        std::uniform_int_distribution<int32_t> RangeEndDist;

        EEngineOperation NextOperation() {
            bool isModification = IsModification(Rng);
            return isModification ? EEngineOperation::SetRange : EEngineOperation::Query;
        }

        TTestPlanBase NextQuery() {
            auto range = NextRange();
            if (range.first == range.second) {
                ++range.second;
            }
            return TTestPlanBase { EEngineOperation::Query, range.first, range.second };
        }

        std::pair<size_t, size_t> NextRange() {
            auto left = RangeStartDist(Rng);
            auto right = RangeEndDist(Rng);
            if (left > right) {
                std::swap(left, right);
            }
            return std::make_pair(left, right);
        }
    };

    class TRangeUpdatePlanGenerator : public ITestPlanBaseGenerator {
    private:
        using TBase = ITestPlanBaseGenerator;

    public:
        using TBase::TBase;

        virtual TTestPlanBase Next() override {
            const auto operation = NextOperation();
            switch (operation) {
                case EEngineOperation::SetRange: {
                    const auto range = NextRange();
                    return TTestPlanBase { EEngineOperation::SetRange, range.first, range.second };
                }
                case EEngineOperation::Query: {
                    return NextQuery();
                }
            }
            #if defined(_MSC_VER)
                __assume(0);
            #else
                __builtin_unreachable();
            #endif
        }

        virtual ~TRangeUpdatePlanGenerator() = default;
    };

    class TPointUpdatePlanGenerator: public ITestPlanBaseGenerator {
    private:
        using TBase = ITestPlanBaseGenerator;

    public:
        using TBase::TBase;

        virtual TTestPlanBase Next() override {
            const auto operation = NextOperation();
            switch (operation) {
                case EEngineOperation::SetRange: {
                    const auto point = RangeStartDist(Rng);
                    // For the point modification, the right part doesn't make sense.
                    // `point + 1` is, however, a convenient value, allowing us to run
                    // the range update engine versions without changes.
                    return TTestPlanBase { EEngineOperation::SetRange, (size_t)point, (size_t)point + 1 };
                }
                case EEngineOperation::Query: {
                    return NextQuery();
                }
            }
        }

        virtual ~TPointUpdatePlanGenerator() = default;
    };

    template<typename TValue, typename TOperationResult>
    struct TTestPlanItem: public TTestPlanBase {
    private:
        using TBase = TTestPlanBase;

    public:
        std::optional<TValue> Value;
        std::optional<TOperationResult> ExpectedResult;

        TTestPlanItem() = default;

        TTestPlanItem(const TTestPlanBase& base, const std::optional<TValue> value, const std::optional<TOperationResult> expectedResult)
            : TBase(base)
            , Value(value)
            , ExpectedResult(expectedResult)
        {
        }

        bool operator==(const TTestPlanItem& other) const {
            return Operation == other.Operation && Left == other.Left && Right == other.Right && Value == other.Value && ExpectedResult == other.ExpectedResult;
        }
    };
} // namespace NProust
