#pragma once

#include <cassert>
#include <vector>

#include "../interface.h"

namespace NProust {
    template<typename TValue, typename TOperationResult, typename TOperation>
    class TNaiveRangeUpdateEngine: public IRangeUpdateEngine<TValue, TOperationResult, TOperation> {
        using TBase = IRangeUpdateEngine<TValue, TOperationResult, TOperation>;

    public:
        TNaiveRangeUpdateEngine() = default;

        virtual void Init(const size_t length) override {
            Elements = std::vector<TOperationResult>(length, TOperationResult());
            const auto baseElement = TOperation::template FromRepeated<TValue, TOperationResult>(TValue(), 1);
            std::fill(Elements.begin(), Elements.end(), baseElement);
        }

        virtual void SetRange(const size_t rangeStart, const size_t rangeEnd, const TValue& value) override {
            assert(rangeStart <= rangeEnd && "The range start must be smaller than its end");
            assert(rangeEnd <= Elements.size() && "The range goes beyond the allocated capacity");
            const auto element = TOperation::template FromRepeated<TValue, TOperationResult>(value, 1);
            std::fill(Elements.begin() + rangeStart, Elements.begin() + rangeEnd, element);
        }

        virtual TOperationResult ComputeRange(const size_t rangeStart, const size_t rangeEnd) const override {
            assert(rangeStart < rangeEnd && "The range start must be smaller than its end");
            assert(rangeStart != rangeEnd && "The operation result for an empty set is undefined");
            assert(rangeEnd <= Elements.size() && "The range goes beyond the allocated capacity");
            auto result = Elements[rangeStart];
            for (size_t i = rangeStart + 1; i < rangeEnd; ++i) {
                result = TOperation::Merge(result, Elements[i]);
            }
            return result;
        }

    private:
        std::vector<TOperationResult> Elements;
    };

    template<typename TValue, typename TOperationResult, typename TOperation>
    std::unique_ptr<IRangeUpdateEngine<TValue, TOperationResult, TOperation>> CreateNaiveRangeUpdateEngine() {
        return std::make_unique<TNaiveRangeUpdateEngine<TValue, TOperationResult, TOperation>>();
    }

} // namespace NProust
