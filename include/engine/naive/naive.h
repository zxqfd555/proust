#pragma once

#include <cassert>
#include <vector>

#include "../interface.h"

namespace NProust {
    template<typename TValue, typename TOperationResult, typename TOperation>
    class TNaiveEngine: public IEngine<TValue, TOperationResult, TOperation> {
        using TBase = IEngine<TValue, TOperationResult, TOperation>;

    public:
        TNaiveEngine() = default;

        virtual TVersion Init(const size_t length) override {
            Elements = std::vector<TOperationResult>(length, TOperationResult());
            CurrentVersion += 1;
            return CurrentVersion;
        }

        virtual TVersion SetRange(const size_t rangeStart, const size_t rangeEnd, const TValue& value) override {
            assert(rangeStart <= rangeEnd && "The range start must be smaller than its end");
            assert(rangeEnd <= Elements.size() && "The range goes beyond the allocated capacity");
            const auto element = TOperation::template Repeat<TValue, TOperationResult>(value, 1);
            std::fill(Elements.begin() + rangeStart, Elements.begin() + rangeEnd, element);
            CurrentVersion += 1;
            return CurrentVersion;
        }

        virtual TOperationResult ComputeRange(const size_t rangeStart, const size_t rangeEnd) override {
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
        TVersion CurrentVersion{0};
        std::vector<TOperationResult> Elements;
    };

    template<typename TValue, typename TOperationResult, typename TOperation>
    std::unique_ptr<IEngine<TValue, TOperationResult, TOperation>> CreateNaiveEngine() {
        return std::make_unique<TNaiveEngine<TValue, TOperationResult, TOperation>>();
    }

} // namespace NProust
