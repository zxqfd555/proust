#pragma once
#include "../interface.h"

namespace NProust {
    template<typename TValue>
    class TSegmentTreeEngine: public IEngine<TValue> {
    public:
        virtual TVersion Init(const size_t length) override {
            return 0;  // TODO
        }

        virtual TVersion Modify(const size_t left, const size_t right, const TValue value) override {
            return 0;  // TODO
        }

        virtual TValue Query(const size_t left, const size_t right) override {
            return TValue();  // TODO
        }
    };

    template<typename TValue>
    std::unique_ptr<IEngine<TValue>> CreateSegmentTreeEngine() {
        return std::make_unique<TSegmentTreeEngine<TValue>>();
    }
} // namespace NProust
