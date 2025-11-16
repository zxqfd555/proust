#pragma once

#include <cstdint>
#include <memory>

namespace NProust {
    using TVersion = uint64_t;

    template<typename TValue>
    class IEngine {
    public:
        virtual TVersion Init(const size_t length) = 0;
        virtual TVersion Modify(const size_t left, const size_t right, const TValue value) = 0;        
        virtual TValue Query(const size_t left, const size_t right) = 0;

        virtual ~IEngine() = default;
    };
} // namespace NProust
