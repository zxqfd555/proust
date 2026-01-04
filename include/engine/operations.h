#pragma once

namespace NProust {
    struct TSumOperation {
        template<typename TValue, typename TOperationResult>
        inline static TOperationResult Repeat(const TValue& v, const size_t times) noexcept(noexcept(TOperationResult(v * times))) {
            return TOperationResult(v * times);
        }

        template<typename TOperationResult>
        inline static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) noexcept(noexcept(lhs + rhs)) {
            return lhs + rhs;
        }

        template<typename TOperationResult>
        inline static TOperationResult Undo(const TOperationResult& lhs, const TOperationResult& rhs) noexcept(noexcept(lhs - rhs)) {
            return lhs - rhs;
        }

        template<typename TOperationResult>
        inline static TOperationResult MergeMany(const TOperationResult* data, size_t fromIndex, size_t toIndex) {
            TOperationResult result = 0;
            for (size_t index = fromIndex; index < toIndex; ++index) {
                result += data[index];
            }
            return result;
        }
    };

    struct TBitwiseAndOperation {
        template<typename TValue, typename TOperationResult>
        static TOperationResult Repeat(const TValue& v, const size_t times) noexcept(noexcept(TOperationResult(v))) {
            return TOperationResult(v);
        }

        template<typename TOperationResult>
        static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) noexcept(noexcept(lhs & rhs)) {
            return lhs & rhs;
        }
    };

    struct TBitwiseOrOperation {
        template<typename TValue, typename TOperationResult>
        static TOperationResult Repeat(const TValue& v, const size_t times) noexcept(noexcept(TOperationResult(v))) {
            return TOperationResult(v);
        }

        template<typename TOperationResult>
        static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) noexcept(noexcept(lhs | rhs)) {
            return lhs | rhs;
        }
    };

    struct TBitwiseXorOperation {
        template<typename TValue, typename TOperationResult>
        inline static TOperationResult Repeat(const TValue& v, const size_t times) noexcept(noexcept(TOperationResult(v)) && noexcept(TOperationResult(0))) {
            return (times & 1) ? TOperationResult(v) : TOperationResult(0);
        }

        template<typename TOperationResult>
        inline static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) noexcept(noexcept(lhs ^ rhs)) {
            return lhs ^ rhs;
        }

        template<typename TOperationResult>
        inline static TOperationResult Undo(const TOperationResult& lhs, const TOperationResult& rhs) noexcept(noexcept(lhs ^ rhs)) {
            return lhs ^ rhs;
        }
    };

    struct TMaxOperation {
        template<typename TValue, typename TOperationResult>
        static TOperationResult Repeat(const TValue& v, const size_t times) noexcept(noexcept(TOperationResult(v))) {
            return TOperationResult(v);
        }

        template<typename TOperationResult>
        static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) noexcept(noexcept(lhs > rhs)) {
            return lhs > rhs ? lhs : rhs;
        }
    };

    struct TMinOperation {
        template<typename TValue, typename TOperationResult>
        static TOperationResult Repeat(const TValue& v, const size_t times) noexcept(noexcept(TOperationResult(v))) {
            return TOperationResult(v);
        }

        template<typename TOperationResult>
        static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) noexcept(noexcept(lhs < rhs)) {
            return lhs < rhs ? lhs : rhs;
        }
    };
} // namespace NProust