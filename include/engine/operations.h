namespace NProust {
    struct TSumOperation {
        template<typename TValue, typename TOperationResult>
        static TOperationResult Repeat(const TValue& v, const size_t times) {
            return TOperationResult(v * times);
        }

        template<typename TOperationResult>
        static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) {
            return lhs + rhs;
        }
    };

    struct TBitwiseAndOperation {
        template<typename TValue, typename TOperationResult>
        static TOperationResult Repeat(const TValue& v, const size_t times) {
            return TOperationResult(v);
        }

        template<typename TOperationResult>
        static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) {
            return lhs & rhs;
        }
    };

    struct TBitwiseOrOperation {
        template<typename TValue, typename TOperationResult>
        static TOperationResult Repeat(const TValue& v, const size_t times) {
            return TOperationResult(v);
        }

        template<typename TOperationResult>
        static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) {
            return lhs | rhs;
        }
    };

    struct TBitwiseXorOperation {
        template<typename TValue, typename TOperationResult>
        static TOperationResult Repeat(const TValue& v, const size_t times) {
            return times % 2 ? TOperationResult(v) : TOperationResult(0);
        }

        template<typename TOperationResult>
        static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) {
            return lhs ^ rhs;
        }
    };

    struct TMaxOperation {
        template<typename TValue, typename TOperationResult>
        static TOperationResult Repeat(const TValue& v, const size_t times) {
            return TOperationResult(v);
        }

        template<typename TOperationResult>
        static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) {
            return std::max(lhs, rhs);
        }
    };

    struct TMinOperation {
        template<typename TValue, typename TOperationResult>
        static TOperationResult Repeat(const TValue& v, const size_t times) {
            return TOperationResult(v);
        }

        template<typename TOperationResult>
        static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) {
            return std::min(lhs, rhs);
        }
    };
} // namespace NProust