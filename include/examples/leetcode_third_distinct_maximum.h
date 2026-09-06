// Based on: https://leetcode.com/problems/third-maximum-number/description/
// but with O(log N) per subarray query time and O(log N) modifications as a bonus ;)

#pragma once

#include <array>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>

#include "engine/robust/engine.h"
#include "engine/operations.h"
#include "serialization/common.h"

namespace NProustExamples {
    template<size_t Capacity>
    struct TTopKState {
        uint32_t Size = 0;  // Not a size_t for tighter data packing.
        std::array<int32_t, Capacity> Values{};

        bool operator==(const TTopKState& other) const noexcept {
            if (Size != other.Size) {
                return false;
            }
            for (uint32_t index = 0; index < Size; ++index) {
                if (Values[index] != other.Values[index]) {
                    return false;
                }
            }
            return true;
        }
    };

    template<size_t StatisticsRequired = 3>
    struct TInverseOrderStatisticOperation {
        using TState = TTopKState<StatisticsRequired>;

        template<typename TValue, typename TOperationResult>
        inline static TOperationResult FromRepeated(const TValue& v, size_t /*times*/) noexcept {
            if constexpr (std::is_same_v<TValue, int32_t> && std::is_same_v<TOperationResult, TState>) {
                // Since each value is counted only once, the result contains only a single element.
                TOperationResult result;
                result.Size = 1;
                result.Values[0] = v;
                return result;
            } else {
                static_assert(sizeof(TValue) == 0, "Unsupported type for TOrderStatisticOperation::FromRepeated");
            }
        }

        template<typename TOperationResult>
        inline static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) noexcept {
            if constexpr (std::is_same_v<TOperationResult, TState>) {
                // A bounded two-way merge of two descending, distinct sequences, stopping after
                // StatisticsRequired values. No allocation, ever.
                //
                // The body is deliberately branch-free: each front is read through an "exhausted ->
                // INT32_MIN" guard, the larger front is taken with a max, and the two cursors advance
                // by comparison results (compiled to cmov/setcc, not jumps). The data-dependent
                // branches of the straightforward three-way if-chain mispredict on essentially every
                // iteration -- the comparison outcome is random -- and this merge is the single
                // hottest loop of the order-statistic benchmarks. A tie (both fronts equal) takes the
                // value once and advances both cursors, which is exactly the duplicate-dropping the
                // three-way form did. The INT32_MIN guard is only a comparison key, never stored: a
                // *real* INT32_MIN at a front is still taken, because the loop condition guarantees
                // at least one cursor is in bounds, and the cursor of an exhausted side may step past
                // its Size without harm -- the guard keeps reporting INT32_MIN for it.
                TOperationResult result;
                uint32_t leftIndex = 0;
                uint32_t rightIndex = 0;
                uint32_t resultSize = 0;

                while (resultSize < StatisticsRequired && (leftIndex < lhs.Size || rightIndex < rhs.Size)) {
                    const int32_t leftFront = leftIndex < lhs.Size ? lhs.Values[leftIndex] : std::numeric_limits<int32_t>::min();
                    const int32_t rightFront = rightIndex < rhs.Size ? rhs.Values[rightIndex] : std::numeric_limits<int32_t>::min();
                    const int32_t taken = leftFront > rightFront ? leftFront : rightFront;
                    result.Values[resultSize++] = taken;
                    leftIndex += (leftFront == taken);
                    rightIndex += (rightFront == taken);
                }

                result.Size = resultSize;
                return result;
            } else {
                static_assert(sizeof(TOperationResult) == 0, "Unsupported type for TOrderStatisticOperation::Merge");
            }
        }

        template<typename TOperationResult, typename TValue>
        inline static TOperationResult FromMany(const TValue* data, size_t fromIndex, size_t toIndex) noexcept {
            if constexpr (std::is_same_v<TValue, int32_t> && std::is_same_v<TOperationResult, TState>) {
                TOperationResult result;
                TOperationResult currentBlock;

                for (size_t index = fromIndex; index < toIndex; ++index) {
                    if (currentBlock.Size < StatisticsRequired) {
                        currentBlock.Values[currentBlock.Size++] = data[index];
                    } else {
                        std::sort(currentBlock.Values.data(), currentBlock.Values.data() + currentBlock.Size, std::greater<int32_t>());
                        currentBlock.Size = std::unique(currentBlock.Values.data(), currentBlock.Values.data() + currentBlock.Size) - currentBlock.Values.data();
                        result = Merge(result, currentBlock);

                        currentBlock.Size = 1;
                        currentBlock.Values[0] = data[index];
                    }
                }

                std::sort(currentBlock.Values.data(), currentBlock.Values.data() + currentBlock.Size, std::greater<int32_t>());
                currentBlock.Size = std::unique(currentBlock.Values.data(), currentBlock.Values.data() + currentBlock.Size) - currentBlock.Values.data();
                result = Merge(result, currentBlock);

                return result;
            } else {
                static_assert(sizeof(TOperationResult) == 0, "Unsupported type for TOrderStatisticOperation::FromMany");
            }
        }
    };

    // The task modifies one element at a time, so prefer the point engine over the range one.
    template<size_t StatisticsRequired = 3>
    inline std::unique_ptr<NProust::IPointUpdateEngine<int32_t, TTopKState<StatisticsRequired>, TInverseOrderStatisticOperation<StatisticsRequired>>> CreateInverseOrderStatisticsPointEngine() {
        return NProust::CreateRobustPointUpdateEngine<int32_t, TTopKState<StatisticsRequired>, TInverseOrderStatisticOperation<StatisticsRequired>>();
    };

    template<size_t StatisticsRequired = 3>
    inline std::unique_ptr<NProust::IRangeUpdateEngine<int32_t, TTopKState<StatisticsRequired>, TInverseOrderStatisticOperation<StatisticsRequired>>> CreateInverseOrderStatisticsEngine() {
        return NProust::CreateRobustRangeUpdateEngine<int32_t, TTopKState<StatisticsRequired>, TInverseOrderStatisticOperation<StatisticsRequired>>();
    };

    // Instantiate these directly to avoid the factories' indirect call per operation.
    template<size_t StatisticsRequired = 3>
    using TInverseOrderStatisticsPointEngine = NProust::TRobustPointUpdateEngine<int32_t, TTopKState<StatisticsRequired>, TInverseOrderStatisticOperation<StatisticsRequired>>;
    template<size_t StatisticsRequired = 3>
    using TInverseOrderStatisticsRangeEngine = NProust::TRobustRangeUpdateEngine<int32_t, TTopKState<StatisticsRequired>, TInverseOrderStatisticOperation<StatisticsRequired>>;

    // A brute force solution to the inverse order statistic problem, with modifications.
    // An implementation of IEngine, adapted to be used as a dataset generator.
    template<size_t StatisticsRequired = 3>
    class TNaiveInverseOrderStatisticsEngine: public NProust::IRangeUpdateEngine<int32_t, TTopKState<StatisticsRequired>, TInverseOrderStatisticOperation<StatisticsRequired>> {
    public:
        virtual void Init(const size_t length) override {
            Elements.resize(length);
            std::fill(Elements.begin(), Elements.end(), 0);
        }

        virtual void SetRange(const size_t left, const size_t right, const int32_t& value) override {
            std::fill(Elements.begin() + left, Elements.begin() + right, value);
        }

        virtual TTopKState<StatisticsRequired> ComputeRange(const size_t left, const size_t right) const override {
            TTopKState<StatisticsRequired> result;

            for (size_t index = left; index < right; ++index) {
                const int32_t value = Elements[index];

                // Insertion point in the descending array: number of elements strictly greater.
                uint32_t position = result.Size;
                while (position > 0 && result.Values[position - 1] < value) {
                    --position;
                }
                if (position > 0 && result.Values[position - 1] == value) {
                    continue;  // already present, the values are distinct
                }
                if (position >= StatisticsRequired) {
                    continue;  // too small to make it into the top-K
                }

                uint32_t last = std::min<uint32_t>(result.Size, StatisticsRequired - 1);
                for (uint32_t shifted = last; shifted > position; --shifted) {
                    result.Values[shifted] = result.Values[shifted - 1];
                }
                result.Values[position] = value;
                if (result.Size < StatisticsRequired) {
                    ++result.Size;
                }
            }

            return result;
        }

    private:
        std::vector<int32_t> Elements;
    };

    template<size_t StatisticsRequired = 3>
    inline std::unique_ptr<NProust::IRangeUpdateEngine<int32_t, TTopKState<StatisticsRequired>, TInverseOrderStatisticOperation<StatisticsRequired>>> CreateNaiveInverseOrderStatisticsEngine() {
        return std::make_unique<TNaiveInverseOrderStatisticsEngine<StatisticsRequired>>();
    };
} // namespace NProustExamples

namespace NProust {
    // Serialize exactly like a std::vector<int32_t> (a size_t length followed by the
    // values), so datasets written by the old vector-based state are read back without
    // a format change or regeneration. This partial specialization is more specialized
    // than the trivially-copyable primary, so it is the one selected for TTopKState.
    template<size_t Capacity>
    struct TTypeSerializer<NProustExamples::TTopKState<Capacity>> {
        static void Write(auto& writer, const NProustExamples::TTopKState<Capacity>& v) {
            size_t size = v.Size;
            writer.WriteRaw(&size, sizeof(size));
            writer.WriteRaw(v.Values.data(), sizeof(int32_t) * size);
        }

        static void Read(auto& reader, NProustExamples::TTopKState<Capacity>& v) {
            size_t size = 0;
            reader.ReadRaw(&size, sizeof(size));
            v.Size = static_cast<uint32_t>(size);
            reader.ReadRaw(v.Values.data(), sizeof(int32_t) * size);
        }
    };
} // namespace NProust
