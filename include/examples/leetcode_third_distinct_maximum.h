// Based on: https://leetcode.com/problems/third-maximum-number/description/
// but with O(log N) per subarray query time and O(log N) modifications as a bonus ;)

#pragma once

#include "engine/segtree/segtree.h"
#include "engine/operations.h"

namespace NProustExamples {
    using TInverseOrderStatisticState = std::vector<int32_t>;

    template<size_t StatisticsRequired = 3>
    struct TInverseOrderStatisticOperation {
        template<typename TValue, typename TOperationResult>
        static TOperationResult Repeat(const TValue& v, size_t times) noexcept {
            if constexpr (std::is_same_v<TValue, int32_t> && std::is_same_v<TOperationResult, TInverseOrderStatisticState>) {
                // Since each value is counted only once, the result contains only a single element.
                return std::vector<int32_t>(1, v);
            } else {
                static_assert(sizeof(TValue) == 0, "Unsupported type for TOrderStatisticOperation::Repeat");
            }
        }

        template<typename TOperationResult>
        static TOperationResult Merge(const TOperationResult& lhs, const TOperationResult& rhs) noexcept {
            if constexpr (std::is_same_v<TOperationResult, TInverseOrderStatisticState>) {
                std::vector<int32_t> newMaximums;
                newMaximums.reserve(StatisticsRequired);

                auto leftCandidateIt = lhs.begin();
                auto rightCandidateIt = rhs.begin();
                for (size_t index = 0; index < StatisticsRequired && (leftCandidateIt != lhs.end() || rightCandidateIt != rhs.end()); ++index) {
                    if (leftCandidateIt == lhs.end() || (rightCandidateIt < rhs.end() && *rightCandidateIt >= *leftCandidateIt)) {
                        if (leftCandidateIt < lhs.end() && *leftCandidateIt == *rightCandidateIt) {
                            // Avoiding duplicates
                            ++leftCandidateIt;
                        }
                        newMaximums.push_back(*(rightCandidateIt++));
                    } else if (rightCandidateIt == rhs.end() || (leftCandidateIt < lhs.end() && *leftCandidateIt > *rightCandidateIt)) {
                        newMaximums.push_back(*(leftCandidateIt++));
                    }
                }

                return newMaximums;
            } else {
                static_assert(sizeof(TOperationResult) == 0, "Unsupported type for TOrderStatisticOperation::Merge");
            }
        }
    };

    template<size_t StatisticsRequired = 3>
    inline std::unique_ptr<NProust::IEngine<int32_t, TInverseOrderStatisticState, TInverseOrderStatisticOperation<StatisticsRequired>>> CreateInverseOrderStatisticsEngine() {
        return NProust::CreateRobustEngine<int32_t, TInverseOrderStatisticState, TInverseOrderStatisticOperation<StatisticsRequired>>();
    };

    // A brute force solution to the inverse order statistic problem, with modifications.
    // An implementation of IEngine, adapted to be used as a dataset generator.
    template<size_t StatisticsRequired = 3>
    class TNaiveInverseOrderStatisticsEngine: public NProust::IEngine<int32_t, TInverseOrderStatisticState, TInverseOrderStatisticOperation<StatisticsRequired>> {
    public:
        virtual NProust::TVersion Init(const size_t length) override {
            Elements.resize(length);
            std::fill(Elements.begin(), Elements.end(), 0);
            return 0;  // no recovery is needed, hence ignoring
        }

        virtual NProust::TVersion SetRange(const size_t left, const size_t right, const int32_t& value) override {
            std::fill(Elements.begin() + left, Elements.begin() + right, value);
            return 0;
        }

        virtual TInverseOrderStatisticState ComputeRange(const size_t left, const size_t right) const override {
            std::vector<int32_t> maximums;
            maximums.reserve(StatisticsRequired + 1);

            for (size_t index = left; index < right; ++index) {
                maximums.push_back(Elements[index]);
                size_t maxIndex = maximums.size() - 1;
                while (maxIndex > 0 && maximums[maxIndex] > maximums[maxIndex - 1]) {
                    std::swap(maximums[maxIndex], maximums[maxIndex--]);
                }

                if (maxIndex > 0 && maximums[maxIndex] == maximums[maxIndex - 1]) {
                    for (size_t index = maxIndex; index + 1 < maximums.size(); ++index) {
                        maximums[index] = maximums[index + 1];
                    }
                    maximums.pop_back();
                }

                if (maximums.size() > StatisticsRequired) {
                    maximums.resize(StatisticsRequired);
                }
            }

            return maximums;
        }

    private:
        std::vector<int32_t> Elements;
    };

    template<size_t StatisticsRequired = 3>
    inline std::unique_ptr<NProust::IEngine<int32_t, TInverseOrderStatisticState, TInverseOrderStatisticOperation<StatisticsRequired>>> CreateNaiveInverseOrderStatisticsEngine() {
        return std::make_unique<TNaiveInverseOrderStatisticsEngine<StatisticsRequired>>();
    };
} // namespace NProustExamples
