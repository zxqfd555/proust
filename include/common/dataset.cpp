#include "dataset.h"

#include "engine/operations.h"
#include "examples/leetcode_maximum_subarray.h"
#include "examples/leetcode_third_distinct_maximum.h"

namespace NProust {
    template<class TValue>
    std::function<TValue(std::mt19937&)> SummationTaskElementGenerator(const size_t length) {
        const auto minPossibleValue = std::numeric_limits<TValue>::min() / (TValue)length;
        const auto maxPossibleValue = std::numeric_limits<TValue>::max() / (TValue)length;
        return NProust::CreateRandomIntGenerator(minPossibleValue, maxPossibleValue);
    }

    template<class TValue>
    std::function<TValue(std::mt19937&)> RegularTaskElementGenerator() {
        const auto minPossibleValue = std::numeric_limits<TValue>::min();
        const auto maxPossibleValue = std::numeric_limits<TValue>::max();
        return NProust::CreateRandomIntGenerator(minPossibleValue, maxPossibleValue);
    }

    std::optional<EDatasetType> DatasetTypeFromString(const std::string_view s) {
        if (s == "SumInt32") {
            return EDatasetType::SumInt32;
        }
        if (s == "MaxInt32") {
            return EDatasetType::MaxInt32;
        }
        if (s == "MaxSumInt32") {
            return EDatasetType::MaxSumInt32;
        }
        if (s == "OrderStatistic16") {
            return EDatasetType::OrderStatistic16;
        }
        return std::nullopt;
    }

    void GenerateDataset(const EDatasetType type, const size_t length, const size_t queryCount, const std::string fname) {
        auto baseGenerator = TRangeUpdatePlanGenerator(length);

        switch (type) {
            case EDatasetType::SumInt32: {
                GenerateTestPlanIntoFile<int32_t, int32_t, NProust::TSumOperation>(
                    length,
                    queryCount,
                    baseGenerator,
                    SummationTaskElementGenerator<int32_t>(length),
                    fname
                );
                break;
            }
            case EDatasetType::MaxInt32: {
                GenerateTestPlanIntoFile<int32_t, int32_t, NProust::TMaxOperation>(
                    length,
                    queryCount,
                    baseGenerator,
                    RegularTaskElementGenerator<int32_t>(),
                    fname
                );
                break;
            }
            case EDatasetType::MaxSumInt32: {
                GenerateTestPlanIntoFile<int32_t, NProustExamples::TMaxSumState, NProustExamples::TMaxSumStateOperation>(
                    length,
                    queryCount,
                    baseGenerator,
                    SummationTaskElementGenerator<int32_t>(length),
                    fname,
                    NProustExamples::CreateNaiveMaxSumEngine()
                );
                break;
            }
            case EDatasetType::OrderStatistic16: {
                GenerateTestPlanIntoFile<int32_t, std::vector<int32_t>, NProustExamples::TInverseOrderStatisticOperation<16>>(
                    length,
                    queryCount,
                    baseGenerator,
                    RegularTaskElementGenerator<int32_t>(),
                    fname,
                    NProustExamples::CreateNaiveInverseOrderStatisticsEngine<16>()
                );
                break;
            }
        }
    }
}
