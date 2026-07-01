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
        if (s == "SumDouble") {
            return EDatasetType::SumDouble;
        }
        if (s == "MaxDouble") {
            return EDatasetType::MaxDouble;
        }
        if (s == "XorInt32") {
            return EDatasetType::XorInt32;
        }
        if (s == "GcdInt32") {
            return EDatasetType::GcdInt32;
        }
        if (s == "AndInt32") {
            return EDatasetType::AndInt32;
        }
        if (s == "OrInt32") {
            return EDatasetType::OrInt32;
        }
        if (s == "SumInt16") {
            return EDatasetType::SumInt16;
        }
        if (s == "SumInt64") {
            return EDatasetType::SumInt64;
        }
        if (s == "XorInt16") {
            return EDatasetType::XorInt16;
        }
        if (s == "XorInt64") {
            return EDatasetType::XorInt64;
        }
        return std::nullopt;
    }

    std::optional<EBaseGeneratorType> BaseGeneratorTypeFromString(const std::string_view s) {
        if (s == "Point") {
            return EBaseGeneratorType::Point;
        }
        if (s == "Range") {
            return EBaseGeneratorType::Range;
        }
        return std::nullopt;
    }

    void GenerateDataset(const EDatasetType type, const size_t length, const size_t queryCount, const std::string fname, ITestPlanBaseGenerator& baseGenerator) {
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
                GenerateTestPlanIntoFile<int32_t, NProustExamples::TTopKState<16>, NProustExamples::TInverseOrderStatisticOperation<16>>(
                    length,
                    queryCount,
                    baseGenerator,
                    RegularTaskElementGenerator<int32_t>(),
                    fname,
                    NProustExamples::CreateNaiveInverseOrderStatisticsEngine<16>()
                );
                break;
            }
            case EDatasetType::SumDouble: {
                // Integer-valued doubles bounded so every partial sum stays exact (see
                // CreateExactDoubleSumGenerator): keeps the bit-exact ground-truth check valid
                // despite floating-point non-associativity.
                GenerateTestPlanIntoFile<double, double, NProust::TSumOperation>(
                    length,
                    queryCount,
                    baseGenerator,
                    CreateExactDoubleSumGenerator(-1000, 1000),
                    fname
                );
                break;
            }
            case EDatasetType::MaxDouble: {
                GenerateTestPlanIntoFile<double, double, NProust::TMaxOperation>(
                    length,
                    queryCount,
                    baseGenerator,
                    CreateRandomDoubleGenerator(-1.0e6, 1.0e6),
                    fname
                );
                break;
            }
            case EDatasetType::XorInt32: {
                GenerateTestPlanIntoFile<int32_t, int32_t, NProust::TBitwiseXorOperation>(
                    length,
                    queryCount,
                    baseGenerator,
                    RegularTaskElementGenerator<int32_t>(),
                    fname
                );
                break;
            }
            case EDatasetType::GcdInt32: {
                // Positive values so that a single element's result (itself) matches the non-negative
                // convention of std::gcd over a multi-element range.
                GenerateTestPlanIntoFile<int32_t, int32_t, NProust::TGcdOperation>(
                    length,
                    queryCount,
                    baseGenerator,
                    CreateRandomIntGenerator<int32_t>(1, 1'000'000'000),
                    fname
                );
                break;
            }
            case EDatasetType::AndInt32: {
                GenerateTestPlanIntoFile<int32_t, int32_t, NProust::TBitwiseAndOperation>(
                    length,
                    queryCount,
                    baseGenerator,
                    RegularTaskElementGenerator<int32_t>(),
                    fname
                );
                break;
            }
            case EDatasetType::OrInt32: {
                GenerateTestPlanIntoFile<int32_t, int32_t, NProust::TBitwiseOrOperation>(
                    length,
                    queryCount,
                    baseGenerator,
                    RegularTaskElementGenerator<int32_t>(),
                    fname
                );
                break;
            }
            case EDatasetType::SumInt16: {
                // Full-range int16: sums wrap mod 2^16 in both the engine and the naive ground truth.
                GenerateTestPlanIntoFile<int16_t, int16_t, NProust::TSumOperation>(
                    length, queryCount, baseGenerator, RegularTaskElementGenerator<int16_t>(), fname);
                break;
            }
            case EDatasetType::SumInt64: {
                GenerateTestPlanIntoFile<int64_t, int64_t, NProust::TSumOperation>(
                    length, queryCount, baseGenerator, SummationTaskElementGenerator<int64_t>(length), fname);
                break;
            }
            case EDatasetType::XorInt16: {
                GenerateTestPlanIntoFile<int16_t, int16_t, NProust::TBitwiseXorOperation>(
                    length, queryCount, baseGenerator, RegularTaskElementGenerator<int16_t>(), fname);
                break;
            }
            case EDatasetType::XorInt64: {
                GenerateTestPlanIntoFile<int64_t, int64_t, NProust::TBitwiseXorOperation>(
                    length, queryCount, baseGenerator, RegularTaskElementGenerator<int64_t>(), fname);
                break;
            }
        }
    }
}
