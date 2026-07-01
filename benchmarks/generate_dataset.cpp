#include <chrono>
#include <iostream>
#include <fstream>
#include <random>
#include <string_view>
#include <span>

#include "common/dataset.h"
#include "engine/operations.h"
#include "serialization/dataset.h"

#include "args_parsing.h"

int main(int argc, char* argv[]) {
    std::span<char*> args(argv, argc);

    std::optional<std::string> outputFn;
    std::optional<NProust::EDatasetType> datasetType;
    std::optional<size_t> length;
    std::optional<size_t> queryCount;
    std::optional<NProust::EBaseGeneratorType> baseGeneratorType;
    std::optional<size_t> maxRangeWidth;
    std::optional<double> pModification;

    for (auto it = args.begin() + 1; it != args.end(); ++it) {
        std::string_view arg = *it;
        if (arg == DATASET_FILE_PARAM) {
            ParseStringArgument(it, args.end(), outputFn);
        } else if (arg == DATASET_TYPE_PARAM) {
            ParseDatasetTypeArgument(it, args.end(), datasetType);
        } else if (arg == LENGTH_PARAM) {
            ParseIntegralArgument(it, args.end(), length);
        } else if (arg == QUERY_COUNT_PARAM) {
            ParseIntegralArgument(it, args.end(), queryCount);
        } else if (arg == GENERATOR_TYPE_PARAM) {
            ParseBaseGeneratorTypeArgument(it, args.end(), baseGeneratorType);
        } else if (arg == MAX_RANGE_WIDTH_PARAM) {
            ParseIntegralArgument(it, args.end(), maxRangeWidth);
        } else if (arg == P_MODIFICATION_PARAM) {
            std::string_view argName = *it;
            if (++it == args.end()) {
                spdlog::critical("{} requires a value", argName);
                std::terminate();
            }
            pModification = std::stod(std::string(*it));
        } else {
            spdlog::critical("Unknown argument: {}", arg);
            std::terminate();
        }
    }

    CheckFieldPresence(DATASET_TYPE_PARAM, datasetType);
    CheckFieldPresence(DATASET_FILE_PARAM, outputFn);
    CheckFieldPresence(LENGTH_PARAM, length);
    CheckFieldPresence(QUERY_COUNT_PARAM, queryCount);
    CheckFieldPresence(GENERATOR_TYPE_PARAM, baseGeneratorType);

    spdlog::info("Proceeding to the dataset generation. Target file: {}. Structure capacity: {}. Query count: {}.", *outputFn, *length, *queryCount);

    const size_t maxWidth = maxRangeWidth.value_or(0);
    const double pMod = pModification.value_or(0.5);
    std::unique_ptr<NProust::ITestPlanBaseGenerator> baseGenerator;
    switch (*baseGeneratorType) {
        case (NProust::EBaseGeneratorType::Point): {
            baseGenerator = std::make_unique<NProust::TPointUpdatePlanGenerator>(*length, pMod, maxWidth);
            break;
        }
        case (NProust::EBaseGeneratorType::Range): {
            baseGenerator = std::make_unique<NProust::TRangeUpdatePlanGenerator>(*length, pMod, maxWidth);
            break;
        }
    }
    NProust::GenerateDataset(*datasetType, *length, *queryCount, *outputFn, *baseGenerator);

    return 0;
}
