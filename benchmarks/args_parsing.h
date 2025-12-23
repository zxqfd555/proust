#pragma once

#include <chrono>
#include <string_view>
#include <span>

const std::string DATASET_FILE_PARAM = "--dataset-file";
const std::string DATASET_TYPE_PARAM = "--dataset-type";
const std::string LENGTH_PARAM = "--length";
const std::string QUERY_COUNT_PARAM = "--query-count";

template<class TValue>
void ParseStringArgument(auto& iterator, const auto& endIterator, TValue& dst) {
    std::string_view argName = *iterator;
    if (++iterator == endIterator) {
        spdlog::critical("{} requires a value", argName);
        std::terminate();
    }
    dst = *iterator;
}

template<class TValue>
void ParseIntegralArgument(auto& iterator, const auto& endIterator, TValue& dst) {
    std::string_view argName = *iterator;
    if (++iterator == endIterator) {
        spdlog::critical("{} requires a value", argName);
        std::terminate();
    }
    dst = std::stoull(*iterator);
}

void ParseDatasetTypeArgument(auto& iterator, const auto& endIterator, std::optional<NProust::EDatasetType>& datasetType) {
    std::string_view rawDatasetType;
    ParseStringArgument(iterator, endIterator, rawDatasetType);
    datasetType = NProust::DatasetTypeFromString(rawDatasetType);
    if (!datasetType) {
        spdlog::critical("Unknown dataset type: {}", *iterator);
        std::terminate();
    }
}

template<class TValue>
void CheckFieldPresence(const std::string& fieldName, const std::optional<TValue>& fieldValue) {
    if (fieldValue == std::nullopt) {
        spdlog::critical("'{}' is not specified", fieldName);
        std::terminate();
    }
}
