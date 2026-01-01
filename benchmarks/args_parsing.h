#pragma once

#include <chrono>
#include <string_view>
#include <span>

#include "engine/robust/layout.h"
#include "engine/robust/topology.h"

constexpr std::string_view DATASET_FILE_PARAM = "--dataset-file";
constexpr std::string_view DATASET_TYPE_PARAM = "--dataset-type";
constexpr std::string_view LENGTH_PARAM = "--length";
constexpr std::string_view QUERY_COUNT_PARAM = "--query-count";
constexpr std::string_view DATA_LAYOUT_PARAM = "--data-layout";
constexpr std::string_view NODE_TOPOLOGY_PARAM = "--node-topology";
constexpr std::string_view LEAF_SIZE_LOG_PARAM = "--leaf-size-log";
constexpr std::string_view WITH_SENTINEL_PARAM = "--with-sentinel";
constexpr std::string_view GENERATOR_TYPE_PARAM = "--generator-type";

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

void ParseDataLayoutArgument(auto& iterator, const auto& endIterator, std::optional<NProust::EDataLayout>& dataLayout) {
    std::string_view rawDataLayout;
    ParseStringArgument(iterator, endIterator, rawDataLayout);
    dataLayout = NProust::DataLayoutFromString(rawDataLayout);
    if (!dataLayout) {
        spdlog::critical("Unknown data layout: {}", *iterator);
        std::terminate();
    }
}

void ParseNodeTopologyArgument(auto& iterator, const auto& endIterator, std::optional<NProust::ENodeTopology>& nodeTopology) {
    std::string_view rawNodeTopology;
    ParseStringArgument(iterator, endIterator, rawNodeTopology);
    nodeTopology = NProust::NodeTopologyFromString(rawNodeTopology);
    if (!nodeTopology) {
        spdlog::critical("Unknown node topology: {}", *iterator);
        std::terminate();
    }
}

void ParseBaseGeneratorTypeArgument(auto& iterator, const auto& endIterator, std::optional<NProust::EBaseGeneratorType>& baseGeneratorType) {
    std::string_view rawBaseGeneratorType;
    ParseStringArgument(iterator, endIterator, rawBaseGeneratorType);
    baseGeneratorType = NProust::BaseGeneratorTypeFromString(rawBaseGeneratorType);
    if (!baseGeneratorType) {
        spdlog::critical("Unknown base generator type: {}", *iterator);
        std::terminate();
    }
}

template<class TValue>
void CheckFieldPresence(const std::string_view fieldName, const std::optional<TValue>& fieldValue) {
    if (fieldValue == std::nullopt) {
        spdlog::critical("'{}' is not specified", fieldName);
        std::terminate();
    }
}
