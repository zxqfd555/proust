#pragma once

#include <chrono>
#include <string_view>
#include <span>

#include "engine/segtree/layout.h"
#include "engine/segtree/topology.h"

const std::string DATASET_FILE_PARAM = "--dataset-file";
const std::string DATASET_TYPE_PARAM = "--dataset-type";
const std::string LENGTH_PARAM = "--length";
const std::string QUERY_COUNT_PARAM = "--query-count";
const std::string DATA_LAYOUT_PARAM = "--data-layout";
const std::string NODE_TOPOLOGY_PARAM = "--node-topology";
const std::string LEAF_SIZE_LOG_PARAM = "--leaf-size-log";

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

template<class TValue>
void CheckFieldPresence(const std::string& fieldName, const std::optional<TValue>& fieldValue) {
    if (fieldValue == std::nullopt) {
        spdlog::critical("'{}' is not specified", fieldName);
        std::terminate();
    }
}
