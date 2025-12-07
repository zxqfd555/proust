#include <chrono>
#include <iostream>
#include <fstream>
#include <random>
#include <string_view>
#include <span>

#include "engine/segtree/segtree.h"
#include "engine/operations.h"
#include "serialization/dataset.h"

template<class TValue, class TOperationResult, class TOperation>
void RunDataset(NProust::TStreamingDatasetReader<TValue, TOperationResult>& streamer) {
    const auto runStartedAt = std::chrono::high_resolution_clock::now();

    size_t readCount = 0;
    auto engine = NProust::CreateRobustEngine<TValue, TOperationResult, TOperation>();
    engine->Init(streamer.GetLength());
    while (auto maybeItem = streamer.NextQuery()) {
        if (!maybeItem.has_value()) {
            break;
        }
        const auto& planItem = *maybeItem;
        switch (planItem.Operation) {
            case NProust::EEngineOperation::SetRange: {
                engine->SetRange(planItem.Left, planItem.Right, *planItem.Value);
                break;
            }
            case NProust::EEngineOperation::Query: {
                const auto result = engine->ComputeRange(planItem.Left, planItem.Right);
                if (result != *planItem.ExpectedResult) {
                    spdlog::critical("The implementation is incorrect: the benchmark doesn't make sense.");
                    std::terminate();
                }
                break;
            }
        };
        ++readCount;
    }

    const auto runFinishedAt = std::chrono::high_resolution_clock::now();
    const auto runTimeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(runFinishedAt - runStartedAt);
    const auto durationPerQuery = std::chrono::duration_cast<std::chrono::nanoseconds>(runFinishedAt - runStartedAt).count() / readCount;
    spdlog::info("The run has finished successfully. {} queries processed in {} ms ({} ns on average per query)", readCount, runTimeElapsed.count(), durationPerQuery);
}

int main(int argc, char* argv[]) {
    std::span<char*> args(argv, argc);

    std::string datasetFn;
    std::string datasetType;

    for (auto it = args.begin() + 1; it != args.end(); ++it) {
        std::string_view arg = *it;

        if (arg == "--dataset-file") {
            if (++it == args.end()) {
                std::cerr << "--dataset-file requires a value" << std::endl;
                return 1;
            }
            datasetFn = *it;
        }
        else if (arg == "--dataset-type") {
            if (++it == args.end()) {
                std::cerr << "--dataset-type requires a value" << std::endl;
                return 1;
            }
            datasetType = *it;
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return 1;
        }
    }

    if (datasetType == "sum_i32") {
        auto datasetStreamer = NProust::TStreamingDatasetReader<int32_t, int32_t>(datasetFn);
        RunDataset<int32_t, int32_t, NProust::TSumOperation>(datasetStreamer);
    } else {
        std::cerr << "Unknown dataset type" << std::endl;
        return 1;
    }

    return 0;
}
