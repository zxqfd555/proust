#include <cassert>
#include <chrono>
#include <iostream>
#include <fstream>
#include <random>
#include <string_view>
#include <span>

#include "common/dataset.h"
#include "engine/segtree/segtree.h"
#include "engine/operations.h"
#include "serialization/dataset.h"

int main(int argc, char* argv[]) {
    std::span<char*> args(argv, argc);

    std::string outputFn;
    std::string datasetType;
    size_t length;
    size_t queryCount;

    for (auto it = args.begin() + 1; it != args.end(); ++it) {
        std::string_view arg = *it;

        if (arg == "--output-file") {
            if (++it == args.end()) {
                std::cerr << "--output-file requires a value" << std::endl;
                return 1;
            }
            outputFn = *it;
        }
        else if (arg == "--dataset-type") {
            if (++it == args.end()) {
                std::cerr << "--dataset-type requires a value" << std::endl;
                return 1;
            }
            datasetType = *it;
        }
        else if (arg == "--length") {
            if (++it == args.end()) {
                std::cerr << "--length requires a value" << std::endl;
                return 1;
            }
            length = std::stoull(*it);
        }
        else if (arg == "--query-count") {
            if (++it == args.end()) {
                std::cerr << "--query-count requires a value" << std::endl;
                return 1;
            }
            queryCount = std::stoull(*it);
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return 1;
        }
    }

    if (datasetType == "sum_i32") {
        NProust::GenerateTestPlanIntoFile<int32_t, int32_t, NProust::TSumOperation>(length, queryCount, NProust::CreateRandomIntGenerator(-100, 100), outputFn);
    } else {
        std::cerr << "Unknown dataset type: " << datasetType << std::endl;
        return 1;
    }

    return 0;
}
