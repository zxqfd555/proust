#include <chrono>
#include <iostream>
#include <random>

#include "common/dataset.h"
#include "engine/segtree/segtree.h"
#include "engine/operations.h"

template<typename TValue, typename TOperationResult, typename TOperation>
void StressTest(const size_t length, const size_t queryCount, std::function<TValue(std::mt19937&)> valueGenerator) {
    const auto testPlan = NProust::GenerateTestPlan<TValue, TOperationResult, TOperation>(length, queryCount, valueGenerator);
    const auto testStartedAt = std::chrono::high_resolution_clock::now();

    auto robustEngine = NProust::CreateRobustEngine<TValue, TOperationResult, TOperation>();
    robustEngine->Init(length);
    for (const auto& planItem : testPlan) {
        switch (planItem.Operation) {
            case NProust::EEngineOperation::SetRange: {
                assert(planItem.Value.has_value());
                robustEngine->SetRange(planItem.Left, planItem.Right, *planItem.Value);
                break;
            }
            case NProust::EEngineOperation::Query: {
                const auto result = robustEngine->ComputeRange(planItem.Left, planItem.Right);
                assert(result == *planItem.ExpectedResult);
                break;
            }
        }
    }
    const auto testEndedAt = std::chrono::high_resolution_clock::now();
    const auto testDuration = std::chrono::duration_cast<std::chrono::milliseconds>(testEndedAt - testStartedAt);

    spdlog::info("Stress test passed. Time elapsed: {} ms", testDuration.count());
}

int main() {
    StressTest<int64_t, int64_t, NProust::TSumOperation>(
        1000,
        1000000,
        [](std::mt19937& generator) {
            static std::uniform_int_distribution<int64_t> valueDistribution(-100, 100);
            auto value = valueDistribution(generator);
            return value;
        }
    );
    return 0;
}
