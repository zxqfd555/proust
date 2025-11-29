#include <chrono>
#include <iostream>
#include <random>

#include "engine/naive/naive.h"
#include "engine/segtree/segtree.h"
#include "engine/operations.h"

enum class EEngineOperation { 
    SetRange,
    Query,
};

template<typename TValue, typename TOperationResult>
struct TTestPlanItem {
    EEngineOperation Operation;
    size_t Left;
    size_t Right;
    std::optional<TValue> Value;
    std::optional<TOperationResult> ExpectedResult;

    TTestPlanItem() = default;

    TTestPlanItem(const EEngineOperation operation, const size_t left, const size_t right, const std::optional<TValue> value, const std::optional<TOperationResult> expectedResult)
        : Operation(operation)
        , Left(left)
        , Right(right)
        , Value(value)
        , ExpectedResult(expectedResult)
    {
    }
};

template<typename TValue, typename TOperationResult, typename TOperation>
void StressTest(const size_t length, const size_t queryCount, std::function<TValue(std::mt19937&)> valueGenerator) {
    const auto testStartedAt = std::chrono::high_resolution_clock::now();

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> opDist(0, 1);
    std::uniform_int_distribution<int> leftDist(0, length - 1);
    std::uniform_int_distribution<int> rightDist(0, length);

    auto naiveEngine = NProust::CreateNaiveEngine<TValue, TOperationResult, TOperation>();

    naiveEngine->Init(length);
    std::vector<TTestPlanItem<TValue, TOperationResult>> testPlan;
    testPlan.reserve(queryCount);
    for (size_t querySeqIdx = 0; querySeqIdx < queryCount; ++querySeqIdx) {
        size_t left = leftDist(rng);
        size_t right = rightDist(rng);
        if (left > right) {
            std::swap(left, right);
        }

        EEngineOperation operation = static_cast<EEngineOperation>(opDist(rng));
        switch (operation) {
            case EEngineOperation::SetRange: {
                const auto value = valueGenerator(rng);
                naiveEngine->SetRange(left, right, value);
                testPlan.emplace_back(operation, left, right, value, std::nullopt);
                break;
            }
            case EEngineOperation::Query: {
                if (left == right) {
                    right += 1;
                }
                const auto expectedResult = naiveEngine->ComputeRange(left, right);
                testPlan.emplace_back(operation, left, right, std::nullopt, expectedResult);
                break;
            }
        }
    }
    auto generationEndedAt = std::chrono::high_resolution_clock::now();
    auto generationDuration = std::chrono::duration_cast<std::chrono::milliseconds>(generationEndedAt - testStartedAt);
    std::cerr << "The test plan has been generated. Time elapsed: " << generationDuration << std::endl;

    auto robustEngine = NProust::CreateRobustEngine<TValue, TOperationResult, TOperation>();
    robustEngine->Init(length);
    for (const auto& planItem : testPlan) {
        switch (planItem.Operation) {
            case EEngineOperation::SetRange: {
                assert(planItem.Value.has_value());
                robustEngine->SetRange(planItem.Left, planItem.Right, *planItem.Value);
                break;
            }
            case EEngineOperation::Query: {
                const auto result = robustEngine->ComputeRange(planItem.Left, planItem.Right);
                assert(result == *planItem.ExpectedResult);
                break;
            }
        }
    }
    auto testEndedAt = std::chrono::high_resolution_clock::now();
    auto testDuration = std::chrono::duration_cast<std::chrono::milliseconds>(testEndedAt - generationEndedAt);

    std::cerr << "Stress test passed. Time elapsed: " << testDuration << std::endl;
}

int main() {
    StressTest<int64_t, int64_t, NProust::TSumOperation>(
        100000,
        1000000,
        [](std::mt19937& generator) {
            static std::uniform_int_distribution<int64_t> valueDistribution(-100, 100);
            auto value = valueDistribution(generator);
            return value;
        }
    );
    return 0;
}
