// Building an engine from an array: InitFrom against Init followed by one modification per element,
// for the operations and the engines of the runner's datasets. Both structures are then checked
// against each other on random queries, so a fast build that is wrong does not go unnoticed.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include "engine/interface.h"
#include "engine/operations.h"
#include "engine/robust/engine.h"
#include "examples/leetcode_maximum_subarray.h"
#include "examples/leetcode_third_distinct_maximum.h"

using namespace NProust;

namespace {
    constexpr size_t LENGTH = 1'000'000;
    constexpr size_t CHECKED_QUERIES = 100'000;
    constexpr int PASSES = 5;

    std::vector<int32_t> MakeValues() {
        std::mt19937 generator(42);
        std::uniform_int_distribution<int32_t> distribution(-1'000'000, 1'000'000);
        std::vector<int32_t> values(LENGTH);
        for (int32_t& value : values) {
            value = distribution(generator);
        }
        return values;
    }

    template<class TEngine>
    void AssignOne(TEngine& engine, const size_t index, const int32_t value) {
        if constexpr (requires { engine.SetPoint(index, value); }) {
            engine.SetPoint(index, value);
        } else {
            engine.SetRange(index, index + 1, value);
        }
    }

    // Nanoseconds per element of the fastest pass; each pass builds a new engine.
    template<class TBuild>
    double BestNsPerElement(TBuild&& build) {
        double best = 1e18;
        for (int pass = 0; pass < PASSES; ++pass) {
            const auto started = std::chrono::steady_clock::now();
            build();
            const std::chrono::duration<double, std::nano> elapsed = std::chrono::steady_clock::now() - started;
            best = std::min(best, elapsed.count() / LENGTH);
        }
        return best;
    }

    template<class TEngine>
    void Measure(const char* name, const std::vector<int32_t>& values) {
        const double fromArray = BestNsPerElement([&] {
            auto engine = std::make_unique<TEngine>();
            engine->InitFrom(values);
        });
        const double oneByOne = BestNsPerElement([&] {
            auto engine = std::make_unique<TEngine>();
            engine->Init(LENGTH);
            for (size_t index = 0; index < LENGTH; ++index) {
                AssignOne(*engine, index, values[index]);
            }
        });

        auto built = std::make_unique<TEngine>();
        built->InitFrom(values);
        auto assigned = std::make_unique<TEngine>();
        assigned->Init(LENGTH);
        for (size_t index = 0; index < LENGTH; ++index) {
            AssignOne(*assigned, index, values[index]);
        }
        std::mt19937_64 generator(7);
        size_t mismatches = 0;
        for (size_t query = 0; query < CHECKED_QUERIES; ++query) {
            size_t left = generator() % LENGTH;
            size_t right = generator() % LENGTH;
            if (left > right) {
                std::swap(left, right);
            }
            mismatches += !(built->ComputeRange(left, right + 1) == assigned->ComputeRange(left, right + 1));
        }
        std::printf("%-22s | %9.2f | %9.2f | %6.1fx | %zu mismatches\n", name, fromArray, oneByOne, oneByOne / fromArray, mismatches);
    }

    using TMaxSum = NProustExamples::TMaxSumState;
    using TMaxSumOp = NProustExamples::TMaxSumStateOperation;
    using TTop16 = NProustExamples::TTopKState<16>;
    using TTop16Op = NProustExamples::TInverseOrderStatisticOperation<16>;
    using TSumOp = TSumOperation;
    using TMaxOp = TMaxOperation;

    template<class TValue, class TResult, class TOperation>
    using TRangeEngine = TRobustRangeUpdateEngine<TValue, TResult, TOperation, TBinaryHeapNodeTopology<4>, TSoALayout<TValue, TResult>>;
} // namespace

int main() {
    const std::vector<int32_t> values = MakeValues();
    std::printf("N=%zu, best of %d passes. ns per element.\n", LENGTH, PASSES);
    std::printf("%-22s | %9s | %9s | %7s |\n", "engine", "InitFrom", "one by one", "ratio");
    Measure<TRangeEngine<int32_t, int64_t, TSumOp>>("range sum", values);
    Measure<TRangeEngine<int32_t, int32_t, TMaxOp>>("range max", values);
    Measure<TRangeEngine<int32_t, TMaxSum, TMaxSumOp>>("range max-subarray", values);
    Measure<TRangeEngine<int32_t, TTop16, TTop16Op>>("range order-statistic", values);
    Measure<TFatPointUpdateEngine<int32_t, int64_t, TSumOp, 4>>("point sum", values);
    Measure<TRobustPointUpdateEngine<int32_t, int32_t, TMaxOp, 4>>("point max", values);
    Measure<TRobustPointUpdateEngine<int32_t, TMaxSum, TMaxSumOp, 4>>("point max-subarray", values);
    Measure<TRobustPointUpdateEngine<int32_t, TTop16, TTop16Op, 4>>("point order-statistic", values);
    return 0;
}
