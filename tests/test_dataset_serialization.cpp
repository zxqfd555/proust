#include <random>

#include <gtest/gtest.h>

#include "common/dataset.h"
#include "engine/robust/engine.h"
#include "engine/operations.h"
#include "serialization/dataset.h"

TEST(DatasetSerialization, Simple) {
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);

    const size_t length = 1000;
    const size_t queryCount = 2000;

    auto valueGenerator = [](std::mt19937& generator) {
        static std::uniform_int_distribution<int64_t> valueDistribution(-100, 100);
        return valueDistribution(generator);
    };

    NProust::TStreamingDatasetWriter<int64_t, int64_t> writer(queryCount, length, stream);
    NProust::TRangeUpdatePlanGenerator baseGenerator(length);
    NProust::GenerateTestPlan<int64_t, int64_t, NProust::TSumOperation>(length, queryCount, baseGenerator, valueGenerator, writer);

    stream.seekg(0);
    NProust::TStreamingDatasetReader<int64_t, int64_t> reader(stream);

    EXPECT_EQ(reader.GetLength(), length);
    size_t readCount = 0;
    while (auto maybeItem = reader.NextQuery()) {
        if (!maybeItem.has_value()) {
            break;
        }
        const auto& planItem = *maybeItem;
        EXPECT_GE(planItem.Right, planItem.Left);
        switch (planItem.Operation) {
            case NProust::EEngineOperation::SetRange: {
                EXPECT_TRUE(planItem.Value.has_value());
                EXPECT_FALSE(planItem.ExpectedResult.has_value());
                break;
            }
            case NProust::EEngineOperation::Query: {
                EXPECT_FALSE(planItem.Value.has_value());
                EXPECT_TRUE(planItem.ExpectedResult.has_value());
                break;
            }
        };
        ++readCount;
    }

    EXPECT_EQ(readCount, queryCount);
}
