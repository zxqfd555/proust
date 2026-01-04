#include <gtest/gtest.h>

#include "engine/robust/engine.h"
#include "engine/operations.h"

TEST(InMemorySimple, SingleElement) {
    auto engine = NProust::CreateRobustRangeUpdateEngine<int64_t, int64_t, NProust::TSumOperation>();
    engine->Init(1);
    EXPECT_EQ(engine->ComputeRange(0, 1), 0);

    engine->SetRange(0, 1, 1);
    EXPECT_EQ(engine->ComputeRange(0, 1), 1);

    engine->SetRange(0, 0, 2);
    EXPECT_EQ(engine->ComputeRange(0, 1), 1);

    engine->SetRange(0, 1, 3);
    EXPECT_EQ(engine->ComputeRange(0, 1), 3);
}

TEST(InMemorySimple, SingleElementPointUpdate) {
    auto engine = std::make_unique<NProust::TRobustPointUpdateEngine<int64_t, int64_t, NProust::TSumOperation>>();
    engine->Init(1);
    EXPECT_EQ(engine->ComputeRange(0, 1), 0);

    engine->SetPoint(0, 1);
    EXPECT_EQ(engine->ComputeRange(0, 1), 1);

    engine->SetPoint(0, 2);
    EXPECT_EQ(engine->ComputeRange(0, 1), 2);

    engine->SetPoint(0, 3);
    EXPECT_EQ(engine->ComputeRange(0, 1), 3);
}

TEST(InMemorySimple, SingleElementPointUpdateBitwiseAnd) {
    auto engine = std::make_unique<NProust::TRobustPointUpdateEngine<int64_t, int64_t, NProust::TBitwiseAndOperation>>();
    engine->Init(1);
    EXPECT_EQ(engine->ComputeRange(0, 1), 0);

    engine->SetPoint(0, 1);
    EXPECT_EQ(engine->ComputeRange(0, 1), 1);
}

TEST(InMemorySimple, SingleElementPointUpdateSmallLeafs) {
    auto engine = std::make_unique<NProust::TRobustPointUpdateEngine<int64_t, int64_t, NProust::TSumOperation, 0>>();
    engine->Init(1);
    engine->SetPoint(0, 1);
    EXPECT_EQ(engine->ComputeRange(0, 1), 1);
}

TEST(InMemorySimple, TwoElements) {
    auto engine = NProust::CreateRobustRangeUpdateEngine<int64_t, int64_t, NProust::TSumOperation>();
    engine->Init(2);
    EXPECT_EQ(engine->ComputeRange(0, 1), 0);
    EXPECT_EQ(engine->ComputeRange(1, 2), 0);
    EXPECT_EQ(engine->ComputeRange(0, 2), 0);

    engine->SetRange(0, 1, 1);
    EXPECT_EQ(engine->ComputeRange(0, 1), 1);
    EXPECT_EQ(engine->ComputeRange(1, 2), 0);
    EXPECT_EQ(engine->ComputeRange(0, 2), 1);

    engine->SetRange(0, 0, 2);
    EXPECT_EQ(engine->ComputeRange(0, 1), 1);
    EXPECT_EQ(engine->ComputeRange(1, 2), 0);
    EXPECT_EQ(engine->ComputeRange(0, 2), 1);

    engine->SetRange(0, 2, 3);
    EXPECT_EQ(engine->ComputeRange(0, 1), 3);
    EXPECT_EQ(engine->ComputeRange(0, 2), 6);
    EXPECT_EQ(engine->ComputeRange(1, 2), 3);

    engine->SetRange(0, 1, 4);
    EXPECT_EQ(engine->ComputeRange(0, 1), 4);
    EXPECT_EQ(engine->ComputeRange(0, 2), 7);
    EXPECT_EQ(engine->ComputeRange(1, 2), 3);

    engine->SetRange(1, 2, 5);
    EXPECT_EQ(engine->ComputeRange(0, 1), 4);
    EXPECT_EQ(engine->ComputeRange(0, 2), 9);
    EXPECT_EQ(engine->ComputeRange(1, 2), 5);
}

TEST(InMemorySimple, TwoElementsPointUpdate) {
    auto engine = std::make_unique<NProust::TRobustPointUpdateEngine<int64_t, int64_t, NProust::TSumOperation>>();
    engine->Init(2);
    EXPECT_EQ(engine->ComputeRange(0, 1), 0);
    EXPECT_EQ(engine->ComputeRange(1, 2), 0);
    EXPECT_EQ(engine->ComputeRange(0, 2), 0);

    engine->SetPoint(0, 1);
    EXPECT_EQ(engine->ComputeRange(0, 1), 1);
    EXPECT_EQ(engine->ComputeRange(1, 2), 0);
    EXPECT_EQ(engine->ComputeRange(0, 2), 1);

    engine->SetPoint(1, 3);
    EXPECT_EQ(engine->ComputeRange(0, 1), 1);
    EXPECT_EQ(engine->ComputeRange(0, 2), 4);
    EXPECT_EQ(engine->ComputeRange(1, 2), 3);

    engine->SetPoint(0, 4);
    EXPECT_EQ(engine->ComputeRange(0, 1), 4);
    EXPECT_EQ(engine->ComputeRange(0, 2), 7);
    EXPECT_EQ(engine->ComputeRange(1, 2), 3);

    engine->SetPoint(1, 5);
    EXPECT_EQ(engine->ComputeRange(0, 1), 4);
    EXPECT_EQ(engine->ComputeRange(0, 2), 9);
    EXPECT_EQ(engine->ComputeRange(1, 2), 5);
}

TEST(InMemorySimple, GloballyAddedValues) {
    auto engine = NProust::CreateRobustRangeUpdateEngine<int64_t, int64_t, NProust::TSumOperation>();
    engine->Init(10);
    engine->SetRange(0, 10, 10);
    EXPECT_EQ(engine->ComputeRange(0, 1), 10);
    EXPECT_EQ(engine->ComputeRange(0, 4), 40);
    EXPECT_EQ(engine->ComputeRange(3, 8), 50);
    EXPECT_EQ(engine->ComputeRange(9, 10), 10);
}

TEST(InMemorySimple, LazyPropagation) {
    auto engine = NProust::CreateRobustRangeUpdateEngine<int64_t, int64_t, NProust::TSumOperation>();
    engine->Init(8);
    engine->SetRange(0, 8, 10);
    engine->SetRange(0, 4, 5);
    EXPECT_EQ(engine->ComputeRange(0, 8), 60);
}

TEST(InMemorySimple, FatTreeEngine_1) {
    auto engine = std::make_unique<NProust::TFatPointUpdateEngine<int64_t, int64_t, NProust::TSumOperation, 1>>();
    engine->Init(16);
    EXPECT_EQ(engine->ComputeRange(0, 1), 0);

    engine->SetPoint(4, 1);
    EXPECT_EQ(engine->ComputeRange(0, 5), 1);
    EXPECT_EQ(engine->ComputeRange(0, 4), 0);

    engine->SetPoint(6, 1);
    EXPECT_EQ(engine->ComputeRange(0, 4), 0);
    EXPECT_EQ(engine->ComputeRange(0, 5), 1);
    EXPECT_EQ(engine->ComputeRange(0, 6), 1);
    EXPECT_EQ(engine->ComputeRange(0, 7), 2);
    EXPECT_EQ(engine->ComputeRange(0, 8), 2);

    engine->SetPoint(6, 2);
    EXPECT_EQ(engine->ComputeRange(0, 6), 1);
    EXPECT_EQ(engine->ComputeRange(0, 7), 3);
    EXPECT_EQ(engine->ComputeRange(0, 8), 3);

    engine->SetPoint(11, 3);
    EXPECT_EQ(engine->ComputeRange(0, 12), 6);
}

TEST(InMemorySimple, FatTreeEngine_2) {
    auto engine = std::make_unique<NProust::TFatPointUpdateEngine<int64_t, int64_t, NProust::TSumOperation, 2>>();
    engine->Init(16);
    EXPECT_EQ(engine->ComputeRange(0, 1), 0);

    engine->SetPoint(4, 1);
    EXPECT_EQ(engine->ComputeRange(0, 5), 1);
    EXPECT_EQ(engine->ComputeRange(0, 4), 0);

    engine->SetPoint(6, 1);
    EXPECT_EQ(engine->ComputeRange(0, 4), 0);
    EXPECT_EQ(engine->ComputeRange(0, 5), 1);
    EXPECT_EQ(engine->ComputeRange(0, 6), 1);
    EXPECT_EQ(engine->ComputeRange(0, 7), 2);
    EXPECT_EQ(engine->ComputeRange(0, 8), 2);

    engine->SetPoint(6, 2);
    EXPECT_EQ(engine->ComputeRange(0, 6), 1);
    EXPECT_EQ(engine->ComputeRange(0, 7), 3);
    EXPECT_EQ(engine->ComputeRange(0, 8), 3);

    engine->SetPoint(11, 3);
    EXPECT_EQ(engine->ComputeRange(0, 12), 6);
}

TEST(InMemorySimple, FatTreeEngineSmallLength) {
    auto engine = std::make_unique<NProust::TFatPointUpdateEngine<int64_t, int64_t, NProust::TSumOperation, 4>>();
    engine->Init(1);
    EXPECT_EQ(engine->ComputeRange(0, 1), 0);

    engine->SetPoint(0, 1);
    EXPECT_EQ(engine->ComputeRange(0, 1), 1);
}
