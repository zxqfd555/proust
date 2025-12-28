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
    auto engine = NProust::CreateRobustPointUpdateEngine<int64_t, int64_t, NProust::TSumOperation>();
    engine->Init(1);
    EXPECT_EQ(engine->ComputeRange(0, 1), 0);

    engine->SetPoint(0, 1);
    EXPECT_EQ(engine->ComputeRange(0, 1), 1);

    engine->SetPoint(0, 2);
    EXPECT_EQ(engine->ComputeRange(0, 1), 2);

    engine->SetPoint(0, 3);
    EXPECT_EQ(engine->ComputeRange(0, 1), 3);
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
    auto engine = NProust::CreateRobustPointUpdateEngine<int64_t, int64_t, NProust::TSumOperation>();
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
