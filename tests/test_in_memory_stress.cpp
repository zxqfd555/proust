#include <chrono>
#include <iostream>
#include <random>

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include "common/dataset.h"
#include "engine/robust/engine.h"
#include "engine/operations.h"

#include "utils.h"

using NProust::CreateRandomIntGenerator;

TEST(InMemoryStress, IntSmallRangeSum) {
    StressTest<int64_t, int64_t, NProust::TSumOperation>(16, 10000000, CreateRandomIntGenerator(-100, 100));
    StressTest<int64_t, int64_t, NProust::TSumOperation>(16, 10000000, CreateRandomIntGenerator(-100, 100), NProust::EBaseGeneratorType::Point);
}

TEST(InMemoryStress, IntSmallRangeAnd) {
    StressTest<int64_t, int64_t, NProust::TBitwiseAndOperation>(16, 10000000, CreateRandomIntGenerator(0, 65535));
    StressTest<int64_t, int64_t, NProust::TBitwiseAndOperation>(16, 10000000, CreateRandomIntGenerator(0, 65535), NProust::EBaseGeneratorType::Point);
}

TEST(InMemoryStress, IntSmallRangeOr) {
    StressTest<int64_t, int64_t, NProust::TBitwiseOrOperation>(16, 10000000, CreateRandomIntGenerator(0, 65535));
    StressTest<int64_t, int64_t, NProust::TBitwiseOrOperation>(16, 10000000, CreateRandomIntGenerator(0, 65535), NProust::EBaseGeneratorType::Point);
}

TEST(InMemoryStress, IntSmallRangeXor) {
    StressTest<int64_t, int64_t, NProust::TBitwiseXorOperation>(16, 10000000, CreateRandomIntGenerator(0, 65535));
    StressTest<int64_t, int64_t, NProust::TBitwiseXorOperation>(16, 10000000, CreateRandomIntGenerator(0, 65535), NProust::EBaseGeneratorType::Point);
}

TEST(InMemoryStress, IntSmallRangeMin) {
    StressTest<int64_t, int64_t, NProust::TMinOperation>(16, 10000000, CreateRandomIntGenerator(-100, 100));
    StressTest<int64_t, int64_t, NProust::TMinOperation>(16, 10000000, CreateRandomIntGenerator(-100, 100), NProust::EBaseGeneratorType::Point);
}

TEST(InMemoryStress, IntSmallRangeMax) {
    StressTest<int64_t, int64_t, NProust::TMaxOperation>(16, 10000000, CreateRandomIntGenerator(-100, 100));
    StressTest<int64_t, int64_t, NProust::TMaxOperation>(16, 10000000, CreateRandomIntGenerator(-100, 100), NProust::EBaseGeneratorType::Point);
}

TEST(InMemoryStress, IntRangeSum) {
    StressTest<int64_t, int64_t, NProust::TSumOperation>(1000, 1000000, CreateRandomIntGenerator(-100, 100));
    StressTest<int64_t, int64_t, NProust::TSumOperation>(1000, 1000000, CreateRandomIntGenerator(-100, 100), NProust::EBaseGeneratorType::Point);
}

TEST(InMemoryStress, IntRangeAnd) {
    StressTest<int64_t, int64_t, NProust::TBitwiseAndOperation>(1000, 1000000, CreateRandomIntGenerator(0, 65535));
    StressTest<int64_t, int64_t, NProust::TBitwiseAndOperation>(1000, 1000000, CreateRandomIntGenerator(0, 65535), NProust::EBaseGeneratorType::Point);
}

TEST(InMemoryStress, IntRangeOr) {
    StressTest<int64_t, int64_t, NProust::TBitwiseOrOperation>(1000, 1000000, CreateRandomIntGenerator(0, 65535));
    StressTest<int64_t, int64_t, NProust::TBitwiseOrOperation>(1000, 1000000, CreateRandomIntGenerator(0, 65535), NProust::EBaseGeneratorType::Point);
}

TEST(InMemoryStress, IntRangeXor) {
    StressTest<int64_t, int64_t, NProust::TBitwiseXorOperation>(1000, 1000000, CreateRandomIntGenerator(0, 65535));
    StressTest<int64_t, int64_t, NProust::TBitwiseXorOperation>(1000, 1000000, CreateRandomIntGenerator(0, 65535), NProust::EBaseGeneratorType::Point);
}

TEST(InMemoryStress, IntRangeMin) {
    StressTest<int64_t, int64_t, NProust::TMinOperation>(1000, 1000000, CreateRandomIntGenerator(-100, 100));
    StressTest<int64_t, int64_t, NProust::TMinOperation>(1000, 1000000, CreateRandomIntGenerator(-100, 100), NProust::EBaseGeneratorType::Point);
}

TEST(InMemoryStress, IntRangeMax) {
    StressTest<int64_t, int64_t, NProust::TMaxOperation>(1000, 1000000, CreateRandomIntGenerator(-100, 100));
    StressTest<int64_t, int64_t, NProust::TMaxOperation>(1000, 1000000, CreateRandomIntGenerator(-100, 100), NProust::EBaseGeneratorType::Point);
}

TEST(InMemoryStress, PreorderTopology) {
    // idempotent operation
    for (size_t length = 1; length <= 100; ++length) {
        StressTest<int64_t, int64_t, NProust::TMaxOperation, NProust::TPreorderNodeTopology<0>>(length, 1000, CreateRandomIntGenerator(-100, 100));
        StressTest<int64_t, int64_t, NProust::TMaxOperation, NProust::TPreorderNodeTopology<1>>(length, 1000, CreateRandomIntGenerator(-100, 100));
        StressTest<int64_t, int64_t, NProust::TMaxOperation, NProust::TPreorderNodeTopology<2>>(length, 1000, CreateRandomIntGenerator(-100, 100));
        StressTest<int64_t, int64_t, NProust::TMaxOperation, NProust::TPreorderNodeTopology<3>>(length, 1000, CreateRandomIntGenerator(-100, 100));
        StressTest<int64_t, int64_t, NProust::TMaxOperation, NProust::TPreorderNodeTopology<4>>(length, 1000, CreateRandomIntGenerator(-100, 100));
    }

    // non-idempotent operation
    for (size_t length = 1; length <= 100; ++length) {
        StressTest<int64_t, int64_t, NProust::TSumOperation, NProust::TPreorderNodeTopology<0>>(length, 1000, CreateRandomIntGenerator(-100, 100));
        StressTest<int64_t, int64_t, NProust::TSumOperation, NProust::TPreorderNodeTopology<1>>(length, 1000, CreateRandomIntGenerator(-100, 100));
        StressTest<int64_t, int64_t, NProust::TSumOperation, NProust::TPreorderNodeTopology<2>>(length, 1000, CreateRandomIntGenerator(-100, 100));
        StressTest<int64_t, int64_t, NProust::TSumOperation, NProust::TPreorderNodeTopology<3>>(length, 1000, CreateRandomIntGenerator(-100, 100));
        StressTest<int64_t, int64_t, NProust::TSumOperation, NProust::TPreorderNodeTopology<4>>(length, 1000, CreateRandomIntGenerator(-100, 100));
    }
}

TEST(InMemoryStress, HeapTopology) {
    // idempotent operation
    for (size_t length = 1; length <= 100; ++length) {
        StressTest<int64_t, int64_t, NProust::TMaxOperation, NProust::TBinaryHeapNodeTopology<0>>(length, 1000, CreateRandomIntGenerator(-100, 100));
        StressTest<int64_t, int64_t, NProust::TMaxOperation, NProust::TBinaryHeapNodeTopology<1>>(length, 1000, CreateRandomIntGenerator(-100, 100));
        StressTest<int64_t, int64_t, NProust::TMaxOperation, NProust::TBinaryHeapNodeTopology<2>>(length, 1000, CreateRandomIntGenerator(-100, 100));
        StressTest<int64_t, int64_t, NProust::TMaxOperation, NProust::TBinaryHeapNodeTopology<3>>(length, 1000, CreateRandomIntGenerator(-100, 100));
        StressTest<int64_t, int64_t, NProust::TMaxOperation, NProust::TBinaryHeapNodeTopology<4>>(length, 1000, CreateRandomIntGenerator(-100, 100));
    }

    // non-idempotent operation
    for (size_t length = 1; length <= 100; ++length) {
        StressTest<int64_t, int64_t, NProust::TSumOperation, NProust::TBinaryHeapNodeTopology<0>>(length, 1000, CreateRandomIntGenerator(-100, 100));
        StressTest<int64_t, int64_t, NProust::TSumOperation, NProust::TBinaryHeapNodeTopology<1>>(length, 1000, CreateRandomIntGenerator(-100, 100));
        StressTest<int64_t, int64_t, NProust::TSumOperation, NProust::TBinaryHeapNodeTopology<2>>(length, 1000, CreateRandomIntGenerator(-100, 100));
        StressTest<int64_t, int64_t, NProust::TSumOperation, NProust::TBinaryHeapNodeTopology<3>>(length, 1000, CreateRandomIntGenerator(-100, 100));
        StressTest<int64_t, int64_t, NProust::TSumOperation, NProust::TBinaryHeapNodeTopology<4>>(length, 1000, CreateRandomIntGenerator(-100, 100));
    }
}
