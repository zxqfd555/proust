#include <iostream>

#include "engine/naive/naive.h"
#include "engine/operations.h"

int main() {
    auto engine = NProust::CreateNaiveEngine<int64_t, int64_t, NProust::TBitwiseAndOperation>();

    std::cout << engine->Init(1) << std::endl;
    return 0;
}
