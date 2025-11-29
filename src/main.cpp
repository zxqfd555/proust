#include <iostream>

#include "engine/segtree/segtree.h"
#include "engine/operations.h"

int main() {
    auto engine = NProust::CreateRobustEngine<int64_t, int64_t, NProust::TSumOperation>();

    std::cout << engine->Init(1) << std::endl;
    return 0;
}
