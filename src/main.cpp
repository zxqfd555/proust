#include <iostream>

#include "segtree/segtree.h"

int main() {
    auto engine = NProust::CreateSegmentTreeEngine<int64_t>();
    std::cout << engine->Init(1) << std::endl;
    return 0;
}
