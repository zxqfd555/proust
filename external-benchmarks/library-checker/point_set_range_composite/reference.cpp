// Independent reference for point_set_range_composite: a plain bottom-up segment tree of
// compositions, in 64-bit arithmetic with % -- slow but simple, for the max tests.
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
    const uint64_t MOD = 998244353;
    struct F { uint64_t a = 1, b = 0; };
    F Then(const F& first, const F& second) { return {first.a * second.a % MOD, (second.a * first.b + second.b) % MOD}; }
}

int main() {
    int n, q;
    if (std::scanf("%d %d", &n, &q) != 2) return 1;
    int size = 1;
    while (size < n) size <<= 1;
    std::vector<F> tree(2 * size);
    for (int i = 0; i < n; ++i) {
        unsigned long long a, b;
        std::scanf("%llu %llu", &a, &b);
        tree[size + i] = {a, b};
    }
    for (int i = size - 1; i > 0; --i) tree[i] = Then(tree[2 * i], tree[2 * i + 1]);
    while (q--) {
        int t;
        unsigned long long x, y, z;
        std::scanf("%d %llu %llu %llu", &t, &x, &y, &z);
        if (t == 0) {
            int i = size + (int)x;
            tree[i] = {y, z};
            for (i >>= 1; i; i >>= 1) tree[i] = Then(tree[2 * i], tree[2 * i + 1]);
        } else {
            F left, right;
            for (int l = size + (int)x, r = size + (int)y; l < r; l >>= 1, r >>= 1) {
                if (l & 1) left = Then(left, tree[l++]);
                if (r & 1) right = Then(tree[--r], right);
            }
            const F all = Then(left, right);
            std::printf("%llu\n", (unsigned long long)((all.a * z + all.b) % MOD));
        }
    }
}
