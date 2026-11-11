// O(N) per query brute force: the ground truth for small stress tests.
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    constexpr uint64_t MOD = 998244353;
    int n, q;
    if (std::scanf("%d %d", &n, &q) != 2) {
        return 1;
    }
    std::vector<uint64_t> a(n), b(n);
    for (int i = 0; i < n; ++i) {
        std::scanf("%lu %lu", &a[i], &b[i]);
    }
    for (int i = 0; i < q; ++i) {
        int type, l, r;
        std::scanf("%d %d %d", &type, &l, &r);
        if (type == 0) {
            uint64_t c, d;
            std::scanf("%lu %lu", &c, &d);
            for (int j = l; j < r; ++j) {
                a[j] = c;
                b[j] = d;
            }
        } else {
            uint64_t x;
            std::scanf("%lu", &x);
            for (int j = l; j < r; ++j) {
                x = (a[j] * x + b[j]) % MOD;
            }
            std::printf("%lu\n", x);
        }
    }
    return 0;
}
