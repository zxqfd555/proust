// Brute force for point_set_range_composite: O(r - l) per query.
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    const uint64_t mod = 998244353;
    int n, q;
    if (std::scanf("%d %d", &n, &q) != 2) return 1;
    std::vector<uint64_t> a(n), b(n);
    for (int i = 0; i < n; ++i) std::scanf("%llu %llu", (unsigned long long*)&a[i], (unsigned long long*)&b[i]);
    while (q--) {
        int t;
        unsigned long long x, y, z;
        std::scanf("%d %llu %llu %llu", &t, &x, &y, &z);
        if (t == 0) {
            a[x] = y;
            b[x] = z;
        } else {
            uint64_t v = z;
            for (unsigned long long i = x; i < y; ++i) v = (a[i] * v + b[i]) % mod;
            std::printf("%llu\n", (unsigned long long)v);
        }
    }
}
