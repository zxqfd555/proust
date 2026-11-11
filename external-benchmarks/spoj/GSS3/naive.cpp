// O(N) per query brute force for stress testing.
#include <cstdio>
#include <vector>
#include <algorithm>
int main() {
    int n; scanf("%d", &n);
    std::vector<long long> a(n);
    for (auto& x : a) scanf("%lld", &x);
    int m; scanf("%d", &m);
    while (m--) {
        int c, x, y; scanf("%d %d %d", &c, &x, &y);
        if (c == 0) { a[x - 1] = y; continue; }
        if (x > y) std::swap(x, y);
        long long best = a[x - 1], cur = 0;
        for (int i = x - 1; i < y; ++i) { cur += a[i]; best = std::max(best, cur); cur = std::max(cur, 0LL); }
        printf("%lld\n", best);
    }
}
