// An independent O(log^2 N) solution, written without the library: a textbook recursive lazy
// segment tree with an assignment tag and binary exponentiation. It is the ground truth on
// max-size tests (where naive.cpp is too slow) and a baseline for timing.
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
    constexpr uint64_t MOD = 998244353;

    struct F {
        uint64_t a = 1, b = 0;
    };

    F Then(const F& f, const F& g) {  // g(f(x))
        return {f.a * g.a % MOD, (g.a * f.b + g.b) % MOD};
    }

    F Pow(F f, int k) {
        F r;
        while (k) {
            if (k & 1) r = Then(r, f);
            f = Then(f, f);
            k >>= 1;
        }
        return r;
    }

    int n;
    std::vector<F> sum, tag;
    std::vector<char> has;

    void Apply(int v, int len, const F& f) {
        sum[v] = Pow(f, len);
        tag[v] = f;
        has[v] = 1;
    }

    void Push(int v, int lo, int hi) {
        if (has[v]) {
            const int mid = (lo + hi) / 2;
            Apply(2 * v, mid - lo, tag[v]);
            Apply(2 * v + 1, hi - mid, tag[v]);
            has[v] = 0;
        }
    }

    void Build(int v, int lo, int hi, const std::vector<F>& init) {
        if (hi - lo == 1) {
            sum[v] = init[lo];
            return;
        }
        const int mid = (lo + hi) / 2;
        Build(2 * v, lo, mid, init);
        Build(2 * v + 1, mid, hi, init);
        sum[v] = Then(sum[2 * v], sum[2 * v + 1]);
    }

    void Assign(int v, int lo, int hi, int l, int r, const F& f) {
        if (r <= lo || hi <= l) return;
        if (l <= lo && hi <= r) {
            Apply(v, hi - lo, f);
            return;
        }
        Push(v, lo, hi);
        const int mid = (lo + hi) / 2;
        Assign(2 * v, lo, mid, l, r, f);
        Assign(2 * v + 1, mid, hi, l, r, f);
        sum[v] = Then(sum[2 * v], sum[2 * v + 1]);
    }

    F Query(int v, int lo, int hi, int l, int r) {
        if (r <= lo || hi <= l) return F{};
        if (l <= lo && hi <= r) return sum[v];
        Push(v, lo, hi);
        const int mid = (lo + hi) / 2;
        return Then(Query(2 * v, lo, mid, l, r), Query(2 * v + 1, mid, hi, l, r));
    }
} // namespace

int main() {
    int q;
    if (std::scanf("%d %d", &n, &q) != 2) return 1;
    std::vector<F> init(n);
    for (auto& f : init) std::scanf("%lu %lu", &f.a, &f.b);
    sum.assign(4 * n, F{});
    tag.assign(4 * n, F{});
    has.assign(4 * n, 0);
    Build(1, 0, n, init);
    for (int i = 0; i < q; ++i) {
        int type, l, r;
        std::scanf("%d %d %d", &type, &l, &r);
        if (type == 0) {
            F f;
            std::scanf("%lu %lu", &f.a, &f.b);
            Assign(1, 0, n, l, r, f);
        } else {
            uint64_t x;
            std::scanf("%lu", &x);
            const F f = Query(1, 0, n, l, r);
            std::printf("%lu\n", (f.a * x + f.b) % MOD);
        }
    }
    return 0;
}
