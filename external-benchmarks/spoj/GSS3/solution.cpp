// SPOJ GSS3. 2^B-element leaves (Kadane on edges) under a bottom-up tree, branchless walk.
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <emmintrin.h>

struct S { int best, pre, suf, sum; };
const int NEG = -6e8;  // NEG + NEG + any real sum fits int

static inline int mx(int a, int b) { return a > b ? a : b; }
static inline S M(const S& a, const S& b) {
    return S{mx(mx(a.best, b.best), a.suf + b.pre), mx(a.pre, a.sum + b.pre),
             mx(b.suf, b.sum + a.suf), a.sum + b.sum};
}
static inline int pk(int m, int a, int b) { return b ^ ((a ^ b) & m); }  // m ? a : b
static inline S sel(int m, const S& a, const S& b) {
    return S{pk(m, a.best, b.best), pk(m, a.pre, b.pre), pk(m, a.suf, b.suf), pk(m, a.sum, b.sum)};
}

static inline S K(const int* v, int l, int r) {
    int sum = v[l], pre = sum, best = sum, e = sum;
    for (int i = l + 1; i < r; ++i) {
        e = mx(e + v[i], v[i]), best = mx(best, e), sum += v[i], pre = mx(pre, sum);
    }
    return S{best, pre, e, sum};
}

#define B 3  // leaf = 2^B elements
static int a[50000 + (1 << B)], H, n;
static S t[2 * (65536 >> B)];
static inline S leaf(int b) { return K(a, b << B, (b + 1) << B < n ? (b + 1) << B : n); }

// SSE2 marks token starts, so numbers parse independently (8-byte load + SWAR, <= 5 digits).
const uint64_t L8 = 0x0101010101010101;
static char* in;
static uint64_t *sm, bits;
static size_t k;
static inline int rd() {
    while (!bits) bits = sm[++k];
    uint64_t w, d;
    memcpy(&w, in + k * 64 + __builtin_ctzll(bits), 8);
    bits &= bits - 1;
    int neg = (w & 255) == '-';
    d = (w >> 8 * neg) ^ (L8 * '0');
    d <<= 64 - (__builtin_ctzll((((d & L8 * 0x7f) + L8 * 0x76) | d) & L8 * 0x80) & ~7);
    d = (d * 10 + (d >> 8)) & 0xFF00FF00FF00FF;
    d = (d * 100 + (d >> 16)) & 0xFFFF0000FFFF;
    d = (d * 10000 + (d >> 32)) & 0xFFFFFFFF;
    return neg ? -(int)d : (int)d;
}

static char ob[1 << 16], *o = ob;
static inline void wr(int x) {
    if (x < 0) *o++ = '-', x = -x;
    char d[12], *q = d;
    do *q++ = '0' + x % 10, x /= 10; while (x);
    while (q != d) *o++ = *--q;
    *o++ = '\n';
    if (o > ob + sizeof ob - 16) write(1, ob, o - ob), o = ob;
}

int main() {
    // Input is followed by >= 64 zero bytes (a file is mapped over a longer zero region).
    struct stat st;
    size_t len = 0;
    if (!fstat(0, &st) && S_ISREG(st.st_mode) && st.st_size > 0) {
        len = st.st_size;
        in = (char*)mmap(0, len + 4096, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        mmap(in, len, PROT_READ, MAP_PRIVATE | MAP_FIXED | MAP_POPULATE, 0, 0);
    } else {
        in = (char*)calloc(1 << 24, 1);
        for (ssize_t r; (r = read(0, in + len, (1 << 24) - 64 - len)) > 0;) len += r;
    }
    size_t nc = (len + 63) / 64;
    sm = (uint64_t*)malloc(nc * 8 + 8);
    uint64_t carry = 1;  // bit i: byte i > ' ' and byte i - 1 <= ' '
    for (size_t c = 0; c < nc; ++c) {
        uint64_t sep = 0;
        for (int j = 0; j < 4; ++j) {
            __m128i v = _mm_loadu_si128((__m128i*)(in + 64 * c + 16 * j));
            sep |= (uint64_t)_mm_movemask_epi8(_mm_cmplt_epi8(v, _mm_set1_epi8('!'))) << 16 * j;
        }
        sm[c] = ~sep & (sep << 1 | carry);
        carry = sep >> 63;
    }
    bits = sm[0];

    n = rd();
    for (int i = 0; i < n; ++i) a[i] = rd();
    int nb = (n + (1 << B) - 1) >> B;
    for (H = 1; H < nb; H <<= 1) {}
    S id = {NEG, NEG, NEG, 0};
    for (int b = 0; b < H; ++b) t[H + b] = b < nb ? leaf(b) : id;
    for (int i = H - 1; i; --i) t[i] = M(t[2 * i], t[2 * i + 1]);

    for (int m = rd(); m--;) {
        int c = rd(), x = rd() - 1, y = rd();
        if (c == 0) {
            a[x] = y;
            int i = H + (x >> B);
            t[i] = leaf(x >> B);
            for (i >>= 1; i; i >>= 1) t[i] = M(t[2 * i], t[2 * i + 1]);
            continue;
        }
        if (x >= y) { int l = y - 1; y = x + 1; x = l; }  // tolerate x > y
        int bl = x >> B, br = (y - 1) >> B;
        if (bl == br) { wr(K(a, x, y).best); continue; }
        S L = K(a, x, (bl + 1) << B), R = K(a, br << B, y);
        for (int l = H + bl + 1, r = H + br; l < r; l >>= 1, r >>= 1) {
            L = sel(-(l & 1), M(L, t[l]), L);
            R = sel(-(r & 1), M(t[r - 1], R), R);
            l += l & 1, r -= r & 1;
        }
        wr(M(L, R).best);
    }
    write(1, ob, o - ob);
}
