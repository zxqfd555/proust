#!/usr/bin/env python3
"""Test generator for range_set_range_composite.

Usage: gen.py MODE SEED [N] [Q]

Modes mirror the official generators (random, max_random, fragment, slide_window) and add
edge-case shapes: empty ranges, whole-array ranges, tiny value pools (equal functions
everywhere: lazy tags equal to the current content, the same power asked for many lengths),
and extreme coefficients (a = 1, a = MOD - 1, b = 0, b = MOD - 1).
"""

import random
import sys

MOD = 998244353
EXTREME_A = [1, 2, MOD - 1, MOD - 2]
EXTREME_B = [0, 1, MOD - 1]


def main():
    mode = sys.argv[1]
    rng = random.Random(int(sys.argv[2]))
    n = int(sys.argv[3]) if len(sys.argv) > 3 else rng.randint(1, 80)
    q = int(sys.argv[4]) if len(sys.argv) > 4 else rng.randint(1, 200)

    if mode == 'pool':
        pool = [(rng.randint(1, MOD - 1), rng.randint(0, MOD - 1)) for _ in range(rng.randint(1, 3))]
        fn = lambda: rng.choice(pool)
    elif mode == 'extreme':
        fn = lambda: (rng.choice(EXTREME_A), rng.choice(EXTREME_B))
    else:
        def fn():
            if rng.random() < 0.1:
                return rng.choice(EXTREME_A), rng.choice(EXTREME_B)
            return rng.randint(1, MOD - 1), rng.randint(0, MOD - 1)

    def pair():
        l, r = sorted((rng.randint(0, n), rng.randint(0, n)))
        return l, r

    def short_range():
        l = rng.randint(0, n - 1)
        return l, min(n, l + rng.randint(1, 10))

    out = [f'{n} {q}']
    out.extend('%d %d' % fn() for _ in range(n))

    window = max(1, n // 2)
    slide = 0 if seed_parity(sys.argv[2]) else n - window
    stride = 3 if slide == 0 else -3
    for i in range(q):
        if mode == 'slide' and i % 6 == 0:
            l = max(0, min(slide, n - window))
            out.append('0 %d %d %d %d' % (l, l + window, *fn()))
            slide += stride
            continue

        t = rng.randint(0, 1)
        if mode == 'empty' and rng.random() < 0.5:
            x = rng.randint(0, n)
            l, r = x, x
        elif mode == 'full' and rng.random() < 0.5:
            l, r = 0, n
        elif mode in ('fragment', 'slide') and t == 0:
            l, r = short_range()
        else:
            l, r = pair()

        if t == 0:
            out.append('0 %d %d %d %d' % (l, r, *fn()))
        else:
            x = rng.choice([0, 1, MOD - 1]) if rng.random() < 0.1 else rng.randint(0, MOD - 1)
            out.append('1 %d %d %d' % (l, r, x))

    sys.stdout.write('\n'.join(out) + '\n')


def seed_parity(seed):
    return int(seed) % 2 == 0


if __name__ == '__main__':
    main()
