# gen.py MODE SEED [N] [M] -- random GSS3 input. Modes vary value signs and the whitespace layout.
import random, sys
mode, seed = sys.argv[1], int(sys.argv[2])
rng = random.Random(seed)
n = int(sys.argv[3]) if len(sys.argv) > 3 else rng.randint(1, 40)
m = int(sys.argv[4]) if len(sys.argv) > 4 else rng.randint(1, 40)
lo, hi = {'neg': (-10000, -1), 'pos': (1, 10000), 'small': (-3, 3)}.get(mode, (-10000, 10000))
val = lambda: rng.randint(lo, hi) if mode != 'extreme' else rng.choice((-10000, 10000))
sep = (lambda: rng.choice([' ', '  ', '\n', ' \r\n', '\t'])) if mode == 'messy' else (lambda: ' ')
out = [str(n), '\n', sep().join(str(val()) for _ in range(n)), '\n', str(m), '\n']
for _ in range(m):
    if rng.random() < 0.5:
        out.append(f"0{sep()}{rng.randint(1, n)}{sep()}{val()}\n")
    else:
        x, y = rng.randint(1, n), rng.randint(1, n)
        if mode != 'swapped': x, y = min(x, y), max(x, y)
        if mode == 'whole': x, y = 1, n
        out.append(f"1{sep()}{x}{sep()}{y}\n")
if mode == 'messy': out.append('\n\n  ')
sys.stdout.write(''.join(out))
