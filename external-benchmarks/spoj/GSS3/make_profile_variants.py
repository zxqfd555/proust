# Builds server-profiling variants of solution.cpp into variants/. Each one repeats a component
# REPS times silently, then does one normal pass, so the verdict stays AC and
#   component time = (variant time - plain solution time) / REPS.
#   p0: whole work (tokenize + parse + build + ops)   p1: tokenize + parse only
#   p2: build + ops on pre-parsed input               p3: p2 without the query walk
#   p4: p2 without the update climb
#
#   python3 make_profile_variants.py [SOURCE [PREFIX [VARIANT...]]]   (default: solution.cpp, all)
import re, os, sys
REPS = 20
SRC = sys.argv[1] if len(sys.argv) > 1 else 'solution.cpp'
PREFIX = sys.argv[2] if len(sys.argv) > 2 else ''
ONLY = set(sys.argv[3:])
src = open(SRC).read()
s = re.sub(r'[ ]*//.*', '', src)            # strip comments (no '//' inside strings here)
s = re.sub(r'\n\s*\n+', '\n', s)

def cut(a, b):
    i, j = s.index(a), s.index(b)
    return s[:i], s[i:j], s[j:]

# Silent passes: output buffer is recycled without writing.
s = s.replace('static char ob[1 << 16], *o = ob;', 'static char ob[1 << 16], *o = ob;\nstatic int out;')
s = s.replace('if (o > ob + sizeof ob - 16) write(1, ob, o - ob), o = ob;',
              'if (o > ob + sizeof ob - 16) { if (out) write(1, ob, o - ob); o = ob; }')
s = s.replace('    write(1, ob, o - ob);\n}', '    if (out) write(1, ob, o - ob);\n    o = ob;\n}')
# Split main: load | tokenize | solve.
head, load, rest = cut('int main() {', '    size_t nc = (len + 63) / 64;')
tok, solve = rest.split('    bits = sm[0];\n', 1)
tok_body = tok.replace('    size_t nc = (len + 63) / 64;\n    sm = (uint64_t*)malloc(nc * 8 + 8);\n', '')
load = load.replace('int main() {\n', '').replace('    size_t len = 0;\n', '')
solve = solve.rstrip()[:-1]  # drop main's closing brace
assert 'n = rd();' in solve
prog = head + 'static size_t len, nc;\nstatic void tokenize() {\n' + tok_body + '    k = 0, bits = sm[0];\n}\n'
prog += 'static void solve() {\n' + solve + '}\n'

# Same solve, but reading a pre-parsed input instead of rd().
arr = solve.replace('    n = rd();\n    for (int i = 0; i < n; ++i) a[i] = rd();',
                    '    n = N0;\n    memcpy(a, A0, n * 4);')
arr = arr.replace('    for (int m = rd(); m--;) {\n        int c = rd(), x = rd() - 1, y = rd();',
                  '    for (int q = 0; q < M0; ++q) {\n        int c = C0[q], x = X0[q] - 1, y = Y0[q];')
WITH_ARRAYS = 'rd()' not in arr  # p2..p4 need the plain op loop; pipelined sources get p0/p1 only
pre = ('static int N0, M0, A0[50000], C0[50000], X0[50000], Y0[50000];\n'
       'static void preparse() {\n    tokenize();\n    N0 = rd();\n    for (int i = 0; i < N0; ++i) A0[i] = rd();\n'
       '    M0 = rd();\n    for (int q = 0; q < M0; ++q) C0[q] = rd(), X0[q] = rd(), Y0[q] = rd();\n}\n')
main_head = 'int main() {\n' + load + '    nc = (len + 63) / 64;\n    sm = (uint64_t*)malloc(nc * 8 + 8);\n'
final = '    out = 1;\n    tokenize();\n    solve();\n}\n'
if WITH_ARRAYS:
  walk = re.search(r'        for \(int l = H \+ bl \+ 1.*?\n        }\n', arr, re.S).group(0)
  climb = '            for (i >>= 1; i; i >>= 1) t[i] = M(t[2 * i], t[2 * i + 1]);\n'
  assert climb in arr
variants = {
    'p0': prog + main_head + f'    for (int r = 0; r < {REPS}; ++r) tokenize(), solve();\n' + final,
    'p1': prog.replace('static size_t len, nc;', 'static size_t len, nc;\nstatic volatile int sink;') + main_head +
          f'    for (int r = 0; r < {REPS}; ++r) {{\n        tokenize();\n        int t = 0, n0 = rd();\n'
          '        for (int i = 0; i < n0; ++i) t += rd();\n        for (int q = rd() * 3; q--;) t += rd();\n        sink = t;\n    }\n' + final,
}
# p2..p4 run every pass (the final one too) from the pre-parsed input, without solve(); the
# removed part is only skipped in silent passes, so the answer stays correct.
if WITH_ARRAYS:
    prog2 = prog[:prog.index('static void solve() {')]
    for name, body in (('p2', arr), ('p3', arr.replace(walk, '        if (out)\n' + walk)),
                       ('p4', arr.replace(climb, '            if (out)\n' + climb))):
        variants[name] = (prog2 + pre + 'static void solve2() {\n' + body + '}\n' + main_head +
                          f'    preparse();\n    for (int r = 0; r < {REPS}; ++r) solve2();\n    out = 1;\n    solve2();\n}}\n')
os.makedirs('variants', exist_ok=True)
for name, code in variants.items():
    if ONLY and name not in ONLY:
        continue
    path = f'variants/{PREFIX}{name}_profile.cpp'
    open(path, 'w').write(code)
    print(path, len(code), len(code) + code.count('\n'))
