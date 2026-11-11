#!/usr/bin/env bash
# ./stress.sh [ITERATIONS] -- solution.cpp (as C++14) against naive.cpp.
set -euo pipefail
cd "$(dirname "$0")"
IT=${1:-500}; W=$(mktemp -d); CXX=${CXX:-g++}
# Pasting from Windows turns every LF into CRLF, so the limit is checked for bytes + lines.
for f in solution.cpp solution.c; do
  test $(( $(wc -c < $f) + $(wc -l < $f) )) -le 5000 || { echo "$f exceeds 5000 bytes with CRLF"; exit 1; }
done
$CXX -std=c++14 -O2 -Wall -Wextra naive.cpp -o $W/naive 2>/dev/null
$CXX -std=c++14 -O2 -Wall -Wextra -Werror -Wno-unused-result solution.cpp -o $W/sol
$CXX -std=c++14 -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all solution.cpp -o $W/san
# The C port of the same solution (strict C99 is the pickiest of SPOJ's C dialects).
gcc -std=c99 -O2 -Wall -Wextra -Werror -Wno-unused-result solution.c -o $W/solc
gcc -std=c99 -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all solution.c -o $W/sanc
chk() { $W/naive < $1 > $W/e; for b in $W/sol $W/san $W/solc $W/sanc; do $b < $1 > $W/a || { echo "CRASH $b $2"; cp $1 failed.in; exit 1; }
        cmp -s $W/e $W/a || { echo "WA $b $2"; diff $W/e $W/a | head; cp $1 failed.in; exit 1; }; done; }
for mode in random neg pos small extreme messy swapped whole; do
  for ((s = 1; s <= IT; ++s)); do python3 gen.py $mode $s > $W/in; chk $W/in ${mode}_$s; done; echo "$mode ok"
done
for n in 1 2 7 8 9 15 16 17 63 64 65 1000 8192 50000; do
  for mode in random neg extreme; do python3 gen.py $mode $n $n 3000 > $W/in; chk $W/in n${n}_$mode; done
done; echo "sizes ok"
for mode in random neg pos extreme whole; do python3 gen.py $mode 7 50000 50000 > $W/in; chk $W/in max_$mode; done
echo "max ok"; echo ALL OK
