#!/usr/bin/env bash
# Stress test of submit.cpp (exactly the file to be submitted) against naive.cpp and reference.cpp.
#
#   ./stress.sh [ITERATIONS]
#
# Stages:
#   1. Small random tests in every generator mode against the brute force, under ASan/UBSan,
#      with build variants that stress the power store, the fold paths and the lookahead.
#   2. Exhaustive tiny shapes: every N in 1..70 (around and past the 16-element block), and N
#      around powers of 16.
#   3. Max-size tests shaped like the official ones, against the independent lazy tree.
set -euo pipefail

cd "$(dirname "$0")"
ITERATIONS=${1:-300}
WORK=${WORK:-$(mktemp -d)}
CXX=${CXX:-g++-12}
FLAGS="-std=c++23 -march=native"
SAN="-O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all"

python3 ../bundle.py solution.cpp -I ../../../include -o submit.cpp

echo "building into $WORK"
$CXX $FLAGS -O2 naive.cpp -o "$WORK/naive"
$CXX $FLAGS -O2 reference.cpp -o "$WORK/reference"
$CXX $FLAGS -O2 submit.cpp -o "$WORK/submit"
# Sanitized variants of the submitted file: default, the least power store that works (collected
# all the time), every partial fold vectorized or none, no lookahead, and the asserts kept alive.
$CXX $FLAGS $SAN submit.cpp -o "$WORK/san_default"
$CXX $FLAGS $SAN -DTIGHT_STORE submit.cpp -o "$WORK/san_tight"
$CXX $FLAGS $SAN -DSCALAR_RUN=0 submit.cpp -o "$WORK/san_vector_folds"
$CXX $FLAGS $SAN -DSCALAR_RUN=16 submit.cpp -o "$WORK/san_scalar_folds"
$CXX $FLAGS $SAN -DLOOKAHEAD=1 submit.cpp -o "$WORK/san_lookahead1"
$CXX $FLAGS $SAN -DKEEP_ASSERTS submit.cpp -o "$WORK/san_asserts"

check() {  # check NAME INPUT EXPECTED_BINARY BINARIES...
    local name=$1 input=$2 expected=$3
    shift 3
    "$expected" < "$input" > "$WORK/expected.out"
    for binary in "$@"; do
        if ! "$binary" < "$input" > "$WORK/actual.out" 2> "$WORK/stderr.out"; then
            echo "CRASH: $(basename "$binary") on $name"; cat "$WORK/stderr.out" | head -30
            cp "$input" "failed_$name.in"; exit 1
        fi
        if ! cmp -s "$WORK/expected.out" "$WORK/actual.out"; then
            echo "WRONG ANSWER: $(basename "$binary") on $name"
            diff "$WORK/expected.out" "$WORK/actual.out" | head -10
            cp "$input" "failed_$name.in"; exit 1
        fi
    done
}

SMALL=("$WORK/submit" "$WORK/san_default" "$WORK/san_tight" "$WORK/san_vector_folds" "$WORK/san_scalar_folds" "$WORK/san_lookahead1")

echo "== stage 0: the statement's example"
check example example.in "$WORK/naive" "${SMALL[@]}"
printf '14005\n470\n8275\n5500\n111111\n' | cmp -s - <("$WORK/submit" < example.in) || { echo "example output differs from statement"; exit 1; }

echo "== stage 1: $ITERATIONS random small tests per mode"
for mode in random fragment slide empty full pool extreme; do
    for ((seed = 1; seed <= ITERATIONS; ++seed)); do
        python3 gen.py "$mode" "$seed" > "$WORK/in"
        check "${mode}_$seed" "$WORK/in" "$WORK/naive" "${SMALL[@]}"
    done
    echo "   $mode: ok"
done

echo "== stage 2: every N in 1..70, several seeds each, plus N in 200..1100"
for ((n = 1; n <= 70; ++n)); do
    for seed in 1 2 3; do
        for mode in random pool full; do
            python3 gen.py "$mode" "$((n * 1000 + seed))" "$n" 300 > "$WORK/in"
            check "n${n}_${mode}_$seed" "$WORK/in" "$WORK/naive" "${SMALL[@]}" "$WORK/san_asserts"
        done
    done
done
for n in 200 255 256 257 511 512 513 1000 1024 1100 4095 4096 4097 65535 65536 65537; do
    for mode in random fragment slide pool; do
        python3 gen.py "$mode" "$n" "$n" 3000 > "$WORK/in"
        check "n${n}_$mode" "$WORK/in" "$WORK/naive" "${SMALL[@]}" "$WORK/san_asserts"
    done
done
echo "   ok"

echo "== stage 3: max tests (N = Q = 500000) against the independent lazy tree"
for mode in random fragment slide pool extreme full; do
    for seed in 1 2; do
        python3 gen.py "$mode" "$seed" 500000 500000 > "$WORK/max_${mode}_$seed.in"
        check "max_${mode}_$seed" "$WORK/max_${mode}_$seed.in" "$WORK/reference" "$WORK/submit" "$WORK/san_tight"
    done
    echo "   $mode: ok"
done

echo "ALL OK"
