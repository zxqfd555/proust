#!/usr/bin/env python3
"""Compare builds of the project over every benchmark the README reports.

Each build is a CMake build directory holding the benchmark binaries. One round runs every cell
(a benchmark in a given configuration) once per build, in an order that flips from round to round,
so that the thermal drift of a laptop acts on all the builds alike. Results are appended to a JSON
Lines file as they come, so a long comparison can be watched, interrupted and resumed.

    compare_builds.py run --build base=DIR --build cand=DIR[:IMPLEMENTATION[:LEAF_SIZE_LOG[:sentinel]]] --datasets DIR --out r.jsonl [--rounds 7]
    compare_builds.py noise r.jsonl --reference base --out noise.json
    compare_builds.py report r.jsonl --reference base [--noise noise.json]

Other work on the machine only ever slows a measurement down, and it can spoil several rounds in a
row, enough to move a median. Hence two defenses. With --probe, a short fixed workload runs around
every play of a cell: if it comes out slower than --quiet-ratio times the best of the recent probes,
the machine is busy, so the run waits for it to calm down, and a play that ended busy is thrown away
and repeated. Only recent probes count, so a lasting slowdown (a thermal state, a power mode) becomes
the new normal after a few minutes instead of stopping the run: builds are compared within a round.
And the statistic of a cell is the median of the fastest half of its values, not of all of them.

A verdict compares medians against the noise threshold of that metric: `noise` writes the largest
difference seen between builds of the same code (run them side by side first, with a copy shifted in
memory among them: code layout alone moves these numbers by a few percent).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict, deque
from pathlib import Path

TASKS = {
    "sum": "SumInt32",
    "max": "MaxInt32",
    "max-subarray": "MaxSumInt32",
    "order-statistic": "OrderStatistic16",
}
MODIFICATIONS = {"point": "Point", "range": "Range"}
MODES = ["per-operation", "average"]

# The configuration of the README's result tables.
RUNNER_CONFIG = ["--data-layout", "SoA", "--node-topology", "heap", "--leaf-size-log", "4"]
RUNNER_METRICS = {
    "per-operation": ["avg_ns_per_query", "peak_memory_kb", "wall_ns_p50", "wall_ns_p99", "modify_tsc_p50", "modify_tsc_p99", "query_tsc_p50", "query_tsc_p99"],
    "average": ["avg_ns_per_query", "peak_memory_kb"],
}
COUNTERS = ["cycles", "instructions", "branch-misses", "L1d-misses", "LLC-misses"]
FLOAT = r"([\d.]+)"
MAX_REPLAYS = 5
PROBE_WINDOW = 20


def parse_external(text):
    for match in re.finditer(rf"^(robust fat|fenwick|algorithmica|raw-upstream)\s+{FLOAT}\s+{FLOAT}\s+{FLOAT}", text, re.M):
        for name, value in zip(("update", "range", "prefix"), match.groups()[1:]):
            yield f"{match[1]}/{name}", float(value)


def parse_fence(text):
    floor = re.search(rf"fence floor.*?: {FLOAT} ns", text)
    if floor:
        yield "floor", float(floor[1])
    for match in re.finditer(rf"^(.+?)\s+throughput\s+{FLOAT} ns/op\s+latency\(fenced\)\s+{FLOAT}", text, re.M):
        yield f"{match[1]}/throughput", float(match[2])
        yield f"{match[1]}/fenced", float(match[3])


def parse_branching(text):
    row = rf"^((?:robust|algorithmica) B=\d+(?: \(log\d\))?)\s*\|\s*{FLOAT}\s+{FLOAT}\s+{FLOAT}\s*\|\s*{FLOAT}\s+{FLOAT}\s+{FLOAT}\s*\|"
    names = ("bare-update", "bare-range", "bare-prefix", "fenced-update", "fenced-range", "fenced-prefix")
    for match in re.finditer(row, text, re.M):
        for name, value in zip(names, match.groups()[1:]):
            yield f"{match[1]}/{name}", float(value)


def parse_query(text):
    for match in re.finditer(rf"^(.+?)\s*\|\s*{FLOAT}\s*\|\s*{FLOAT}\s*\|\s*{FLOAT}\s*\|", text, re.M):
        for name, value in zip(("update", "bare-range", "fenced-range"), match.groups()[1:]):
            yield f"{match[1]}/{name}", float(value)


def parse_maxsum(text):
    for match in re.finditer(rf"^(QUERY|UPDATE)\s*: before\(2N\)\s+{FLOAT} ns \| after\(group\)\s+{FLOAT} ns", text, re.M):
        yield f"{match[1].lower()}/before", float(match[2])
        yield f"{match[1].lower()}/after", float(match[3])


def parse_init(text):
    for match in re.finditer(rf"^(.+?)\s*\|\s*{FLOAT}\s*\|\s*{FLOAT}\s*\|", text, re.M):
        yield f"{match[1]}/from-array", float(match[2])
        yield f"{match[1]}/one-by-one", float(match[3])


MICROBENCHMARKS = {
    "external_microbench": parse_external,
    "fence_experiment": parse_fence,
    "branching_microbench": parse_branching,
    "query_microbench": parse_query,
    "maxsum_microbench": parse_maxsum,
    "init_microbench": parse_init,
}


def all_cells():
    for task in TASKS:
        for modification in MODIFICATIONS:
            for mode in MODES:
                yield f"runner/{task}/{modification}/{mode}"
    for name in MICROBENCHMARKS:
        yield f"micro/{name}"


class Runner:
    def __init__(self, args):
        self.builds = {}
        self.implementations = {}
        self.configs = {}
        for item in args.build:
            name, spec = item.split("=", 1)
            directory, *knobs = spec.split(":")
            self.builds[name] = directory
            self.implementations[name] = knobs[0] if knobs and knobs[0] else "robust"
            config = list(RUNNER_CONFIG)
            if len(knobs) > 1:
                config[config.index("--leaf-size-log") + 1] = knobs[1]
            if len(knobs) > 2 and knobs[2] == "sentinel":
                config.append("--with-sentinel")
            self.configs[name] = config
        self.datasets = Path(args.datasets)
        self.cpu = str(args.cpu)
        self.out = Path(args.out)
        self.perf_stat = args.perf_stat
        self.probe_binary = args.probe
        self.quiet_ratio = args.quiet_ratio
        self.recent_probes = deque(maxlen=PROBE_WINDOW)

    def play(self, cell, build):
        if cell.startswith("runner/"):
            return self.play_runner(cell, build)
        return self.play_micro(cell, build)

    def binary(self, build, name):
        return str(Path(self.builds[build]) / "benchmarks" / name)

    def play_runner(self, cell, build):
        _, task, modification, mode = cell.split("/")
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "run.csv"
            command = ["taskset", "-c", self.cpu]
            if self.perf_stat:
                command.append(self.perf_stat)
            command += [
                self.binary(build, "runner"),
                "--dataset-file", str(self.datasets / f"{task}_{modification}.dataset"),
                "--dataset-type", TASKS[task],
                "--generator-type", MODIFICATIONS[modification],
                *self.configs[build],
                "--implementation", self.implementations[build],
                "--measurement-mode", mode,
                "--csv-file", str(csv_path),
                "--csv-label", build,
            ]
            done = subprocess.run(command, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"{cell} failed in {build}:\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}")
            with open(csv_path) as source:
                row = list(csv.DictReader(source))[-1]
        values = {metric: float(row[metric]) for metric in RUNNER_METRICS[mode]}
        for line in done.stderr.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] in COUNTERS:
                values[parts[0]] = float(parts[1])
        return values

    def play_micro(self, cell, build):
        name = cell.split("/")[1]
        done = subprocess.run(["taskset", "-c", self.cpu, self.binary(build, name)], capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"{cell} failed in {build}:\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}")
        for count in re.findall(r"(\d+) mismatches", done.stdout):
            if int(count):
                raise RuntimeError(f"{cell} reports mismatches in {build}:\n{done.stdout}")
        values = dict(MICROBENCHMARKS[name](done.stdout))
        if not values:
            raise RuntimeError(f"{cell}: nothing parsed from:\n{done.stdout}")
        return values

    def probe(self):
        """True if the machine looks quiet; the best time so far is the yardstick."""
        if not self.probe_binary:
            return True
        done = subprocess.run(["taskset", "-c", self.cpu, self.probe_binary], capture_output=True, text=True, check=True)
        elapsed = float(done.stdout.split()[0])
        self.recent_probes.append(elapsed)
        return elapsed <= self.quiet_ratio * min(self.recent_probes)

    def wait_until_quiet(self):
        waited = 0
        while not self.probe():
            waited += 30
            if waited == 30 or waited % 600 == 0:
                print(f"  machine is busy, waiting ({waited} s so far)", flush=True)
            time.sleep(30)

    def record(self, handle, cell, build, attempt):
        """One play, repeated while the machine turns out to have been busy during it."""
        for _ in range(MAX_REPLAYS):
            self.wait_until_quiet()
            values = self.play(cell, build)
            if self.probe():
                break
            print(f"  busy during {cell} in {build}, playing again", flush=True)
        handle.write(json.dumps({"cell": cell, "build": build, "attempt": attempt, "values": values}) + "\n")
        handle.flush()

    def round(self, handle, cells, attempt, seen):
        names = list(self.builds)
        if attempt % 2:
            names.reverse()
        for cell in cells:
            # A benchmark added after a build was made is simply not compared on it.
            if cell.startswith("micro/") and not all(Path(self.binary(build, cell.split("/")[1])).exists() for build in names):
                continue
            for build in names:
                if (cell, build, attempt) not in seen:
                    self.record(handle, cell, build, attempt)
                    seen.add((cell, build, attempt))
            print(f"  round {attempt}: {cell}", flush=True)


def load(path):
    with open(path) as source:
        return [json.loads(line) for line in source if line.strip()]


def fast_half_median(values):
    """The median of the fastest half: other work on the machine can only add to a measurement."""
    ordered = sorted(values)
    return statistics.median(ordered[:max(1, (len(ordered) + 1) // 2)])


def collect(rows):
    """{(cell, metric): {build: [values]}}"""
    table = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for metric, value in row["values"].items():
            table[(row["cell"], metric)][row["build"]].append(value)
    return table


def command_run(args):
    runner = Runner(args)
    out = Path(args.out)
    rows = load(out) if out.exists() else []
    seen = {(row["cell"], row["build"], row["attempt"]) for row in rows}
    cells = [cell for cell in all_cells() if not args.only or re.search(args.only, cell)]
    for task in TASKS:
        for modification in MODIFICATIONS:
            subprocess.run(["cat", str(runner.datasets / f"{task}_{modification}.dataset")], stdout=subprocess.DEVNULL, check=True)
    with open(out, "a") as handle:
        for attempt in range(args.rounds):
            runner.round(handle, cells, attempt, seen)


def command_noise(args):
    table = collect(load(args.results))
    thresholds = {}
    for (cell, metric), by_build in table.items():
        medians = [fast_half_median(values) for values in by_build.values()]
        base = fast_half_median(by_build[args.reference])
        spread = max(abs(median - base) for median in medians) / base * 100 if base else 0.0
        thresholds[f"{cell}|{metric}"] = round(spread, 3)
    with open(args.out, "w") as sink:
        json.dump(thresholds, sink, indent=0, sort_keys=True)
    values = sorted(thresholds.values())
    print(f"{len(values)} thresholds; median {statistics.median(values):.2f}%, max {values[-1]:.2f}%")


def command_report(args):
    table = collect(load(args.results))
    noise = json.load(open(args.noise)) if args.noise else {}
    candidates = sorted({build for by_build in table.values() for build in by_build} - {args.reference})
    for candidate in candidates:
        better, worse, same = [], [], 0
        lines = []
        for (cell, metric), by_build in sorted(table.items()):
            if candidate not in by_build or args.reference not in by_build:
                continue
            base = fast_half_median(by_build[args.reference])
            value = fast_half_median(by_build[candidate])
            if base == 0:
                continue
            delta = (value - base) / base * 100
            threshold = max(noise.get(f"{cell}|{metric}", args.default_noise), args.min_noise)
            verdict = "=" if abs(delta) <= threshold else ("BETTER" if delta < 0 else "WORSE")
            if verdict == "BETTER":
                better.append((cell, metric, delta))
            elif verdict == "WORSE":
                worse.append((cell, metric, delta))
            else:
                same += 1
            lines.append(f"{cell:45s} {metric:32s} {base:14.2f} {value:14.2f} {delta:+7.2f}% +-{threshold:5.2f}% {verdict}")
        print(f"=== {candidate} vs {args.reference}: {len(better)} better, {len(worse)} worse, {same} within noise")
        if args.full:
            print("\n".join(lines))
        else:
            for name, rows in (("better", better), ("WORSE", worse)):
                for cell, metric, delta in sorted(rows, key=lambda row: row[2]):
                    print(f"  {name:6s} {delta:+7.2f}%  {cell}  {metric}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run")
    run.add_argument("--build", action="append", required=True, help="NAME=BUILD_DIR, repeated")
    run.add_argument("--datasets", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--rounds", type=int, default=7)
    run.add_argument("--probe", help="path to the quiet_probe binary, to detect a busy machine")
    run.add_argument("--quiet-ratio", type=float, default=1.15)
    run.add_argument("--cpu", type=int, default=2)
    run.add_argument("--perf-stat", help="path to the perf_stat binary, for hardware counters")
    run.add_argument("--only", help="regular expression selecting the cells")
    run.set_defaults(handler=command_run)

    noise = commands.add_parser("noise")
    noise.add_argument("results")
    noise.add_argument("--reference", required=True)
    noise.add_argument("--out", required=True)
    noise.set_defaults(handler=command_noise)

    report = commands.add_parser("report")
    report.add_argument("results")
    report.add_argument("--reference", required=True)
    report.add_argument("--noise")
    report.add_argument("--default-noise", type=float, default=3.0)
    report.add_argument("--min-noise", type=float, default=0.5)
    report.add_argument("--full", action="store_true")
    report.set_defaults(handler=command_report)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
