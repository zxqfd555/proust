#!/usr/bin/env python3
"""Orchestrate the C++ benchmarking pipeline of the range query engine.

The project (see README.md) decouples dataset *creation* from dataset
*execution*: a slow but trustworthy naive engine writes a dataset once, and a
fast engine replays it whenever the implementation changes. This script drives
that workflow from a single, extendable entry point.

Three modes are exposed:

  generate   Produce the benchmarking datasets with the naive engine. This is
             slow and is meant to be run once. It accepts an optional task and
             an optional modification type; when either is omitted, the whole
             cross product is generated.

  execute    Replay existing datasets against the fast engine, several times,
             appending every run to a CSV log under a caller-supplied label.
             Run this from time to time, whenever the engine is improved.

  report     Turn the CSV log into a Markdown report: one table per (task,
             modification) pair, either for a specified label or for the best
             run found across all labels.

The README fixes four tasks and two modification flavors:

  Tasks (the associative ``Func`` being benchmarked):
    sum              -- element sum
    max              -- maximum value
    max-subarray     -- maximum subrange sum
    order-statistic  -- K-th order statistic (K = 16)

  Modification type (how an assignment touches the structure):
    point            -- single-element assignment (a special, cheaper case)
    range            -- the generic interval assignment

In all cases the structure holds 1_000_000 elements and the plan contains
10_000_000 queries, matching the methodology section of the README.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
import sys
from pathlib import Path

# The repository root is taken from this file's location so that paths resolve
# the same way regardless of the caller's working directory.
REPO_ROOT = Path(__file__).resolve().parent

# Friendly task names mapped to the dataset-type identifiers the C++ binaries
# understand (see NProust::DatasetTypeFromString).
TASKS: dict[str, str] = {
    "sum": "SumInt32",
    "max": "MaxInt32",
    "max-subarray": "MaxSumInt32",
    "order-statistic": "OrderStatistic16",
}

# Modification types mapped to the base generator identifiers the binaries
# understand (see NProust::BaseGeneratorTypeFromString).
MODIFICATIONS: dict[str, str] = {
    "point": "Point",
    "range": "Range",
}

# Engine implementations. Each maps to the concrete runner knobs that define it.
# Only "robust" exists today; the indirection is here so that adding e.g. a
# "fat" engine later is a one-line change rather than a new code path.
IMPLEMENTATIONS: dict[str, dict] = {
    "robust": {
        "data-layout": "SoA",
        "node-topology": "heap",
        "leaf-size-log": 4,
        "with-sentinel": False,
    },
    # An external reference baseline: a Fenwick tree fixed to the int32 sum problem with
    # point modifications (see include/engine/external/fenwick.h). It ignores the layout /
    # topology / leaf-size knobs — they are passed only because the runner CLI requires them.
    # It is only meaningful for `--task sum --modification point`.
    "fenwick": {
        "data-layout": "SoA",
        "node-topology": "heap",
        "leaf-size-log": 4,
        "with-sentinel": False,
    },
    # Another external baseline: the "wide segment tree" from Algorithmica/HPC — a B=16, AVX2
    # structure (see include/engine/external/algorithmica.h). Same caveats as fenwick: the knobs
    # below are ignored, and it only models `--task sum --modification point`.
    "algorithmica": {
        "data-layout": "SoA",
        "node-topology": "heap",
        "leaf-size-log": 4,
        "with-sentinel": False,
    },
}

# A short, human-readable description of every task, used as the preamble of the
# corresponding report tables (kept to one or two sentences, per requirement).
TASK_DESCRIPTIONS: dict[str, str] = {
    "sum": "Range-sum task: each query returns the sum of all elements in the half-open range [l, r).",
    "max": "Range-maximum task: each query returns the largest element in the half-open range [l, r).",
    "max-subarray": "Maximum-subarray-sum task: each query returns the largest sum achievable by any contiguous subsegment inside [l, r).",
    "order-statistic": "K-th order-statistic task (K = 16): each query returns the sixteen largest values inside [l, r).",
}

MODIFICATION_DESCRIPTIONS: dict[str, str] = {
    "point": "Modifications are single-point assignments.",
    "range": "Modifications are range assignments over [l, r).",
}

# The fixed problem dimensions from the README's benchmarking methodology.
DEFAULT_LENGTH = 1_000_000
DEFAULT_QUERY_COUNT = 10_000_000

# Where the generated datasets land. The template is filled per (task,
# modification) pair so a single invocation can produce the whole matrix
# without files clobbering each other. The "datasets" directory is gitignored.
DEFAULT_OUTPUT_TEMPLATE = "datasets/{task}_{modification}.dataset"

# The build configuration whose binaries we drive. Datasets and benchmarks are
# meant to be produced/run from an optimized build.
DEFAULT_BUILD_CONFIG = "Release"

# Defaults for the execute/report modes.
DEFAULT_CSV_FILE = "benchmark_results.csv"
DEFAULT_REPORT_FILE = "benchmark_report.md"
DEFAULT_RUNS = 7

# The percentile suffixes the runner emits, in order. They must match the
# TARGET_PERCENTILES table in benchmarks/runner.cpp.
PERCENTILE_NAMES = ["50", "85", "95", "99", "99.9", "99.99", "99.999"]

# Runs within a batch are ranked by this column to find the representative one.
SELECTION_COLUMN = "wall_ns_p99.999"

# Two-sided 95% Student's t critical values, indexed by degrees of freedom. Used
# for the confidence intervals so we avoid a SciPy dependency. For seven runs
# (six degrees of freedom) this is the textbook 2.447.
T_95: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131,
    20: 2.086, 30: 2.042, 60: 2.000,
}


# Above this many nanoseconds a latency reads more naturally in microseconds, so
# reports promote the unit (and rescale the value) past this threshold. TSC-cycle
# and memory columns are unaffected — only the "ns" unit is rescaled.
NS_TO_US_THRESHOLD = 1000


def format_ns(value: float) -> tuple[str, str]:
    """Render a nanosecond quantity, switching to microseconds past the threshold.

    Returns ``(rendered_value, unit)`` so the caller can place the unit wherever
    it likes. Values at or below the threshold stay in whole nanoseconds; larger
    ones are shown in microseconds with two decimals.
    """
    if value > NS_TO_US_THRESHOLD:
        return f"{value / 1000.0:,.2f}", "us"
    return f"{round(value):,}", "ns"


def format_measurement(column_unit: str, point: float, halfwidth: float | None) -> tuple[str, str, str]:
    """Render a (point, CI) measurement pair, returning ``(point, ci, unit)``.

    Only the ``ns`` unit is subject to the microsecond promotion; every other
    unit is rendered verbatim. The CI follows whatever unit the point estimate
    settles on, so the two never disagree.
    """
    if column_unit == "ns":
        point_str, unit = format_ns(point)
        if halfwidth is None:
            return point_str, "—", unit
        # The CI follows the point's unit, even when the half-width alone would
        # not have crossed the microsecond threshold — the two must agree.
        ci_str = f"{halfwidth / 1000.0:,.2f}" if unit == "us" else f"{round(halfwidth):,}"
        return point_str, f"± {ci_str}", unit

    point_str = f"{round(point):,}"
    ci_str = "—" if halfwidth is None else f"± {round(halfwidth):,}"
    return point_str, ci_str, column_unit


def measurement_columns() -> list[tuple[str, str, str]]:
    """The measurements rendered in each report table, as (column, label, unit)."""
    columns: list[tuple[str, str, str]] = [
        ("avg_ns_per_query", "Average latency", "ns"),
        ("peak_memory_kb", "Peak memory", "KB"),
    ]
    for name in PERCENTILE_NAMES:
        columns.append((f"wall_ns_p{name}", f"Wall-clock p{name}", "ns"))
    for name in PERCENTILE_NAMES:
        columns.append((f"modify_tsc_p{name}", f"Modify p{name}", "TSC"))
    for name in PERCENTILE_NAMES:
        columns.append((f"query_tsc_p{name}", f"Query p{name}", "TSC"))
    return columns


def t_value(dof: int) -> float:
    """95% t critical value, falling back to the largest tabulated dof below."""
    if dof in T_95:
        return T_95[dof]
    best = 1.96  # normal-approximation asymptote for large samples
    for key in sorted(T_95):
        if key <= dof:
            best = T_95[key]
    return best


def confidence_interval_halfwidth(values: list[float]) -> float | None:
    """95% CI half-width for the mean of ``values``; None when undefined (n < 2)."""
    n = len(values)
    if n < 2:
        return None
    return t_value(n - 1) * statistics.stdev(values) / math.sqrt(n)


class Benchmarker:
    """Drives the C++ ``generate_dataset`` and ``runner`` binaries."""

    def __init__(
        self,
        build_config: str = DEFAULT_BUILD_CONFIG,
        output_template: str = DEFAULT_OUTPUT_TEMPLATE,
        length: int = DEFAULT_LENGTH,
        query_count: int = DEFAULT_QUERY_COUNT,
        repo_root: Path = REPO_ROOT,
    ) -> None:
        self.repo_root = repo_root
        self.build_config = build_config
        self.output_template = output_template
        self.length = length
        self.query_count = query_count

    # -- paths ---------------------------------------------------------------

    def _binary_path(self, name: str) -> Path:
        """Locate a benchmark binary inside the chosen build tree."""
        path = self.repo_root / "build" / self.build_config / "benchmarks" / name
        if not path.exists():
            raise FileNotFoundError(
                f"Benchmark binary not found: {path}. "
                f"Build it first, e.g. `make {self.build_config.lower()}`."
            )
        return path

    def _output_path(self, task: str, modification: str) -> Path:
        """Resolve the dataset file for a (task, modification) pair."""
        rendered = self.output_template.format(task=task, modification=modification)
        path = Path(rendered)
        if not path.is_absolute():
            path = self.repo_root / path
        return path

    def _resolve_csv(self, csv_file: str | None) -> Path:
        path = Path(csv_file) if csv_file else self.repo_root / DEFAULT_CSV_FILE
        if not path.is_absolute():
            path = self.repo_root / path
        return path

    # -- generation mode -----------------------------------------------------

    def generate(
        self,
        tasks: list[str] | None = None,
        modifications: list[str] | None = None,
    ) -> None:
        """Generate datasets for the requested tasks and modification types.

        ``None`` for either argument expands to the full set, so the default is
        to generate every (task, modification) combination.
        """
        tasks = tasks or list(TASKS)
        modifications = modifications or list(MODIFICATIONS)

        binary = self._binary_path("generate_dataset")
        for task in tasks:
            for modification in modifications:
                self._generate_one(binary, task, modification)

    def _generate_one(self, binary: Path, task: str, modification: str) -> None:
        output = self._output_path(task, modification)
        output.parent.mkdir(parents=True, exist_ok=True)

        command = [
            str(binary),
            "--dataset-file", str(output),
            "--dataset-type", TASKS[task],
            "--length", str(self.length),
            "--query-count", str(self.query_count),
            "--generator-type", MODIFICATIONS[modification],
        ]

        print(
            f"[generate] task={task} modification={modification} -> {output}",
            flush=True,
        )
        subprocess.run(command, check=True)

    # -- execution mode ------------------------------------------------------

    def execute(
        self,
        label: str,
        tasks: list[str] | None = None,
        modifications: list[str] | None = None,
        implementation: str = "robust",
        runs: int = DEFAULT_RUNS,
        csv_file: str | None = None,
        measurement_mode: str | None = None,
    ) -> None:
        """Replay datasets against the fast engine and append runs to the CSV.

        Every (task, modification) selected is run ``runs`` times; each run
        appends one row to the CSV under ``label``. ``runs`` must be a positive
        odd number, so that the report can pick an unambiguous median run.
        """
        if not label:
            raise ValueError("a label is required for execution runs")
        if runs < 1 or runs % 2 == 0:
            raise ValueError(f"the number of runs must be a positive odd number, got {runs}")
        if implementation not in IMPLEMENTATIONS:
            raise ValueError(f"unknown implementation: {implementation}")

        tasks = tasks or list(TASKS)
        modifications = modifications or list(MODIFICATIONS)
        config = IMPLEMENTATIONS[implementation]
        csv_path = self._resolve_csv(csv_file)
        binary = self._binary_path("runner")

        for task in tasks:
            for modification in modifications:
                dataset_path = self._output_path(task, modification)
                if not dataset_path.exists():
                    raise FileNotFoundError(
                        f"Dataset not found: {dataset_path}. "
                        f"Generate it first: `{Path(sys.argv[0]).name} generate "
                        f"--task {task} --modification {modification}`."
                    )
                for run_index in range(runs):
                    self._execute_one(
                        binary, task, modification, implementation,
                        config, label, csv_path, run_index, runs,
                        measurement_mode,
                    )

    def _execute_one(
        self,
        binary: Path,
        task: str,
        modification: str,
        implementation: str,
        config: dict,
        label: str,
        csv_path: Path,
        run_index: int,
        runs: int,
        measurement_mode: str | None = None,
    ) -> None:
        command = [
            str(binary),
            "--dataset-file", str(self._output_path(task, modification)),
            "--dataset-type", TASKS[task],
            "--generator-type", MODIFICATIONS[modification],
            "--data-layout", config["data-layout"],
            "--node-topology", config["node-topology"],
            "--leaf-size-log", str(config["leaf-size-log"]),
            "--implementation", implementation,
            "--csv-label", label,
            "--csv-file", str(csv_path),
        ]
        if measurement_mode is not None:
            command += ["--measurement-mode", measurement_mode]
        if config.get("with-sentinel"):
            command.append("--with-sentinel")

        print(
            f"[execute] label={label} task={task} modification={modification} "
            f"impl={implementation} run={run_index + 1}/{runs} -> {csv_path}",
            flush=True,
        )
        subprocess.run(command, check=True)

    # -- reporting mode ------------------------------------------------------

    def report(
        self,
        label: str | None = None,
        csv_file: str | None = None,
        output: str | None = None,
    ) -> None:
        """Render the CSV log into a Markdown report (one table per cell).

        With ``label`` set, every cell shows that label's runs. Without it, each
        cell independently shows the best run found across all labels.
        """
        csv_path = self._resolve_csv(csv_file)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"CSV log not found: {csv_path}. Run the `execute` mode first."
            )

        with open(csv_path, newline="") as handle:
            rows = list(csv.DictReader(handle))

        markdown = self._build_report(rows, label, csv_path)

        out_path = Path(output) if output else self.repo_root / DEFAULT_REPORT_FILE
        if not out_path.is_absolute():
            out_path = self.repo_root / out_path
        out_path.write_text(markdown)
        print(f"[report] wrote {out_path}", flush=True)

    def _build_report(self, rows: list[dict], label: str | None, csv_path: Path) -> str:
        if label is None:
            selection = "the best run across all labels (per cell)"
        else:
            selection = f"label `{label}`"

        lines = [
            "# Benchmark report",
            "",
            f"_Source: `{csv_path.name}`. Showing {selection}._",
            "",
            "_Each cell reports the representative run: of the recorded (odd) runs, "
            "the median by wall-clock 99.999th-percentile latency. The 95% column "
            "is the confidence interval of the mean across that batch's runs._",
            "",
            "_Latencies above 1,000 ns are reported in microseconds (us)._",
            "",
        ]

        # Name the hardware the numbers came from: the CSV records it per run, and a
        # report only makes sense against the processor that produced it.
        cpu_models = sorted({row["cpu_model"] for row in rows if row.get("cpu_model")})
        if cpu_models:
            lines.append(f"_Processor: {', '.join(cpu_models)}._")
            lines.append("")

        for task in TASKS:
            lines.append(f"## {task}")
            lines.append("")
            for modification in MODIFICATIONS:
                lines.extend(self._render_cell(rows, task, modification, label))
                lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def _render_cell(
        self,
        rows: list[dict],
        task: str,
        modification: str,
        label: str | None,
    ) -> list[str]:
        heading = f"### {task} — {modification} modifications"
        preamble = f"{TASK_DESCRIPTIONS[task]} {MODIFICATION_DESCRIPTIONS[modification]}"

        group = self._select_group(rows, task, modification, label)
        if group is None:
            return [heading, "", preamble, "", "_No runs recorded for this cell._"]

        group_rows, group_label, implementation = group
        summary, _, selection_value = self._summarize_group(group_rows)

        selection_str, selection_unit = format_ns(selection_value)
        caption = (
            f"_Label `{group_label}`, implementation `{implementation}`, "
            f"{len(group_rows)} run(s); selected at {SELECTION_COLUMN} "
            f"= {selection_str} {selection_unit}._"
        )

        table = [
            "| Measurement | Representative | 95% CI |",
            "| --- | ---: | ---: |",
        ]
        for column, name, unit in measurement_columns():
            point, halfwidth = summary[column]
            point_str, ci, effective_unit = format_measurement(unit, point, halfwidth)
            table.append(f"| {name} ({effective_unit}) | {point_str} | {ci} |")

        return [heading, "", preamble, "", caption, "", *table]

    def _select_group(
        self,
        rows: list[dict],
        task: str,
        modification: str,
        label: str | None,
    ) -> tuple[list[dict], str, str] | None:
        """Pick the (rows, label, implementation) group to show in a cell.

        Groups are keyed by (label, implementation). When a label is given we
        restrict to it; otherwise all labels compete. The winning group is the
        one whose representative run has the lowest selection-column latency.
        """
        dataset_type = TASKS[task]
        generator_type = MODIFICATIONS[modification]

        groups: dict[tuple[str, str], list[dict]] = {}
        for row in rows:
            if row.get("dataset_type") != dataset_type:
                continue
            if row.get("generator_type") != generator_type:
                continue
            if label is not None and row.get("label") != label:
                continue
            if not row.get(SELECTION_COLUMN):
                continue
            key = (row.get("label", ""), row.get("implementation", ""))
            groups.setdefault(key, []).append(row)

        if not groups:
            return None

        best_key = None
        best_value = None
        for key, group_rows in groups.items():
            _, _, selection_value = self._summarize_group(group_rows)
            if best_value is None or selection_value < best_value:
                best_value = selection_value
                best_key = key

        group_label, implementation = best_key
        return groups[best_key], group_label, implementation

    def _summarize_group(
        self,
        group_rows: list[dict],
    ) -> tuple[dict[str, tuple[float, float | None]], dict, float]:
        """Representative point estimates and CIs for one group of runs.

        The representative run is the median when the group is sorted by the
        selection column; its value is the point estimate for every measurement,
        and the CI is computed across all runs in the group.
        """
        ordered = sorted(group_rows, key=lambda row: float(row[SELECTION_COLUMN]))
        median_row = ordered[len(ordered) // 2]

        summary: dict[str, tuple[float, float | None]] = {}
        for column, _, _ in measurement_columns():
            values = [float(row[column]) for row in group_rows if row.get(column)]
            point = float(median_row[column])
            summary[column] = (point, confidence_interval_halfwidth(values))

        return summary, median_row, float(median_row[SELECTION_COLUMN])


def odd_int(value: str) -> int:
    """argparse type: a positive odd integer."""
    number = int(value)
    if number < 1 or number % 2 == 0:
        raise argparse.ArgumentTypeError(f"must be a positive odd number, got {value}")
    return number


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Orchestrate the C++ range-query benchmarking pipeline.",
    )
    parser.add_argument(
        "--build-config",
        default=DEFAULT_BUILD_CONFIG,
        help="Build configuration whose binaries are driven (default: %(default)s).",
    )
    parser.add_argument(
        "--output-template",
        default=DEFAULT_OUTPUT_TEMPLATE,
        help=(
            "Dataset path template, with {task} and {modification} placeholders "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--length",
        type=int,
        default=DEFAULT_LENGTH,
        help="Structure size (default: %(default)s).",
    )
    parser.add_argument(
        "--query-count",
        type=int,
        default=DEFAULT_QUERY_COUNT,
        help="Number of queries per dataset (default: %(default)s).",
    )

    modes = parser.add_subparsers(dest="mode", required=True)

    generate = modes.add_parser(
        "generate",
        help="Generate datasets with the naive engine (run once).",
    )
    generate.add_argument(
        "--task",
        choices=sorted(TASKS),
        default=None,
        help="Task to generate for; omit to generate for all tasks.",
    )
    generate.add_argument(
        "--modification",
        choices=sorted(MODIFICATIONS),
        default=None,
        help="Modification type to generate for; omit to generate for both.",
    )

    execute = modes.add_parser(
        "execute",
        help="Replay datasets against the fast engine (run after improvements).",
    )
    execute.add_argument(
        "--task",
        choices=sorted(TASKS),
        default=None,
        help="Task to run; omit to run all tasks.",
    )
    execute.add_argument(
        "--modification",
        choices=sorted(MODIFICATIONS),
        default=None,
        help="Modification type to run; omit to run both.",
    )
    execute.add_argument(
        "--implementation",
        choices=sorted(IMPLEMENTATIONS),
        default="robust",
        help="Engine implementation to benchmark (default: %(default)s).",
    )
    execute.add_argument(
        "--runs",
        type=odd_int,
        default=DEFAULT_RUNS,
        help="Number of runs per cell; must be odd (default: %(default)s).",
    )
    execute.add_argument(
        "--label",
        required=True,
        help="Label recorded with every run, to distinguish it in the CSV log.",
    )
    execute.add_argument(
        "--measurement-mode",
        choices=["per-operation", "average"],
        default=None,
        help=(
            "How the runner times operations: 'per-operation' brackets each op in RDTSC fences "
            "for percentiles and the modify/query split; 'average' times whole batches for clean "
            "throughput, without the fence overhead. Omit to use the runner default."
        ),
    )
    execute.add_argument(
        "--csv-file",
        default=None,
        help=f"CSV log to append to (default: {DEFAULT_CSV_FILE}).",
    )

    report = modes.add_parser(
        "report",
        help="Render the CSV log into a Markdown report.",
    )
    report.add_argument(
        "--label",
        default=None,
        help="Show this label's runs; omit to show the best run per cell.",
    )
    report.add_argument(
        "--csv-file",
        default=None,
        help=f"CSV log to read (default: {DEFAULT_CSV_FILE}).",
    )
    report.add_argument(
        "--output",
        default=None,
        help=f"Markdown file to write (default: {DEFAULT_REPORT_FILE}).",
    )

    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    benchmarker = Benchmarker(
        build_config=args.build_config,
        output_template=args.output_template,
        length=args.length,
        query_count=args.query_count,
    )

    if args.mode == "generate":
        benchmarker.generate(
            tasks=[args.task] if args.task else None,
            modifications=[args.modification] if args.modification else None,
        )
    elif args.mode == "execute":
        benchmarker.execute(
            label=args.label,
            tasks=[args.task] if args.task else None,
            modifications=[args.modification] if args.modification else None,
            implementation=args.implementation,
            runs=args.runs,
            csv_file=args.csv_file,
            measurement_mode=args.measurement_mode,
        )
    elif args.mode == "report":
        benchmarker.report(
            label=args.label,
            csv_file=args.csv_file,
            output=args.output,
        )
    else:  # argparse enforces a valid mode; this guards future additions.
        print(f"Unknown mode: {args.mode}", file=sys.stderr)
        sys.exit(2)
