#!/usr/bin/env python3

import argparse
import csv
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_IFDEFS = (
    "-DMAIN_32 -DBASELINES -DGPULSMOPT -DREGKEYGEN "
    "-DTOTALRUNS=3 -DINITIAL_BUILD_SIZE=22 "
    "-DINITIAL_PROBE_SIZE=20 -DROUNDS_NUMBER=4 "
    "-DRANGE_BUILD_SIZE_LOG=22 -DRANGE_PROBE_SIZE_LOG=17 "
    "-DRANGE_KEY_RANGE_MULTIPLIER_LOG=2 "
    "-DRANGE_EXPECTED_HITS_LOG=8 -DRANGE_SORT_PROBE=0"
)


METRIC_NAMES = {
    "insert": "Insert update time, excluding setup zero",
    "delete": "Delete update time",
    "hit_lookup": "Hit lookup probe",
    "miss_lookup": "Miss lookup probe",
    "successor_hit": "Successor hit probe",
    "successor_miss": "Successor miss probe",
    "range_sum": "Range sum probe",
    "range_build": "Range build time",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_divisors(text: str) -> list[int]:
    values = []
    for item in text.replace(",", " ").split():
        value = int(item)
        if value < 1:
            raise argparse.ArgumentTypeError("divisors must be >= 1")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("at least one divisor is required")
    return values


def run_command(cmd: list[str], cwd: Path, log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed, see {log_path}")


def read_metric_rows(csv_path: Path) -> dict[str, list[float]]:
    buckets = {name: [] for name in METRIC_NAMES.values()}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            exp = (row.get("EXPERIMENT") or "").strip()
            desc = (row.get("DESCRIPTION") or "").strip()
            try:
                value = float((row.get("VALUE") or "").strip())
            except ValueError:
                continue
            if exp == "batches":
                do_insert = (row.get("do_insert") or "").strip()
                do_delete = (row.get("do_delete") or "").strip()
                if desc == "insert_or_delete_time_ms":
                    if do_insert == "1" and do_delete == "0" and value != 0:
                        buckets[METRIC_NAMES["insert"]].append(value)
                    elif do_insert == "0" and do_delete == "1":
                        buckets[METRIC_NAMES["delete"]].append(value)
                elif desc == "probe_time_ms":
                    buckets[METRIC_NAMES["hit_lookup"]].append(value)
                elif desc == "probe_miss_time_ms":
                    buckets[METRIC_NAMES["miss_lookup"]].append(value)
                elif desc == "successor_hits_probe_time_ms":
                    buckets[METRIC_NAMES["successor_hit"]].append(value)
                elif desc == "successor_misses_probe_time_ms":
                    buckets[METRIC_NAMES["successor_miss"]].append(value)
            elif exp == "range_query":
                if desc == "probe_time_ms":
                    buckets[METRIC_NAMES["range_sum"]].append(value)
                elif desc == "build_time_ms":
                    buckets[METRIC_NAMES["range_build"]].append(value)
    return buckets


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "avg": None,
            "median": None,
            "min": None,
            "max": None,
            "samples": 0,
        }
    return {
        "avg": sum(values) / len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "samples": len(values),
    }


def summarize_csv(csv_path: Path) -> dict[str, dict[str, float | int | None]]:
    return {name: summarize(values)
            for name, values in read_metric_rows(csv_path).items()}


def objective_score(
    summary: dict[str, dict[str, float | int | None]],
    objective: str,
) -> float:
    def avg(metric: str) -> float:
        value = summary[metric]["avg"]
        return float("inf") if value is None else float(value)

    if objective == "insert":
        return avg(METRIC_NAMES["insert"])
    if objective == "balanced":
        return (
            avg(METRIC_NAMES["insert"])
            + avg(METRIC_NAMES["delete"])
            + avg(METRIC_NAMES["hit_lookup"])
            + avg(METRIC_NAMES["miss_lookup"])
            + avg(METRIC_NAMES["range_sum"])
            + avg(METRIC_NAMES["successor_hit"])
            + avg(METRIC_NAMES["successor_miss"])
        )
    return avg(METRIC_NAMES[objective])


def write_summary_csv(
    output_path: Path,
    results: list[dict[str, object]],
) -> None:
    fields = [
        "divisor",
        "score",
        "metric",
        "avg_ms",
        "median_ms",
        "min_ms",
        "max_ms",
        "samples",
        "updates_csv",
        "build_dir",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            summary = result["summary"]
            for metric, stats in summary.items():
                writer.writerow({
                    "divisor": result["divisor"],
                    "score": result["score"],
                    "metric": metric,
                    "avg_ms": stats["avg"],
                    "median_ms": stats["median"],
                    "min_ms": stats["min"],
                    "max_ms": stats["max"],
                    "samples": stats["samples"],
                    "updates_csv": result["updates_csv"],
                    "build_dir": result["build_dir"],
                })


def print_table(results: list[dict[str, object]]) -> None:
    metric_order = [
        METRIC_NAMES["insert"],
        METRIC_NAMES["delete"],
        METRIC_NAMES["hit_lookup"],
        METRIC_NAMES["miss_lookup"],
        METRIC_NAMES["successor_hit"],
        METRIC_NAMES["successor_miss"],
        METRIC_NAMES["range_sum"],
        METRIC_NAMES["range_build"],
    ]
    print("\nDivisor summary, avg ms")
    header = ["divisor", "score"] + metric_order
    print("\t".join(header))
    for result in results:
        row = [str(result["divisor"]), f"{result['score']:.6f}"]
        summary = result["summary"]
        for metric in metric_order:
            value = summary[metric]["avg"]
            row.append("" if value is None else f"{float(value):.6f}")
        print("\t".join(row))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tune GPULSMOpt's run drain divisor with FliX."
    )
    parser.add_argument(
        "--divisors",
        type=parse_divisors,
        default=parse_divisors("1,2,4,8,20,40"),
        help="Comma or space separated values.",
    )
    parser.add_argument(
        "--objective",
        choices=[
            "insert",
            "delete",
            "hit_lookup",
            "miss_lookup",
            "successor_hit",
            "successor_miss",
            "range_sum",
            "range_build",
            "balanced",
        ],
        default="insert",
    )
    parser.add_argument(
        "--cuda-compiler",
        default="/usr/local/cuda-12.8/bin/nvcc",
    )
    parser.add_argument("--build-root", default="/tmp/flix-gpulsmopt-tune")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--ifdefs", default=DEFAULT_IFDEFS)
    parser.add_argument("--keep-builds", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    flix_dir = root / "FliX"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else (
        root / "tuning_results" / f"gpulsmopt-drain-{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    build_root = Path(args.build_root)
    build_root.mkdir(parents=True, exist_ok=True)

    results = []
    for divisor in args.divisors:
        build_dir = build_root / f"divisor-{divisor}"
        log_path = out_dir / f"divisor-{divisor}.log"
        if build_dir.exists() and not args.keep_builds:
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)

        ifdefs = f"{args.ifdefs} -DGPULSMOPT_RUN_DRAIN_DIVISOR={divisor}"
        cmake_cmd = [
            "cmake",
            "-S",
            str(flix_dir),
            "-B",
            str(build_dir),
            f"-DCMAKE_CUDA_COMPILER={args.cuda_compiler}",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DIFDEFS={ifdefs}",
        ]
        build_cmd = ["cmake", "--build", str(build_dir), "-j"]
        run_cmd = ["./index_prototype"]

        run_command(cmake_cmd, root, log_path)
        cache_dir = build_dir / "data_cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        run_command(build_cmd, root, log_path)
        updates_path = build_dir / "updates.csv"
        if updates_path.exists():
            updates_path.unlink()
        run_command(run_cmd, build_dir, log_path)
        if not updates_path.exists():
            raise RuntimeError(f"missing updates.csv for divisor {divisor}")

        copied_csv = out_dir / f"updates_divisor_{divisor}.csv"
        shutil.copy2(updates_path, copied_csv)
        summary = summarize_csv(copied_csv)
        score = objective_score(summary, args.objective)
        results.append({
            "divisor": divisor,
            "score": score,
            "summary": summary,
            "updates_csv": str(copied_csv),
            "build_dir": str(build_dir),
        })
        write_summary_csv(out_dir / "summary.csv", results)
        (out_dir / "results.json").write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )

    results.sort(key=lambda item: item["score"])
    write_summary_csv(out_dir / "summary.csv", results)
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    print_table(results)
    print(f"\nBest divisor for {args.objective}: {results[0]['divisor']}")
    print(f"Summary CSV: {out_dir / 'summary.csv'}")
    print(f"Raw CSVs/logs: {out_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
