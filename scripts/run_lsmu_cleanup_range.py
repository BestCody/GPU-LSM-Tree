#!/usr/bin/env python3
"""Build and run the LSMu post-deletion range sweep with cleanup."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RESULT = re.compile(r"^RESULT (?P<body>.+)$")


def run(command: list[str], *, log: Path | None = None) -> None:
    process = subprocess.Popen(
        command, cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert process.stdout is not None
    lines: list[str] = []
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    code = process.wait()
    if log is not None:
        log.write_text("".join(lines))
    if code:
        raise subprocess.CalledProcessError(code, command)


def parse_result(line: str) -> dict[str, str] | None:
    match = RESULT.match(line.strip())
    if not match:
        return None
    fields: dict[str, str] = {}
    for item in match.group("body").split():
        key, value = item.split("=", 1)
        fields[key] = value
    return fields


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2,
                        choices=(1, 2))
    parser.add_argument("--build", type=Path,
                        default=ROOT / "build" / "lsmu-cleanup-range")
    arguments = parser.parse_args()

    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    build = arguments.build.resolve()
    configure = [
        "cmake", "-S", str(ROOT / "FliX"), "-B", str(build),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc",
        "-DCMAKE_CUDA_ARCHITECTURES=120",
        "-DFLIX_BUILD_LSMU_CLEANUP_RANGE_EXPERIMENT=ON",
    ]
    run(configure, log=output / "configure.log")
    run(["cmake", "--build", str(build), "--target",
         "lsmu_cleanup_range_experiment", "-j2"],
        log=output / "build.log")

    executable = build / "lsmu_cleanup_range_experiment"
    rows: list[dict[str, str]] = []
    for repetition in range(1, arguments.repetitions + 1):
        log = output / f"repetition_{repetition}.log"
        command = [str(executable), "--repetition", str(repetition)]
        process = subprocess.Popen(
            command, cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert process.stdout is not None
        lines: list[str] = []
        for line in process.stdout:
            print(line, end="", flush=True)
            lines.append(line)
            result = parse_result(line)
            if result is not None:
                rows.append(result)
        code = process.wait()
        log.write_text("".join(lines))
        if code:
            raise subprocess.CalledProcessError(code, command)

    columns = sorted({key for row in rows for key in row})
    with (output / "measurements.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, columns)
        writer.writeheader()
        writer.writerows(rows)

    summary: list[dict[str, object]] = []
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["operation"], row["live"], row.get("expected", ""))
        groups.setdefault(key, []).append(row)
    for (operation, live, expected), group in sorted(groups.items()):
        gpu_field = "reported_gpu_ms" if expected else "gpu_ms"
        wall_field = "reported_wall_ms" if expected else "wall_ms"
        gpu = [float(row[gpu_field]) for row in group]
        wall = [float(row[wall_field]) for row in group]
        record: dict[str, object] = {
            "operation": operation,
            "state": int(group[0]["state"]),
            "live": int(live),
            "expected": int(expected) if expected else None,
            "repetitions": len(group),
            "mean_reported_gpu_ms": statistics.fmean(gpu),
            "minimum_reported_gpu_ms": min(gpu),
            "maximum_reported_gpu_ms": max(gpu),
            "mean_reported_wall_ms": statistics.fmean(wall),
        }
        if expected:
            record["mean_cleanup_gpu_ms"] = statistics.fmean(
                float(row["cleanup_gpu_ms"]) for row in group)
            record["mean_range_gpu_ms"] = statistics.fmean(
                float(row["range_gpu_ms"]) for row in group)
        summary.append(record)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    sources = [
        ROOT / "FliX" / "CMakeLists.txt",
        ROOT / "tests" / "lsmu_cleanup_range_experiment.cu",
        ROOT / "FliX" / "impl_lsm_tree.cuh",
        ROOT / "FliX" / "lsm_sort_context.cuh",
        Path(__file__).resolve(),
    ]
    metadata = {
        "state": "complete",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repetitions": arguments.repetitions,
        "initial_log": 27,
        "batch_log": 20,
        "query_limit_log": 24,
        "minimum_live_log": 20,
        "cleanup_policy": (
            "Clean once before the first post-deletion range batch at each "
            "state; add the same measured cleanup cost to each independently "
            "reported range-size case."
        ),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(output), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FAIL {error}", file=sys.stderr)
        raise
