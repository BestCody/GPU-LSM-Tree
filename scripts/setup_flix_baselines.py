#!/usr/bin/env python3
"""Build and check the repository's FliX backends on small inputs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = {
    "gpulsmopt": "GPULSMOPT",
    "lsmu": "LSM_TREE",
    "gpu_btree": "GPU_BTREE",
    "flix": None,
    "slabhash": "HASHTABLE_SLAB",
    "warpcore": "HASHTABLE_WARPCORE",
    "sorted_array": "SORTED_ARRAY",
}


def command(args, log, cwd, timeout=600):
    with log.open("w") as output:
        output.write(json.dumps([str(arg) for arg in args]) + "\n")
        output.flush()
        subprocess.run(args, cwd=cwd, stdout=output, stderr=subprocess.STDOUT,
                       timeout=timeout, check=True)


def cuda_root(explicit):
    candidates = [explicit, os.environ.get("CUDAToolkit_ROOT"),
                  os.environ.get("CUDA_HOME")]
    executable = shutil.which("nvcc")
    if executable:
        candidates.append(str(Path(executable).resolve().parents[1]))
    candidates.append("/usr/local/cuda")
    candidates.extend(str(p) for p in sorted(Path("/usr/local").glob("cuda-*"),
                                             reverse=True))
    for value in candidates:
        if value and (Path(value) / "bin/nvcc").is_file():
            return Path(value).resolve()
    raise RuntimeError("CUDA compiler missing; pass --cuda-root")


def verify_dependencies():
    manifest = json.loads((ROOT / "FliX/dependencies.json").read_text())
    for name, expected in manifest["files"].items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Missing or changed pinned dependency: {name}")
    return manifest["sources"]


def flags(backend, args):
    result = ["MAIN_32", "TOTALRUNS=1", "DENSEKEYGEN", f"XVAL={args.x}",
              f"YVAL={args.y}", "DIV=8", "NODESIZE=5", "CACHE_LINE_SIZE=0",
              "MAX_NODE=32", "DEFINE_TILE_SIZE=32", "TILE_INSERTS",
              "TILE_INSERTS_C", "INSERTS_TILE_BULK_ONLY", "TILE_DELETES",
              "DELETES_TILE_BULK", f"INITIAL_BUILD_SIZE={args.build_log}",
              f"INITIAL_PROBE_SIZE={args.probe_log}",
              f"ROUNDS_NUMBER={args.rounds}", "UPDATE_INSERT_PERCENT=25",
              "UPDATE_DELETE_PERCENT=25", "PERFORM_SUCCESSOR_PROBES=0",
              f"RANGE_BUILD_SIZE_LOG={args.build_log}",
              f"RANGE_PROBE_SIZE_LOG={min(args.probe_log, 10)}",
              "RANGE_EXPECTED_HITS_LOG=3", "RANGE_KEY_RANGE_MULTIPLIER_LOG=2"]
    if BACKENDS[backend]:
        result += ["BASELINES", BACKENDS[backend], "UNSORTED_PROBES_CHECKS"]
    return " ".join("-D" + value for value in result)


def configure_and_build(backend, args, toolkit, architecture):
    folder = args.build_root / backend
    folder.mkdir(parents=True, exist_ok=True)
    record = {"backend": backend, "state": "building", "flags": flags(backend, args)}
    started = time.monotonic()
    try:
        command(["cmake", "-S", ROOT / "FliX", "-B", folder,
                 f"-DCMAKE_CUDA_COMPILER={toolkit / 'bin/nvcc'}",
                 f"-DCUDAToolkit_ROOT={toolkit}",
                 f"-DCUDA_TOOLKIT_ROOT_DIR={toolkit}",
                 f"-DBIN2C={toolkit / 'bin/bin2c'}",
                 f"-DCMAKE_CUDA_ARCHITECTURES={architecture}",
                 f"-DFLIX_BENCHMARK_KEY_BITS={args.key_bits}",
                 "-DFLIX_ENABLE_PROFILING=OFF", "-DFLIX_USE_KEY_CACHE=OFF",
                 "-DCMAKE_BUILD_TYPE=Release", f"-DIFDEFS={record['flags']}"],
                folder / "configure.log", ROOT)
        command(["cmake", "--build", folder, "--target", "index_prototype",
                 "--parallel", str(args.jobs)], folder / "build.log", ROOT)
        record["state"] = "built"
        record["binary_sha256"] = hashlib.sha256(
            (folder / "index_prototype").read_bytes()).hexdigest()
    except (OSError, subprocess.SubprocessError) as error:
        record["state"] = "failed"
        record["error"] = str(error)
    record["build_seconds"] = time.monotonic() - started
    print(backend + ": " + record["state"], flush=True)
    return record


def verify_output(folder, backend, rounds, sanitized):
    log = (folder / "run.log").read_text()
    if re.search(r"->\s*SKIP|\bFAIL\b|mismatch|illegal memory|CUDA error|"
                 r"Assertion.*failed", log, re.IGNORECASE):
        raise RuntimeError("Harness reported a correctness failure or skipped case")
    if sanitized and "ERROR SUMMARY: 0 errors" not in log:
        raise RuntimeError("Memcheck did not finish with zero errors")
    files = list(folder.glob("*.csv"))
    rows = []
    for path in files:
        with path.open() as stream:
            rows.extend(csv.DictReader(stream, skipinitialspace=True))
    if not rows:
        raise RuntimeError("Harness produced no result rows")
    update_rows = [row for row in rows if row.get("step", "").strip()]
    states = sorted({int(row["step"]) for row in update_rows})
    if states != list(range(2 * rounds + 1)):
        raise RuntimeError(f"Missing update states: {states}")
    signatures = []
    for state in states:
        state_rows = [row for row in update_rows if int(row["step"]) == state]
        row = state_rows[0]
        if (row.get("lookup_timing") != "complete_unsorted_v1" or
                row.get("probe_input_order") != "unsorted"):
            raise RuntimeError("Missing complete unsorted lookup timing contract")
        metrics = {r["DESCRIPTION"]: float(r["VALUE"]) for r in state_rows}
        for count_key, total_key, prefix in [
                ("hit_query_count", "probe_time_ms", "probe"),
                ("miss_query_count", "probe_miss_time_ms", "probe_miss"),
                ("deleted_query_count", "deleted_keys_probe_time_ms", "deleted_keys_probe")]:
            total = metrics[total_key]
            wall = metrics[prefix + "_wall_time_ms"]
            components = [metrics[prefix + "_" + part + "_time_ms"]
                          for part in ("prepare", "search", "restore")]
            if any(not math.isfinite(v) or v < 0 for v in [total, wall] + components):
                raise RuntimeError(f"Invalid timing at state {state}: {prefix}")
            if not math.isclose(total, sum(components), rel_tol=1e-4, abs_tol=1e-4):
                raise RuntimeError(f"Lookup components do not span the total: {prefix}")
            if int(row[count_key]) > 0 and (total <= 0 or wall <= 0):
                raise RuntimeError(f"Untimed lookup batch at state {state}: {prefix}")
        signatures.append({key: row[key] for key in (
            "run", "step", "workload_key_bits", "workload_min_key", "workload_max_key",
            "workload_keys_checksum", "request_checksum", "checksum_format",
            "hit_query_count", "miss_query_count", "deleted_query_count")})
    return {"update_states": states, "csv_rows": len(rows),
            "input_signature": signatures,
            "input_signature_sha256": hashlib.sha256(
                json.dumps(signatures, sort_keys=True).encode()).hexdigest()}


def run_check(record, args, toolkit, sanitized):
    backend = record["backend"]
    build = args.build_root / backend
    folder = Path(tempfile.mkdtemp(
        prefix="memcheck-" if sanitized else "check-", dir=build))
    result_key = "memcheck" if sanitized else "correctness"
    record[result_key] = {"directory": str(folder)}
    cmd = [build / "index_prototype"]
    if sanitized:
        cmd = [toolkit / "bin/compute-sanitizer", "--tool", "memcheck",
               "--error-exitcode", "99"] + cmd
    command(cmd, folder / "run.log", folder, args.timeout)
    details = verify_output(folder, backend, args.rounds, sanitized)
    record[result_key].update(details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", choices=BACKENDS,
                        default=list(BACKENDS))
    parser.add_argument("--build-root", type=Path,
                        default=ROOT / "build/flix_baselines")
    parser.add_argument("--cuda-root")
    parser.add_argument("--arch", help="CUDA architecture; detected from GPU 0 by default")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--parallel-builds", type=int, default=2)
    parser.add_argument("--build-log", type=int, default=15)
    parser.add_argument("--probe-log", type=int, default=16)
    parser.add_argument("--key-bits", type=int, default=31,
                        help="shared key domain; all selected backends must support it")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--x", type=int, default=25)
    parser.add_argument("--y", type=int, default=90)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--memcheck", action="store_true")
    parser.add_argument("--recheck", action="store_true",
                        help="repeat checks even when a validated binary is unchanged")
    args = parser.parse_args()
    if min(args.jobs, args.parallel_builds, args.rounds, args.timeout) < 1:
        parser.error("jobs, parallel-builds, rounds and timeout must be positive")
    if not (1 <= args.build_log <= 28 and 1 <= args.probe_log <= 28):
        parser.error("build-log and probe-log must be in [1, 28]")
    if not (2 <= args.key_bits <= 32):
        parser.error("key-bits must be in [2, 32]")
    key_count = (1 << args.build_log) + ((1 << args.build_log) // (4 * args.rounds)) * args.rounds
    if key_count > (1 << args.key_bits) - 4:
        parser.error("key domain is too small for the distinct build and insertion keys")
    if (1 << args.build_log) // (4 * args.rounds) == 0:
        parser.error("build-log and rounds must yield nonempty update batches")
    if not (1 <= args.x <= 99 and 0 <= args.y <= 100):
        parser.error("the dense generator requires x in [1, 99] and y in [0, 100]")
    sources = verify_dependencies()
    toolkit = cuda_root(args.cuda_root)
    architecture = args.arch or subprocess.check_output(
        ["nvidia-smi", "--id=0", "--query-gpu=compute_cap", "--format=csv,noheader"],
        text=True).strip().replace(".", "")
    if not re.fullmatch(r"\d+", architecture):
        parser.error("arch must be a numeric CUDA architecture")
    args.build_root = args.build_root.resolve()
    args.build_root.mkdir(parents=True, exist_ok=True)
    status_path = args.build_root / "status.json"
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    if (previous.get("cuda_root") != str(toolkit) or
            previous.get("architecture") != architecture):
        previous = {}
    summary = {"purpose": "setup and correctness, not performance evaluation",
               "sources": sources, "cuda_root": str(toolkit),
               "architecture": architecture, "workload_key_bits": args.key_bits,
               "backends": {}}
    def save():
        (args.build_root / "status.json").write_text(json.dumps(summary, indent=2) + "\n")
    with ThreadPoolExecutor(max_workers=args.parallel_builds) as pool:
        futures = {pool.submit(configure_and_build, backend, args, toolkit, architecture):
                   backend for backend in dict.fromkeys(args.backends)}
        for future in as_completed(futures):
            record = future.result()
            summary["backends"][record["backend"]] = record
            save()
    for backend in dict.fromkeys(args.backends):
        record = summary["backends"][backend]
        if record["state"] != "built" or args.build_only:
            continue
        try:
            prior = previous.get("backends", {}).get(backend, {})
            if (not args.recheck and prior.get("state") == "passed" and
                    prior.get("binary_sha256") == record["binary_sha256"] and
                    prior.get("flags") == record["flags"] and
                    (not args.memcheck or "memcheck" in prior)):
                reusable = True
                for key in ("correctness", "memcheck"):
                    if key not in prior:
                        continue
                    try:
                        verify_output(Path(prior[key]["directory"]), backend,
                                      args.rounds, key == "memcheck")
                    except (OSError, ValueError, RuntimeError, KeyError):
                        reusable = False
                        break
                if reusable:
                    for key in ("correctness", "memcheck"):
                        if key in prior:
                            record[key] = prior[key]
                    record["state"] = "passed"
                    record["checks_reused_for_unchanged_binary"] = True
                    save()
                    print(backend + ": passed (unchanged binary; checks reused)", flush=True)
                    continue
            run_check(record, args, toolkit, False)
            if args.memcheck:
                run_check(record, args, toolkit, True)
            record["state"] = "passed"
        except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
            record["state"] = "failed"
            record["error"] = str(error)
        save()
        print(backend + ": " + record["state"], flush=True)
    cohort = [r for r in summary["backends"].values() if r["state"] == "passed"]
    if cohort:
        expected = cohort[0]["correctness"]["input_signature_sha256"]
        mismatched = [r["backend"] for r in cohort
                      if r["correctness"]["input_signature_sha256"] != expected or
                      ("memcheck" in r and
                       r["memcheck"]["input_signature_sha256"] != expected)]
        summary["input_comparison"] = {
            "state": "failed" if mismatched else "passed",
            "backends": [r["backend"] for r in cohort],
            "input_signature_sha256": expected,
            "mismatched_backends": mismatched}
        save()
        if mismatched:
            print("Input checksums differ: " + ", ".join(mismatched), flush=True)
            return 1
        print(f"Input checksums match across {len(cohort)} dynamic backends.", flush=True)
    print("Status: " + str(args.build_root / "status.json"), flush=True)
    return int(any(r["state"] == "failed" for r in summary["backends"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
