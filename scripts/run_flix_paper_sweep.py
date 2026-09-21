#!/usr/bin/env python3
"""Run the common paper matrix and original FliX workloads through their adapters."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

import setup_flix_baselines as setup

ROOT = setup.ROOT
PROTOCOL = "flix_paper_suite_v1"
BACKENDS = list(setup.BACKENDS)
DYNAMIC = list(BACKENDS)
RANGES = {"gpulsmopt", "lsmu", "gpu_btree", "flix", "sorted_array"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def source_signature():
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT).decode().split("\0")
    files = {name: sha(ROOT / name) for name in sorted(set(names))
             if (ROOT / name).is_file() and
             (Path(name).suffix in {".cu", ".cuh", ".h", ".hpp", ".hxx", ".cpp", ".cmake", ".sh", ".py", ".json"}
              or Path(name).name == "CMakeLists.txt")}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["paper", "flix", "both"], default="paper")
    parser.add_argument("--systems", nargs="+", default=["all"],
                        help="all, both (GPULSMOpt/LSMu), or backend names")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", type=Path, default=ROOT / "results/flix_paper_suite_v1")
    parser.add_argument("--build-root", type=Path, default=ROOT / "build/flix_paper_suite")
    parser.add_argument("--batch-logs", nargs="+", help="space-separated public batch logs")
    parser.add_argument("--insert-limit-log", type=int)
    parser.add_argument("--query-limit-log", type=int)
    parser.add_argument("--range-chunk-log", type=int, default=0,
                        help="0 submits the full range-query batch (default); N limits each call to 2^N queries")
    parser.add_argument("--stop-after-r", type=int, default=0)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--adopt-warmups-from", type=Path,
                        help="use completed warmup_00 cases from an earlier suite as one timing sample")
    parser.add_argument("--adopt-warmup-batch-logs", nargs="+", type=int, default=[],
                        help="paper batch logs whose completed warmups should replace new repetitions")
    parser.add_argument("--adopt-warmup-cases", nargs="+", default=[],
                        help="additional case IDs whose completed warmups should replace new repetitions")
    parser.add_argument("--skip-completed-from", type=Path,
                        help="skip cases with all required, validated runs in an earlier suite")
    parser.add_argument("--skip-bulk", action="store_true")
    parser.add_argument("--skip-ranges", action="store_true")
    parser.add_argument("--skip-deletions", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--memcheck", action="store_true",
                        help="one separate sanitizer replay per case; excluded from results")
    parser.add_argument("--plan", action="store_true", help="print matrix without building/running")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--cuda-root")
    parser.add_argument("--cuda-arch", dest="arch")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--parallel-builds", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--flix-build-log", type=int)
    parser.add_argument("--flix-probe-log", type=int)
    parser.add_argument("--flix-rounds", type=int, default=5)
    parser.add_argument("--xy", nargs="+", default=["25:25", "25:90"], help="FliX X:Y pairs")
    args = parser.parse_args()
    full = args.mode == "full"
    args.insert_limit_log = args.insert_limit_log if args.insert_limit_log is not None else (27 if full else 18)
    args.query_limit_log = args.query_limit_log if args.query_limit_log is not None else (24 if full else 18)
    args.repetitions = args.repetitions if args.repetitions is not None else (3 if full else 2)
    args.warmups = args.warmups if args.warmups is not None else (1 if full else 0)
    args.flix_build_log = args.flix_build_log if args.flix_build_log is not None else (26 if full else 15)
    args.flix_probe_log = args.flix_probe_log if args.flix_probe_log is not None else (27 if full else 16)
    args.batch_logs = [int(x) for part in (args.batch_logs or ["15 16 17 18 19 20 21 22 23 24 25 26 27" if full else "16 17"])
                       for x in part.replace(",", " ").split()]
    args.batch_logs = list(dict.fromkeys(args.batch_logs))
    names = [x for part in args.systems for x in part.replace(",", " ").split()]
    args.systems = list(dict.fromkeys(BACKENDS if names == ["all"] else
                                    ["gpulsmopt", "lsmu"] if names == ["both"] else names))
    if not args.systems or any(x not in BACKENDS for x in args.systems):
        parser.error("systems must be all, both, or " + ", ".join(BACKENDS))
    try:
        args.xy = [tuple(map(int, pair.split(":"))) for pair in args.xy]
        args.xy = list(dict.fromkeys(args.xy))
        if any(len(pair) != 2 or not (1 <= pair[0] <= 99 and 0 <= pair[1] <= 100) for pair in args.xy):
            raise ValueError()
    except ValueError:
        parser.error("xy must contain X:Y pairs with X in [1,99], Y in [0,100]")
    if not (1 <= args.query_limit_log <= args.insert_limit_log <= 27):
        parser.error("require 1 <= query-limit-log <= insert-limit-log <= 27")
    if not (0 <= args.range_chunk_log <= args.query_limit_log):
        parser.error("range-chunk-log must be 0 (full batch) or positive and no larger than query-limit-log")
    if any(not 1 <= b <= args.insert_limit_log for b in args.batch_logs):
        parser.error("batch logs must be positive and at most insert-limit-log")
    if min(args.repetitions, args.jobs, args.parallel_builds, args.timeout, args.flix_rounds) < 1 or min(args.warmups, args.stop_after_r) < 0:
        parser.error("counts must be positive; warmups/stop-after-r may be zero")
    if not (1 <= args.flix_build_log <= 28 and 1 <= args.flix_probe_log <= 28):
        parser.error("FliX input size logs must be in [1,28]")
    if (1 << args.flix_build_log) // (4 * args.flix_rounds) == 0:
        parser.error("FliX configuration produces empty update batches")
    args.output = args.output.resolve(); args.build_root = args.build_root.resolve()
    if args.skip_completed_from is not None:
        args.skip_completed_from = args.skip_completed_from.resolve()
    if args.adopt_warmups_from is not None:
        args.adopt_warmups_from = args.adopt_warmups_from.resolve()
    if bool(args.adopt_warmups_from) != bool(args.adopt_warmup_batch_logs or args.adopt_warmup_cases):
        parser.error("--adopt-warmups-from requires batch logs or case IDs to adopt")
    return args


def cases_for(args):
    cases = []
    if args.family in ("paper", "both"):
        for backend in args.systems:
            for b in args.batch_logs:
                cases.append({"family": "paper", "backend": backend, "batch_log": b, "kind": "main"})
            if not args.skip_bulk:
                cases.append({"family": "paper", "backend": backend,
                              "batch_log": args.insert_limit_log, "kind": "bulk"})
    if args.family in ("flix", "both"):
        for backend in args.systems:
            for x, y in args.xy:
                cases.append({"family": "flix", "backend": backend, "batch_log": 16,
                              "kind": f"x{x}_y{y}", "x": x, "y": y})
    for case in cases:
        case["id"] = f"{case['family']}/{case['kind']}_b{case['batch_log']}/{case['backend']}"
    return cases


def skip_completed_cases(cases, args):
    if args.skip_completed_from is None:
        return cases, []
    previous = args.skip_completed_from
    previous_manifest = previous / "run_manifest.json"
    if not previous_manifest.is_file():
        raise RuntimeError(f"Missing previous suite manifest: {previous_manifest}")
    old_manifest = json.loads(previous_manifest.read_text())
    if old_manifest.get("protocol") != PROTOCOL:
        raise RuntimeError("Previous suite uses a different protocol")
    old_settings = old_manifest["settings"]
    ignored = {"output", "build_root", "plan", "no_plots", "build_only",
               "jobs", "parallel_builds", "timeout", "skip_completed_from",
               "adopt_warmups_from", "adopt_warmup_batch_logs", "adopt_warmup_cases"}
    current_settings = json.loads(json.dumps(
        {k: v for k, v in vars(args).items() if k not in ignored}))
    # Changing the number of repetitions only drops later runs. The old range
    # chunk size is allowed to differ because range cases must be rerun below.
    for key, value in current_settings.items():
        if key in {"repetitions", "warmups", "range_chunk_log"}:
            continue
        if old_settings.get(key) != value:
            raise RuntimeError(f"Previous suite has a different {key} setting")
    if old_settings["repetitions"] < args.repetitions or old_settings["warmups"] < args.warmups:
        raise RuntimeError("Previous suite has too few runs to skip completed cases")
    old_cases = {case["id"]: case for case in old_manifest["cases"]}
    labels = [f"warmup_{r:02}" for r in range(args.warmups)]
    labels += [f"rep_{r:02}" for r in range(args.repetitions)]
    if args.memcheck:
        labels.append("memcheck")
    remaining, skipped = [], []
    for case in cases:
        if old_cases.get(case["id"]) != case:
            raise RuntimeError(f"Previous suite has a different case: {case['id']}")
        completions = []
        for label in labels:
            folder = previous / case["id"] / label
            marker = folder / "completion.json"
            if not marker.is_file():
                break
            try:
                record = json.loads(marker.read_text())
                for artifact, expected in record["artifacts"].items():
                    if sha(folder / artifact) != expected:
                        raise RuntimeError(f"Changed artifact: {folder / artifact}")
                validate_case(folder, case, args, label == "memcheck")
            except (KeyError, ValueError, OSError, RuntimeError) as error:
                print(f"Rerun {case['id']}: {error}", flush=True)
                break
            completions.append({"label": label, "sha256": sha(marker)})
        if len(completions) == len(labels):
            skipped.append({"case": case, "completions": completions})
        else:
            remaining.append(case)
    return remaining, skipped


def adopt_warmup_cases(cases, args):
    if args.adopt_warmups_from is None:
        return cases, [], []
    source = args.adopt_warmups_from
    previous = json.loads((source / "run_manifest.json").read_text())
    if previous.get("protocol") != PROTOCOL:
        raise RuntimeError("Adopted warmups use a different protocol")
    for key in ("family", "mode", "insert_limit_log", "query_limit_log",
                "range_chunk_log", "skip_ranges", "skip_deletions", "stop_after_r"):
        if previous["settings"].get(key) != getattr(args, key):
            raise RuntimeError(f"Adopted warmups use a different {key} setting")
    available = {case["id"] for case in previous["cases"]}
    if set(args.adopt_warmup_cases) - available:
        raise RuntimeError("Requested adopted warmup case is absent from the source suite")
    remaining, adopted, records = [], [], []
    for case in cases:
        selected_by_log = (case["family"] == "paper" and case["kind"] == "main" and
                           case["batch_log"] in args.adopt_warmup_batch_logs)
        if ((not selected_by_log and case["id"] not in args.adopt_warmup_cases) or
                case["id"] not in available):
            remaining.append(case)
            continue
        folder = source / case["id"] / "warmup_00"
        completion = folder / "completion.json"
        if not completion.is_file():
            raise RuntimeError(f"Missing completed warmup: {completion}")
        record = json.loads(completion.read_text())
        for artifact, expected in record["artifacts"].items():
            if sha(folder / artifact) != expected:
                raise RuntimeError(f"Changed adopted warmup artifact: {folder / artifact}")
        validation = validate_case(folder, case, args)
        if validation != record["validation"]:
            raise RuntimeError(f"Changed adopted warmup validation: {folder}")
        adopted.append({"case": case, "completion_sha256": sha(completion)})
        records.append(("warmup_00", case, record))
    if not adopted:
        raise RuntimeError("No completed warmups match the requested batch logs")
    validate_cohort(records)
    return remaining, adopted, records


def build_case(case, args, toolkit, arch):
    settings = argparse.Namespace(build_log=args.flix_build_log, probe_log=args.flix_probe_log,
                                  rounds=args.flix_rounds, x=case.get("x",25), y=case.get("y",90))
    defines = setup.flags(case["backend"], settings)
    defines += f" -DPAPER_LSM_BATCH_LOG={case['batch_log']} -DGPULSMOPT_BATCH_CAPACITY=1048576"
    identity = {"backend": case["backend"], "batch_log": case["batch_log"],
                "family": case["family"], "flags": defines, "arch": arch, "toolkit": str(toolkit)}
    build_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    folder = args.build_root / build_id; folder.mkdir(parents=True, exist_ok=True)
    target = "paper_sweep" if case["family"] == "paper" else "index_prototype"
    setup.command(["cmake", "-S", ROOT / "FliX", "-B", folder,
        f"-DCMAKE_CUDA_COMPILER={toolkit / 'bin/nvcc'}", f"-DCUDAToolkit_ROOT={toolkit}",
        f"-DCUDA_TOOLKIT_ROOT_DIR={toolkit}", f"-DBIN2C={toolkit / 'bin/bin2c'}",
        f"-DCMAKE_CUDA_ARCHITECTURES={arch}", "-DCMAKE_BUILD_TYPE=Release",
        "-DFLIX_USE_KEY_CACHE=OFF", "-DFLIX_ENABLE_PROFILING=OFF",
        "-DFLIX_BENCHMARK_KEY_BITS=31", "-DFLIX_BUILD_COMMON_PAPER_SWEEP=" +
        ("ON" if case["family"] == "paper" else "OFF"), f"-DIFDEFS={defines}"], folder / "configure.log", ROOT)
    setup.command(["cmake", "--build", folder, "--target", target, "--parallel", str(args.jobs)],
                  folder / "build.log", ROOT, max(600, args.timeout))
    print("Built " + case["id"], flush=True)
    return {"binary": str(folder / target), "sha256": sha(folder / target), "flags": defines}


def read_measurements(folder):
    with (folder / "measurements.csv").open() as f:
        return list(csv.DictReader(f))


def validate_case(folder, case, args, sanitized=False):
    if case["family"] == "flix":
        result = setup.verify_output(folder, case["backend"], args.flix_rounds, sanitized)
        return {"input_signature": result["input_signature"], "rows": result["csv_rows"]}
    # setup.command writes the command line before the program's output.
    log = (folder / "run.log").read_text().split("\n", 1)[-1]
    if re.search(r"\bERROR\b(?! SUMMARY: 0)|VALIDATION_FAILURE|mismatch|illegal memory", log, re.I):
        raise RuntimeError("Paper driver reported a failure")
    if sanitized and "ERROR SUMMARY: 0 errors" not in log:
        raise RuntimeError("Missing clean sanitizer completion")
    if not (folder / "complete_common").exists():
        raise RuntimeError("Missing paper completion marker")
    rows = read_measurements(folder)
    if not rows: raise RuntimeError("No paper measurements")
    b = case["batch_log"]; states = 1 << (args.insert_limit_log - b)
    if args.stop_after_r: states = min(states, args.stop_after_r)
    dynamic = case["backend"] in DYNAMIC
    expected_updates = states - 1 if dynamic and case["kind"] == "main" else 0
    expected_deletes = expected_updates if not args.skip_deletions else 0
    counts = {op: sum(r["operation"] == op for r in rows) for op in {r['operation'] for r in rows}}
    if counts.get("insert", 0) != expected_updates or counts.get("delete", 0) != expected_deletes:
        raise RuntimeError(f"Incomplete updates: {counts}")
    expected_builds = 1 if dynamic or case["kind"] == "bulk" else states
    if counts.get("build", 0) + counts.get("bulk_build", 0) != expected_builds:
        raise RuntimeError("Incomplete build states")
    if counts.get("destroy") != 1: raise RuntimeError("Missing destruction measurement")
    query_states = 1 if case["kind"] == "bulk" else sum(
        r * (1 << b) <= (1 << args.query_limit_log) or r == states
        for r in range(1, states + 1))
    after_delete = sum((r - 1) * (1 << b) <= (1 << args.query_limit_log) or r == 2
                       for r in range(2, states + 1)) if expected_deletes else 0
    range_states = (int(args.insert_limit_log <= args.query_limit_log) if case["kind"] == "bulk"
                    else min(states, (1 << args.query_limit_log) // (1 << b)))
    expected_ranges = 2 * range_states if case["backend"] in RANGES and b <= 20 and not args.skip_ranges else 0
    expected_counts = {"lookup": 2 * query_states, "lookup_deleted": expected_deletes,
                       "lookup_after_delete": 2 * after_delete, "range_sum": expected_ranges,
                       "range_sum_after_delete": 2 * min(states - 1, (1 << args.query_limit_log) // (1 << b))
                       if expected_deletes and case["backend"] in RANGES and b <= 20 and not args.skip_ranges else 0}
    capabilities = json.loads((folder / "capabilities.json").read_text())
    if capabilities['dynamic'] != dynamic:
        raise RuntimeError("Update capability differs between adapter and runner")
    if capabilities['range'] != (case['backend'] in RANGES):
        raise RuntimeError("Range capability differs between adapter and runner")
    if capabilities.get('range_timing') != 'complete_unsorted_range_v1':
        raise RuntimeError("Incorrect range timing contract")
    if capabilities['range'] and capabilities.get('range_processing') != 'enumerate_records_sum_v1':
        raise RuntimeError("Range processing must enumerate records before summing")
    if any(counts.get(op, 0) != count for op, count in expected_counts.items()):
        raise RuntimeError(f"Incomplete lookup/range states: {counts}; expected {expected_counts}")
    identities = set(); signature = []
    for row in rows:
        if row["protocol"] != "common_initialized_v1": raise RuntimeError("Incorrect protocol")
        if row["operation"].startswith("range_sum"):
            if row.get("range_processing") != 'enumerate_records_sum_v1':
                raise RuntimeError("Range row is missing the record-enumeration contract")
            items = int(row["items"])
            call_items = min(items, 1 << args.range_chunk_log) if args.range_chunk_log else items
            calls = (items + call_items - 1) // call_items
            if (int(row.get("range_api_calls", -1)) != calls or
                    int(row.get("range_max_call_items", -1)) != call_items):
                raise RuntimeError("Range calls do not match the configured public batch size")
        key = tuple(row[k] for k in ("operation", "state", "resident_elements", "items", "scenario"))
        if key in identities: raise RuntimeError("Duplicate operation row")
        identities.add(key)
        times = [float(row[k]) for k in ("time_ms", "wall_ms", "prepare_ms", "search_ms", "restore_ms")]
        if any(not math.isfinite(v) or v < 0 for v in times) or times[0] <= 0:
            raise RuntimeError("Invalid operation time")
        if not math.isclose(times[0], sum(times[2:]), rel_tol=1e-4, abs_tol=1e-4):
            raise RuntimeError("Incomplete operation timing")
        if row["operation"] != "destroy":
            signature.append({k: row[k] for k in ("operation", "state", "resident_elements", "items",
                "scenario", "input_sum", "input_xor", "checksum_sum", "checksum_xor")})
    return {"rows": len(rows), "counts": counts, "input_signature": signature}


def run_case(case, build, args, toolkit, label, sanitized=False):
    folder = args.output / case["id"] / label
    completion = folder / "completion.json"
    if completion.exists():
        record = json.loads(completion.read_text())
        if record['binary_sha256'] != build['sha256']:
            raise RuntimeError(f"Changed binary for completed run: {folder}; choose a new output directory")
        for file, expected in record['artifacts'].items():
            if sha(folder / file) != expected: raise RuntimeError(f"Changed result artifact: {folder / file}")
        validate_case(folder, case, args, sanitized)
        print("Resume: " + case['id'] + "/" + label, flush=True)
        return record
    if folder.exists():
        archived = folder.with_name(folder.name + f".incomplete-{time.time_ns()}")
        folder.rename(archived)
    folder.mkdir(parents=True)
    command = [build['binary']]
    if case['family'] == 'paper':
        command += ['--output', str(folder), '--insert-limit-log', str(args.insert_limit_log),
                    '--query-limit-log', str(args.query_limit_log), '--range-chunk-log', str(args.range_chunk_log),
                    '--stop-after-r', str(args.stop_after_r)]
        if case['kind'] == 'bulk': command += ['--bulk-only']
        if args.skip_ranges: command += ['--skip-ranges']
        if args.skip_deletions: command += ['--skip-deletions']
    if sanitized:
        command = [str(toolkit / 'bin/compute-sanitizer'), '--tool', 'memcheck', '--error-exitcode', '99'] + command
    started = time.monotonic()
    setup.command(command, folder / 'run.log', folder, args.timeout)
    validation = validate_case(folder, case, args, sanitized)
    record = {'binary_sha256': build['sha256'], 'command': command,
              'process_seconds': time.monotonic()-started, 'validation': validation,
              'artifacts': {p.name: sha(p) for p in folder.iterdir() if p.is_file()}}
    save(completion, record)
    print("Passed: " + case['id'] + '/' + label, flush=True)
    return record


def validate_cohort(records):
    shared = {}; compared = 0
    for identity, case, record in records:
        for entry in record['validation']['input_signature']:
            if case['family']=='flix':
                key = (case['family'], case['kind'], entry['step'])
                values = entry
            else:
                key = ('paper', case['kind'], case['batch_log']) + tuple(entry[k] for k in
                      ('operation','state','resident_elements','items','scenario'))
                values = entry
            if key in shared:
                if shared[key] != values: raise RuntimeError(f"Input or range checksum disagreement: {key}")
                compared += 1
            else: shared[key] = values
    return compared


def main():
    args = parse_args(); all_cases = cases_for(args)
    all_cases, adopted, adopted_records = adopt_warmup_cases(all_cases, args)
    cases, skipped = skip_completed_cases(all_cases, args)
    if not cases: raise RuntimeError("No supported cases selected")
    if args.plan:
        print(json.dumps({'protocol':PROTOCOL, 'cases':cases,
                          'skipped_cases':[item['case']['id'] for item in skipped],
                          'adopted_warmups':[item['case']['id'] for item in adopted],
                          'repetitions':args.repetitions,
                          'range_chunk_log':args.range_chunk_log,
                          'warmups':args.warmups, 'memcheck':args.memcheck},indent=2)); return
    setup.verify_dependencies()
    toolkit = setup.cuda_root(args.cuda_root)
    arch = args.arch or subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=compute_cap',
                  '--format=csv,noheader'],text=True).strip().replace('.','')
    if not re.fullmatch(r'\d+',arch): raise RuntimeError('Invalid architecture')
    gpu = subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=name,uuid,driver_version',
                                  '--format=csv,noheader'],text=True).strip()
    if adopted:
        old_gpu = json.loads((args.adopt_warmups_from / 'run_manifest.json').read_text())['gpu']
        if old_gpu != gpu:
            raise RuntimeError("Adopted warmups were measured on a different GPU or driver")
    manifest = {'protocol':PROTOCOL, 'source_sha256':source_signature(), 'cases':cases,
                'skipped_from':str(args.skip_completed_from) if skipped else None,
                'previous_manifest_sha256':sha(args.skip_completed_from / 'run_manifest.json') if skipped else None,
                'skipped_cases':skipped,
                'adopted_warmups':adopted,
                'adopted_from':str(args.adopt_warmups_from) if adopted else None,
                'adopted_manifest_sha256':sha(args.adopt_warmups_from / 'run_manifest.json') if adopted else None,
                'toolkit':str(toolkit), 'arch':arch, 'gpu':gpu,
                'settings':{k:v for k,v in vars(args).items() if k not in
                    {'output','build_root','plan','no_plots','build_only','jobs','parallel_builds','timeout',
                     'skip_completed_from','adopt_warmups_from','adopt_warmup_batch_logs',
                     'adopt_warmup_cases'}}}
    manifest = json.loads(json.dumps(manifest))
    args.output.mkdir(parents=True,exist_ok=True)
    path = args.output / 'run_manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError('Output belongs to different source/configuration. Use a new --output directory.')
    elif any(args.output.iterdir()):
        raise RuntimeError('Nonempty output directory has no suite manifest; historical results will not be overwritten.')
    else: save(path,manifest)
    builds = {}; build_groups = {}
    for case in cases:
        key = tuple(case.get(k) for k in ('family', 'backend', 'batch_log', 'x', 'y'))
        build_groups.setdefault(key, []).append(case)
    with ThreadPoolExecutor(max_workers=args.parallel_builds) as pool:
        futures = {pool.submit(build_case, group[0], args, toolkit, arch): group
                   for group in build_groups.values()}
        for future in as_completed(futures):
            build = future.result()
            for case in futures[future]: builds[case['id']] = build
    save(args.output / 'builds.json',builds)
    if args.build_only: return
    records = []
    previous_records = []
    for item in skipped:
        for completion in item['completions']:
            folder = args.skip_completed_from / item['case']['id'] / completion['label']
            previous_records.append((completion['label'], item['case'],
                                     json.loads((folder / 'completion.json').read_text())))
    labels = [(f'warmup_{r:02}',False) for r in range(args.warmups)]
    labels += [(f'rep_{r:02}',False) for r in range(args.repetitions)]
    if args.memcheck: labels.append(('memcheck',True))
    for iteration,(label,sanitized) in enumerate(labels):
        ordered = cases[iteration % len(cases):] + cases[:iteration % len(cases)]
        for case in ordered:
            record = run_case(case, builds[case['id']], args, toolkit, label, sanitized)
            records.append((label,case,record))
        compared = validate_cohort(records)
        save(args.output / 'validation.json', {'state':'partial','completed_runs':len(records),
                                              'matching_backend_and_repetition_rows':compared})
    compared = validate_cohort(records + previous_records + adopted_records)
    save(args.output / 'validation.json', {'state':'passed',
                                          'completed_runs':len(records) + len(previous_records) + len(adopted_records),
                                          'reused_runs':len(previous_records),
                                          'adopted_single_samples':len(adopted_records),
                                          'matching_backend_and_repetition_rows':compared})
    from flix_paper_reporting import summarize
    summarize(args.output, no_plots=args.no_plots)
    print('Paper suite complete: ' + str(args.output),flush=True)


if __name__=='__main__':
    try: main()
    except (RuntimeError,ValueError,OSError,subprocess.SubprocessError) as error:
        print('ERROR: '+str(error),file=sys.stderr); raise SystemExit(1)
