#!/usr/bin/env python3
"""Run independent experiment commands in order and continue after failures.

The queue manifest is JSON with this shape:

{
  "experiments": [
    {"id": "example", "repetitions": 2,
     "command": ["python3", "example.py"]}
  ]
}

Each command receives its own output directory through the environment variable
CELLLSM_EXPERIMENT_OUTPUT. A failed command is recorded and the next experiment
still runs. The queue itself exits unsuccessfully when any experiment failed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
MAX_NEW_EXPERIMENT_REPETITIONS = 2


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_manifest(manifest: dict) -> list[dict]:
    experiments = manifest.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("manifest must contain a nonempty experiments list")
    seen: set[str] = set()
    for experiment in experiments:
        identifier = experiment.get("id")
        command = experiment.get("command")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("every experiment needs a nonempty id")
        if identifier in seen:
            raise ValueError(f"duplicate experiment id: {identifier}")
        seen.add(identifier)
        if not isinstance(command, list) or not command or not all(
                isinstance(item, str) and item for item in command):
            raise ValueError(f"{identifier} needs a nonempty command array")
        if experiment.get("measured", True):
            repetitions = experiment.get("repetitions")
            if not isinstance(repetitions, int) or not (
                    1 <= repetitions <= MAX_NEW_EXPERIMENT_REPETITIONS):
                raise ValueError(
                    f"{identifier} must request one or two measured repetitions")
        validation_file = experiment.get("validation_file")
        if validation_file is not None and not isinstance(validation_file, str):
            raise ValueError(f"{identifier} validation_file must be a string")
    return experiments


def expand_command(command: list[str], output: Path, queue_root: Path) -> list[str]:
    replacements = {
        "{output}": str(output),
        "{queue_root}": str(queue_root),
        "{repo}": str(ROOT),
    }
    expanded = []
    for item in command:
        for placeholder, value in replacements.items():
            item = item.replace(placeholder, value)
        expanded.append(item)
    return expanded


def process_matches(pid: int, expected: str) -> bool:
    path = Path("/proc") / str(pid) / "cmdline"
    try:
        command = path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except FileNotFoundError:
        return False
    # A completed process ID can be reused by the operating system. In that
    # case the prerequisite process is no longer running; the success record
    # below, rather than the reused PID, decides whether the queue may start.
    return expected in command


def wait_for_prerequisite(manifest: dict, status_path: Path) -> dict:
    wait = manifest.get("wait_for_process")
    if not wait:
        return {"state": "not_requested"}
    pid = int(wait["pid"])
    expected = str(wait["command_contains"])
    interval = max(1, int(wait.get("poll_seconds", 30)))
    record = {
        "state": "waiting",
        "pid": pid,
        "command_contains": expected,
        "started_utc": utc_now(),
    }
    write_json(status_path, {"state": "waiting_for_prerequisite", "wait": record})
    while process_matches(pid, expected):
        time.sleep(interval)
    record["state"] = "finished"
    record["finished_utc"] = utc_now()
    success_file = wait.get("success_file")
    if success_file:
        path = (ROOT / success_file).resolve()
        record["success_file"] = str(path)
        record["success_file_present"] = path.is_file()
        if path.is_file():
            record["success_file_sha256"] = sha256(path)
            try:
                record["success_record"] = json.loads(path.read_text())
            except json.JSONDecodeError:
                record["success_record"] = None
        required_state = wait.get("require_state")
        if required_state is not None:
            record["required_state"] = required_state
            record["requirement_met"] = (
                (record.get("success_record") or {}).get("state") == required_state)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    output = args.output.resolve()
    manifest = json.loads(manifest_path.read_text())
    experiments = validate_manifest(manifest)
    if args.dry_run:
        print(json.dumps({"manifest": str(manifest_path), "experiments": experiments},
                         indent=2))
        return 0

    output.mkdir(parents=True, exist_ok=False)
    frozen_manifest = output / "queue_manifest.json"
    frozen_manifest.write_bytes(manifest_path.read_bytes())
    source_hashes = {}
    for name in manifest.get("source_files", []):
        path = (ROOT / name).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"queue source file is missing: {path}")
        source_hashes[str(path.relative_to(ROOT))] = sha256(path)
    write_json(output / "queue_source_hashes.json", source_hashes)
    queue_status = output / "status.json"
    prerequisite = wait_for_prerequisite(manifest, queue_status)
    if prerequisite.get("requirement_met") is False:
        write_json(queue_status, {
            "state": "failed_prerequisite",
            "updated_utc": utc_now(),
            "prerequisite": prerequisite,
            "experiments": [],
        })
        return 1

    results: list[dict] = []
    active: subprocess.Popen | None = None

    def stop_active(signum: int, _frame: object) -> None:
        if active is not None and active.poll() is None:
            active.terminate()
        write_json(queue_status, {
            "state": "interrupted",
            "signal": signum,
            "updated_utc": utc_now(),
            "prerequisite": prerequisite,
            "experiments": results,
        })
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, stop_active)
    signal.signal(signal.SIGTERM, stop_active)

    for position, experiment in enumerate(experiments, start=1):
        identifier = experiment["id"]
        directory = output / f"{position:02d}_{identifier}"
        directory.mkdir()
        command = expand_command(experiment["command"], directory, output)
        record = {
            "id": identifier,
            "position": position,
            "state": "running",
            "command": command,
            "started_utc": utc_now(),
            "output": str(directory),
        }
        results.append(record)
        write_json(queue_status, {
            "state": "running",
            "current": identifier,
            "updated_utc": utc_now(),
            "prerequisite": prerequisite,
            "experiments": results,
        })
        changed_sources = [name for name, expected in source_hashes.items()
                           if not (ROOT / name).is_file() or
                           sha256(ROOT / name) != expected]
        if changed_sources:
            record["state"] = "failed"
            record["error"] = "queue source changed after launch"
            record["changed_sources"] = changed_sources
            record["finished_utc"] = utc_now()
            write_json(directory / "completion.json", record)
            continue
        environment = dict(os.environ)
        environment["CELLLSM_EXPERIMENT_OUTPUT"] = str(directory)
        environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
        with (directory / "run.log").open("w") as log:
            try:
                active = subprocess.Popen(command, cwd=ROOT, env=environment,
                                          stdout=log, stderr=subprocess.STDOUT)
                return_code = active.wait()
                record["return_code"] = return_code
                record["state"] = "passed" if return_code == 0 else "failed"
                validation_name = experiment.get("validation_file")
                if return_code == 0 and validation_name:
                    validation_path = Path(expand_command(
                        [validation_name], directory, output)[0])
                    record["validation_file"] = str(validation_path)
                    if not validation_path.is_file():
                        record["state"] = "failed"
                        record["error"] = "required validation file is missing"
                    else:
                        try:
                            validation = json.loads(validation_path.read_text())
                            record["validation_sha256"] = sha256(validation_path)
                            record["validation_state"] = validation.get("state")
                            expected = experiment.get("require_validation_state", "passed")
                            if validation.get("state") != expected:
                                record["state"] = "failed"
                                record["error"] = (
                                    "validation state does not match " + repr(expected))
                        except (OSError, json.JSONDecodeError) as error:
                            record["state"] = "failed"
                            record["error"] = f"invalid validation file: {error!r}"
            except BaseException as error:
                record["state"] = "failed"
                record["error"] = repr(error)
            finally:
                active = None
                record["finished_utc"] = utc_now()
                write_json(directory / "completion.json", record)
        # Deliberately continue: experiments are independent evidence.

    failures = [record["id"] for record in results if record["state"] != "passed"]
    final = {
        "state": "passed" if not failures else "complete_with_failures",
        "updated_utc": utc_now(),
        "prerequisite": prerequisite,
        "failed_experiments": failures,
        "experiments": results,
    }
    write_json(queue_status, final)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
