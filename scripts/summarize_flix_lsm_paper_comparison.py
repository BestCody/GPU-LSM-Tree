#!/usr/bin/env python3
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SYSTEM_COLORS = {"GPULSMOpt": "#1772b4", "LSMu": "#d95f02"}


def load_csvs(pattern):
    paths = sorted(glob.glob(pattern, recursive=True))
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_operation(result_dir, prefix):
    frames = []
    for directory, fallback in (("gpulsmopt", "GPULSMOpt"),
                                ("lsmu", "LSMu")):
        frame = load_csvs(os.path.join(
            result_dir, directory, f"{prefix}_b*.csv"))
        if frame.empty:
            continue
        if "system" not in frame.columns:
            frame.insert(0, "system", fallback)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_range(result_dir):
    frames = []
    for directory, fallback in (("gpulsmopt", "GPULSMOpt"),
                                ("lsmu", "LSMu")):
        base = os.path.join(result_dir, directory)
        frame = load_csvs(os.path.join(base, "range_b*.csv"))
        if frame.empty:
            frame = load_csvs(os.path.join(base, "count_range_b*.csv"))
        if frame.empty:
            continue
        frame = frame[frame["operation"] == "range"].copy()
        if "system" not in frame.columns:
            frame.insert(0, "system", fallback)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def validate_range_checksums(frame):
    if frame.empty:
        return 0
    required = {"checksum_sum", "checksum_xor"}
    if not required.issubset(frame.columns):
        raise RuntimeError("range results do not contain validation checksums")
    systems = set(frame["system"])
    if not {"GPULSMOpt", "LSMu"}.issubset(systems):
        return 0
    keys = [
        "batch_log", "r", "resident_elements",
        "expected_hits", "chunk_size",
    ]
    gpulsmopt = frame[frame["system"] == "GPULSMOpt"]
    lsmu = frame[frame["system"] == "LSMu"]
    paired = gpulsmopt.merge(
        lsmu, on=keys, suffixes=("_gpulsmopt", "_lsmu"),
        how="outer", indicator=True, validate="one_to_one")
    unpaired = paired[paired["_merge"] != "both"]
    if not unpaired.empty:
        first = unpaired.iloc[0]
        identity = ", ".join(f"{key}={first[key]}" for key in keys)
        raise RuntimeError(
            f"unpaired range result ({first['_merge']}): {identity}")
    mismatched = paired[
        (paired["checksum_sum_gpulsmopt"] != paired["checksum_sum_lsmu"]) |
        (paired["checksum_xor_gpulsmopt"] != paired["checksum_xor_lsmu"])
    ]
    if not mismatched.empty:
        first = mismatched.iloc[0]
        identity = ", ".join(f"{key}={first[key]}" for key in keys)
        raise RuntimeError(f"range checksum mismatch: {identity}")
    return len(paired)


def harmonic_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    return float(len(values) / np.sum(1.0 / values)) if len(values) else np.nan


def aggregate(frame, groups):
    rows = []
    for keys, group in frame.groupby(groups, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rates = group["rate_mops"].to_numpy(dtype=float)
        row = dict(zip(groups, keys))
        row.update({
            "minimum_rate_mops": float(np.min(rates)),
            "maximum_rate_mops": float(np.max(rates)),
            "harmonic_mean_rate_mops": harmonic_mean(rates),
            "states": int(len(rates)),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def graph_insertion(summary, output):
    if summary.empty:
        return
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for system, group in summary.groupby("system"):
        group = group.sort_values("batch_log")
        color = SYSTEM_COLORS.get(system)
        axis.fill_between(
            group["batch_log"], group["minimum_rate_mops"],
            group["maximum_rate_mops"], color=color, alpha=0.14)
        axis.plot(
            group["batch_log"], group["harmonic_mean_rate_mops"],
            "o-", color=color, label=system)
    axis.set_title("Batch insertion throughput (paper Table II metric)")
    axis.set_xlabel("log2(external batch size)")
    axis.set_ylabel("M records/s")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def graph_lookup(summary, output):
    if summary.empty:
        return
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    scenarios = ("all_existing", "none_existing")
    for axis, scenario in zip(axes, scenarios):
        selected = summary[summary["scenario"] == scenario]
        for system, group in selected.groupby("system"):
            group = group.sort_values("batch_log")
            axis.fill_between(
                group["batch_log"], group["minimum_rate_mops"],
                group["maximum_rate_mops"],
                color=SYSTEM_COLORS.get(system), alpha=0.14)
            axis.plot(
                group["batch_log"], group["harmonic_mean_rate_mops"],
                "o-", color=SYSTEM_COLORS.get(system), label=system)
        axis.set_title(scenario.replace("_", " "))
        axis.set_xlabel("log2(external batch size)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("M queries/s")
    axes[0].legend()
    figure.suptitle("Point lookup throughput (paper Table III metric)")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def graph_range(summary, output):
    if summary.empty:
        return
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, expected_hits in zip(axes, (8, 1024)):
        selected = summary[summary["expected_hits"] == expected_hits]
        for system, group in selected.groupby("system"):
            group = group.sort_values("batch_log")
            axis.fill_between(
                group["batch_log"], group["minimum_rate_mops"],
                group["maximum_rate_mops"],
                color=SYSTEM_COLORS.get(system), alpha=0.14)
            axis.plot(
                group["batch_log"], group["harmonic_mean_rate_mops"],
                "o-", color=SYSTEM_COLORS.get(system), label=system)
        axis.set_title(f"L={expected_hits}")
        axis.set_ylabel("M queries/s")
        axis.set_xlabel("log2(external batch size)")
        axis.grid(alpha=0.25)
    axes[0].legend()
    figure.suptitle("Range-enumeration throughput (paper Table IV metric)")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def graph_effective_insertion(insertion, output):
    selected = insertion[insertion["batch_log"].isin([17, 18, 19, 20])]
    if selected.empty:
        return
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=False)
    for axis, batch_log in zip(axes.flat, (17, 18, 19, 20)):
        batch = selected[selected["batch_log"] == batch_log]
        for system, group in batch.groupby("system"):
            group = group.sort_values("resident_elements")
            stride = max(1, len(group) // 256)
            sample = group.iloc[::stride]
            if not sample.empty and sample.index[-1] != group.index[-1]:
                sample = pd.concat([sample, group.tail(1)])
            axis.plot(
                sample["resident_elements"] / 1e6,
                sample["effective_rate_mops"],
                color=SYSTEM_COLORS.get(system), label=system)
        axis.set_title(f"b=2^{batch_log}")
        axis.set_xlabel("Resident records (millions)")
        axis.set_ylabel("M records/s")
        axis.grid(alpha=0.25)
    handles = []
    labels = []
    for axis in axes.flat:
        current_handles, current_labels = axis.get_legend_handles_labels()
        for handle, label in zip(current_handles, current_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        figure.legend(handles, labels, loc="upper right")
    figure.suptitle("Cumulative effective insertion rate (paper Figure 4b)")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def graph_batch_latency(insertion, output):
    selected = insertion[insertion["batch_log"] == 19]
    if selected.empty:
        return
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for system, group in selected.groupby("system"):
        group = group.sort_values("r")
        axis.plot(
            group["r"], group["time_ms"], linewidth=1,
            color=SYSTEM_COLORS.get(system), label=system)
    axis.set_title("Per-batch insertion latency, b=2^19 (paper Figure 4a)")
    axis.set_xlabel("Submitted batches")
    axis.set_ylabel("Milliseconds")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def graph_bulk(bulk, output):
    if bulk.empty:
        return
    figure, axis = plt.subplots(figsize=(7, 5))
    colors = [SYSTEM_COLORS.get(value, "#777777") for value in bulk["system"]]
    axis.bar(bulk["system"], bulk["rate_mops"], color=colors)
    axis.set_title("Bulk-build throughput")
    axis.set_ylabel("M records/s")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_markdown_table(output, frame, columns):
    output.write("| " + " | ".join(columns) + " |\n")
    output.write("|" + "|".join("---" for _ in columns) + "|\n")
    for row in frame.itertuples(index=False):
        values = []
        mapping = row._asdict()
        for column in columns:
            value = mapping[column]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        output.write("| " + " | ".join(values) + " |\n")


def main():
    if len(sys.argv) not in (2, 3) or (
            len(sys.argv) == 3 and sys.argv[2] != "--validate-only"):
        raise SystemExit(
            "usage: summarize_flix_lsm_paper_comparison.py "
            "RESULT_DIR [--validate-only]")
    result_dir = os.path.abspath(sys.argv[1])
    validate_only = len(sys.argv) == 3
    summary_dir = os.path.join(result_dir, "summary")
    graph_dir = os.path.join(result_dir, "graphs")
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    insertion = load_operation(result_dir, "insertion")
    lookup = load_operation(result_dir, "lookup")
    range_results = load_range(result_dir)
    validated_ranges = validate_range_checksums(range_results)
    if validate_only:
        print(f"Validated {validated_ranges} paired range states")
        return
    bulk = load_csvs(os.path.join(result_dir, "*", "bulk_build.csv"))
    cleanup = load_csvs(os.path.join(
        result_dir, "lsmu", "cleanup_b*", "cleanup.csv"))

    insertion_summary = aggregate(insertion, ["system", "batch_log"])
    lookup_summary = aggregate(
        lookup, ["system", "batch_log", "scenario"])
    range_summary = aggregate(
        range_results, ["system", "batch_log", "expected_hits"])

    insertion_summary.to_csv(
        os.path.join(summary_dir, "batch_insertion.csv"), index=False)
    lookup_summary.to_csv(
        os.path.join(summary_dir, "lookup.csv"), index=False)
    range_summary.to_csv(
        os.path.join(summary_dir, "range.csv"), index=False)
    if not bulk.empty:
        bulk.to_csv(os.path.join(summary_dir, "bulk_build.csv"), index=False)
    if not cleanup.empty:
        cleanup.to_csv(os.path.join(summary_dir, "cleanup_lsmu.csv"), index=False)

    graph_bulk(
        bulk,
        os.path.join(graph_dir, "bulk_build_throughput_n2p27.png"))
    graph_insertion(
        insertion_summary,
        os.path.join(
            graph_dir, "batch_insertion_throughput_by_batch_size.png"))
    graph_effective_insertion(
        insertion,
        os.path.join(
            graph_dir, "cumulative_effective_insertion_throughput.png"))
    graph_batch_latency(
        insertion,
        os.path.join(
            graph_dir, "per_batch_insertion_latency_b2p19.png"))
    graph_lookup(
        lookup_summary,
        os.path.join(
            graph_dir, "point_lookup_throughput_hits_vs_misses.png"))
    graph_range(
        range_summary,
        os.path.join(
            graph_dir, "range_enumeration_throughput_l8_vs_l1024.png"))

    report_path = os.path.join(result_dir, "REPORT.md")
    with open(report_path, "w", encoding="utf-8") as report:
        report.write("# FliX GPULSMOpt versus LSMu paper-protocol sweep\n\n")
        report.write(
            "All primary rates use the metrics from GPU LSM "
            "(1707.05354v2): min/max/harmonic-mean throughput across "
            "resident states, cumulative effective insertion rate, and "
            "bulk-build throughput. Each state has one timed sample.\n\n")
        report.write(
            "Both systems receive identical generated keys, probes, range "
            "bounds, and query chunk sizes. GPULSMOpt retains a fixed 2^20 "
            "maximum admission tile when external batches are larger. Its "
            "level-zero capacity is 16 times the smaller of the fixed "
            "external batch size and that admission tile.\n\n")
        report.write(
            "RANGE is enumeration with checksum for both systems, not "
            "materialized output. COUNT is omitted because GPULSMOpt does "
            f"not expose a count operation. {validated_ranges} paired "
            "range states passed two independent 64-bit result checksums. "
            "Cleanup "
            "is reported only for LSMu because GPULSMOpt has no equivalent "
            "explicit cleanup operation.\n\n")
        if not bulk.empty:
            report.write("## Bulk build\n\n")
            write_markdown_table(
                report, bulk,
                ["system", "elements", "gpu_time_ms", "rate_mops",
                 "gpu_resident_bytes"])
            report.write("\n")
        if not insertion_summary.empty:
            report.write("## Batch insertion summary\n\n")
            write_markdown_table(
                report, insertion_summary,
                ["system", "batch_log", "minimum_rate_mops",
                 "maximum_rate_mops", "harmonic_mean_rate_mops", "states"])
            report.write("\n")
        if not lookup_summary.empty:
            report.write("## Lookup summary\n\n")
            write_markdown_table(
                report, lookup_summary,
                ["system", "batch_log", "scenario",
                 "minimum_rate_mops", "maximum_rate_mops",
                 "harmonic_mean_rate_mops", "states"])
            report.write("\n")
        if not range_summary.empty:
            report.write("## Range summary\n\n")
            write_markdown_table(
                report, range_summary,
                ["system", "batch_log", "expected_hits",
                 "minimum_rate_mops", "maximum_rate_mops",
                 "harmonic_mean_rate_mops", "states"])
            report.write("\n")
        if not cleanup.empty:
            report.write("## LSMu cleanup\n\n")
            write_markdown_table(
                report, cleanup,
                ["system", "batch_log", "resident_batches",
                 "resident_elements", "stale_percent", "survivors",
                 "cleanup_wall_ms", "cleanup_rate_mops", "query_count",
                 "before_lookup_ms", "after_lookup_ms", "speedup",
                 "amortized_speedup"])
            report.write("\n")
        report.write("## Figures\n\n")
        figures = (
            (
                "bulk_build_throughput_n2p27.png",
                "Bulk-build throughput at N = 2^27",
                "GPULSMOpt and LSMu bulk-build throughput for 2^27 records",
            ),
            (
                "batch_insertion_throughput_by_batch_size.png",
                "Batch-insertion throughput by external batch size",
                "Minimum, maximum, and harmonic-mean insertion throughput "
                "across resident states",
            ),
            (
                "cumulative_effective_insertion_throughput.png",
                "Cumulative effective insertion throughput",
                "Cumulative insertion throughput versus resident records "
                "for b in {2^17, 2^18, 2^19, 2^20}",
            ),
            (
                "per_batch_insertion_latency_b2p19.png",
                "Per-batch insertion latency for b = 2^19",
                "Insertion latency at every resident state for external "
                "batch size 2^19",
            ),
            (
                "point_lookup_throughput_hits_vs_misses.png",
                "Point-lookup throughput: all hits versus all misses",
                "Point-lookup throughput across resident states, separated "
                "into all-existing and none-existing probes",
            ),
            (
                "range_enumeration_throughput_l8_vs_l1024.png",
                "Range-enumeration throughput: L = 8 versus L = 1024",
                "Range-enumeration-with-checksum throughput across resident "
                "states for expected result lengths 8 and 1024",
            ),
        )
        for name, heading, description in figures:
            path = os.path.join(graph_dir, name)
            if not os.path.exists(path):
                continue
            image_path = os.path.abspath(path)
            report.write(
                f"### {heading}\n\n"
                f"![{description}]({image_path})\n\n"
                f"{description}.\n\n")
    print("Wrote", report_path)


if __name__ == "__main__":
    main()
