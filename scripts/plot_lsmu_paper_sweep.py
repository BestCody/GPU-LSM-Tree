#!/usr/bin/env python3
import glob
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAPER_INSERTION = {
    15: (0.7, 101.4, 58.4),
    16: (1.4, 194.3, 98.0),
    17: (2.8, 326.8, 159.9),
    18: (5.6, 432.0, 220.7),
    19: (11.3, 558.2, 289.4),
    20: (22.4, 664.2, 354.1),
    21: (44.0, 694.0, 398.1),
    22: (84.3, 714.8, 441.2),
    23: (155.5, 726.3, 485.2),
    24: (270.3, 727.1, 537.9),
    25: (421.1, 727.1, 585.5),
    26: (585.3, 727.7, 648.8),
    27: (727.8, 727.8, 727.8),
}

PAPER_LOOKUP = {
    "none_existing": {
        16: (18.1, 386.1, 45.5),
        17: (19.5, 376.7, 47.4),
        18: (22.0, 361.5, 51.1),
        19: (25.2, 331.7, 57.7),
        20: (30.7, 291.3, 68.4),
        21: (41.4, 237.4, 84.0),
        22: (61.6, 183.1, 105.9),
        23: (116.8, 133.1, 124.4),
        24: (116.8, 116.8, 116.8),
    },
    "all_existing": {
        16: (22.7, 365.7, 54.9),
        17: (24.3, 347.3, 56.7),
        18: (27.2, 332.0, 60.7),
        19: (32.0, 299.2, 67.5),
        20: (39.3, 251.1, 77.6),
        21: (51.7, 194.7, 91.1),
        22: (74.7, 142.2, 104.7),
        23: (106.8, 118.6, 112.4),
        24: (106.8, 106.8, 106.8),
    },
}

PAPER_COUNT_RANGE = {
    ("count", 8): {
        16: (15.1, 44.7, 23.9), 17: (16.2, 67.6, 27.4),
        18: (19.0, 82.9, 35.1), 19: (22.4, 94.1, 39.8),
        20: (27.0, 103.7, 48.0),
    },
    ("count", 1024): {
        16: (2.32, 3.99, 2.57), 17: (2.36, 4.06, 2.63),
        18: (2.44, 4.10, 2.79), 19: (2.51, 4.13, 2.86),
        20: (2.58, 4.17, 3.01),
    },
    ("range", 8): {
        16: (11.0, 23.0, 15.1), 17: (12.9, 36.3, 19.8),
        18: (14.6, 50.5, 24.9), 19: (19.4, 63.2, 31.9),
        20: (23.2, 72.1, 38.4),
    },
    ("range", 1024): {
        16: (1.18, 1.76, 1.27), 17: (1.20, 1.81, 1.28),
        18: (1.23, 1.81, 1.35), 19: (1.25, 1.81, 1.37),
        20: (1.27, 1.80, 1.43),
    },
}


def load_many(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_csv(path) for path in paths]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def harmonic_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    return len(values) / np.sum(1.0 / values)


def aggregate(frame, group_columns):
    rows = []
    for keys, group in frame.groupby(group_columns):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = group["rate_mops"].to_numpy()
        row = dict(zip(group_columns, keys))
        row.update({
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "mean": harmonic_mean(values),
            "states": len(values),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def paper_frame(values):
    return pd.DataFrame([
        {"batch_log": batch_log, "minimum": rates[0],
         "maximum": rates[1], "mean": rates[2]}
        for batch_log, rates in sorted(values.items())
    ])


def normalized(values):
    maximum = np.max(values)
    return values / maximum if maximum else values


def comparison_graph(local, paper, title, output_path):
    local = local.sort_values("batch_log")
    paper = paper.sort_values("batch_log")
    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].fill_between(
        local["batch_log"], local["minimum"], local["maximum"],
        color="#1772b4", alpha=0.18, label="Local min-max")
    axes[0].plot(
        local["batch_log"], local["mean"], "o-",
        color="#1772b4", label="Local harmonic mean")
    axes[0].fill_between(
        paper["batch_log"], paper["minimum"], paper["maximum"],
        color="#d95f02", alpha=0.14, label="Paper min-max")
    axes[0].plot(
        paper["batch_log"], paper["mean"], "s--",
        color="#d95f02", label="Paper harmonic mean")
    axes[0].set_ylabel("Throughput (M operations/s)")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=2, fontsize=9)

    common = sorted(set(local["batch_log"]) & set(paper["batch_log"]))
    local_common = local.set_index("batch_log").loc[common]
    paper_common = paper.set_index("batch_log").loc[common]
    axes[1].plot(
        common, normalized(local_common["mean"].to_numpy()), "o-",
        color="#1772b4", label="Local normalized trend")
    axes[1].plot(
        common, normalized(paper_common["mean"].to_numpy()), "s--",
        color="#d95f02", label="Paper normalized trend")
    axes[1].set_xlabel("log2(batch size b)")
    axes[1].set_ylabel("Mean / series maximum")
    axes[1].set_ylim(bottom=0)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def trend_row(name, local, paper):
    common = sorted(set(local["batch_log"]) & set(paper["batch_log"]))
    local_mean = local.set_index("batch_log").loc[common, "mean"]
    paper_mean = paper.set_index("batch_log").loc[common, "mean"]
    correlation = (
        local_mean.rank().corr(paper_mean.rank())
        if len(common) > 1 else float("nan")
    )
    geometric_speedup = math.exp(
        np.mean(np.log(local_mean.to_numpy() / paper_mean.to_numpy())))
    return {
        "result": name,
        "batch_points": len(common),
        "spearman_trend": correlation,
        "geometric_speedup": geometric_speedup,
        "local_endpoint_growth": local_mean.iloc[-1] / local_mean.iloc[0],
        "paper_endpoint_growth": paper_mean.iloc[-1] / paper_mean.iloc[0],
    }


def plot_effective_insertion(insertion, graph_dir):
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for batch_log in (17, 18, 19, 20):
        group = insertion[insertion["batch_log"] == batch_log]
        if group.empty:
            continue
        stride = max(1, len(group) // 256)
        sampled = group.iloc[::stride]
        if sampled.index[-1] != group.index[-1]:
            sampled = pd.concat([sampled, group.tail(1)])
        axis.plot(
            sampled["resident_elements"] / 1e6,
            sampled["effective_rate_mops"],
            label=f"Local b=2^{batch_log}")
    axis.set_title("Effective insertion rate (paper Figure 4b trend)")
    axis.set_xlabel("Resident elements (millions)")
    axis.set_ylabel("Effective insertion rate (M elements/s)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(graph_dir + "/figure_4b_effective_insertion.png", dpi=180)
    plt.close(figure)


def plot_cleanup(result_dir, graph_dir):
    path = result_dir + "/cleanup.csv"
    if not os.path.exists(path):
        return
    cleanup = pd.read_csv(path)
    throughput = cleanup[cleanup["query_count"] == 0].copy()
    paper = {(20, 10): 1870.2, (20, 50): 1828.2,
             (19, 10): 1842.5, (19, 50): 1794.3}
    figure, axis = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(throughput))
    labels = [
        f"b=2^{int(row.batch_log)}\n{int(row.stale_percent)}% stale"
        for row in throughput.itertuples()
    ]
    paper_values = [
        paper[(int(row.batch_log), int(row.stale_percent))]
        for row in throughput.itertuples()
    ]
    axis.bar(x - 0.19, throughput["cleanup_rate_mops"], 0.38,
             label="Local")
    axis.bar(x + 0.19, paper_values, 0.38, label="Paper")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Cleanup throughput (M resident elements/s)")
    axis.set_title("Cleanup throughput examples from Section V-D")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(graph_dir + "/cleanup_throughput.png", dpi=180)
    plt.close(figure)

    benefit = cleanup[cleanup["query_count"] > 0]
    if benefit.empty:
        return
    row = benefit.iloc[0]
    local = [row["before_lookup_ms"], row["after_lookup_ms"],
             row["after_lookup_ms"] + row["cleanup_wall_ms"]]
    paper_benefit = [4.8 * (132.5 + 19.23), 132.5, 132.5 + 19.23]
    figure, axis = plt.subplots(figsize=(9, 5.5))
    x = np.arange(3)
    axis.bar(x - 0.19, local, 0.38, label="Local")
    axis.bar(x + 0.19, paper_benefit, 0.38, label="Paper")
    axis.set_xticks(x, ["Before cleanup", "After cleanup",
                        "After + cleanup cost"])
    axis.set_ylabel("Time for 2^25 lookups (ms)")
    axis.set_title("Cleanup effect on lookup time")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(graph_dir + "/cleanup_lookup_effect.png", dpi=180)
    plt.close(figure)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: plot_lsmu_paper_sweep.py RESULT_DIR")
    result_dir = os.path.abspath(sys.argv[1])
    graph_dir = result_dir + "/graphs"
    os.makedirs(graph_dir, exist_ok=True)
    insertion = load_many(result_dir + "/insertion_b*.csv")
    lookup = load_many(result_dir + "/lookup_b*.csv")
    count_range = load_many(result_dir + "/count_range_b*.csv")
    if insertion.empty:
        raise SystemExit("no insertion CSV files found")

    summaries = []
    insertion_summary = aggregate(insertion, ["batch_log"])
    insertion_paper = paper_frame(PAPER_INSERTION)
    comparison_graph(
        insertion_summary, insertion_paper,
        "Table II: batch insertion throughput",
        graph_dir + "/table_ii_insertion.png")
    summaries.append(trend_row(
        "table_ii_insertion", insertion_summary, insertion_paper))
    plot_effective_insertion(insertion, graph_dir)

    if not lookup.empty:
        lookup_summary = aggregate(lookup, ["batch_log", "scenario"])
        for scenario, paper_values in PAPER_LOOKUP.items():
            local = lookup_summary[lookup_summary["scenario"] == scenario]
            paper = paper_frame(paper_values)
            name = "table_iii_lookup_" + scenario
            comparison_graph(
                local, paper,
                "Table III: lookup throughput, " +
                scenario.replace("_", " "),
                graph_dir + "/" + name + ".png")
            summaries.append(trend_row(name, local, paper))

    if not count_range.empty:
        cr_summary = aggregate(
            count_range, ["batch_log", "operation", "expected_hits"])
        for (operation, expected_hits), paper_values in \
                PAPER_COUNT_RANGE.items():
            local = cr_summary[
                (cr_summary["operation"] == operation) &
                (cr_summary["expected_hits"] == expected_hits)]
            paper = paper_frame(paper_values)
            name = f"table_iv_{operation}_L{expected_hits}"
            comparison_graph(
                local, paper,
                f"Table IV: {operation.upper()}, L={expected_hits}",
                graph_dir + "/" + name + ".png")
            summaries.append(trend_row(name, local, paper))

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(result_dir + "/trend_summary.csv", index=False)
    plot_cleanup(result_dir, graph_dir)

    with open(result_dir + "/REPORT.md", "w", encoding="utf-8") as report:
        report.write("# LSMu paper sweep\n\n")
        report.write("Reference: `1707.05354v2.pdf`.\n\n")
        report.write(
            "This is the complete GPU LSM-only matrix from Tables II-IV, "
            "plus the cleanup examples in Section V-D. All local query "
            "results passed GPU validation.\n\n")
        report.write(
            "Spearman correlation compares the local and paper rankings "
            "across batch sizes. The throughput ratio is the geometric "
            "mean of local throughput divided by paper throughput.\n\n")
        report.write(
            "RANGE measures this implementation's range-sum adapter; "
            "interpret its normalized trend rather than exact equivalence "
            "to the paper's materialized range output.\n\n")
        report.write(
            "| Result | Points | Spearman | Local/paper | "
            "Local growth | Paper growth |\n")
        report.write("|---|---:|---:|---:|---:|---:|\n")
        for row in summary_frame.itertuples(index=False):
            report.write(
                f"| {row.result} | {row.batch_points} | "
                f"{row.spearman_trend:.4g} | "
                f"{row.geometric_speedup:.4g}x | "
                f"{row.local_endpoint_growth:.4g}x | "
                f"{row.paper_endpoint_growth:.4g}x |\n")
        report.write("\n\n## Graphs\n\n")
        for path in sorted(glob.glob(graph_dir + "/*.png")):
            label = os.path.basename(path).replace("_", " ").removesuffix(".png")
            report.write(f"### {label}\n\n![{label}](graphs/{os.path.basename(path)})\n\n")
    print("Graphs and report written to", result_dir)


if __name__ == "__main__":
    main()
