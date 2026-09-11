# FliX: Flipped-Indexing for Scalable GPU Queries and Updates

This repository contains the source code and experimental framework for the paper:  
**_FliX: Flipped-Indexing for Scalable GPU Queries and Updates_**

FliX is a high-performance GPU-resident indexing structure designed to support high-speed concurrent updates alongside fast point and range queries.

## Authors

* **Rosina Kharal** — University of Waterloo, Canada
* **Justus Henneberg** — Johannes Gutenberg University of Mainz, Germany
* **Trevor Brown** — University of Waterloo, Canada
* **Felix Schuhknecht** — Johannes Gutenberg University of  Mainz, Germany

## Previous Work

This code base builds on the original work of Justus Henneberg from the **Coarse-Granular RTIndeX** (_cgRX_) repository:

https://gitlab.rlp.net/juhenneb/coarse-granular-rtindex

The current repository extends that earlier foundation of **cgRX** with the **FliX** implementation, additional benchmark support, new baselines, and updated experiments.

## Project & Code Structure

The codebase utilizes a benchmark-driven architecture where the specific index implementation is toggled at compile-time via macros.

### Execution Flow:
```text
main.cu
└── benchmark_updates (Kernel/Host Wrapper)
    ├── Loads one benchmark configuration
    ├── Selects data structure (Compile-time Macro)
    │   ├── FliX        -> impl_cg_rtx_index_updates.cuh
    │   ├── LSMu        -> impl_lsm_tree.cuh
    │   ├── GPU-BTree   -> impl_tree_awad.cuh
    │   ├── SlabHash    -> impl_hashtable_slab.cuh
    │   └── WarpCore    -> impl_hashtable_warpcore.cuh
    └── Executes benchmark operations
        ├── index.insert
        ├── index.remove
        ├── index.lookup
        ├── index.successor (FliX and LSMu only)
        └── index.rebuild   (FliX only)
```

The scripts in `runscripts_experiments/` automate the process of recompiling and running the benchmark for each structure sequentially to generate comparisons and output files for each.

## Benchmarked Data Structures

If you use these baselines in your research, please cite the original publications:

| Index Type | Reference | Link |
| :--- | :--- | :--- |
| **FliX** | *FliX: Flipped-Indexing for Scalable GPU Queries and Updates* | TBD |
| **LSMu** | Ashkiani et al., *GPU LSM: A Dynamic Dictionary Data Structure for the GPU* | [DOI](https://ieeexplore.ieee.org/document/8425197) |
| **GPU-BTree** | Awad et al., *Engineering a High-Performance GPU B-Tree* | [DOI](https://dl.acm.org/doi/10.1145/3293883.3295706) |
| **Hash_Slab** | Ashkiani et al., *A Dynamic Hash Table for the GPU* | [DOI](https://doi.org/10.1109/IPDPS.2018.00052) |
| **Hash_Warpcore** | Jünger et al., *WarpCore: A Library for Fast Hash Tables on GPUs* | [DOI](https://ieeexplore.ieee.org/document/9406635) |

## Installation Requirements

### Hardware
* **NVIDIA GPU**: Support for concurrent kernels and sufficient VRAM (24GB+ recommended).
* **Tested On**: NVIDIA RTX A6000 / RTX 6000 Ada.
* **Memory**: 32 GB System RAM.
* **OS**: 64-bit Linux 

### Software
* **CUDA Toolkit**: 12.8 or newer.
* **Compiler**: `gcc` / `g++` version 12.1 or newer.
* **Driver**: NVIDIA Driver version 555.42 or newer.


## Usage and Experiments

### Local baseline setup

From the GPU-LSM-Tree repository root, build and validate the available
backends with:

```bash
python3 scripts/setup_flix_baselines.py --memcheck
```

This uses the existing FliX executable and generators with small correctness
inputs. It builds GPULSMOpt, LSMu, GPU B-tree, FliX, SlabHash, WarpCore, and
the sorted array in separate directories under `build/flix_baselines`.
GPU checks run serially. The default update case uses 32,768 initial keys,
65,536 main probes, and five insertion/deletion batches each, including for
the sorted array. These are setup checks, not the
paper's performance evaluation.

`status.json` and per-backend logs record compilation and correctness outcomes.
A skipped workload, missing result states, incomplete lookup timing, different
input checksums, or a memcheck error fails the check.
CUDA and the GPU architecture are detected automatically; `--cuda-root`,
`--arch`, and `--build-root` override discovery. Use `--backends flix gpu_btree`
to check selected implementations, or `--build-only` to compile without running
GPU checks. Public input sizes and the X/Y distribution are configurable;
see `--help`. Successful checks are reused if the executable hash and flags
are unchanged and the saved logs still pass validation; `--recheck` repeats them.

The harness submits unsorted updates and probes. All dynamic backends use the
same deterministic FliX generators, public batch sizes, and common key domain
(31 bits by default). `--key-bits` changes that domain for all backends together;
an unsupported domain fails compilation. The CSV records checksums of the full
generated key sequence and each step's ordered update/query requests and values.
The runner checks that these signatures match across all selected dynamic
backends and between normal and sanitizer runs, including the sorted array.

FliX's bulk update kernels require ordered inputs; the harness sorts key/value
pairs or deletion keys on the GPU inside the update timer. LSMu capacity includes
tombstones and the padding consumed by each public update call. The hash checks
cover distinct-key growth and deletion, not overwriting live keys.

Update-benchmark lookup rows use `lookup_timing=complete_unsorted_v1`.
`probe_time_ms`, `probe_miss_time_ms`, and `deleted_keys_probe_time_ms` now cover
the complete resident lookup: unsorted GPU inputs through answers in their
original order. Required scratch growth, permutation generation, sorting,
result initialization, the backend call, and answer restoration are inside one
CUDA-event interval. A synchronized wall-clock duration is also exported for
each lookup class. Per-run scratch is reused within that run; the first use is
charged, and no untimed lookup warms that index beforehand.

The `*_prepare_time_ms`, `*_search_time_ms`, and `*_restore_time_ms` columns
partition the total; do not add them to it. The search component covers the
backend API, including any internal routing or workspace allocation. The old
update-benchmark `sort_time_ms` column is replaced by these components. Main
hit queries, future-insertion misses, and post-delete misses all use this same
pipeline for every dynamic backend. Transfers and host correctness checking are
outside these resident lookup intervals. Construction/lifecycle and peak-memory
accounting remain separate evaluation work; these totals are not those metrics.

The main executable uses release optimization without host profiling by default.
`FLIX_ENABLE_PROFILING=ON` enables the previous instrumentation. The common paper
driver uses this wrapper too; the legacy paper protocol keeps its original timers.

The imported SlabHash, modified WarpCore, and helper libraries come from
[FliX-Full](https://github.com/rkharal/FliX-Full/tree/4acb4b5eab91851e6d29b0015752f23cd6af187b).
The matching [OptiX 8.0 headers](https://github.com/NVIDIA/optix-dev/tree/f60c1e44f18426f426a2ed948f28515b3cf67b8a)
are under `ext/optix`; an external SDK is unnecessary for this checkout's
headers, while a compatible NVIDIA driver is still required. Source revisions
and imported-file hashes are recorded in `FliX/dependencies.json` and checked
by the setup runner. Existing GPU B-tree and FliX index implementations remain
the selected local versions.

### Common paper sweep

The existing paper entry point selects all seven dynamic adapters, including
the sorted array. Inspect a plan without building or running:

```bash
scripts/run_flix_lsm_paper_comparison.sh --family both --mode smoke --plan
```

From the repository root, a small validation suite with two timed repetitions
and a separate memory-sanitizer replay is:

```bash
scripts/run_flix_lsm_paper_comparison.sh --family both --mode smoke \
  --batch-logs 16 --insert-limit-log 18 --query-limit-log 18 \
  --repetitions 2 --memcheck --output results/flix_paper_pilot
```

`--family paper` selects the controlled batch/growth matrix; `--family flix`
selects the original X/Y generator and update trace. `--xy 25:25 25:90` is the
default pair of uniform and concentrated cases. FliX input sizes and rounds
are configurable through `--flix-build-log`, `--flix-probe-log`, and
`--flix-rounds`. `--systems` accepts `all`, `both` (GPULSMOpt/LSMu), or any of
`gpulsmopt lsmu gpu_btree flix slabhash warpcore sorted_array`.

The common paper protocol, `common_initialized_v1`, builds the first batch for
every backend, then grows through identical prefixes using the same public
batch size. This supplies FliX's required initial bucket layout. Initial
construction is reported separately from insertion. The historical bijective
key mapping is shifted by two to avoid reserved keys; values are their original
insertion ordinals. Every point-lookup answer is checked against that mapping.
Deletes remove growth batches in reverse order, followed by checks of both
deleted keys and surviving keys. Live-key overwrite semantics are not assumed.

Common ranges enumerate visible records and sum their values in place of
writing an output list. GPULSMOpt, LSMu, GPU B-tree, FliX, and sorted array
participate; their input and result checksums must agree, including after
deletions. Each query must visit every visible matching record; bucket sums
and value-prefix shortcuts do not satisfy this contract. Bounds are inclusive
and output arithmetic is modulo 2^32. Results carry the processing tag
`enumerate_records_sum_v1`; the CSV operation names remain `range_sum` and
`range_sum_after_delete`. Historical untagged results remain separate.

GPULSMOpt's native range traversal already resolves versions and visits
individual records. Its output sink reduces their values. Returning records
can reuse that traversal, but needs per-query output sizing and offsets,
full-key reconstruction from the region and suffix, and writes to output
storage. Producing key-ordered lists also needs output ordering. Those costs
are omitted by the sum-output experiment, so its timings do not establish
the performance of a materialized range API. The sparse-record path already
enumerates into temporary output before summing. The paper family's
`--skip-ranges`, `--skip-deletions`, and
`--skip-bulk` select subsets. Explicit LSM cleanup stays in the legacy family.

The local sorted-array update implementation uses CUB radix sorting and parallel
merging for insertion. Deletion sorts the request keys, binary-searches that
list for each resident record, and compacts the surviving key/value pairs with
CUB selection. With N resident records and D deletion requests, this costs
O(N log(D + 1)) membership work plus sorting and compaction. Insertion copies
the resident records into the merged output; deletion also requires space for
the surviving output. Scratch buffers grow geometrically and are reused within
a trace. All sorting, workspace growth, and the survivor-count readback needed
by deletion occur inside the update call and its timer. Adapter memory
snapshots include retained scratch space. Bulk-build and lookup measurements
remain available as static reference measurements.

Sorted-array insertion adds the supplied key/value pairs; duplicate keys are
retained, and point lookup returns a matching value without promising which
duplicate wins. Range and exact-key sums include every matching record.
Deletion removes every occurrence of a requested key; absent and repeated
requests are allowed. Thus the common distinct-key workloads are comparable,
but this adapter does not claim newest-value overwrite semantics. Inputs are
preserved; updates and queries use the caller's CUDA stream. Callers must order
operations on an instance, including when switching streams. The adapter
supports 32- and 64-bit keys and at most INT_MAX resident records or requests.
Historical static-only sorted-array results remain unchanged and must not be
combined with the new dynamic update results.

The imported `coarse_granular_range_queries.cuh` remains unchanged from
FliX-Full commit `4acb4b5eab91851e6d29b0015752f23cd6af187b`, with its source hash
in `dependencies.json`. The active adapter uses the local
`coarse_granular_range_enumeration.cuh`, which reuses the upstream node helpers
and scans all matching records, including fully covered interior buckets.
The visitor sends each key/value pair to an output sink; the measured sink
sums values. It no longer precomputes bucket sums or allocates a sum buffer.
The complete call is timed from unsorted GPU bounds to answers in input order.
SlabHash and WarpCore have no implemented range method.

Both families use the complete unsorted lookup wrapper. GPU inputs are ready
before timing; required sorting, workspace growth, searching, and answer-order
restoration are included. Required update sorting is also timed. The common
paper driver records CUDA-event and synchronized wall times, construction, and
index destruction. Original FliX X/Y construction is not timed; required FliX
rebuild maintenance is exported separately. These results do **not** measure
transfer-inclusive lifecycle time or allocation peaks. Adapter memory snapshots
exclude harness buffers and are not interchangeable with peak memory.

Each repetition uses a fresh process and index. Backends execute serially;
compilation may run in parallel. `--warmups` controls excluded warmup traces.
`--mode full` selects the larger historical size matrix with five repetitions
and one warmup; it is never selected by default. All build settings are recorded.

Resume by repeating the same command. Completed runs are reused only if source,
configuration, binary, validation, and result hashes agree. Incomplete runs are
archived before retrying; historical data is not overwritten. Use a new output
directory after changing source or settings. Reports in `summary/` contain raw
operations, per-state variability, per-repetition trace totals, and plots for
every participating backend. Warmups and sanitizer timings are excluded. Compare
aggregated rates only where item counts and state coverage match.

In the paper family, `batch_log` is the public batch-size exponent. In the FliX
family, the `b16` suffix records LSMu's internal batch geometry; the actual public
update sizes come from the X/Y harness and appear in `current_batch_size` and the
normalized report's `items` column. They are identical across backends and need
not be powers of two.

The original two-backend driver and reporting remain accessible through
`--family legacy`, with their original options and timing semantics. Do not
merge those results with the new protocol. No backend algorithm is changed by
the common sweep.

### Original experiment scripts

To reproduce the experimental results from the paper:

1. Navigate to the `runscripts_experiments` directory.
2. Run the desired benchmark scripts
3. Results will be saved to the `results/` directory for visualization.

> **Note:** Plotting scripts and additional run configurations are currently being integrated into the repository. Please check back for updates as we finalize the repo.

## License and Attribution

```cpp
 =============================================================================
 Authors:       Justus Henneberg, Rosina Kharal
 Copyright (c) 2025-2026 Justus Henneberg, Rosina Kharal
 SPDX-License-Identifier: GPL-3.0-or-later
 =============================================================================
```
```
