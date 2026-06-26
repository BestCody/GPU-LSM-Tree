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