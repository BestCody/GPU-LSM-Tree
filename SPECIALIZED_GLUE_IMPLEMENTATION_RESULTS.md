# Specialized GPULSMOpt glue implementation results

Date: 2026-08-30

## Verdict

The frozen specialized-glue architecture was implemented in the isolated
`specialized-glue-faithful` worktree. After reviewing the completed results,
the user explicitly accepted the retained performance and memory tradeoffs on
2026-08-30 and directed promotion to `main`. `GPULSMOpt` remains the sole
executor and authoritative dictionary state. Mixed-length support is sparse
exact-key and capsule state attached to the restored pending slots and
resident roots; no general dictionary wrapper or mirrored ordinary state is
present.

All exercised semantic, lifecycle, sanitizer, protected-SASS, and memory
accounting checks pass. Two performance acceptance checks do not pass:

- One B20 insertion measured 8.495318% above restored main in the protected
  five-process comparison, versus a 5% limit.
- The completed frozen 134,217,728-row sealed call measured 16.191551 ms GPU
  time. After the separately recorded, user-directed LSMu allocation-boundary
  correction, its five-process median is 9.702464 ms, still above the absolute
  target of less than 9.270 ms.

These results are retained without tuning or changing the frozen algorithms.
Promotion records acceptance of the measured tradeoffs; it does not turn the
failed gates into passing results.

## Implementation provenance and promotion state

- Source worktree during implementation:
  `/tmp/gpulsm-specialized-glue-faithful`
- Source branch: `specialized-glue-faithful`
- Base: `0e91aad702e38d21023ddba710826dff6242c19b`
- Promotion target:
  `/home/hansenl/LSMGPU/GPU-LSM-Tree`
- Pre-promotion backup branch:
  `backup/main-before-specialized-glue-20260830`
- Prototype source:
  `/home/hansenl/LSMGPU/GPU-LSM-Tree/profiling/gpulsmopt-production-closure-20260829`
- Frozen design:
  `/home/hansenl/LSMGPU/GPU-LSM-Tree/profiling/gpulsm-specialized-glue-design-20260829/SPECIALIZED_GLUE_DESIGN.md`

Immediately before promotion, the main worktree was clean at the same base
commit. The 33 files in the production-closure prototype directory remained
present and unmodified, with aggregate content hash
`851dce3e4a5f715cd3928aff55caafec967b39abd4f17e2b7bcf084fa0783469`.
`git diff --check` passed in the implementation worktree.

## Implemented physical boundaries

- The restored pending arrays, resident arena, level masks, and manifest are
  authoritative.
- Ordinary insertion calls the restored `admit_tile`; a sub-epoch ordinary
  call performs no sparse planning, allocation, synchronization, or launch.
- Mixed admission uses restored anchor transport and stores only exceptional
  locators, complete keys, non-inline values, and sparse descriptors.
- Every resident source presents at most one projection row per four-byte
  head. Ambiguous heads contain a sentinel and one exact-key sidecar.
- Completed epochs and resident carries retain restored projection placement;
  sparse refinement gathers all same-head exact and ordinary candidates,
  resolves newest complete keys, and patches only observed placeholders.
- The structural radix-4 sealed planner materializes final roots only and
  publishes once. It delegates a non-fusing epoch to restored publication.
- Exact queries are deferred before accepting a head winner. Canonical, TQRJ
  direct, and TQRJ overflow roles remain intact.
- Ordinary protected lookup writes directly through the restored FliX output
  convention. There is no temporary `found` buffer.
- All-inline bulk construction dispatches directly to restored `bulk_build`;
  mixed bulk builds one projection and sparse sidecars directly.
- Successor uses restored head search until a sentinel requires exact-sidecar
  refinement.
- Range checksum uses the selected capacity-independent rank-roster executor.
- Dispatch is based on source capabilities and observed sparse state, not a
  `key_type == uint32_t` special case.
- Sparse memory reporting includes manifests, retained overlays, current
  workspace, workspace high-water, capsule mappings, live bytes, garbage,
  and segment counts.

The only `alternate bank` wording in the resulting source belongs to the
pre-existing restored GPULSMOpt top-level rollover mechanism. No alternate
sparse or universal ordinary bank was added.

## Prototype reuse

The implementation adapts the validated prototype algorithms onto restored
GPULSMOpt ownership instead of recreating them:

| Current component | Prototype source reused/adapted |
| --- | --- |
| `gpulsm_sparse/common.cuh`, `capsule.cuh` | closure input and capsule encoding, 64-bit capsule addresses |
| `capsule_lifecycle.cuh` | capsule transfer, retained-garbage accounting, dead-at-least-live compaction |
| `pending.cuh` | pending lowering, grouped locators, sparse pending ownership |
| `exact.cuh`, `refine.cuh` | end-aware continuation and exact-sidecar construction |
| `fan_in.cuh` | capacity-independent exact-source fan-in and newest-wins selection |
| `sealed.cuh` | validated structural radix-4 carry planner and sealed builder |
| `read.cuh`, `ordered_read.cuh` | exact comparison, direct/hash ownership, successor refinement |
| `range.cuh`, `range_common.cuh` | accepted rank-roster range implementation |
| edge and one-head tests | prototype collision pair, lifecycle cases, and `one_head_entropy` generator |

Wrapper-specific `CoordinatedDictionary` state, replacement ordinary
kernels, and mirrored manifests were not transplanted.

## Measurement environment

- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97,887 MiB
- Driver: 595.71.05
- CUDA compiler: 12.8, build 35404655
- Architecture: `sm_120`
- Protected paper harness: Release, `-O3`, line info, fast math
- Candidate and restored-main binaries used the same FliX
  `paper_lsmu_sweep.cu` harness and GPU.

## Protected gates

Five fresh processes were used for each protected comparison.

| Measurement | Restored-main median | Candidate median | Delta | Gate | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| One B20 insertion, GPU ms | 0.0888959989 | 0.0964479968 | +8.495318% | within 5% | Fail |
| 128 x B13 cumulative insertion, GPU ms | 6.60016002879 | 6.594912 | -0.079514% | within 10% | Pass |
| B20 hit lookup, GPU ms | 0.0901120007 | 0.0911360011 | +1.136364% | within 5% | Pass |
| B20 miss lookup, GPU ms | 0.0706240013 | 0.0706240013 | 0% | within 5% | Pass |
| B20 bulk build, GPU ms | 7.10256004333 | 6.98291206360 | -1.684575% | within 10% | Pass |
| B20 bulk build, wall ms | 7.211496 | 7.113714 | -1.355918% | within 10% | Pass |
| B20 retained bytes | 825,766,938 | 825,770,698 | +0.000455334% | within 10% | Pass |

Raw B20 insertion samples, in process order:

- Restored main: `0.0881600007`, `0.0880000`, `0.0888959989`,
  `0.0900800005`, `0.0900800005` ms.
- Candidate: `0.0986239985`, `0.0963520035`, `0.1097280011`,
  `0.0922240019`, `0.0964479968` ms.

Raw 128 x B13 cumulative samples:

- Restored main: `6.60016002879`, `6.60691201687`, `6.60598398931`,
  `6.58470397256`, `6.59161601961` ms.
- Candidate: `6.56252793781`, `6.65164794773`, `6.594912`,
  `6.53801601753`, `6.65942402184` ms.

Raw B20 bulk-build GPU samples:

- Restored main: `9.07791996`, `6.73571205`, `6.80713606`,
  `8.96054363`, `7.10256004` ms.
- Candidate: `6.98291206`, `9.49609566`, `7.31372786`,
  `6.76246405`, `6.71871996` ms.

### B20 insertion failure audit

The candidate and baseline each launched exactly these six kernels once:

1. admission quotient count;
2. CUB scan initialization;
3. CUB exclusive scan;
4. admission scatter;
5. admission signature build;
6. admission metadata commit.

No sparse kernel, copy, allocation, plan, or synchronization appeared. The
Nsight trace summed 51.680 microseconds of kernel execution for restored main
and 51.392 microseconds for the candidate. The restored admission kernels and
the two CUB scan kernels were also machine-code identical. There is therefore
no unexpected sparse work authorized for removal; the protected five-process
measurement remains recorded as a failed gate.

## Protected lookup SASS

Machine-code streams extracted from candidate and restored paper binaries are
byte-identical:

| Kernel | Machine words | SHA-256 for both binaries |
| --- | ---: | --- |
| TQRJ hash/overflow | 4,102 | `ee59ec1c7b10947b57c2463cbd5c8cb44f9628f7a623b129d9be183e8243873b` |
| TQRJ direct | 2,048 | `8292123bdfb3fa44365f7a41757a7ccb143569b3d1b536ca072bb77dde0cf132` |
| Canonical with pending | 816 | `85672033c4302dc3f54906889333f9f30153cebd35b79fedcb3ecebef7ed8904` |

The protected SASS gate passes.

## Structural and correctness gates

| Gate | Result |
| --- | --- |
| Mixed pending completed by ordinary suffix | Pass |
| Ordinary pending completed by mixed suffix | Pass |
| Exact heads carried from older and newer resident roots | Pass |
| Literal one-head short/four-byte/long/duplicate/tombstone/resurrection history | Pass |
| 1,000,003 distinct twelve-byte keys under one head | Pass |
| Forced rejection-hash collision (`0xb8bed1a7`) | Pass |
| Key lengths 0 through 102, value lengths 0 through 129, embedded zeros | Pass |
| Capsule transfer and retained-garbage accounting | Pass |
| Dead-at-least-live capsule compaction | Pass |
| Canonical, TQRJ direct, and TQRJ hash exact-query agreement | Pass |
| Exact lookup hit/miss, successor, and rank-roster range checksum | Pass |
| Optimized mixed publication, direct bulk, TQRJ, sealed, pending, and edge matrix | Pass |
| 134,217,728-row sealed correctness and sampled lookups | Pass, 0 errors |
| 134,217,728-row sealed time below 9.270 ms | Fail |

The final optimized sealed measurement was:

```text
rows=134217728
gpu_ms=16.191551
wall_ms=16.199488
fresh_bytes=2262105550
resident_bytes=2262105550
sparse_manifest_bytes=3760
workspace_high_water=1971338506
sampled_lookup_errors=0
```

This is 74.666138% above the absolute target. An earlier same-path candidate
run measured 15.411104 ms GPU and 15.420996 ms wall, also failing the target
by 66.247077%. The untouched restored-main control measured 17.750015 ms GPU
and 17.761334 ms wall. The frozen candidate is correct and faster than that
control, but it does not meet the absolute design gate. No threshold path or
algorithm change was introduced in response.

### User-directed LSMu allocation-boundary correction

After the completed frozen result above was retained, the user explicitly
directed the all-inline sealed path to follow LSMu's allocation timing. The
sealed sort/select workspace and its fixed publication command/receipt are now
allocated from `DictionaryConfig::max_elements` during dictionary
construction, included in `gpu_resident_bytes()`, reused by every qualifying
sealed insertion, and released only with the dictionary. No sealed kernel,
winner rule, planner rule, or publication rule changed.

Configuration and workload were unchanged: a fresh dictionary with
`max_elements=2^27`, `batch_capacity=2^20`, `level_zero_capacity=2^24`, and
one 134,217,728-row paper input. Five independent process results were:

```text
gpu_ms:  9.695232  9.692800  9.714752  9.702752  9.702464
wall_ms: 9.704505  9.702739  9.721636  9.713596  9.714871
median_gpu_ms=9.702464
median_wall_ms=9.713596
fresh_bytes=4233444880
resident_bytes=4233444880
sampled_lookup_errors=0
```

The median is 40.076994% below the retained 16.191551 ms result, but remains
4.665200% above the required 9.270 ms gate. Construction-time retained memory
increased by 1,971,339,330 bytes: 1,971,338,506 bytes for the reusable sealed
workspace and 824 bytes for its fixed device controls. A fresh Nsight Systems
trace measured zero CUDA allocation, free, VMM-create, or VMM-map calls between
the insertion start and stop event records. This is an allocation-lifetime
correction only; the remaining performance gap is retained for a separate
design decision.

A promotion-time rerun of the same restored-main 128M control measured
17.734655 ms GPU time, 17.743534 ms wall time, 2,262,101,790 fresh and resident
bytes, and zero sampled lookup errors. The promoted configuration therefore
retains 1,971,343,090 additional bytes, or 87.146524% more, than that control.
A clean promotion-time rebuild of the post-correction candidate measured
9.711936 ms GPU time, 9.718211 ms wall time, 4,233,444,880 fresh and resident
bytes, and zero sampled lookup errors. This verification sample is consistent
with the retained five-process distribution above.

The final optimized one-head integrated stress measured:

```text
rows=1000003
bulk_gpu_ms=29.128736
bulk_wall_ms=29.136777
full_shared_head_range_gpu_ms=13.946816
capsule_live_bytes=40000120
sparse_workspace_high_water=552126606
errors=0
```

The full-range query used the lexicographic minimum and maximum complete keys
and validated the wrapped 32-bit sum across all 1,000,003 records.

## Sanitizers

- Expanded edge/lifecycle test, memcheck: 0 errors.
- Expanded edge/lifecycle test, racecheck: 0 hazards, 0 errors, 0 warnings.
- Mixed publication/lookup/successor/range test, memcheck: 0 errors.
- Mixed publication/lookup/successor/range test, racecheck: 0 hazards,
  0 errors, 0 warnings.

## Retained limitations and negative evidence

- Operations are serialized and successful-operation semantics are assumed;
  rollback remains outside the research scope.
- Required arbitrary-byte input normalization remains timed.
- The known long-common-prefix negative costs remain retained; no alternate
  matcher or threshold was added.
- Rank-roster range enumeration remains intentionally general and outside the
  performance correction. Its integrated one-head timing is recorded above.
- All deferred negative results in `/home/hansenl/LSMGPU/AGENTS.md` and the
  production-closure reports remain untouched.

Not every decisive performance gate is green. After reviewing that evidence,
the user explicitly accepted the result and directed promotion to `main` on
2026-08-30. The promotion is therefore deliberate rather than silent. Future
performance remedies remain a separate design pass and must not rewrite the
recorded negative results.
