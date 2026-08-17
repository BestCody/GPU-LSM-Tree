#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
SOURCE_DIR="$REPO_DIR/FliX"

RESULT_DIR="$REPO_DIR/results/flix_lsm_paper_comparison"
MODE=full
SYSTEMS=both
CUDA_COMPILER=${CUDACXX:-/usr/local/cuda/bin/nvcc}
CUDA_ARCH=${CUDA_ARCH:-120}
RANGE_CHUNK_LOG=${RANGE_CHUNK_LOG:-18}
INSERT_LIMIT_LOG=
QUERY_LIMIT_LOG=
STOP_AFTER_R=0
SKIP_BULK=0
SKIP_CLEANUP=0
SKIP_PLOTS=0
BATCH_LOGS_TEXT=

usage()
{
  cat <<'EOF'
Usage: scripts/run_flix_lsm_paper_comparison.sh [options]

Run the GPU LSM paper protocol through the FliX adapters for GPULSMOpt
and LSMu. Completed (system, batch-size) runs are skipped on restart.

Options:
  --output DIR             Result directory.
  --mode full|smoke        Full paper matrix or reduced correctness run.
  --systems both|gpulsmopt|lsmu
  --batch-logs "16 20"    Override the mode's log2(batch-size) list.
  --insert-limit-log N     Override maximum inserted element count.
  --query-limit-log N      Override maximum query-state element count.
  --range-chunk-log N      Queries per range timing chunk (default 18).
  --stop-after-r N         Stop each state sweep after N submitted batches.
  --cuda-arch N            CUDA architecture (default 120).
  --skip-bulk              Omit the bulk-build experiment.
  --skip-cleanup           Omit LSMu's paper-specific cleanup experiment.
  --no-plots               Do not create aggregate CSVs and figures.
  -h, --help               Show this help.

The full mode uses the paper's matrix:
  insertion b=2^15..2^27, N=2^27
  lookup b=2^16..2^24, states through N=2^24
  range b=2^16..2^20, L in {8,1024}
  effective insertion b in {2^17,2^18,2^19,2^20}

No timed measurement is repeated. Raw per-state CSVs are retained.
EOF
}

while (($#)); do
  case "$1" in
    --output)
      RESULT_DIR=$2
      shift 2
      ;;
    --mode)
      MODE=$2
      shift 2
      ;;
    --systems)
      SYSTEMS=$2
      shift 2
      ;;
    --batch-logs)
      BATCH_LOGS_TEXT=$2
      shift 2
      ;;
    --insert-limit-log)
      INSERT_LIMIT_LOG=$2
      shift 2
      ;;
    --query-limit-log)
      QUERY_LIMIT_LOG=$2
      shift 2
      ;;
    --range-chunk-log)
      RANGE_CHUNK_LOG=$2
      shift 2
      ;;
    --stop-after-r)
      STOP_AFTER_R=$2
      shift 2
      ;;
    --cuda-arch)
      CUDA_ARCH=$2
      shift 2
      ;;
    --skip-bulk)
      SKIP_BULK=1
      shift
      ;;
    --skip-cleanup)
      SKIP_CLEANUP=1
      shift
      ;;
    --no-plots)
      SKIP_PLOTS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

for value in "$RANGE_CHUNK_LOG" "$STOP_AFTER_R" "$CUDA_ARCH"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "Expected a nonnegative integer, received: $value" >&2
    exit 2
  fi
done

case "$MODE" in
  full)
    DEFAULT_BATCH_LOGS="15 16 17 18 19 20 21 22 23 24 25 26 27"
    : "${INSERT_LIMIT_LOG:=27}"
    : "${QUERY_LIMIT_LOG:=24}"
    BULK_BATCH_LOG=27
    ;;
  smoke)
    DEFAULT_BATCH_LOGS="16 17 18"
    : "${INSERT_LIMIT_LOG:=18}"
    : "${QUERY_LIMIT_LOG:=18}"
    BULK_BATCH_LOG=18
    RANGE_CHUNK_LOG=$((RANGE_CHUNK_LOG < 16 ? RANGE_CHUNK_LOG : 16))
    SKIP_CLEANUP=1
    ;;
  *)
    echo "--mode must be full or smoke" >&2
    exit 2
    ;;
esac

case "$SYSTEMS" in
  both)
    SYSTEM_LIST=(gpulsmopt lsmu)
    ;;
  gpulsmopt|lsmu)
    SYSTEM_LIST=("$SYSTEMS")
    ;;
  *)
    echo "--systems must be both, gpulsmopt, or lsmu" >&2
    exit 2
    ;;
esac

read -r -a BATCH_LOGS <<< "${BATCH_LOGS_TEXT:-$DEFAULT_BATCH_LOGS}"
for value in "$INSERT_LIMIT_LOG" "$QUERY_LIMIT_LOG" "$RANGE_CHUNK_LOG" \
             "$STOP_AFTER_R" "$CUDA_ARCH" "${BATCH_LOGS[@]}"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "Expected a nonnegative integer, received: $value" >&2
    exit 2
  fi
done
if ((INSERT_LIMIT_LOG > 27)); then
  echo "This protocol supports at most the paper's 2^27 inserted records" >&2
  exit 2
fi
if ((QUERY_LIMIT_LOG > INSERT_LIMIT_LOG)); then
  echo "The query limit cannot exceed the insertion limit" >&2
  exit 2
fi
if ((RANGE_CHUNK_LOG > QUERY_LIMIT_LOG)); then
  echo "The range chunk cannot exceed the query-state limit" >&2
  exit 2
fi
for value in "${BATCH_LOGS[@]}"; do
  if ((value > 27)); then
    echo "External batch logs above 27 are outside this protocol" >&2
    exit 2
  fi
done

mkdir -p "$RESULT_DIR/logs" "$RESULT_DIR/gpulsmopt" "$RESULT_DIR/lsmu"

{
  echo "paper=1707.05354v2.pdf"
  echo "protocol=FliX adapters, one timed sample per state"
  echo "repo=$REPO_DIR"
  echo "commit=$(git -C "$REPO_DIR" rev-parse HEAD)"
  echo "dirty=$(test -n "$(git -C "$REPO_DIR" status --porcelain)" && echo 1 || echo 0)"
  echo "mode=$MODE"
  echo "systems=$SYSTEMS"
  echo "batch_logs=${BATCH_LOGS[*]}"
  echo "insert_limit_log=$INSERT_LIMIT_LOG"
  echo "query_limit_log=$QUERY_LIMIT_LOG"
  echo "range_chunk_log=$RANGE_CHUNK_LOG"
  echo "query_operations=lookup,range_enumeration_checksum"
  echo "gpulsmopt_internal_batch_capacity_log=20"
  echo "cuda_compiler=$CUDA_COMPILER"
  echo "cuda_arch=$CUDA_ARCH"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sha256sum \
    "$REPO_DIR/GPULSMOpt.cuh" \
    "$REPO_DIR/gpu_dictionary_adapter.cuh" \
    "$SOURCE_DIR/impl_gpulsmopt.cuh" \
    "$SOURCE_DIR/impl_lsm_tree.cuh" \
    "$SOURCE_DIR/paper_lsmu_sweep.cu"
  nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader
} > "$RESULT_DIR/run_metadata.txt"

want_system()
{
  local candidate=$1
  local selected
  for selected in "${SYSTEM_LIST[@]}"; do
    [[ "$selected" == "$candidate" ]] && return 0
  done
  return 1
}

build_batch()
{
  local batch_log=$1
  local build_dir="$REPO_DIR/build/flix-paper-comparison-b${batch_log}"
  local build_lsmu=OFF
  local build_gpulsmopt=OFF
  want_system lsmu && build_lsmu=ON
  want_system gpulsmopt && build_gpulsmopt=ON

  cmake -S "$SOURCE_DIR" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DFLIX_BUILD_LSMU_PAPER_SWEEP="$build_lsmu" \
    -DFLIX_BUILD_GPULSMOPT_PAPER_SWEEP="$build_gpulsmopt" \
    -DIFDEFS:STRING="-DPAPER_LSM_BATCH_LOG=${batch_log}"

  local targets=()
  want_system gpulsmopt && targets+=(gpulsmopt_paper_sweep)
  want_system lsmu && targets+=(lsmu_paper_sweep)
  cmake --build "$build_dir" --target "${targets[@]}" --parallel "$(nproc)"
}

binary_for()
{
  local system=$1
  local batch_log=$2
  echo "$REPO_DIR/build/flix-paper-comparison-b${batch_log}/${system}_paper_sweep"
}

data_rows()
{
  local path=$1
  if [[ ! -f "$path" ]]; then
    echo -1
    return
  fi
  local lines
  lines=$(wc -l < "$path")
  echo $((lines - 1))
}

validate_main_case()
{
  local system=$1
  local batch_log=$2
  local system_dir="$RESULT_DIR/$system"
  local insertion_states=$((1 << (INSERT_LIMIT_LOG - batch_log)))
  if ((STOP_AFTER_R > 0 && STOP_AFTER_R < insertion_states)); then
    insertion_states=$STOP_AFTER_R
  fi
  local query_states=0
  if ((batch_log >= 16 && batch_log <= 24 &&
       QUERY_LIMIT_LOG >= batch_log)); then
    query_states=$((1 << (QUERY_LIMIT_LOG - batch_log)))
    if ((query_states > insertion_states)); then
      query_states=$insertion_states
    fi
  fi
  local range_states=0
  if ((batch_log >= 16 && batch_log <= 20 &&
       QUERY_LIMIT_LOG >= batch_log)); then
    range_states=$query_states
  fi
  local insertion_rows
  local lookup_rows
  local range_rows
  insertion_rows=$(data_rows "$system_dir/insertion_b${batch_log}.csv")
  lookup_rows=$(data_rows "$system_dir/lookup_b${batch_log}.csv")
  range_rows=$(data_rows "$system_dir/range_b${batch_log}.csv")
  if ((insertion_rows != insertion_states)); then
    echo "$system b=2^$batch_log has $insertion_rows insertion rows; expected $insertion_states" >&2
    exit 1
  fi
  if ((lookup_rows != 2 * query_states)); then
    echo "$system b=2^$batch_log has $lookup_rows lookup rows; expected $((2 * query_states))" >&2
    exit 1
  fi
  if ((range_rows != 2 * range_states)); then
    echo "$system b=2^$batch_log has $range_rows range rows; expected $((2 * range_states))" >&2
    exit 1
  fi
}

run_main_case()
{
  local system=$1
  local batch_log=$2
  local system_dir="$RESULT_DIR/$system"
  local marker="$system_dir/complete_range_b${batch_log}"
  local range_file="$system_dir/range_b${batch_log}.csv"
  if [[ -f "$marker" ]] &&
     head -n 1 "$range_file" 2>/dev/null | \
       rg -q 'checksum_sum,checksum_xor'; then
    validate_main_case "$system" "$batch_log"
    echo "Skipping completed $system b=2^$batch_log"
    return
  fi
  local binary
  binary=$(binary_for "$system" "$batch_log")
  local command=(
    "$binary"
    --output "$system_dir"
    --insert-limit-log "$INSERT_LIMIT_LOG"
    --query-limit-log "$QUERY_LIMIT_LOG"
    --range-chunk-log "$RANGE_CHUNK_LOG"
    --range-only
    --main-only
  )
  if ((STOP_AFTER_R > 0)); then
    command+=(--stop-after-r "$STOP_AFTER_R")
  fi
  echo "Running $system b=2^$batch_log"
  /usr/bin/time -v "${command[@]}" 2>&1 | \
    tee "$RESULT_DIR/logs/${system}_main_b${batch_log}.log"
  validate_main_case "$system" "$batch_log"
}

run_bulk_case()
{
  local system=$1
  local system_dir="$RESULT_DIR/$system"
  if [[ -f "$system_dir/complete_bulk" ]]; then
    echo "Skipping completed $system bulk build"
    return
  fi
  local binary
  binary=$(binary_for "$system" "$BULK_BATCH_LOG")
  echo "Running $system bulk build with 2^$INSERT_LIMIT_LOG records"
  /usr/bin/time -v "$binary" \
    --output "$system_dir" \
    --insert-limit-log "$INSERT_LIMIT_LOG" \
    --query-limit-log "$QUERY_LIMIT_LOG" \
    --range-chunk-log "$RANGE_CHUNK_LOG" \
    --range-only \
    --bulk-only 2>&1 | \
    tee "$RESULT_DIR/logs/${system}_bulk.log"
}

run_cleanup_case()
{
  local batch_log=$1
  local cleanup_dir="$RESULT_DIR/lsmu/cleanup_b${batch_log}"
  if [[ -f "$cleanup_dir/complete" ]]; then
    echo "Skipping completed LSMu cleanup b=2^$batch_log"
    return
  fi
  mkdir -p "$cleanup_dir"
  if [[ -f "$cleanup_dir/cleanup.csv" ]]; then
    unlink "$cleanup_dir/cleanup.csv"
  fi
  local binary
  binary=$(binary_for lsmu "$batch_log")
  echo "Running LSMu paper cleanup b=2^$batch_log"
  /usr/bin/time -v "$binary" \
    --output "$cleanup_dir" --cleanup-only 2>&1 | \
    tee "$RESULT_DIR/logs/lsmu_cleanup_b${batch_log}.log"
  printf 'ok\n' > "$cleanup_dir/complete"
}

BUILT_LOGS=()
ensure_built()
{
  local wanted=$1
  local existing
  for existing in "${BUILT_LOGS[@]}"; do
    [[ "$existing" == "$wanted" ]] && return
  done
  echo "Building FliX paper drivers for b=2^$wanted"
  build_batch "$wanted"
  BUILT_LOGS+=("$wanted")
}

if ((SKIP_BULK == 0)); then
  ensure_built "$BULK_BATCH_LOG"
  for system in "${SYSTEM_LIST[@]}"; do
    run_bulk_case "$system"
  done
fi

for batch_log in "${BATCH_LOGS[@]}"; do
  if ((batch_log > INSERT_LIMIT_LOG)); then
    echo "Skipping b=2^$batch_log: larger than insertion limit"
    continue
  fi
  ensure_built "$batch_log"
  for system in "${SYSTEM_LIST[@]}"; do
    run_main_case "$system" "$batch_log"
  done
  if want_system gpulsmopt && want_system lsmu; then
    python3 "$SCRIPT_DIR/summarize_flix_lsm_paper_comparison.py" \
      "$RESULT_DIR" --validate-only
  fi
done

if ((SKIP_CLEANUP == 0)) && want_system lsmu; then
  for batch_log in 18 19 20; do
    ensure_built "$batch_log"
    run_cleanup_case "$batch_log"
  done
fi

if ((SKIP_PLOTS == 0)); then
  python3 "$SCRIPT_DIR/summarize_flix_lsm_paper_comparison.py" "$RESULT_DIR"
fi

echo "FliX GPU LSM comparison complete: $RESULT_DIR"
