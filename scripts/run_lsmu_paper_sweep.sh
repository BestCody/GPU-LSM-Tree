#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
SOURCE_DIR="$REPO_DIR/FliX"
RESULT_DIR=${1:-"$REPO_DIR/results/lsmu_paper_sweep"}
MODE=${2:-full}
CUDA_COMPILER=${CUDACXX:-/usr/local/cuda/bin/nvcc}
CUDA_ARCH=${CUDA_ARCH:-120}
RANGE_CHUNK_LOG=${RANGE_CHUNK_LOG:-18}

mkdir -p "$RESULT_DIR/logs"

if [[ "$MODE" == "calibration" ]]; then
  BATCH_LOGS=(16)
  INSERT_LIMIT_LOG=18
  QUERY_LIMIT_LOG=18
else
  BATCH_LOGS=(15 16 17 18 19 20 21 22 23 24 25 26 27)
  INSERT_LIMIT_LOG=27
  QUERY_LIMIT_LOG=24
fi

{
  echo "paper=1707.05354v2.pdf"
  echo "repo=$REPO_DIR"
  echo "commit=$(git -C "$REPO_DIR" rev-parse HEAD)"
  echo "mode=$MODE"
  echo "cuda_compiler=$CUDA_COMPILER"
  echo "cuda_arch=$CUDA_ARCH"
  echo "range_chunk_log=$RANGE_CHUNK_LOG"
  nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader
} > "$RESULT_DIR/run_metadata.txt"

build_batch()
{
  local batch_log=$1
  local build_dir="$REPO_DIR/build/lsmu-paper-b${batch_log}"
  cmake -S "$SOURCE_DIR" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DFLIX_BUILD_LSMU_PAPER_SWEEP=ON \
    -DIFDEFS:STRING="-DBASELINES -DLSM_TREE \
-DPAPER_LSM_BATCH_LOG=${batch_log}"
  cmake --build "$build_dir" --target lsmu_paper_sweep \
    --parallel "$(nproc)"
  echo "$build_dir/lsmu_paper_sweep"
}

for batch_log in "${BATCH_LOGS[@]}"; do
  if [[ -f "$RESULT_DIR/complete_b${batch_log}" ]]; then
    echo "Skipping completed b=${batch_log}"
    continue
  fi
  echo "Building paper sweep for b=2^${batch_log}"
  binary=$(build_batch "$batch_log" | tail -n 1)
  echo "Running paper sweep for b=2^${batch_log}"
  /usr/bin/time -v "$binary" \
    --output "$RESULT_DIR" \
    --insert-limit-log "$INSERT_LIMIT_LOG" \
    --query-limit-log "$QUERY_LIMIT_LOG" \
    --range-chunk-log "$RANGE_CHUNK_LOG" \
    --main-only \
    2>&1 | tee "$RESULT_DIR/logs/main_b${batch_log}.log"
done

if [[ "$MODE" == "full" ]]; then
  rm -f "$RESULT_DIR/cleanup.csv"
  for batch_log in 18 19 20; do
    echo "Running cleanup sweep for b=2^${batch_log}"
    binary=$(build_batch "$batch_log" | tail -n 1)
    /usr/bin/time -v "$binary" \
      --output "$RESULT_DIR" --cleanup-only \
      2>&1 | tee "$RESULT_DIR/logs/cleanup_b${batch_log}.log"
  done
fi

python3 "$SCRIPT_DIR/plot_lsmu_paper_sweep.py" "$RESULT_DIR"
echo "Paper sweep complete: $RESULT_DIR"
