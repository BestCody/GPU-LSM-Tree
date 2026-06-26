#!/usr/bin/env bash
#------------------------------------------------------------
# HeatMap Trials: Single Threaded + Tiled Tests (Parametrized Node/Cache)
# Runs full experiment + plotting for each DIV in {1,2,4,8}
#------------------------------------------------------------
set -u -o pipefail
cd ..

if [[ $# -ne 7 ]]; then
  echo "Usage: $0 Xval Yval GrowthVal NodeSizeInput CachelineSizeInput BuildSize ProbeSize" >&2
  exit 1
fi

#BUILDSIZE=24
#PROBESIZE=25
Xval=$1
Yval=$2
GrowthVal=$3
NodeSizeInput=$4
CachelineSizeInput=$5
BUILDSIZE=$6
PROBESIZE=$7

# Validate numeric
for v in "$Xval" "$Yval" "$GrowthVal" "$NodeSizeInput" "$CachelineSizeInput"; do
  [[ "$v" =~ ^[0-9]+$ ]] || { echo "Error: all args must be non-negative integers." >&2; exit 1; }
done

# Exactly one of NodeSizeInput or CachelineSizeInput must be zero
if ! { { [[ "$NodeSizeInput" -eq 0 ]] && [[ "$CachelineSizeInput" -ne 0 ]]; } || { [[ "$NodeSizeInput" -ne 0 ]] && [[ "$CachelineSizeInput" -eq 0 ]]; }; }; then
  echo "Error: exactly ONE of NodeSizeInput or CachelineSizeInput must be 0." >&2
  echo "Given: NodeSizeInput=$NodeSizeInput, CachelineSizeInput=$CachelineSizeInput" >&2
  exit 1
fi

# Derive NodeSize
if [[ "$CachelineSizeInput" -eq 0 ]]; then
  NodeSize="$NodeSizeInput"
else
  if [[ "$CachelineSizeInput" -eq 10 ]]; then
    NodeSize=16
  else
    echo "Unsupported CachelineSizeInput=$CachelineSizeInput" >&2
    exit 1
  fi
fi

# Derive MaxNode / TileSize
if [[ "$CachelineSizeInput" -eq 10 ]]; then
  MaxNode=14
  TileSize=16
else
  MaxNode=$((2 ** NodeSize))
  TileSize=$((2 ** NodeSize))
fi

echo "  Running Deletions with common parameters: "
echo "  Xval=$Xval, Yval=$Yval, GrowthVal=$GrowthVal"
echo "  NodeSizeInput=$NodeSizeInput, CachelineSizeInput=$CachelineSizeInput"
echo "  NodeSize=$NodeSize, MaxNode=$MaxNode, TileSize=$TileSize"
echo "------------------------------------------------------------"

# Node/Cache CMake defines
if [[ "$CachelineSizeInput" -eq 0 ]]; then
  NodeCacheString=" -DNODESIZE=${NodeSizeInput}"
else
  NodeCacheString=" -DNODESIZE=0 -DCACHE_LINE_SIZE=${CachelineSizeInput}"
fi

#TARGET_DIR="LOOKUPS_SORTED/4rounds/New_benchmark/VaryBuildProbe/"
#TARGET_DIR="BASELINES_NOV9/NewBenchmark/SORTED/2rounds/200GROWTH/LSM_ALT"

#-----TARGET_DIR="BASELINES_NOV9/NewBenchmark/SORTED/8rounds/200GROWTH"
TARGET_DIR="Lookups/VaryingRounds/Sorted"

mkdir -p "$TARGET_DIR"

# ---------- Build helper ----------
build_and_make() {
  local base_flags="$1"
  local extra_flags="$2"

  local cmake_cmd=(cmake .. -DIFDEFS:STRING="${base_flags} ${extra_flags}")
  local make_clean_cmd=(make clean)
  local make_build_cmd=(make -j$(nproc))

  rm -f CMakeCache.txt build.log
  rm build.log

  echo "Running: ${cmake_cmd[*]}" | tee build.log
  "${cmake_cmd[@]}" 2>&1 | tee -a build.log || { echo "ERROR: cmake failed. Command: ${cmake_cmd[*]}"; exit 1; }

  echo "Running: ${make_clean_cmd[*]}" | tee -a build.log
  "${make_clean_cmd[@]}" 2>&1 | tee -a build.log || { echo "ERROR: make clean failed. Command: ${make_clean_cmd[*]}"; exit 1; }

  echo "Running: ${make_build_cmd[*]}" | tee -a build.log
  "${make_build_cmd[@]}" 2>&1 | tee -a build.log || { echo "ERROR: make build failed. Command: ${make_build_cmd[*]}"; exit 1; }
}

# ======================= OUTER LOOP OVER DIV =======================
#for Div in 1 2 4 8; do

#  BASE_IFDEFS_TILED="-DDENSEKEYGEN -DTOTALRUNS=1 -DMAIN_32 -DDIV=8 -DTILE_INSERTS -DTILE_INSERTS_C -DINSERTS_TILE_BULK_ONLY -DTILE_DELETES -DMAX_NODE=16 -DDEFINE_TILE_SIZE=16"
rm -f data_cache/*.* || true
#for Div in 1 2 4 8; do
#for Rounds in 5 7 10 12; do

for Rounds in 5 6 7 9; do
for Div in 8; do #may test varying DIV sizes for experiments; Div == 8  is optimal 
  echo "------------------------------------------------------------"
  echo "===== Starting experiments with DIV=$Div ====="

  CSV_FILES=()
  # Include DIV in the *first half* of the basename, right after R_X..Y..
  OUT_BASENAME="(OrderedLookups_100MillionBuild_100_X${Xval}Y${Yval}_NS${NodeSizeInput}_CL${CachelineSizeInput}_x${GrowthVal}_32bit_${BUILDSIZE}_${PROBESIZE}_${Rounds}ROUNDS__DIV${Div})"

  #COMMON_FLAGS="-DDENSEKEYGEN -DXVAL=${Xval} -DYVAL=${Yval} -DTOTALRUNS=1 -DMAIN_32 -DDIV=${Div} -DMAX_NODE=${MaxNode} -DDEFINE_TILE_SIZE=${TileSize}${NodeCacheString}"
  #COMMON_FLAGS_BASE="-DDENSEKEYGEN -DXVAL=${Xval} -DYVAL=${Yval} -DTOTALRUNS=1 -DMAIN_32 -DDIV=1 -DMAX_NODE=${MaxNode}"

  # Base flags for Single-threaded and Tiled runs with current DIV
  ##BASE_IFDEFS_SINGLE="-DDENSEKEYGEN -DXVAL=${Xval} -DYVAL=${Yval} -DTOTALRUNS=1 -DMAIN_32 -DDIV=${Div} -DPROCESS_INSERTS_SINGLETHREAD -DALL_SHIFT_RIGHT -DMAX_NODE=${MaxNode} -DDEFINE_TILE_SIZE=${TileSize}${NodeCacheString}"
 # -DNODESIZE=3 -DMAX_NODE=8 -DDEFINE_TILE_SIZE=8 -DDENSEKEYGEN -DTILE_DELETES -DTILE_INSERTS -DTILE_INSERTS_C -DINSERTS_TILE_BULK_ONLY -DTOTALRUNS=1 -DMAIN_32 -DDIV=8 -DXVAL= -DYVAL=90 -DLOOKUPS_CUDA"
  BASE_IFDEFS_TILED="-DDENSEKEYGEN -DXVAL=${Xval} -DYVAL=${Yval} -DTOTALRUNS=5 -DMAIN_32 -DDIV=${Div} -DTILE_INSERTS -DTILE_INSERTS_C -DINSERTS_TILE_BULK_ONLY -DTILE_DELETES -DDELETES_TILE_BULK -DMAX_NODE=${MaxNode} -DDEFINE_TILE_SIZE=${TileSize}${NodeCacheString} -DINITIAL_BUILD_SIZE=${BUILDSIZE} -DINITIAL_PROBE_SIZE=${PROBESIZE}"
  
  
  #BASE_IFDEFS_BASELINES="${COMMON_FLAGS_BASE} -DBASELINES"
  # --- Experiments ---
  #rm -f data_cache/*.* || true
# --------------------------------------- TILED: Maxvalues Buffer
  #OUT_TXT="${OUT_BASENAME}+{NoRebuild}.txt"
  #OUT_CSV="${OUT_BASENAME}+{NoRebuild}.csv"
  rm updates.csv 
  OUT_TXT="${OUT_BASENAME}+{FliX}.txt"
  OUT_CSV="${OUT_BASENAME}+{FliX}.csv"
  build_and_make "$BASE_IFDEFS_TILED" " -DROUNDS_NUMBER=${Rounds}"
  ./index_prototype &> "$OUT_TXT"
  cp updates.csv "$OUT_CSV"
  mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
  CSV_FILES+=("${TARGET_DIR}/${OUT_CSV}")
  echo "Sorted lookups done completed ."

echo "All runs completed successfully."



# previous experiments with rebuilding and ray tracing index layer
if false; then
# --------------------------------------- Ray Tracing LOOKUPS
  #OUT_TXT="${OUT_BASENAME}+{NoRebuild}.txt"
  #OUT_CSV="${OUT_BASENAME}+{NoRebuild}.csv"
  rm updates.csv 
  OUT_TXT="${OUT_BASENAME}+{RayTracing}.txt"
  OUT_CSV="${OUT_BASENAME}+{RayTracing}.csv"
  build_and_make "$BASE_IFDEFS_TILED" "-DTILE_DELETES"
  ./index_prototype &> "$OUT_TXT"
  cp updates.csv "$OUT_CSV"
  mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
  CSV_FILES+=("${TARGET_DIR}/${OUT_CSV}")
  echo "Ray tracing lookups done completed ."
# --------------------------------------- REBUILD Every 1
  OUT_TXT="${OUT_BASENAME}+{RebuildEvery1}.txt"
  OUT_CSV="${OUT_BASENAME}+{RebuildEvery1}.csv"
  build_and_make "$BASE_IFDEFS_TILED" "-DREBUILD_ON"
  ./index_prototype &> "$OUT_TXT"
  cp updates.csv "$OUT_CSV"
  mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
  CSV_FILES+=("${TARGET_DIR}/${OUT_CSV}")
  echo "Rebuild every 1 Execution done completed ."

# ---------------------------------------REBUILD Every 2
  OUT_TXT="${OUT_BASENAME}+{RebuildEvery2}.txt"
  OUT_CSV="${OUT_BASENAME}+{RebuildEvery2}.csv"
  build_and_make "$BASE_IFDEFS_TILED" "-DREBUILD_ON -DREBUILD_INTERVAL=2"
  ./index_prototype &> "$OUT_TXT"
  cp updates.csv "$OUT_CSV"
  mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
  CSV_FILES+=("${TARGET_DIR}/${OUT_CSV}")
  echo "Rebuild every 2 Execution done completed ."

# ---------------------------------------REBUILD Every 4
  OUT_TXT="${OUT_BASENAME}+{RebuildEvery4}.txt"
  OUT_CSV="${OUT_BASENAME}+{RebuildEvery4}.csv"
  build_and_make "$BASE_IFDEFS_TILED" "-DREBUILD_ON -DREBUILD_INTERVAL=4"
  ./index_prototype &> "$OUT_TXT"
  cp updates.csv "$OUT_CSV"
  mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
  CSV_FILES+=("${TARGET_DIR}/${OUT_CSV}")
  echo "Rebuild every 4 Execution done completed ."

# ---------------------------------------REBUILD Every 8
  OUT_TXT="${OUT_BASENAME}+{RebuildEvery8}.txt"
  OUT_CSV="${OUT_BASENAME}+{RebuildEvery8}.csv"
  build_and_make "$BASE_IFDEFS_TILED" "-DREBUILD_ON -DREBUILD_INTERVAL=8"
  ./index_prototype &> "$OUT_TXT"
  cp updates.csv "$OUT_CSV"
  mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
  CSV_FILES+=("${TARGET_DIR}/${OUT_CSV}")
  echo "Rebuild every 8 Execution done completed ."
fi 
