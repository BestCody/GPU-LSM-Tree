#!/usr/bin/env bash
# actual experiment and file generation occurs here
set -u -o pipefail
cd ..

# ------------------ Args ------------------
if [[ $# -ne 7 ]]; then
  echo "Usage: $0 Xval Yval GrowthVal NodeSizeInput CachelineSizeInput BuildSize ProbeSize" >&2
  exit 1
fi

Xval=$1
Yval=$2
GrowthVal=$3
NodeSizeInput=$4
CachelineSizeInput=$5
BUILDSIZE=$6
PROBESIZE=$7

# Validate numeric
for v in "$Xval" "$Yval" "$GrowthVal" "$NodeSizeInput" "$CachelineSizeInput" "$BUILDSIZE" "$PROBESIZE"; do
  [[ "$v" =~ ^[0-9]+$ ]] || { echo "Error: all args must be non-negative integers." >&2; exit 1; }
done

echo "==================== RAW ARGS ===================="
echo "Xval=$Xval"
echo "Yval=$Yval"
echo "GrowthVal=$GrowthVal"
echo "NodeSizeInput=$NodeSizeInput"
echo "CachelineSizeInput=$CachelineSizeInput"
echo "BUILDSIZE=$BUILDSIZE"
echo "PROBESIZE=$PROBESIZE"
echo "PWD=$(pwd)"
echo "=================================================="

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

# Node/Cache CMake defines
if [[ "$CachelineSizeInput" -eq 0 ]]; then
  NodeCacheString="-DNODESIZE=${NodeSizeInput} -DCACHE_LINE_SIZE=0"
else
  NodeCacheString="-DNODESIZE=0 -DCACHE_LINE_SIZE=${CachelineSizeInput}"
fi

echo "==================== DERIVED ====================="
echo "NodeSize=$NodeSize"
echo "MaxNode=$MaxNode"
echo "TileSize=$TileSize"
echo "NodeCacheString=$NodeCacheString"
echo "=================================================="
#TARGET_DIR="KEEPImportantFIles/100MBuild100MProbe/FliXSkewCompare"

TARGET_DIR="Del_Experiments/"

#TARGET_DIR="KEEPImportantFIles/100MBuild100MProbe/FliXSkewCompare/Baselines"
mkdir -p "$TARGET_DIR"

# ---------- Build helper (per-run log, fail fast) ----------
build_and_make() {
  local base_flags="$1"
  local extra_flags="$2"
  local run_tag="$3"

  local cmake_cmd=(cmake .. -DIFDEFS:STRING="${base_flags} ${extra_flags}")
  local make_clean_cmd=(make clean)
  local make_build_cmd=(make -j"$(nproc)")

  rm -f CMakeCache.txt build.log || true
  rm -f build.log || true

  echo "==================== BUILD ${run_tag} ====================" | tee build.log
  echo "PWD=$(pwd)" | tee -a build.log
  echo "IFDEFS(base)=${base_flags}" | tee -a build.log
  echo "IFDEFS(extra)=${extra_flags}" | tee -a build.log
  echo "Running: ${cmake_cmd[*]}" | tee -a build.log
  "${cmake_cmd[@]}" 2>&1 | tee -a build.log || { echo "ERROR: cmake failed. Command: ${cmake_cmd[*]}"; exit 1; }

  # Inspect cache for actual values (if present)
  if [[ -f CMakeCache.txt ]]; then
    echo "---- CMakeCache extract (NODESIZE/CACHE_LINE_SIZE) ----" | tee -a build.log
    grep -E '(^|;)NODESIZE(:|=)|(^|;)CACHE_LINE_SIZE(:|=)' CMakeCache.txt | tee -a build.log || true
    echo "------------------------------------------------------" | tee -a build.log
  else
    echo "WARN: CMakeCache.txt not found in PWD=$(pwd)" | tee -a build.log
  fi

  echo "Running: ${make_clean_cmd[*]}" | tee -a build.log
  "${make_clean_cmd[@]}" 2>&1 | tee -a build.log || { echo "ERROR: make clean failed. Command: ${make_clean_cmd[*]}"; exit 1; }

  echo "Running: ${make_build_cmd[*]}" | tee -a build.log
  "${make_build_cmd[@]}" 2>&1 | tee -a build.log || { echo "ERROR: make build failed. Command: ${make_build_cmd[*]}"; exit 1; }
}

CSV_FILES=()

# NOTE: TL BULK DELETES
COMMON_FLAGS="-DDENSEKEYGEN -DXVAL=${Xval} -DYVAL=${Yval} -DTOTALRUNS=3 -DMAIN_32 -DDIV=8 \
-DTILE_INSERTS -DTILE_INSERTS_C -DINSERTS_TILE_BULK_ONLY -DTILE_DELETES -DDELETES_TILE_BULK \
-DMAX_NODE=${MaxNode} -DDEFINE_TILE_SIZE=32 -DINITIAL_BUILD_SIZE=${BUILDSIZE} -DINITIAL_PROBE_SIZE=${PROBESIZE} \
${NodeCacheString}"

# NOTE: TL Shift Left Deletest
COMMON_FLAGS1="-DDENSEKEYGEN -DXVAL=${Xval} -DYVAL=${Yval} -DTOTALRUNS=3 -DMAIN_32 -DDIV=8 \
-DTILE_INSERTS -DTILE_INSERTS_C -DINSERTS_TILE_BULK_ONLY -DTILE_DELETES  \
-DMAX_NODE=${MaxNode} -DDEFINE_TILE_SIZE=32 -DINITIAL_BUILD_SIZE=${BUILDSIZE} -DINITIAL_PROBE_SIZE=${PROBESIZE} \
${NodeCacheString}"

# NOTE: ST (Single Threaded) Shift Left Deletes - NORMAL DELETES (not bulk)
#CHANGE NODE SIZE MANUALLY to 8 for this since we want to test single threaded deletes on the smallest node size (which is 16 for tile-based, but we want to see 8 as well) 
NodeCacheString2="-DNODESIZE=3 -DCACHE_LINE_SIZE=0"
COMMON_FLAGS2="-DDENSEKEYGEN -DXVAL=${Xval} -DYVAL=${Yval} -DTOTALRUNS=3 -DMAIN_32 -DDIV=4 \
-DTILE_INSERTS -DTILE_INSERTS_C -DINSERTS_TILE_BULK_ONLY  \
-DMAX_NODE=3 -DDEFINE_TILE_SIZE=8 -DINITIAL_BUILD_SIZE=${BUILDSIZE} -DINITIAL_PROBE_SIZE=${PROBESIZE} \
${NodeCacheString2}"


BASE_IFDEFS_BASELINES="${COMMON_FLAGS} -DBASELINES"

echo "==================== FLAGS ======================="
echo "COMMON_FLAGS=${COMMON_FLAGS}"
echo "BASE_IFDEFS_BASELINES=${BASE_IFDEFS_BASELINES}"
echo "=================================================="

rm -f data_cache/*.* || true
# **************************************
# MAIN LOOP
# ************************************* 
for Rounds in 4 8; do
#for Rounds in 4 6 5 8; do
  #OUT_BASENAME="X${Xval}Y${Yval}_100MB_100MP_DIV8_NS${NodeSizeInput}_CL${CachelineSizeInput}_x${GrowthVal}_32bit_${BUILDSIZE}_${PROBESIZE}_${Rounds}ROUNDS"
  OUT_BASENAME="X${Xval}Y${Yval}_DEL_25Build_26Probe_DIVVarying_NS${NodeSizeInput}_CL${CachelineSizeInput}_x${GrowthVal}_32bit_${BUILDSIZE}_${PROBESIZE}_${Rounds}ROUNDS"
  
  rm -f data_cache/*.* || true
  # Save params snapshot
  PARAM_LOG="${TARGET_DIR}/${OUT_BASENAME}_run_params.log"
  {
    echo "RAW: Xval=$Xval Yval=$Yval GrowthVal=$GrowthVal NodeSizeInput=$NodeSizeInput CachelineSizeInput=$CachelineSizeInput BUILDSIZE=$BUILDSIZE PROBESIZE=$PROBESIZE"
    echo "DERIVED: NodeSize=$NodeSize MaxNode=$MaxNode TileSize=$TileSize"
    echo "NodeCacheString=$NodeCacheString"
    echo "COMMON_FLAGS=$COMMON_FLAGS"
    echo "BASE_IFDEFS_BASELINES=$BASE_IFDEFS_BASELINES"
    echo "PWD=$(pwd)"
    date
  } > "$PARAM_LOG"

  

#----------------------------------------------
### REPEAT FOR ALL DELETION EXPERIMENTS
#----------------------------------------------
 #if false; then
  # ------------------------ FLIX ------------------------------
  rm -f updates.csv || true
#if false; then
  OUT_TXT="${OUT_BASENAME}+{ALLBULK_TILED}.txt"
  OUT_CSV="${OUT_BASENAME}+{ALLBULK_TILED}.csv"
  build_and_make "${COMMON_FLAGS}" "-DROUNDS_NUMBER=${Rounds}" "FLIX_R${Rounds}"
  ./index_prototype &> "$OUT_TXT"
  cp updates.csv "$OUT_CSV"
  mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
  CSV_FILES+=("${TARGET_DIR}/${OUT_CSV}")
  echo "FliX TL BUlk Del completed."
#fi

   rm -f updates.csv || true

  OUT_TXT="${OUT_BASENAME}+{TL-SHIFT}.txt"
  OUT_CSV="${OUT_BASENAME}+{TL-SHIFT}.csv"
  build_and_make "${COMMON_FLAGS1}" "-DROUNDS_NUMBER=${Rounds}" "FLIX_R${Rounds}"
  ./index_prototype &> "$OUT_TXT"
  cp updates.csv "$OUT_CSV"
  mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
  CSV_FILES+=("${TARGET_DIR}/${OUT_CSV}")
  echo "FliX TL ShiftLeft done completed."

  rm -f updates.csv || true

  OUT_TXT="${OUT_BASENAME}+{ST-SHIFT}.txt"
  OUT_CSV="${OUT_BASENAME}+{ST-SHIFT}.csv"
  build_and_make "${COMMON_FLAGS2}" "-DROUNDS_NUMBER=${Rounds}" "FLIX_R${Rounds}"
  ./index_prototype &> "$OUT_TXT"
  cp updates.csv "$OUT_CSV"
  mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
  CSV_FILES+=("${TARGET_DIR}/${OUT_CSV}")
  echo "FliX ST ShiftLeft done completed."

# fi

done




# --NOT USED-- 
if false; then
  # ------------------------ LSM_TREE ------------------------------
  rm -f updates.csv || true

  OUT_TXT="${OUT_BASENAME}+{LSMu}.txt"
  OUT_CSV="${OUT_BASENAME}+{LSMu}.csv"
  build_and_make "$BASE_IFDEFS_BASELINES" "-DLSM_TREE -DLSM_CHUNK_SIZE_LOG=16 -DROUNDS_NUMBER=${Rounds}" "LSM_TREE_R${Rounds}"
  ./index_prototype &> "$OUT_TXT"
  cp updates.csv "$OUT_CSV"
  mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
  CSV_FILES+=("${TARGET_DIR}/${OUT_CSV}")
  echo "LSM Tree complete."

  # ------------------------ HASHTABLE: SLAB ------------------------
  rm -f updates.csv || true

  OUT_TXT="${OUT_BASENAME}+{HASHTABLE_SLAB}.txt"
  OUT_CSV="${OUT_BASENAME}+{HASHTABLE_SLAB}.csv"
  build_and_make "$BASE_IFDEFS_BASELINES" "-DHASHTABLE_SLAB -DROUNDS_NUMBER=${Rounds}" "SLAB_R${Rounds}"
  ./index_prototype &> "$OUT_TXT"
  cp updates.csv "$OUT_CSV"
  mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
  CSV_FILES+=("${TARGET_DIR}/${OUT_CSV}")
  echo "SlabHash complete."

  # ------------------------ HASHTABLE: WARPCORE --------------------
  rm -f updates.csv || true

  OUT_TXT="${OUT_BASENAME}+{HASHTABLE_WARPCORE}.txt"
  OUT_CSV="${OUT_BASENAME}+{HASHTABLE_WARPCORE}.csv"
  build_and_make "$BASE_IFDEFS_BASELINES" "-DHASHTABLE_WARPCORE -DROUNDS_NUMBER=${Rounds}" "WARPCORE_R${Rounds}"
  ./index_prototype &> "$OUT_TXT"
  cp updates.csv "$OUT_CSV"
  mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
  CSV_FILES+=("${TARGET_DIR}/${OUT_CSV}")
  echo "Warpcore complete."

  # ------------------------ GPU_BTREE ------------------------------
  rm -f updates.csv || true

  OUT_TXT="${OUT_BASENAME}+{GPU_BTREE}.txt"
  OUT_CSV="${OUT_BASENAME}+{GPU_BTREE}.csv"
  build_and_make "$BASE_IFDEFS_BASELINES" "-DGPU_BTREE -DROUNDS_NUMBER=${Rounds}" "GPU_BTREE_R${Rounds}"
  ./index_prototype &> "$OUT_TXT"
  cp updates.csv "$OUT_CSV"
  mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
  CSV_FILES+=("${TARGET_DIR}/${OUT_CSV}")
  echo "GPU B Tree complete."
 fi
 