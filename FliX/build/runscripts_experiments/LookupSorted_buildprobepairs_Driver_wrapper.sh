#!/usr/bin/env bash
# file: RunLookupSorted_VaryBuildProbe.sh
# Purpose: Drive LookupSorted.sh across multiple (BUILDSIZE, PROBESIZE) pairs.
set -u -o pipefail

# ---- Target (must accept 7 args: X Y Growth Node Cacheline Build Probe) ----
BASELINE_SCRIPT="./LookupsSorted_VaryingRounds.sh"

# ---- Global params ----
GrowthVal=100

# Workloads: (X Y)
declare -a WORKLOADS=(
  "90 90"
  "50 90"
  "25 90"
  "12 90"
  "6 90"
  "3 90"
  # add more if needed: "1 90" "2 90" ...
)

# NodeSize / Cacheline pairs (exactly one must be 0)
declare -a NODE_CACHE_PAIRS=(
  #"3 0"
  "5 0" # best suited for Tiled Lookups
)

# (BUILDSIZE, PROBESIZE) pairs to sweep
declare -a BUILD_PROBE_PAIRS=(
  "25 26"
  #"15 16"
  "24 25"
  #"22 23"
  #"15 16"
  #"10 11"
)

# ---- Sanity ----
if [[ ! -r "$BASELINE_SCRIPT" ]]; then
  echo "ERROR: Missing baseline script: $BASELINE_SCRIPT" >&2
  exit 1
fi
if [[ ! -x "$BASELINE_SCRIPT" ]]; then
  echo "NOTE: Making $BASELINE_SCRIPT executable"
  chmod +x "$BASELINE_SCRIPT" || { echo "ERROR: chmod failed"; exit 1; }
fi

echo "Driver: $0"
echo "Script: $BASELINE_SCRIPT"
echo "GrowthVal: $GrowthVal"
echo "=============================================================="

# ---- Run sweeps ----
for bp in "${BUILD_PROBE_PAIRS[@]}"; do
  set -- $bp
  BUILDSIZE="$1"; PROBESIZE="$2"

  # numeric safety
  for v in "$BUILDSIZE" "$PROBESIZE"; do
    [[ "$v" =~ ^[0-9]+$ ]] || { echo "ERROR: Non-integer Build/Probe size: $v" >&2; exit 1; }
  done

  echo "==> Build/Probe: BUILDSIZE=$BUILDSIZE  PROBESIZE=$PROBESIZE"

  for wl in "${WORKLOADS[@]}"; do
    set -- $wl
    Xval="$1"; Yval="$2"
    echo "  -- Workload: X=$Xval, Y=$Yval"

    for pair in "${NODE_CACHE_PAIRS[@]}"; do
      set -- $pair
      NodeSizeInput="$1"; CachelineSizeInput="$2"

      echo "     >> NodeSizeInput=$NodeSizeInput, CachelineSizeInput=$CachelineSizeInput"
      echo "        Running: $BASELINE_SCRIPT $Xval $Yval $GrowthVal $NodeSizeInput $CachelineSizeInput $BUILDSIZE $PROBESIZE"
      rm -f data_cache/*.* || true

      "$BASELINE_SCRIPT" \
        "$Xval" "$Yval" "$GrowthVal" \
        "$NodeSizeInput" "$CachelineSizeInput" \
        "$BUILDSIZE" "$PROBESIZE" || {
          echo "ERROR: Failure at (Build=$BUILDSIZE, Probe=$PROBESIZE) WL=($Xval,$Yval) Node=$NodeSizeInput Cacheline=$CachelineSizeInput" >&2
          exit 1
        }
    done
    echo "  -- Completed workload X=$Xval, Y=$Yval for (Build=$BUILDSIZE, Probe=$PROBESIZE)"
  done

  echo "Completed sweep for (BUILDSIZE=$BUILDSIZE, PROBESIZE=$PROBESIZE)"
  echo "--------------------------------------------------------------"
done

echo "All runs completed successfully."
