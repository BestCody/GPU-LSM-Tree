#pragma once

// GPULSMOpt: the GPU LSM tree described in the CellLSM paper, for 32-bit keys
// and values.
//
// A key splits into a 16-bit region and a 16-bit suffix. Updates land in up
// to 16 unsorted pending batches grouped by region; a full set is reduced to
// the newest version of each key and published as an immutable sorted run.
// Runs sit in tiers of three slots, each tier four times larger than the one
// before; a fourth run merges with the tier into the next one. Each region of
// a run is further split into 128 cells of 512 keys to bound every search.
//
// celllsm/
//   config.cuh         constants and tier/slot capacity math
//   device_memory.cuh  device, pinned and virtual-memory buffers
//   layout.cuh         run rows, region directory, manifest, row searches
//   pending.cuh        pending batches and their per-region filters
//   publish.cuh        newest-wins reduction, publication plan, manifest switch
//   merge.cuh          merge job planning and the per-cell tournament merge
//   large_batch.cuh    runs built directly from a large sorted input
//   lookup.cuh         point lookup paths
//   range.cuh          range sum
//   successor.cuh      successor
//   index.cuh          class GPULSMOpt, which owns storage and runs operations

#include "celllsm/index.cuh"
