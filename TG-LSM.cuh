#pragma once

#include "gpu_dictionary_adapter.cuh"

#include <cuda_runtime.h>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <vector>

#ifndef TGLSM_CUDA_CHECK
#define TGLSM_CUDA_CHECK(stmt)                                                \
  do {                                                                        \
    cudaError_t err__ = (stmt);                                               \
    if (err__ != cudaSuccess) {                                               \
      throw std::runtime_error(cudaGetErrorString(err__));                    \
    }                                                                         \
  } while (false)
#endif

#ifndef TGLSM_TILE_KEYS
#define TGLSM_TILE_KEYS 1024
#endif
#ifndef TGLSM_C0_FLUSH_TRIGGER
#define TGLSM_C0_FLUSH_TRIGGER (1 << 20)
#endif
#ifndef TGLSM_C0_INDEX_TRIGGER
#define TGLSM_C0_INDEX_TRIGGER (1 << 18)
#endif
#ifndef TGLSM_RAW_SCAN_LIMIT
#define TGLSM_RAW_SCAN_LIMIT (1 << 22)
#endif
#ifndef TGLSM_TARGET_GUARD_KEYS
#define TGLSM_TARGET_GUARD_KEYS (1 << 18)
#endif
#ifndef TGLSM_MAX_GUARDS
#define TGLSM_MAX_GUARDS 4096
#endif
#ifndef TGLSM_MAX_FRAGS_PER_GUARD
#define TGLSM_MAX_FRAGS_PER_GUARD 8
#endif
#ifndef TGLSM_GUARD_COMPACT_KEYS
#define TGLSM_GUARD_COMPACT_KEYS (1 << 20)
#endif

namespace tglsm_detail {

constexpr std::uint32_t kEmptyKey =
    std::numeric_limits<std::uint32_t>::max();
constexpr std::uint8_t kInsert = 1;
constexpr std::uint8_t kTombstone = 0;

struct DeviceKeyBatch {
  const std::uint32_t *keys = nullptr;
  std::size_t count = 0;
};

struct DeviceUpdateBatch {
  const std::uint32_t *keys = nullptr;
  const std::uint32_t *values = nullptr;
  const std::uint8_t *ops = nullptr;
  std::size_t count = 0;
};

struct FragmentMeta {
  std::uint32_t base = 0;
  std::uint32_t len = 0;
  std::uint32_t guard = 0;
  std::uint32_t min_key = 0;
  std::uint32_t max_key = 0;
};

struct ResolveResult {
  std::uint32_t seq = 0;
  std::uint32_t value = 0;
  std::uint8_t op = kTombstone;
  std::uint8_t seen = 0;
};

struct DeviceView {
  const std::uint32_t *guard_hi = nullptr;
  const std::uint32_t *guard_first = nullptr;
  const std::uint32_t *guard_count = nullptr;
  std::uint32_t num_guards = 0;
  const std::uint32_t *frag_base = nullptr;
  const std::uint32_t *frag_len = nullptr;
  const std::uint32_t *frag_min = nullptr;
  const std::uint32_t *frag_max = nullptr;
  std::uint32_t frag_count = 0;
  const std::uint32_t *store_keys = nullptr;
  const std::uint32_t *store_values = nullptr;
  const std::uint8_t *store_ops = nullptr;
  const std::uint32_t *store_seqs = nullptr;
  const std::uint32_t *c0_keys = nullptr;
  const std::uint32_t *c0_values = nullptr;
  const std::uint8_t *c0_ops = nullptr;
  const std::uint32_t *c0_seqs = nullptr;
  std::uint32_t c0_count = 0;
  const std::uint32_t *raw_keys = nullptr;
  const std::uint32_t *raw_values = nullptr;
  const std::uint8_t *raw_ops = nullptr;
  const std::uint32_t *raw_seqs = nullptr;
  std::uint32_t raw_count = 0;
};

__host__ __device__ inline std::size_t
lower_bound_u32(const std::uint32_t *data, std::size_t n,
                std::uint32_t key) {
  std::size_t first = 0;
  while (n > 0) {
    const std::size_t step = n >> 1;
    const std::size_t mid = first + step;
    if (data[mid] < key) {
      first = mid + 1;
      n -= step + 1;
    } else {
      n = step;
    }
  }
  return first;
}

__host__ __device__ inline std::size_t
upper_bound_u32(const std::uint32_t *data, std::size_t n,
                std::uint32_t key) {
  std::size_t first = 0;
  while (n > 0) {
    const std::size_t step = n >> 1;
    const std::size_t mid = first + step;
    if (!(key < data[mid])) {
      first = mid + 1;
      n -= step + 1;
    } else {
      n = step;
    }
  }
  return first;
}

__host__ __device__ inline std::uint64_t
packed_sort_key(std::uint32_t key, std::uint32_t seq) {
  return (static_cast<std::uint64_t>(key) << 32) |
         static_cast<std::uint64_t>(UINT32_MAX - seq);
}

__host__ __device__ inline std::uint32_t
route_guard(const std::uint32_t *guard_hi, std::uint32_t num_guards,
            std::uint32_t key) {
  if (num_guards == 0)
    return 0;
  std::uint32_t g =
      static_cast<std::uint32_t>(lower_bound_u32(guard_hi, num_guards, key));
  return g < num_guards ? g : num_guards - 1;
}

__device__ inline void maybe_take(ResolveResult *r, std::uint32_t seq,
                                  std::uint32_t value, std::uint8_t op) {
  if (!r->seen || seq > r->seq) {
    r->seen = 1;
    r->seq = seq;
    r->value = value;
    r->op = op;
  }
}

__device__ inline ResolveResult resolve_key(const DeviceView &view,
                                            std::uint32_t key) {
  ResolveResult r;
  for (std::uint32_t i = 0; i < view.raw_count; ++i) {
    if (view.raw_keys[i] == key) {
      maybe_take(&r, view.raw_seqs[i], view.raw_values[i],
                 view.raw_ops[i]);
    }
  }
  if (view.c0_count > 0) {
    const std::size_t p =
        lower_bound_u32(view.c0_keys, view.c0_count, key);
    if (p < view.c0_count && view.c0_keys[p] == key) {
      maybe_take(&r, view.c0_seqs[p], view.c0_values[p],
                 view.c0_ops[p]);
    }
  }
  if (view.num_guards == 0 || view.frag_count == 0)
    return r;
  const std::uint32_t g =
      route_guard(view.guard_hi, view.num_guards, key);
  const std::uint32_t first = view.guard_first[g];
  const std::uint32_t count = view.guard_count[g];
  for (std::uint32_t j = 0; j < count; ++j) {
    const std::uint32_t f = first + j;
    if (key < view.frag_min[f] || key > view.frag_max[f])
      continue;
    const std::uint32_t base = view.frag_base[f];
    const std::uint32_t len = view.frag_len[f];
    const std::size_t p =
        lower_bound_u32(view.store_keys + base, len, key);
    if (p < len && view.store_keys[base + p] == key) {
      maybe_take(&r, view.store_seqs[base + p],
                 view.store_values[base + p],
                 view.store_ops[base + p]);
    }
  }
  return r;
}

__device__ inline bool min_candidate(const DeviceView &view,
                                     std::uint32_t key,
                                     std::uint32_t *out_key) {
  bool have = false;
  std::uint32_t best = kEmptyKey;
  if (view.c0_count > 0) {
    const std::size_t p =
        lower_bound_u32(view.c0_keys, view.c0_count, key);
    if (p < view.c0_count) {
      best = view.c0_keys[p];
      have = true;
    }
  }
  for (std::uint32_t i = 0; i < view.raw_count; ++i) {
    const std::uint32_t k = view.raw_keys[i];
    if (k >= key && (!have || k < best)) {
      best = k;
      have = true;
    }
  }
  if (view.num_guards == 0 || view.frag_count == 0) {
    *out_key = best;
    return have;
  }
  const std::uint32_t start =
      route_guard(view.guard_hi, view.num_guards, key);
  for (std::uint32_t g = start; g < view.num_guards; ++g) {
    std::uint32_t local = kEmptyKey;
    bool local_have = false;
    const std::uint32_t first = view.guard_first[g];
    const std::uint32_t count = view.guard_count[g];
    for (std::uint32_t j = 0; j < count; ++j) {
      const std::uint32_t f = first + j;
      if (view.frag_max[f] < key)
        continue;
      const std::uint32_t base = view.frag_base[f];
      const std::uint32_t len = view.frag_len[f];
      const std::size_t p =
          lower_bound_u32(view.store_keys + base, len, key);
      if (p < len) {
        const std::uint32_t k = view.store_keys[base + p];
        if (!local_have || k < local) {
          local = k;
          local_have = true;
        }
      }
    }
    if (local_have) {
      if (!have || local < best) {
        best = local;
        have = true;
      }
      break;
    }
  }
  *out_key = best;
  return have;
}

__device__ inline void range_accum(const DeviceView &view, std::uint32_t lo,
                                   std::uint32_t hi,
                                   std::uint32_t *sum,
                                   std::uint32_t *count) {
  std::uint32_t s = 0;
  std::uint32_t c = 0;
  for (std::uint32_t i = 0; i < view.raw_count; ++i) {
    const std::uint32_t key = view.raw_keys[i];
    if (key < lo || key > hi)
      continue;
    const ResolveResult r = resolve_key(view, key);
    if (r.seen && r.seq == view.raw_seqs[i] && r.op == kInsert) {
      s += view.raw_values[i];
      ++c;
    }
  }
  if (view.c0_count > 0) {
    const std::size_t b =
        lower_bound_u32(view.c0_keys, view.c0_count, lo);
    const std::size_t e =
        upper_bound_u32(view.c0_keys, view.c0_count, hi);
    for (std::size_t p = b; p < e; ++p) {
      const ResolveResult r = resolve_key(view, view.c0_keys[p]);
      if (r.seen && r.seq == view.c0_seqs[p] && r.op == kInsert) {
        s += view.c0_values[p];
        ++c;
      }
    }
  }
  if (view.num_guards == 0 || view.frag_count == 0) {
    *sum = s;
    *count = c;
    return;
  }
  const std::uint32_t g0 =
      route_guard(view.guard_hi, view.num_guards, lo);
  const std::uint32_t g1 =
      route_guard(view.guard_hi, view.num_guards, hi);
  for (std::uint32_t g = g0; g <= g1; ++g) {
    const std::uint32_t first = view.guard_first[g];
    const std::uint32_t frag_n = view.guard_count[g];
    for (std::uint32_t j = 0; j < frag_n; ++j) {
      const std::uint32_t f = first + j;
      if (view.frag_max[f] < lo || view.frag_min[f] > hi)
        continue;
      const std::uint32_t base = view.frag_base[f];
      const std::uint32_t len = view.frag_len[f];
      std::size_t p = lower_bound_u32(view.store_keys + base, len, lo);
      while (p < len) {
        const std::uint32_t key = view.store_keys[base + p];
        if (key > hi)
          break;
        const ResolveResult r = resolve_key(view, key);
        if (r.seen && r.seq == view.store_seqs[base + p] &&
            r.op == kInsert) {
          s += view.store_values[base + p];
          ++c;
        }
        ++p;
      }
    }
  }
  *sum = s;
  *count = c;
}

__global__ void init_append_kernel(std::uint32_t *values,
                                   std::uint8_t *ops,
                                   std::uint32_t *seqs,
                                   std::uint8_t op,
                                   std::uint32_t seq_base,
                                   std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  if (op == kTombstone)
    values[i] = 0;
  ops[i] = op;
  seqs[i] = seq_base + static_cast<std::uint32_t>(i);
}

__global__ void fill_sort_keys_kernel(const std::uint32_t *keys,
                                      const std::uint32_t *seqs,
                                      std::uint64_t *sort_keys,
                                      std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    sort_keys[i] = packed_sort_key(keys[i], seqs[i]);
}

__global__ void lookup_kernel(DeviceView view,
                              const std::uint32_t *queries,
                              std::uint32_t *out_values,
                              std::uint8_t *out_found,
                              std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const ResolveResult r = resolve_key(view, queries[i]);
  const bool found = r.seen && r.op == kInsert;
  out_found[i] = found ? 1u : 0u;
  out_values[i] = found ? r.value : 0u;
}

__global__ void successor_kernel(DeviceView view,
                                 const std::uint32_t *queries,
                                 std::uint32_t *out_keys,
                                 std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  std::uint32_t cur = queries[i];
  for (;;) {
    std::uint32_t candidate = kEmptyKey;
    if (!min_candidate(view, cur, &candidate)) {
      out_keys[i] = kEmptyKey;
      return;
    }
    const ResolveResult r = resolve_key(view, candidate);
    if (r.seen && r.op == kInsert) {
      out_keys[i] = candidate;
      return;
    }
    if (candidate == kEmptyKey) {
      out_keys[i] = kEmptyKey;
      return;
    }
    cur = candidate + 1u;
    if (cur == 0u) {
      out_keys[i] = kEmptyKey;
      return;
    }
  }
}

__global__ void count_kernel(DeviceView view, const std::uint32_t *lo,
                             const std::uint32_t *hi,
                             std::uint32_t *out_counts,
                             std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  if (lo[i] > hi[i]) {
    out_counts[i] = 0;
    return;
  }
  std::uint32_t sum = 0;
  std::uint32_t count = 0;
  range_accum(view, lo[i], hi[i], &sum, &count);
  out_counts[i] = count;
}

__global__ void range_kernel(DeviceView view, const std::uint32_t *lo,
                             const std::uint32_t *hi,
                             std::uint32_t *out_sums,
                             std::uint32_t *out_counts,
                             std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  if (lo[i] > hi[i]) {
    out_sums[i] = 0;
    if (out_counts)
      out_counts[i] = 0;
    return;
  }
  std::uint32_t sum = 0;
  std::uint32_t count = 0;
  range_accum(view, lo[i], hi[i], &sum, &count);
  out_sums[i] = sum;
  if (out_counts)
    out_counts[i] = count;
}

struct IsTombstoneTuple {
  template <class Tuple>
  __host__ __device__ bool operator()(const Tuple &t) const {
    return thrust::get<2>(t) == kTombstone;
  }
};

}

class TGLSM {
public:
  using DeviceKeyBatch = tglsm_detail::DeviceKeyBatch;
  using DeviceUpdateBatch = tglsm_detail::DeviceUpdateBatch;
  static constexpr const char *name = "TG-LSM";

  explicit TGLSM(const DictionaryConfig &config)
      : max_elements_(config.max_elements), batch_size_(config.batch_size) {
    if (max_elements_ > 0x7fffffffu) {
      throw std::invalid_argument(
          "TGLSM currently supports at most 2^31-1 records");
    }
    reserve_front_buffers();
    init_guards(0);
  }

  TGLSM(const TGLSM &) = delete;
  TGLSM &operator=(const TGLSM &) = delete;

  void clear(cudaStream_t stream) {
    std::unique_lock<std::shared_mutex> guard(mutex_);
    log_keys_.clear();
    log_values_.clear();
    log_ops_.clear();
    log_seqs_.clear();
    c0_keys_.clear();
    c0_values_.clear();
    c0_ops_.clear();
    c0_seqs_.clear();
    store_keys_.clear();
    store_values_.clear();
    store_ops_.clear();
    store_seqs_.clear();
    frags_.clear();
    next_seq_ = 1;
    reserve_front_buffers();
    init_guards(stream);
    TGLSM_CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void insert(const DeviceKeyValueBatch &batch, cudaStream_t stream) {
    if (batch.count == 0)
      return;
    if (!batch.values)
      throw std::invalid_argument("TGLSM insert requires values");
    std::unique_lock<std::shared_mutex> guard(mutex_);
    append_records(batch.keys, batch.values, tglsm_detail::kInsert,
                   batch.count, stream);
    maybe_flush_after_write(stream);
  }

  void erase(const DeviceKeyBatch &batch, cudaStream_t stream) {
    if (batch.count == 0)
      return;
    std::unique_lock<std::shared_mutex> guard(mutex_);
    append_records(batch.keys, nullptr, tglsm_detail::kTombstone,
                   batch.count, stream);
    maybe_flush_after_write(stream);
  }

  void lookup(const DeviceLookupBatch &batch, cudaStream_t stream) {
    if (batch.count == 0)
      return;
    std::unique_lock<std::shared_mutex> guard(mutex_);
    prepare_for_read(batch.count, stream);
    const tglsm_detail::DeviceView view = device_view();
    const int block = 256;
    const int grid = static_cast<int>((batch.count + block - 1) / block);
    tglsm_detail::lookup_kernel<<<grid, block, 0, stream>>>(
        view, batch.queries, batch.out_values, batch.out_found,
        batch.count);
    TGLSM_CUDA_CHECK(cudaGetLastError());
    TGLSM_CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void count(const DeviceRangeBatch &batch, cudaStream_t stream) {
    if (batch.count == 0)
      return;
    std::unique_lock<std::shared_mutex> guard(mutex_);
    prepare_for_read(batch.count, stream);
    const tglsm_detail::DeviceView view = device_view();
    const int block = 128;
    const int grid = static_cast<int>((batch.count + block - 1) / block);
    tglsm_detail::count_kernel<<<grid, block, 0, stream>>>(
        view, batch.lo, batch.hi, batch.out_counts, batch.count);
    TGLSM_CUDA_CHECK(cudaGetLastError());
    TGLSM_CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void successor(const DeviceSuccessorBatch &batch, cudaStream_t stream) {
    if (batch.count == 0)
      return;
    std::unique_lock<std::shared_mutex> guard(mutex_);
    prepare_for_read(batch.count, stream);
    const tglsm_detail::DeviceView view = device_view();
    const int block = 128;
    const int grid = static_cast<int>((batch.count + block - 1) / block);
    tglsm_detail::successor_kernel<<<grid, block, 0, stream>>>(
        view, batch.queries, batch.out_keys, batch.count);
    TGLSM_CUDA_CHECK(cudaGetLastError());
    TGLSM_CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void range(const DeviceRangeOutputBatch &batch, cudaStream_t stream) {
    if (batch.query_count == 0)
      return;
    std::unique_lock<std::shared_mutex> guard(mutex_);
    prepare_for_read(batch.query_count, stream);
    const tglsm_detail::DeviceView view = device_view();
    const int block = 128;
    const int grid =
        static_cast<int>((batch.query_count + block - 1) / block);
    tglsm_detail::range_kernel<<<grid, block, 0, stream>>>(
        view, batch.lo, batch.hi, batch.out_sums, batch.out_counts,
        batch.query_count);
    TGLSM_CUDA_CHECK(cudaGetLastError());
    TGLSM_CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void cleanup(cudaStream_t stream) {
    std::unique_lock<std::shared_mutex> guard(mutex_);
    build_c0_index(stream);
    flush_c0_to_fragments(stream);
    maintain_hot_guards(stream);
    TGLSM_CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void maintain(cudaStream_t stream) { cleanup(stream); }

  void drain_to_sheet(cudaStream_t stream) { cleanup(stream); }

  std::size_t sheet_live_count() const {
    std::shared_lock<std::shared_mutex> guard(mutex_);
    return store_keys_.size() + c0_keys_.size() + log_keys_.size();
  }

  std::size_t run_count() const {
    std::shared_lock<std::shared_mutex> guard(mutex_);
    return frags_.size();
  }

  std::size_t segment_count() const {
    std::shared_lock<std::shared_mutex> guard(mutex_);
    return h_guard_hi_.size();
  }

  std::size_t sheet_bucket_count() const {
    std::shared_lock<std::shared_mutex> guard(mutex_);
    return frags_.size();
  }

  std::size_t sheet_capacity() const {
    std::shared_lock<std::shared_mutex> guard(mutex_);
    return store_keys_.capacity();
  }

  std::size_t gpu_resident_bytes() const {
    std::shared_lock<std::shared_mutex> guard(mutex_);
    return device_bytes_all(
        log_keys_, log_values_, log_ops_, log_seqs_, c0_keys_, c0_values_,
        c0_ops_, c0_seqs_, store_keys_, store_values_, store_ops_,
        store_seqs_, d_guard_hi_, d_guard_first_, d_guard_count_,
        d_frag_base_, d_frag_len_, d_frag_min_, d_frag_max_, tmp_keys_,
        tmp_values_, tmp_ops_, tmp_seqs_, sort_keys_, guard_offsets_);
  }

private:
  template <class T> static T *raw_or_null(thrust::device_vector<T> &v) {
    return v.empty() ? nullptr : thrust::raw_pointer_cast(v.data());
  }

  template <class T>
  static const T *raw_or_null(const thrust::device_vector<T> &v) {
    return v.empty() ? nullptr : thrust::raw_pointer_cast(v.data());
  }

  template <class T>
  static std::size_t device_bytes(const thrust::device_vector<T> &v) {
    return v.size() * sizeof(T);
  }

  template <class... Vecs>
  static std::size_t device_bytes_all(const Vecs &...vecs) {
    return (std::size_t{0} + ... + device_bytes(vecs));
  }

  static std::uint32_t checked_u32(std::size_t n, const char *name) {
    if (n > std::numeric_limits<std::uint32_t>::max())
      throw std::overflow_error(name);
    return static_cast<std::uint32_t>(n);
  }

  template <class T>
  void upload_host_vector(thrust::device_vector<T> &dst,
                          const std::vector<T> &src,
                          cudaStream_t stream) {
    dst.resize(src.size());
    if (!src.empty()) {
      TGLSM_CUDA_CHECK(cudaMemcpyAsync(
          raw_or_null(dst), src.data(), src.size() * sizeof(T),
          cudaMemcpyHostToDevice, stream));
    }
  }

  void init_guards(cudaStream_t stream) {
    const std::size_t target = std::max<std::size_t>(
        1, static_cast<std::size_t>(TGLSM_TARGET_GUARD_KEYS));
    std::size_t wanted = max_elements_ == 0 ? 1 : max_elements_ / target;
    wanted = std::max<std::size_t>(1, wanted);
    std::size_t guards = 1;
    while (guards < wanted && guards < TGLSM_MAX_GUARDS)
      guards <<= 1;
    guards = std::min<std::size_t>(guards, TGLSM_MAX_GUARDS);
    h_guard_hi_.resize(guards);
    const std::uint64_t span = 1ull << 32;
    for (std::size_t i = 0; i < guards; ++i) {
      const std::uint64_t hi =
          ((static_cast<std::uint64_t>(i + 1) * span) / guards) - 1;
      h_guard_hi_[i] = static_cast<std::uint32_t>(hi);
    }
    h_guard_hi_.back() = tglsm_detail::kEmptyKey;
    upload_guards(stream);
    rebuild_fragment_directory(stream);
  }

  void reserve_front_buffers() {
    std::size_t front = std::max<std::size_t>(1, batch_size_);
    front = std::max<std::size_t>(front, TGLSM_C0_INDEX_TRIGGER);
    front = std::min<std::size_t>(front, TGLSM_C0_FLUSH_TRIGGER);
    if (max_elements_ != 0)
      front = std::min<std::size_t>(front, max_elements_);
    const std::size_t merge = std::min<std::size_t>(
        max_elements_ == 0 ? front * 2 : max_elements_, front * 2);
    log_keys_.reserve(front);
    log_values_.reserve(front);
    log_ops_.reserve(front);
    log_seqs_.reserve(front);
    c0_keys_.reserve(front);
    c0_values_.reserve(front);
    c0_ops_.reserve(front);
    c0_seqs_.reserve(front);
    tmp_keys_.reserve(merge);
    tmp_values_.reserve(merge);
    tmp_ops_.reserve(merge);
    tmp_seqs_.reserve(merge);
    sort_keys_.reserve(merge);
  }

  void upload_guards(cudaStream_t stream) {
    upload_host_vector(d_guard_hi_, h_guard_hi_, stream);
  }

  void rebuild_fragment_directory(cudaStream_t stream) {
    std::sort(frags_.begin(), frags_.end(),
              [](const tglsm_detail::FragmentMeta &a,
                 const tglsm_detail::FragmentMeta &b) {
                if (a.guard != b.guard)
                  return a.guard < b.guard;
                if (a.min_key != b.min_key)
                  return a.min_key < b.min_key;
                return a.base < b.base;
              });
    const std::size_t guards = h_guard_hi_.size();
    std::vector<std::uint32_t> first(guards, 0);
    std::vector<std::uint32_t> count(guards, 0);
    std::size_t cursor = 0;
    for (std::size_t g = 0; g < guards; ++g) {
      first[g] = checked_u32(cursor, "too many fragments");
      while (cursor < frags_.size() && frags_[cursor].guard == g)
        ++cursor;
      count[g] = checked_u32(cursor - first[g], "too many fragments");
    }
    std::vector<std::uint32_t> base(frags_.size());
    std::vector<std::uint32_t> len(frags_.size());
    std::vector<std::uint32_t> min_key(frags_.size());
    std::vector<std::uint32_t> max_key(frags_.size());
    for (std::size_t i = 0; i < frags_.size(); ++i) {
      base[i] = frags_[i].base;
      len[i] = frags_[i].len;
      min_key[i] = frags_[i].min_key;
      max_key[i] = frags_[i].max_key;
    }
    upload_host_vector(d_guard_first_, first, stream);
    upload_host_vector(d_guard_count_, count, stream);
    upload_host_vector(d_frag_base_, base, stream);
    upload_host_vector(d_frag_len_, len, stream);
    upload_host_vector(d_frag_min_, min_key, stream);
    upload_host_vector(d_frag_max_, max_key, stream);
  }

  void append_records(const std::uint32_t *keys,
                      const std::uint32_t *values,
                      std::uint8_t op, std::size_t count,
                      cudaStream_t stream) {
    if (!keys)
      throw std::invalid_argument("TGLSM update requires keys");
    const std::uint64_t next =
        static_cast<std::uint64_t>(next_seq_) + count;
    if (next > std::numeric_limits<std::uint32_t>::max())
      throw std::overflow_error("TGLSM sequence space exhausted");
    const std::size_t old = log_keys_.size();
    log_keys_.resize(old + count);
    log_values_.resize(old + count);
    log_ops_.resize(old + count);
    log_seqs_.resize(old + count);
    TGLSM_CUDA_CHECK(cudaMemcpyAsync(
        raw_or_null(log_keys_) + old, keys, count * sizeof(std::uint32_t),
        cudaMemcpyDeviceToDevice, stream));
    if (op == tglsm_detail::kInsert) {
      TGLSM_CUDA_CHECK(cudaMemcpyAsync(
          raw_or_null(log_values_) + old, values,
          count * sizeof(std::uint32_t), cudaMemcpyDeviceToDevice,
          stream));
    }
    const int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    tglsm_detail::init_append_kernel<<<grid, block, 0, stream>>>(
        raw_or_null(log_values_) + old, raw_or_null(log_ops_) + old,
        raw_or_null(log_seqs_) + old, op, next_seq_, count);
    TGLSM_CUDA_CHECK(cudaGetLastError());
    next_seq_ = static_cast<std::uint32_t>(next);
  }

  void maybe_flush_after_write(cudaStream_t stream) {
    if (log_keys_.size() < TGLSM_C0_FLUSH_TRIGGER)
      return;
    build_c0_index(stream);
    flush_c0_to_fragments(stream);
    maintain_hot_guards(stream);
  }

  void prepare_for_read(std::size_t query_count, cudaStream_t stream) {
    const std::size_t raw = log_keys_.size();
    if (raw == 0)
      return;
    const bool product_too_large =
        query_count != 0 &&
        raw > static_cast<std::size_t>(TGLSM_RAW_SCAN_LIMIT) / query_count;
    if (raw >= TGLSM_C0_INDEX_TRIGGER || product_too_large) {
      build_c0_index(stream);
    }
    if (c0_keys_.size() >= TGLSM_C0_FLUSH_TRIGGER) {
      flush_c0_to_fragments(stream);
      maintain_hot_guards(stream);
    }
  }

  void fill_sort_keys(thrust::device_vector<std::uint32_t> &keys,
                      thrust::device_vector<std::uint32_t> &seqs,
                      std::size_t n, cudaStream_t stream) {
    sort_keys_.resize(n);
    if (n == 0)
      return;
    const int block = 256;
    const int grid = static_cast<int>((n + block - 1) / block);
    tglsm_detail::fill_sort_keys_kernel<<<grid, block, 0, stream>>>(
        raw_or_null(keys), raw_or_null(seqs), raw_or_null(sort_keys_), n);
    TGLSM_CUDA_CHECK(cudaGetLastError());
  }

  void sort_unique_newest(thrust::device_vector<std::uint32_t> &keys,
                          thrust::device_vector<std::uint32_t> &values,
                          thrust::device_vector<std::uint8_t> &ops,
                          thrust::device_vector<std::uint32_t> &seqs,
                          cudaStream_t stream) {
    const std::size_t n = keys.size();
    if (n == 0)
      return;
    fill_sort_keys(keys, seqs, n, stream);
    auto policy = thrust::cuda::par.on(stream);
    auto sort_values = thrust::make_zip_iterator(
        thrust::make_tuple(keys.begin(), values.begin(), ops.begin(),
                           seqs.begin()));
    thrust::sort_by_key(policy, sort_keys_.begin(), sort_keys_.begin() + n,
                        sort_values);
    auto unique_values = thrust::make_zip_iterator(
        thrust::make_tuple(values.begin(), ops.begin(), seqs.begin()));
    auto end = thrust::unique_by_key(policy, keys.begin(), keys.begin() + n,
                                     unique_values);
    const std::size_t u =
        static_cast<std::size_t>(end.first - keys.begin());
    keys.resize(u);
    values.resize(u);
    ops.resize(u);
    seqs.resize(u);
  }

  void filter_inserts(thrust::device_vector<std::uint32_t> &keys,
                      thrust::device_vector<std::uint32_t> &values,
                      thrust::device_vector<std::uint8_t> &ops,
                      thrust::device_vector<std::uint32_t> &seqs,
                      cudaStream_t stream) {
    const std::size_t n = keys.size();
    if (n == 0)
      return;
    auto policy = thrust::cuda::par.on(stream);
    auto first = thrust::make_zip_iterator(
        thrust::make_tuple(keys.begin(), values.begin(), ops.begin(),
                           seqs.begin()));
    auto last = first + n;
    auto end =
        thrust::remove_if(policy, first, last,
                          tglsm_detail::IsTombstoneTuple{});
    const std::size_t u = static_cast<std::size_t>(end - first);
    keys.resize(u);
    values.resize(u);
    ops.resize(u);
    seqs.resize(u);
  }

  void build_c0_index(cudaStream_t stream) {
    const std::size_t raw_n = log_keys_.size();
    if (raw_n == 0)
      return;
    const std::size_t c0_n = c0_keys_.size();
    const std::size_t n = raw_n + c0_n;
    tmp_keys_.resize(n);
    tmp_values_.resize(n);
    tmp_ops_.resize(n);
    tmp_seqs_.resize(n);
    auto policy = thrust::cuda::par.on(stream);
    if (c0_n > 0) {
      thrust::copy(policy, c0_keys_.begin(), c0_keys_.end(),
                   tmp_keys_.begin());
      thrust::copy(policy, c0_values_.begin(), c0_values_.end(),
                   tmp_values_.begin());
      thrust::copy(policy, c0_ops_.begin(), c0_ops_.end(),
                   tmp_ops_.begin());
      thrust::copy(policy, c0_seqs_.begin(), c0_seqs_.end(),
                   tmp_seqs_.begin());
    }
    thrust::copy(policy, log_keys_.begin(), log_keys_.end(),
                 tmp_keys_.begin() + c0_n);
    thrust::copy(policy, log_values_.begin(), log_values_.end(),
                 tmp_values_.begin() + c0_n);
    thrust::copy(policy, log_ops_.begin(), log_ops_.end(),
                 tmp_ops_.begin() + c0_n);
    thrust::copy(policy, log_seqs_.begin(), log_seqs_.end(),
                 tmp_seqs_.begin() + c0_n);
    sort_unique_newest(tmp_keys_, tmp_values_, tmp_ops_, tmp_seqs_, stream);
    c0_keys_.swap(tmp_keys_);
    c0_values_.swap(tmp_values_);
    c0_ops_.swap(tmp_ops_);
    c0_seqs_.swap(tmp_seqs_);
    tmp_keys_.clear();
    tmp_values_.clear();
    tmp_ops_.clear();
    tmp_seqs_.clear();
    log_keys_.clear();
    log_values_.clear();
    log_ops_.clear();
    log_seqs_.clear();
  }

  void flush_c0_to_fragments(cudaStream_t stream) {
    if (c0_keys_.empty())
      return;
    append_sorted_payload_as_fragments(c0_keys_, c0_values_, c0_ops_,
                                       c0_seqs_, c0_keys_.size(), stream);
    c0_keys_.clear();
    c0_values_.clear();
    c0_ops_.clear();
    c0_seqs_.clear();
  }

  void append_sorted_payload_as_fragments(
      thrust::device_vector<std::uint32_t> &keys,
      thrust::device_vector<std::uint32_t> &values,
      thrust::device_vector<std::uint8_t> &ops,
      thrust::device_vector<std::uint32_t> &seqs, std::size_t n,
      cudaStream_t stream) {
    if (n == 0)
      return;
    const std::size_t guards = h_guard_hi_.size();
    guard_offsets_.resize(guards + 1);
    auto policy = thrust::cuda::par.on(stream);
    thrust::fill(policy, guard_offsets_.begin(), guard_offsets_.begin() + 1,
                 0u);
    thrust::upper_bound(policy, keys.begin(), keys.begin() + n,
                        d_guard_hi_.begin(), d_guard_hi_.end(),
                        guard_offsets_.begin() + 1);
    std::vector<std::uint32_t> offsets(guards + 1);
    TGLSM_CUDA_CHECK(cudaMemcpyAsync(
        offsets.data(), raw_or_null(guard_offsets_),
        offsets.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
        stream));
    TGLSM_CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::uint32_t base =
        checked_u32(store_keys_.size(), "TGLSM payload too large");
    store_keys_.resize(store_keys_.size() + n);
    store_values_.resize(store_values_.size() + n);
    store_ops_.resize(store_ops_.size() + n);
    store_seqs_.resize(store_seqs_.size() + n);
    thrust::copy(policy, keys.begin(), keys.begin() + n,
                 store_keys_.begin() + base);
    thrust::copy(policy, values.begin(), values.begin() + n,
                 store_values_.begin() + base);
    thrust::copy(policy, ops.begin(), ops.begin() + n,
                 store_ops_.begin() + base);
    thrust::copy(policy, seqs.begin(), seqs.begin() + n,
                 store_seqs_.begin() + base);
    std::vector<tglsm_detail::FragmentMeta> added;
    for (std::size_t g = 0; g < guards; ++g) {
      std::uint32_t b = offsets[g];
      const std::uint32_t e = offsets[g + 1];
      while (b < e) {
        const std::uint32_t len =
            std::min<std::uint32_t>(TGLSM_TILE_KEYS, e - b);
        tglsm_detail::FragmentMeta meta;
        meta.base = base + b;
        meta.len = len;
        meta.guard = static_cast<std::uint32_t>(g);
        added.push_back(meta);
        b += len;
      }
    }
    std::vector<std::uint32_t> mins(added.size());
    std::vector<std::uint32_t> maxs(added.size());
    for (std::size_t i = 0; i < added.size(); ++i) {
      const std::uint32_t b = added[i].base - base;
      TGLSM_CUDA_CHECK(cudaMemcpyAsync(
          &mins[i], raw_or_null(keys) + b, sizeof(std::uint32_t),
          cudaMemcpyDeviceToHost, stream));
      TGLSM_CUDA_CHECK(cudaMemcpyAsync(
          &maxs[i], raw_or_null(keys) + b + added[i].len - 1,
          sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream));
    }
    TGLSM_CUDA_CHECK(cudaStreamSynchronize(stream));
    for (std::size_t i = 0; i < added.size(); ++i) {
      added[i].min_key = mins[i];
      added[i].max_key = maxs[i];
      frags_.push_back(added[i]);
    }
    rebuild_fragment_directory(stream);
  }

  bool can_split_guard(std::uint32_t guard,
                       std::size_t live_keys) const {
    (void)guard;
    return h_guard_hi_.size() < TGLSM_MAX_GUARDS &&
           live_keys > 2 * static_cast<std::size_t>(TGLSM_TARGET_GUARD_KEYS);
  }

  bool compact_guard(std::uint32_t guard, cudaStream_t stream) {
    std::vector<tglsm_detail::FragmentMeta> selected;
    std::vector<tglsm_detail::FragmentMeta> kept;
    std::size_t total = 0;
    for (const auto &f : frags_) {
      if (f.guard == guard) {
        selected.push_back(f);
        total += f.len;
      } else {
        kept.push_back(f);
      }
    }
    if (selected.empty())
      return false;
    tmp_keys_.resize(total);
    tmp_values_.resize(total);
    tmp_ops_.resize(total);
    tmp_seqs_.resize(total);
    std::size_t dst = 0;
    for (const auto &f : selected) {
      TGLSM_CUDA_CHECK(cudaMemcpyAsync(
          raw_or_null(tmp_keys_) + dst, raw_or_null(store_keys_) + f.base,
          f.len * sizeof(std::uint32_t), cudaMemcpyDeviceToDevice,
          stream));
      TGLSM_CUDA_CHECK(cudaMemcpyAsync(
          raw_or_null(tmp_values_) + dst, raw_or_null(store_values_) + f.base,
          f.len * sizeof(std::uint32_t), cudaMemcpyDeviceToDevice,
          stream));
      TGLSM_CUDA_CHECK(cudaMemcpyAsync(
          raw_or_null(tmp_ops_) + dst, raw_or_null(store_ops_) + f.base,
          f.len * sizeof(std::uint8_t), cudaMemcpyDeviceToDevice,
          stream));
      TGLSM_CUDA_CHECK(cudaMemcpyAsync(
          raw_or_null(tmp_seqs_) + dst, raw_or_null(store_seqs_) + f.base,
          f.len * sizeof(std::uint32_t), cudaMemcpyDeviceToDevice,
          stream));
      dst += f.len;
    }
    sort_unique_newest(tmp_keys_, tmp_values_, tmp_ops_, tmp_seqs_, stream);
    filter_inserts(tmp_keys_, tmp_values_, tmp_ops_, tmp_seqs_, stream);
    const std::size_t live = tmp_keys_.size();
    bool split = false;
    if (can_split_guard(guard, live) && live > 1) {
      std::uint32_t split_key = 0;
      TGLSM_CUDA_CHECK(cudaMemcpyAsync(
          &split_key, raw_or_null(tmp_keys_) + live / 2,
          sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream));
      TGLSM_CUDA_CHECK(cudaStreamSynchronize(stream));
      const std::uint32_t old_hi = h_guard_hi_[guard];
      const bool above_prev =
          guard == 0 || split_key > h_guard_hi_[guard - 1];
      if (split_key < old_hi && above_prev) {
        h_guard_hi_[guard] = split_key;
        h_guard_hi_.insert(h_guard_hi_.begin() + guard + 1, old_hi);
        for (auto &f : kept) {
          if (f.guard > guard)
            ++f.guard;
        }
        split = true;
      }
    }
    frags_.swap(kept);
    upload_guards(stream);
    if (live > 0) {
      append_sorted_payload_as_fragments(tmp_keys_, tmp_values_, tmp_ops_,
                                         tmp_seqs_, live, stream);
    } else {
      rebuild_fragment_directory(stream);
    }
    tmp_keys_.clear();
    tmp_values_.clear();
    tmp_ops_.clear();
    tmp_seqs_.clear();
    return split || selected.size() > 1;
  }

  void maintain_hot_guards(cudaStream_t stream) {
    for (int pass = 0; pass < 64; ++pass) {
      std::vector<std::size_t> frag_count(h_guard_hi_.size(), 0);
      std::vector<std::size_t> live_count(h_guard_hi_.size(), 0);
      for (const auto &f : frags_) {
        if (f.guard >= frag_count.size())
          continue;
        ++frag_count[f.guard];
        live_count[f.guard] += f.len;
      }
      bool changed = false;
      for (std::size_t g = 0; g < frag_count.size(); ++g) {
        const bool too_many =
            frag_count[g] > TGLSM_MAX_FRAGS_PER_GUARD &&
            live_count[g] <=
                TGLSM_MAX_FRAGS_PER_GUARD *
                    static_cast<std::size_t>(TGLSM_TILE_KEYS);
        const bool too_large =
            live_count[g] > TGLSM_GUARD_COMPACT_KEYS &&
            can_split_guard(static_cast<std::uint32_t>(g), live_count[g]);
        if (too_many || too_large) {
          changed = compact_guard(static_cast<std::uint32_t>(g), stream);
          break;
        }
      }
      if (!changed)
        break;
    }
  }

  tglsm_detail::DeviceView device_view() const {
    tglsm_detail::DeviceView view;
    view.guard_hi = raw_or_null(d_guard_hi_);
    view.guard_first = raw_or_null(d_guard_first_);
    view.guard_count = raw_or_null(d_guard_count_);
    view.num_guards = checked_u32(h_guard_hi_.size(), "too many guards");
    view.frag_base = raw_or_null(d_frag_base_);
    view.frag_len = raw_or_null(d_frag_len_);
    view.frag_min = raw_or_null(d_frag_min_);
    view.frag_max = raw_or_null(d_frag_max_);
    view.frag_count = checked_u32(frags_.size(), "too many fragments");
    view.store_keys = raw_or_null(store_keys_);
    view.store_values = raw_or_null(store_values_);
    view.store_ops = raw_or_null(store_ops_);
    view.store_seqs = raw_or_null(store_seqs_);
    view.c0_keys = raw_or_null(c0_keys_);
    view.c0_values = raw_or_null(c0_values_);
    view.c0_ops = raw_or_null(c0_ops_);
    view.c0_seqs = raw_or_null(c0_seqs_);
    view.c0_count = checked_u32(c0_keys_.size(), "C0 too large");
    view.raw_keys = raw_or_null(log_keys_);
    view.raw_values = raw_or_null(log_values_);
    view.raw_ops = raw_or_null(log_ops_);
    view.raw_seqs = raw_or_null(log_seqs_);
    view.raw_count = checked_u32(log_keys_.size(), "raw log too large");
    return view;
  }

  std::size_t max_elements_ = 0;
  std::size_t batch_size_ = 0;
  std::uint32_t next_seq_ = 1;
  mutable std::shared_mutex mutex_;

  std::vector<std::uint32_t> h_guard_hi_;
  std::vector<tglsm_detail::FragmentMeta> frags_;

  thrust::device_vector<std::uint32_t> log_keys_;
  thrust::device_vector<std::uint32_t> log_values_;
  thrust::device_vector<std::uint8_t> log_ops_;
  thrust::device_vector<std::uint32_t> log_seqs_;

  thrust::device_vector<std::uint32_t> c0_keys_;
  thrust::device_vector<std::uint32_t> c0_values_;
  thrust::device_vector<std::uint8_t> c0_ops_;
  thrust::device_vector<std::uint32_t> c0_seqs_;

  thrust::device_vector<std::uint32_t> store_keys_;
  thrust::device_vector<std::uint32_t> store_values_;
  thrust::device_vector<std::uint8_t> store_ops_;
  thrust::device_vector<std::uint32_t> store_seqs_;

  thrust::device_vector<std::uint32_t> d_guard_hi_;
  thrust::device_vector<std::uint32_t> d_guard_first_;
  thrust::device_vector<std::uint32_t> d_guard_count_;
  thrust::device_vector<std::uint32_t> d_frag_base_;
  thrust::device_vector<std::uint32_t> d_frag_len_;
  thrust::device_vector<std::uint32_t> d_frag_min_;
  thrust::device_vector<std::uint32_t> d_frag_max_;

  thrust::device_vector<std::uint32_t> tmp_keys_;
  thrust::device_vector<std::uint32_t> tmp_values_;
  thrust::device_vector<std::uint8_t> tmp_ops_;
  thrust::device_vector<std::uint32_t> tmp_seqs_;
  thrust::device_vector<std::uint64_t> sort_keys_;
  thrust::device_vector<std::uint32_t> guard_offsets_;
};
