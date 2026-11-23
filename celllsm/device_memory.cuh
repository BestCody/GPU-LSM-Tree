#pragma once

// Device, pinned and virtual-memory buffers.

#include "config.cuh"

namespace gpulsmopt2_detail {

template <class T> inline std::size_t maximum_resident_elements() {
  std::size_t free_bytes{}, total_bytes{};
  CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  (void)free_bytes;
  return total_bytes / sizeof(T);
}

template <class T> class Buffer {
public:
  Buffer() = default;
  explicit Buffer(std::size_t count) { resize(count); }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  ~Buffer() { if (pointer_ && owns_) cudaFree(pointer_); }
  void resize(std::size_t count) {
    if (pointer_ && owns_) CUDA_CHECK(cudaFree(pointer_));
    pointer_ = nullptr;
    count_ = count;
    owns_ = true;
    if (count)
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&pointer_),
                            count * sizeof(T)));
  }
  void attach(T *pointer, std::size_t count) {
    if (pointer_ && owns_) CUDA_CHECK(cudaFree(pointer_));
    pointer_ = pointer;
    count_ = count;
    owns_ = false;
  }
  T *data() { return pointer_; }
  std::size_t size() const { return count_; }
private:
  T *pointer_{};
  std::size_t count_{};
  bool owns_{true};
};

// Pinned receipts support asynchronous publication.
template <class T> class PinnedBuffer {
public:
  explicit PinnedBuffer(std::size_t count) {
    if (!count) return;
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::bad_alloc();
    constexpr std::size_t alignment = 4096u;
    const std::size_t bytes = count * sizeof(T);
    if (bytes > std::numeric_limits<std::size_t>::max() -
                    (alignment - 1u))
      throw std::bad_alloc();
    const std::size_t allocation_bytes =
        (bytes + alignment - 1u) / alignment * alignment;
    pointer_ = static_cast<T *>(
        std::aligned_alloc(alignment, allocation_bytes));
    if (!pointer_) throw std::bad_alloc();
    const cudaError_t error = cudaHostRegister(
        pointer_, allocation_bytes, cudaHostRegisterDefault);
    if (error != cudaSuccess) {
      std::free(pointer_);
      pointer_ = nullptr;
      CUDA_CHECK(error);
    }
  }
  PinnedBuffer(const PinnedBuffer &) = delete;
  PinnedBuffer &operator=(const PinnedBuffer &) = delete;
  ~PinnedBuffer() {
    if (pointer_) {
      cudaHostUnregister(pointer_);
      std::free(pointer_);
    }
  }
  T *data() { return pointer_; }

private:
  T *pointer_{};
};

inline void check_driver(CUresult result, const char *file, int line) {
  if (result == CUDA_SUCCESS) return;
  throw std::runtime_error(
      std::string("CUDA driver error ") + std::to_string(result) + " at " +
      file + ":" + std::to_string(line));
}

#define GPULSMOPT_CU_CHECK(call) \
  ::gpulsmopt2_detail::check_driver((call), __FILE__, __LINE__)

struct VmmFunctions {
  decltype(&cuMemAddressReserve) reserve{};
  decltype(&cuMemAddressFree) free_address{};
  decltype(&cuMemCreate) create{};
  decltype(&cuMemRelease) release{};
  decltype(&cuMemMap) map{};
  decltype(&cuMemUnmap) unmap{};
  decltype(&cuMemSetAccess) set_access{};
  decltype(&cuMemGetAllocationGranularity) granularity{};

  template <class Function>
  static Function load(const char *name) {
    void *pointer = nullptr;
    CUDA_CHECK(cudaGetDriverEntryPoint(
        name, &pointer, cudaEnableDefault, nullptr));
    if (!pointer)
      throw std::runtime_error(std::string("missing CUDA driver entry ") +
                               name);
    return reinterpret_cast<Function>(pointer);
  }

  VmmFunctions()
      : reserve(load<decltype(reserve)>("cuMemAddressReserve")),
        free_address(load<decltype(free_address)>("cuMemAddressFree")),
        create(load<decltype(create)>("cuMemCreate")),
        release(load<decltype(release)>("cuMemRelease")),
        map(load<decltype(map)>("cuMemMap")),
        unmap(load<decltype(unmap)>("cuMemUnmap")),
        set_access(load<decltype(set_access)>("cuMemSetAccess")),
        granularity(load<decltype(granularity)>(
            "cuMemGetAllocationGranularity")) {}
};

inline VmmFunctions &vmm_functions() {
  static VmmFunctions functions;
  return functions;
}

// Own the virtual reservation and its physical mappings.
class VirtualAllocation {
public:
  VirtualAllocation() = default;
  VirtualAllocation(const VirtualAllocation &) = delete;
  VirtualAllocation &operator=(const VirtualAllocation &) = delete;
  VirtualAllocation(VirtualAllocation &&other) noexcept { move_from(other); }
  VirtualAllocation &operator=(VirtualAllocation &&other) noexcept {
    if (this != &other) { reset(); move_from(other); }
    return *this;
  }
  ~VirtualAllocation() { reset(); }

  void reserve(std::size_t bytes) {
    if (!bytes) throw std::bad_alloc();
    if (address_) throw std::logic_error("virtual allocation already reserved");
    CUDA_CHECK(cudaFree(nullptr));
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    property_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    property_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    property_.location.id = device;
    auto &functions = vmm_functions();
    GPULSMOPT_CU_CHECK(functions.granularity(
        &granularity_, &property_, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    reserved_bytes_ = align_up(bytes);
    GPULSMOPT_CU_CHECK(functions.reserve(
        &address_, reserved_bytes_, granularity_, 0u, 0u));
  }

  void attach(void *pointer, std::size_t reserved, std::size_t mapped) {
    reset();
    external_ = true;
    address_ = reinterpret_cast<CUdeviceptr>(pointer);
    reserved_bytes_ = reserved;
    mapped_bytes_ = mapped;
  }

  void map_to(std::size_t bytes) {
    if (bytes > reserved_bytes_) throw std::bad_alloc();
    if (external_) { mapped_bytes_ = bytes; return; }
    const std::size_t target = align_up(bytes);
    if (target <= mapped_bytes_) return;
    const std::size_t extension = target - mapped_bytes_;
    auto &functions = vmm_functions();
    CUmemGenericAllocationHandle handle{};
    GPULSMOPT_CU_CHECK(functions.create(&handle, extension, &property_, 0u));
    bool mapped = false;
    try {
      GPULSMOPT_CU_CHECK(functions.map(
          address_ + mapped_bytes_, extension, 0u, handle, 0u));
      mapped = true;
      CUmemAccessDesc access{};
      access.location = property_.location;
      access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      GPULSMOPT_CU_CHECK(functions.set_access(
          address_ + mapped_bytes_, extension, &access, 1u));
      // Roll back if recording ownership fails.
      mappings_.push_back({mapped_bytes_, extension, handle});
    } catch (...) {
      if (mapped) functions.unmap(address_ + mapped_bytes_, extension);
      functions.release(handle);
      throw;
    }
    mapped_bytes_ = target;
  }

  void shrink(std::size_t bytes) {
    if (external_) { mapped_bytes_ = bytes; return; }
    const std::size_t target = bytes ? align_up(bytes) : 0u;
    auto &functions = vmm_functions();
    while (!mappings_.empty() && mappings_.back().offset >= target) {
      const Mapping mapping = mappings_.back();
      GPULSMOPT_CU_CHECK(functions.unmap(address_ + mapping.offset, mapping.bytes));
      GPULSMOPT_CU_CHECK(functions.release(mapping.handle));
      mappings_.pop_back();
      mapped_bytes_ = mapping.offset;
    }
    if (mapped_bytes_ != target)
      throw std::invalid_argument("unaligned virtual buffer shrink");
  }

  void reset() noexcept {
    if (address_ && !external_) {
      auto &functions = vmm_functions();
      for (auto it = mappings_.rbegin(); it != mappings_.rend(); ++it) {
        functions.unmap(address_ + it->offset, it->bytes);
        functions.release(it->handle);
      }
      functions.free_address(address_, reserved_bytes_);
    }
    address_ = 0u;
    granularity_ = reserved_bytes_ = mapped_bytes_ = 0u;
    external_ = false;
    mappings_.clear();
  }

  std::uint8_t *data() const {
    return reinterpret_cast<std::uint8_t *>(static_cast<std::uintptr_t>(address_));
  }
  std::size_t mapped_bytes() const { return mapped_bytes_; }
  bool external() const { return external_; }

private:
  struct Mapping {
    std::size_t offset, bytes;
    CUmemGenericAllocationHandle handle;
  };
  std::size_t align_up(std::size_t bytes) const {
    if (!granularity_ || bytes >
        std::numeric_limits<std::size_t>::max() - (granularity_ - 1u))
      throw std::overflow_error("virtual allocation size overflow");
    return (bytes + granularity_ - 1u) / granularity_ * granularity_;
  }
  void move_from(VirtualAllocation &other) noexcept {
    address_ = std::exchange(other.address_, 0u);
    granularity_ = std::exchange(other.granularity_, 0u);
    reserved_bytes_ = std::exchange(other.reserved_bytes_, 0u);
    mapped_bytes_ = std::exchange(other.mapped_bytes_, 0u);
    external_ = std::exchange(other.external_, false);
    property_ = other.property_;
    mappings_ = std::move(other.mappings_);
  }

  CUdeviceptr address_{};
  std::size_t granularity_{};
  std::size_t reserved_bytes_{};
  std::size_t mapped_bytes_{};
  bool external_{};
  CUmemAllocationProp property_{};
  std::vector<Mapping> mappings_;
};

template <class T> class VirtualBuffer {
public:
  VirtualBuffer() = default;
  VirtualBuffer(std::size_t maximum_count, std::size_t initial_count) {
    if (!maximum_count || maximum_count >
        std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::bad_alloc();
    maximum_count_ = maximum_count;
    allocation_.reserve(maximum_count * sizeof(T));
    grow(initial_count);
  }
  VirtualBuffer(const VirtualBuffer &) = delete;
  VirtualBuffer &operator=(const VirtualBuffer &) = delete;
  VirtualBuffer(VirtualBuffer &&) = delete;
  VirtualBuffer &operator=(VirtualBuffer &&) = delete;

  void attach(T *pointer, std::size_t maximum_count, std::size_t current_count) {
    if (!pointer || !maximum_count || current_count > maximum_count)
      throw std::invalid_argument("invalid virtual buffer attachment");
    if (maximum_count > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::bad_alloc();
    allocation_.attach(pointer, maximum_count * sizeof(T), current_count * sizeof(T));
    maximum_count_ = maximum_count;
  }

  void grow(std::size_t requested_count) {
    if (requested_count <= size()) return;
    if (requested_count > maximum_count_) throw std::bad_alloc();
    std::size_t target = requested_count;
    if (!allocation_.external() && size()) {
      const std::size_t doubled = size() > maximum_count_ / 2u
          ? maximum_count_ : size() * 2u;
      target = std::max(target, doubled);
    }
    allocation_.map_to(target * sizeof(T));
  }

  void shrink(std::size_t requested_count) {
    if (requested_count > size())
      throw std::invalid_argument("virtual buffer shrink growth");
    allocation_.shrink(requested_count * sizeof(T));
  }

  T *data() { return reinterpret_cast<T *>(allocation_.data()); }
  std::size_t size() const { return allocation_.mapped_bytes() / sizeof(T); }
private:
  VirtualAllocation allocation_;
  std::size_t maximum_count_{};
};

}  // namespace gpulsmopt2_detail

namespace gpulsm_sparse {

inline void check(cudaError_t error, const char *where) {
  if (error != cudaSuccess)
    throw std::runtime_error(
        std::string(where) + ": " + cudaGetErrorString(error));
}

template <class T>
class Buffer {
 public:
  Buffer() = default;
  explicit Buffer(std::size_t count) { reset(count); }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  Buffer(Buffer &&other) noexcept { swap(other); }
  Buffer &operator=(Buffer &&other) noexcept {
    if (this != &other) {
      clear();
      swap(other);
    }
    return *this;
  }
  ~Buffer() { clear(); }

  void reset(std::size_t count) {
    clear();
    owns_ = true;
    count_ = count;
    if (count) check(cudaMalloc(&pointer_, count * sizeof(T)), "cudaMalloc");
  }

  void attach(T *pointer, std::size_t count) {
    clear();
    pointer_ = pointer;
    count_ = count;
    owns_ = false;
  }

  void clear() noexcept {
    if (pointer_ && owns_) cudaFree(pointer_);
    pointer_ = nullptr;
    count_ = 0u;
    owns_ = true;
  }

  T *data() const { return pointer_; }
  std::size_t size() const { return count_; }
  std::size_t bytes() const { return count_ * sizeof(T); }

  void swap(Buffer &other) noexcept {
    std::swap(pointer_, other.pointer_);
    std::swap(count_, other.count_);
    std::swap(owns_, other.owns_);
  }

 private:
  T *pointer_{};
  std::size_t count_{};
  bool owns_{true};
};

}  // namespace gpulsm_sparse
