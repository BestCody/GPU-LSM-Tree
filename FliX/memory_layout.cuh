// =============================================================================
// File: memory_layout.cuh
// Author: Justus Henneberg
// Description: Implements memory_layout     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef MEMORY_LAYOUT_CUH
#define MEMORY_LAYOUT_CUH

#include "definitions.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace coop = cooperative_groups;


namespace memory_layout {

template <typename key_type>
HOSTDEVICEQUALIFIER INLINEQUALIFIER
key_type extract(const void* buffer, bigsize byte_offset);

template <>
HOSTDEVICEQUALIFIER INLINEQUALIFIER
uint32_t extract<uint32_t>(const void* buffer, bigsize byte_offset) {
    auto full_offset = reinterpret_cast<const uint8_t *>(buffer) + byte_offset;
    return (uint32_t) *reinterpret_cast<const uint32_t *>(full_offset);
}

template <>
HOSTDEVICEQUALIFIER INLINEQUALIFIER
uint64_t extract<uint64_t>(const void* buffer, bigsize byte_offset) {
    auto full_offset = reinterpret_cast<const uint8_t *>(buffer) + byte_offset;
    auto lower = reinterpret_cast<const uint32_t *>(full_offset + 0);
    auto upper = reinterpret_cast<const uint32_t *>(full_offset + 4);
    return ((uint64_t) *upper << 32u | (uint64_t) *lower);
}


template <typename key_type>
HOSTDEVICEQUALIFIER INLINEQUALIFIER
void set(void* buffer, bigsize byte_offset, key_type key);

template <>
HOSTDEVICEQUALIFIER INLINEQUALIFIER
void set<uint32_t>(void* buffer, bigsize byte_offset, uint32_t key) {
    auto full_offset = reinterpret_cast<uint8_t *>(buffer) + byte_offset;
    *reinterpret_cast<uint32_t *>(full_offset) = key;
}

template <>
HOSTDEVICEQUALIFIER INLINEQUALIFIER
void set<uint64_t>(void* buffer, bigsize byte_offset, uint64_t key) {
    auto full_offset = reinterpret_cast<uint8_t *>(buffer) + byte_offset;
    auto lower = reinterpret_cast<uint32_t *>(full_offset + 0);
    auto upper = reinterpret_cast<uint32_t *>(full_offset + 4);
    *lower = key;
    *upper = key >> 32u;
}


template <typename key_type_, typename value_type_, typename size_type_>
struct row_store {
    using key_type = key_type_;
    using value_type = value_type_;
    using size_type = size_type_;

    const uint8_t* _buf;
    size_type _size;

    static constexpr smallsize stride_bytes = sizeof(key_type) + sizeof(smallsize);

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    row_store(const void* buf, size_type size)
            : _buf{static_cast<const uint8_t *>(buf)}, _size{size} {}

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    size_type size() const {
        return _size;
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    key_type extract_key(size_type i) const {
        return extract<key_type>(_buf, i * stride_bytes);
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    value_type extract_value(size_type i) const {
        return extract<value_type>(_buf, i * stride_bytes + sizeof(key_type));
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    const uint8_t* key_pointer() const {
        return _buf;
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr smallsize key_stride() const {
        return stride_bytes;
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    const uint8_t* value_pointer() const {
        return _buf + sizeof(key_type);
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr smallsize value_stride() const {
        return stride_bytes;
    }
};


template <typename key_type_, typename value_type_, typename size_type_>
struct col_store {
    using key_type = key_type_;
    using value_type = value_type_;
    using size_type = size_type_;

    const key_type* _keys;
    const value_type* _offsets;
    size_type _size;

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    col_store(const key_type *keys, const value_type *offsets, size_type size)
            : _keys{keys}, _offsets{offsets}, _size{size} {}

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    size_type size() const {
        return _size;
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    key_type extract_key(size_type i) const {
        return _keys[i];
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    value_type extract_value(size_type i) const {
        return _offsets[i];
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    const uint8_t* key_pointer() const {
        return (const uint8_t*) _keys;
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr smallsize key_stride() const {
        return sizeof(key_type);
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    const uint8_t* value_pointer() const {
        return (const uint8_t*) _offsets;
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr smallsize value_stride() const {
        return sizeof(smallsize);
    }
};


template <typename key_type, typename value_type, typename size_type, bool use_row_layout>
using store_type = std::conditional_t<
    use_row_layout,
    row_store<key_type, value_type, size_type>,
    col_store<key_type, value_type, size_type>>;


template <typename store_type_>
struct rq {
    const store_type_& store;

    using key_type = typename store_type_::key_type;
    using value_type = typename store_type_::value_type;
    using size_type = typename store_type_::size_type;

    DEVICEQUALIFIER INLINEQUALIFIER
    explicit rq(const store_type_& store) : store(store) {}

    DEVICEQUALIFIER INLINEQUALIFIER
    value_type range_query(
            key_type lower,
            key_type upper,
            size_type initial_offset
    ) {
        value_type local_agg = 0;
        for (size_type i = initial_offset; i < store.size(); ++i) {
            auto k = store.extract_key(i);
            if (k < lower) continue;
            if (k > upper) break;
            local_agg += store.extract_value(i);
        }
        return local_agg;
    }


    template <typename cg_type>
    DEVICEQUALIFIER INLINEQUALIFIER
    value_type cooperative_range_query(
            cg_type tile,
            key_type lower,
            key_type upper,
            size_type initial_offset
    ) {
        size_type local_offset = initial_offset + tile.thread_rank();
        value_type local_agg = 0;
        for (;; local_offset += tile.size()) {
            bool is_valid = local_offset < store.size();

            auto key = is_valid ? store.extract_key(local_offset) : 0;
            bool overstepped = !is_valid || key > upper;
            bool is_in_range = !overstepped && key >= lower;

            auto value = is_in_range ? store.extract_value(local_offset) : 0;

            local_agg += value;
            if (tile.any(overstepped)) break;
        }
        return coop::reduce(tile, local_agg, coop::plus<value_type>());
    }

};

}

#endif //MEMORY_LAYOUT_CUH
