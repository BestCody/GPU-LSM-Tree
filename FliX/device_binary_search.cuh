// =============================================================================
// File: device_binary_search.cuh
// Author: Justus Henneberg
// Description: Implements device_binary_search     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef DEVICE_BINARY_SEARCH_CUH
#define DEVICE_BINARY_SEARCH_CUH


template <typename element_type>
DEVICEQUALIFIER INLINEQUALIFIER
smallsize device_binary_search(element_type key, const element_type* buf, smallsize size) {
    smallsize match_index = 0;
    for (smallsize skip = smallsize(1u) << 30u; skip != 0; skip >>= 1u) {
        if (match_index + skip >= size)
            continue;

        if (buf[match_index + skip] <= key)
            match_index += skip;
    }
    return match_index;
}


template <typename element_type>
DEVICEQUALIFIER INLINEQUALIFIER
smallsize reverse_device_binary_search(element_type key, const element_type* buf, smallsize initial_offset, smallsize initial_skip) {
    smallsize match_index = initial_offset;
    for (smallsize skip = initial_skip; skip != 0; skip >>= 1u) {
        if (match_index < skip)
            continue;

        if (buf[match_index - skip] >= key)
            match_index -= skip;
    }
    return match_index;
}


template <typename element_type>
DEVICEQUALIFIER INLINEQUALIFIER
smallsize reverse_device_binary_search(element_type key, const element_type* buf, smallsize entries_count) {
    return reverse_device_binary_search(key, buf, entries_count - 1, smallsize(1) << ilog2_gpu(entries_count));
}


template <typename key_type, typename size_type, bool branchless>
DEVICEQUALIFIER INLINEQUALIFIER
size_type opt_reverse_device_binary_search(
        const uint8_t* sorted_entries,
        smallsize sorted_entries_stride,
        key_type key,
        size_type initial_offset,
        size_type initial_skip
) {
    size_type match_offset = initial_offset;
    for (size_type skip = initial_skip; skip > 0; skip >>= 1u) {
        if (match_offset < skip)
            continue;

        auto current = mem::extract<key_type>(sorted_entries, (match_offset - skip) * sorted_entries_stride);

        if constexpr (branchless) {
            match_offset -= bool(current >= key) * skip;
        } else {
            if (current >= key) {
                match_offset -= skip;
            }
        }
    }
    return match_offset;
}


template <typename key_type, typename size_type, bool branchless>
DEVICEQUALIFIER INLINEQUALIFIER
size_type opt_reverse_device_binary_search(
        const uint8_t* sorted_entries,
        size_type sorted_entries_count,
        smallsize sorted_entries_stride,
        key_type key
) {
    return opt_reverse_device_binary_search<key_type, size_type, branchless>(sorted_entries, sorted_entries_stride, key, sorted_entries_count - 1, size_type(1) << ilog2_gpu(sorted_entries_count));
}


template <typename key_type, typename size_type, bool branchless>
DEVICEQUALIFIER INLINEQUALIFIER
size_type opt_reverse_device_binary_search(
        const key_type* sorted_entries,
        size_type sorted_entries_count,
        key_type key
) {
    return opt_reverse_device_binary_search<key_type, size_type, branchless>((const uint8_t*) sorted_entries, sorted_entries_count, (smallsize) sizeof(key_type), key);
}

#endif
