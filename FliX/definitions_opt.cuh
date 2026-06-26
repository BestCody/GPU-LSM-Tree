// =============================================================================
// File: definitions_opt.cuh
// Author: Justus Henneberg
// Description: Implements definitions_opt     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef DEFINITIONS_OPT_CUH
#define DEFINITIONS_OPT_CUH

#include <cmath>

#include <cub/cub.cuh>


namespace opt {
    struct f {
        static constexpr uint8_t none = 0;
        // opt_sorted_array/static_tree
        static constexpr uint8_t cul = 1;
        static constexpr uint8_t culp = cul | 2;
        // opt_sorted_array/static_tree/opt_tree
        static constexpr uint8_t slb = 4;
        static constexpr uint8_t sulb = slb | 8;
        static constexpr uint8_t all = culp | sulb;

        static constexpr bool cache_upper_levels(uint8_t flags) {
            return flags & cul;
        }
        static constexpr bool cache_upper_levels_partial(uint8_t flags) {
            return flags & 2;
        }
        static constexpr bool sort_lookups_block(uint8_t flags) {
            return flags & slb;
        }
        static constexpr bool sort_unsort_lookups_block(uint8_t flags) {
            return flags & 8;
        }
    };


    struct empty_type {};


    // wrapper around cub::BlockExchange
    template <typename payload_type, uint32_t threads_per_block, uint32_t items_per_thread>
    class block_exchange {

        using exchange_type = cub::BlockExchange<payload_type, threads_per_block, items_per_thread>;

        uint8_t* _shmem;

    public:
        static constexpr smallsize temp_storage_bytes = (smallsize) sizeof(typename exchange_type::TempStorage);

        DEVICEQUALIFIER INLINEQUALIFIER
        block_exchange(uint8_t* shmem) : _shmem(shmem) {}

        template <typename index_type>
        DEVICEQUALIFIER INLINEQUALIFIER
        void to_striped(payload_type (&payloads)[items_per_thread], index_type (&ranks)[items_per_thread]) {
            auto ex = exchange_type(*reinterpret_cast<typename exchange_type::TempStorage*>(_shmem));
            ex.ScatterToStriped(payloads, ranks);
        }

        template <typename index_type>
        DEVICEQUALIFIER INLINEQUALIFIER
        void to_blocked(payload_type (&payloads)[items_per_thread], index_type (&ranks)[items_per_thread]) {
            auto ex = exchange_type(*reinterpret_cast<typename exchange_type::TempStorage*>(_shmem));
            ex.ScatterToBlocked(payloads, ranks);
        }
    };


    // specialization for empty_type
    template <uint32_t threads_per_block, uint32_t items_per_thread>
    class block_exchange<empty_type, threads_per_block, items_per_thread> {
    public:
        static constexpr uint32_t temp_storage_bytes = 0;

        DEVICEQUALIFIER INLINEQUALIFIER
        block_exchange(uint8_t* shmem) {}

        template <typename index_type>
        DEVICEQUALIFIER INLINEQUALIFIER
        void to_striped(empty_type (&)[items_per_thread], index_type (&)[items_per_thread]) {
            // do nothing
        }

        template <typename index_type>
        DEVICEQUALIFIER INLINEQUALIFIER
        void to_blocked(empty_type (&)[items_per_thread], index_type (&)[items_per_thread]) {
            // do nothing
        }
    };


    // modified variant of cub::BlockRadixSort to allow arbitrary payloads
    // however, this variant only works for non-negative integer keys since we skip the normalization step
    template <typename key_type, uint32_t threads_per_block, uint32_t items_per_thread, uint32_t radix_bits = 4, typename ...payload_types>
    class block_radix_sort {

        // cub uses int internally, so we also use int
        using index_type = int;

        using rank_type = cub::BlockRadixRank<threads_per_block, radix_bits, false>;

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr smallsize compute_temp_bytes() {
            return std::max({
                (smallsize) sizeof(typename rank_type::TempStorage),
                block_exchange<key_type, threads_per_block, items_per_thread>::temp_storage_bytes,
                block_exchange<payload_types, threads_per_block, items_per_thread>::temp_storage_bytes...
            });
        };

        uint8_t* _shmem;

        template <typename payload_type, bool to_striped>
        DEVICEQUALIFIER INLINEQUALIFIER
        void exchange(payload_type (&payloads)[items_per_thread], index_type (&ranks)[items_per_thread]) {
            __syncthreads();

            auto ex = block_exchange<payload_type, threads_per_block, items_per_thread>(_shmem);
            if constexpr (to_striped) {
                ex.to_striped(payloads, ranks);
            } else {
                ex.to_blocked(payloads, ranks);
            }
        }

    public:
        static constexpr smallsize temp_storage_bytes = compute_temp_bytes();

        DEVICEQUALIFIER INLINEQUALIFIER
        block_radix_sort(uint8_t* shmem) : _shmem(shmem) {}

        DEVICEQUALIFIER INLINEQUALIFIER
        void to_striped(key_type (&keys)[items_per_thread], payload_types (&...payloads)[items_per_thread]) {
            uint32_t begin_bit = 0;
            uint32_t end_bit = sizeof(key_type) * 8;

            while (true) {
                uint32_t bit_count = std::min(radix_bits, end_bit - begin_bit);

                cub::BFEDigitExtractor<key_type> digit_extractor(begin_bit, bit_count);

                index_type ranks[items_per_thread];
                rank_type(*reinterpret_cast<typename rank_type::TempStorage*>(_shmem))
                        .RankKeys(keys, ranks, digit_extractor);
                begin_bit += radix_bits;

                if (begin_bit >= end_bit) {
                    exchange<key_type, true>(keys, ranks);
                    (exchange<payload_types, true>(payloads, ranks), ...);
                    break;
                }

                exchange<key_type, false>(keys, ranks);
                (exchange<payload_types, false>(payloads, ranks), ...);

                __syncthreads();
            }
        }
    };


    template <typename key_type, uint16_t threads_per_block, uint16_t items_per_thread, bool has_range_query>
    struct algorithms {
        using upper = typename std::conditional<has_range_query, key_type, empty_type>::type;
        using sort = block_radix_sort<key_type, threads_per_block, items_per_thread, 4, uint32_t, upper>;
        using result_shuffle = block_exchange<smallsize, threads_per_block, items_per_thread>;
        constexpr static size_t temp_storage_bytes = std::max(sort::temp_storage_bytes, result_shuffle::temp_storage_bytes);
    };
}


DEVICEQUALIFIER INLINEQUALIFIER
smallsize ilog2_gpu(smallsize value) {
    return value == 0 ? 0 : 31u - __clz(value);
}

#endif
