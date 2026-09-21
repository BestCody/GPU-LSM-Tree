#pragma once

#include "../ext/moderngpu/src/moderngpu/kernel_segsort.hxx"

#include <limits>
#include <stdexcept>
#include <vector>

// Keep ModernGPU's temporary allocations between COUNT/RANGE calls, just as
// the other LSM work buffers are retained. Reuse is ordered on the call's stream.
class lsm_sort_context final : public mgpu::standard_context_t
{
    struct allocation
    {
        void *pointer = nullptr;
        size_t bytes = 0;
        bool in_use = false;
    };

    std::vector<allocation> allocations;

public:
    explicit lsm_sort_context(cudaStream_t stream)
        : mgpu::standard_context_t(false, stream) {}

    ~lsm_sort_context()
    {
        for (const auto &buffer : allocations)
            cudaFree(buffer.pointer);
    }

    void set_stream(cudaStream_t stream)
    {
        if (_stream != stream)
        {
            const auto status = cudaStreamSynchronize(_stream);
            if (status != cudaSuccess)
                throw mgpu::cuda_exception_t(status);
            _stream = stream;
        }
    }

    void *alloc(size_t bytes, mgpu::memory_space_t space) override
    {
        if (space != mgpu::memory_space_device)
            return mgpu::standard_context_t::alloc(bytes, space);
        if (bytes == 0)
            return nullptr;

        // Prefer the smallest free allocation that already fits.
        allocation *chosen = nullptr;
        for (auto &buffer : allocations)
            if (!buffer.in_use && buffer.bytes >= bytes &&
                (!chosen || buffer.bytes < chosen->bytes))
                chosen = &buffer;

        if (!chosen)
        {
            // Grow a free allocation before adding another retained buffer.
            for (auto &buffer : allocations)
                if (!buffer.in_use &&
                    (!chosen || buffer.bytes > chosen->bytes))
                    chosen = &buffer;
            if (!chosen)
            {
                allocations.emplace_back();
                chosen = &allocations.back();
            }
            if (chosen->pointer)
            {
                mgpu::standard_context_t::free(chosen->pointer, space);
                chosen->pointer = nullptr;
                chosen->bytes = 0;
            }
            chosen->pointer = mgpu::standard_context_t::alloc(bytes, space);
            chosen->bytes = bytes;
        }
        chosen->in_use = true;
        return chosen->pointer;
    }

    void free(void *pointer, mgpu::memory_space_t space) override
    {
        if (space != mgpu::memory_space_device)
        {
            mgpu::standard_context_t::free(pointer, space);
            return;
        }
        if (!pointer)
            return;
        for (auto &buffer : allocations)
            if (buffer.pointer == pointer)
            {
                buffer.in_use = false;
                return;
            }
        throw std::logic_error("Unknown ModernGPU workspace allocation");
    }

    size_t size_in_bytes() const
    {
        size_t bytes = 0;
        for (const auto &buffer : allocations)
            bytes += buffer.bytes;
        return bytes;
    }

    template <typename Key, typename Value, typename Compare>
    size_t max_sort_items() const
    {
        // ModernGPU's merge frames round up to a power-of-two number of
        // tiles. The rounded frame must also fit its signed 32-bit indices.
        using launch = typename mgpu::detail::segsort_t<
            mgpu::empty_t, Key, Value, Compare>::launch_t;
        const size_t tile = launch::nv(*this);
        const size_t tile_limit = std::numeric_limits<int>::max() / tile;
        size_t tiles = 1;
        while (tiles <= tile_limit / 2)
            tiles *= 2;
        return tile * tiles;
    }
};
