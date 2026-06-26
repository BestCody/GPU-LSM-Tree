// =============================================================================
// File: cuda_buffer.cuh
// Author: Justus Henneberg
// Description: Implements cuda_buffer     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef CUDA_BUFFER_H
#define CUDA_BUFFER_H

#include "definitions.cuh"

#include <cassert>
#include <vector>

template <typename T>
struct cuda_buffer_base {

    void swap(cuda_buffer_base<T>& other) {
        std::swap(num_elements, other.num_elements);
        std::swap(raw_ptr, other.raw_ptr);
    }

    cuda_buffer_base() {};
    cuda_buffer_base(const cuda_buffer_base<T>& other) = delete;
    cuda_buffer_base(cuda_buffer_base<T>&& other) {
        swap(other);
    }

    cuda_buffer_base<T>& operator=(const cuda_buffer_base<T>& other) = delete;
    cuda_buffer_base<T>& operator=(cuda_buffer_base<T>&& other) {
        swap(other);
        return *this;
    }

    CUdeviceptr cu_ptr() {
        return (CUdeviceptr)raw_ptr;
    }

    const T* ptr() const {
        return (const T*)raw_ptr;
    }

    T* ptr() {
        return (T*)raw_ptr;
    }

    operator const T*() const {
        return (const T*)raw_ptr;
    }

    operator T*() {
        return (T*)raw_ptr;
    }

    size_t size_in_bytes() {
        return num_elements * sizeof(T);
    }

    void debug_dump(size_t count) {
        size_t actual_size = std::min(count, num_elements);
        std::vector<T> temp(actual_size);
        cudaMemcpy(temp.data(), raw_ptr, actual_size * sizeof(T), cudaMemcpyDeviceToHost);
        for (const auto& entry : temp) {
            std::cerr << entry << " ";
        }
        if (actual_size == 0) {
            std::cerr << "EMPTY";
        }
        std::cerr << std::endl;
    }

    size_t num_elements{0};
    void* raw_ptr{nullptr};
};


template <typename T>
struct cuda_buffer_async : public cuda_buffer_base<T> {

    using cuda_buffer_base<T>::raw_ptr;
    using cuda_buffer_base<T>::num_elements;
    using cuda_buffer_base<T>::size_in_bytes;

    ~cuda_buffer_async() {
        if (raw_ptr) free(0);
    }

    void alloc(size_t size, cudaStream_t stream) {
        assert(raw_ptr == nullptr);
        num_elements = size;
        cudaMallocAsync((void**)&raw_ptr, size_in_bytes(), stream);
    }

    void free(cudaStream_t stream) {
        cudaFreeAsync(raw_ptr);
        raw_ptr = nullptr;
        num_elements = 0;
    }

    void zero(cudaStream_t stream) {
        cudaMemsetAsync(raw_ptr, 0, size_in_bytes(), stream);
    }
};


template <typename T>
struct cuda_buffer : public cuda_buffer_base<T> {

    using cuda_buffer_base<T>::raw_ptr;
    using cuda_buffer_base<T>::num_elements;
    using cuda_buffer_base<T>::size_in_bytes;

    ~cuda_buffer() {
        if (raw_ptr) free();
    }

    void resize(size_t size) {
        if (raw_ptr) free();
        alloc(size);
    }

    void alloc(size_t size) {
        assert(raw_ptr == nullptr);
        num_elements = size;
        cudaMalloc((void**)&raw_ptr, size_in_bytes());
    }

    void free() {
        cudaFree(raw_ptr);
        raw_ptr = nullptr;
        num_elements = 0;
    }

    template <typename U>
    void alloc_for_size(const std::vector<U> &vt) {
        alloc(vt.size());
    }

    void alloc_and_upload(const std::vector<T> &vt) {
        alloc_for_size(vt);
        upload((const T*)vt.data(), vt.size());
    }

    void upload(const T* t, size_t count) {
        assert(raw_ptr != nullptr);
        assert(count <= num_elements);
        cudaMemcpy(raw_ptr, (void *)t, count * sizeof(T), cudaMemcpyHostToDevice);
    }

    T download_first_item() {
        assert(raw_ptr != nullptr);
        T temp;
        download(&temp, 1);
        return temp;
    }

    std::vector<T> download(size_t count) {
        assert(raw_ptr != nullptr);
        assert(count <= num_elements);
        std::vector<T> temp(count);
        download(temp.data(), count);
        return temp;
    }

    void download(T* t, size_t count) {
        assert(raw_ptr != nullptr);
        assert(count <= num_elements);
        cudaMemcpy((void *)t, raw_ptr, count * sizeof(T), cudaMemcpyDeviceToHost);
    }

    void zero() {
        assert(raw_ptr != nullptr);
        cudaMemset(raw_ptr, 0, size_in_bytes());
    }
};

#endif
