#pragma once

#include "Allocator.hpp"
#include "CudaAllocator.h"
#include "StandardAllocator.h"

template <typename T>
CudaAllocator<T>::CudaAllocator(const size_t size)
  : Allocator<T, cuda_free>(size, [](void* ptr) { cudaFree(ptr); })
{
  m_raw_data = nullptr;
  CUDA_SAFE_CALL(cudaMalloc((void**)&m_raw_data, size * sizeof(T)));
  m_data.reset(m_raw_data);
}

template <typename T>
CudaAllocator<T>::CudaAllocator(const StandardAllocator<T>& allocator)
  : CudaAllocator<T>(allocator.size())
{
  CUDA_SAFE_CALL(cudaMemcpy((void*)m_raw_data,
    (void*)allocator.raw(),
    allocator.size() * sizeof(T),
    cudaMemcpyHostToDevice));
}


template <typename T>
CudaAllocator<T>::CudaAllocator(const CudaAllocator<T>& allocator)
  : CudaAllocator<T>(allocator.size())
{
  CUDA_SAFE_CALL(cudaMemcpy((void*)m_raw_data,
    (void*)allocator.raw(),
    allocator.size() * sizeof(T),
    cudaMemcpyDeviceToDevice));
}

template <typename T>
CudaAllocator<T>::CudaAllocator(CudaAllocator<T>&& allocator)
  : CudaAllocator<T>(0)
{
  m_data = std::move(allocator.m_data);
  m_raw_data = allocator.m_raw_data;
  m_size = allocator.m_size;
}

template <typename T>
CudaAllocator<T>& CudaAllocator<T>::operator=(CudaAllocator<T> allocator)
{
  swap(*this, allocator);
  return *this;
}

template <typename T>
void CudaAllocator<T>::fill(const T& value)
{
  CUDA_SAFE_CALL(cudaMemset((void*)m_raw_data, value, this->size() * sizeof(T)));
}

template <typename T>
CudaAllocator<T>::~CudaAllocator()
{
  m_raw_data = nullptr;
}