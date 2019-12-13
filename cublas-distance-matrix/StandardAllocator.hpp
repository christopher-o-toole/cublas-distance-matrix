#pragma once

#include "Allocator.hpp"
#include "StandardAllocator.h"
#include "CudaAllocator.h"

#include <algorithm>

template <typename T>
StandardAllocator<T>::StandardAllocator(const size_t size)
  : Allocator<T[]>(size)
{
  m_data.reset(new T[size]{});
}

template <typename T>
StandardAllocator<T>::StandardAllocator(StandardAllocator<T>&& allocator)
  : StandardAllocator<T>(0)
{
  m_data = std::move(allocator.m_data);
  m_raw_data = std::move(allocator.m_raw_data);
  m_size = std::move(allocator.m_size);
}

template <typename T>
StandardAllocator<T>::StandardAllocator(const StandardAllocator<T>& allocator)
  : StandardAllocator<T>(allocator.size())
{
  std::copy_n(allocator.m_data.get(), alloator.size(), m_data.get());
}

template <typename T>
StandardAllocator<T>::StandardAllocator(const CudaAllocator<T>& allocator)
  : StandardAllocator<T>(allocator.size())
{
  CUDA_SAFE_CALL(cudaMemcpy((void*)&m_data[0],
    (const void*)allocator.get(),
    allocator.size() * sizeof(T),
    cudaMemcpyDeviceToHost));
}

template <typename T>
StandardAllocator<T>& StandardAllocator<T>::operator=(StandardAllocator<T> allocator)
{
  swap(*this, allocator);
  return *this;
}

template <typename T>
void StandardAllocator<T>::fill(const T& value)
{
  memset(&this->m_data[0], value, this->size());
}

template <typename T>
StandardAllocator<T>::~StandardAllocator()
{
  m_raw_data = nullptr;
}