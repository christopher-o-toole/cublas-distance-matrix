#pragma once

#include "cuda_util.h"

#include <stdio.h>
#include <memory>
#include <cstring>

template <typename T,  
          class Deleter = std::default_delete<T>,
          typename BaseType = std::remove_all_extents<T>::type>
class Allocator
{
protected:
  std::unique_ptr<T, Deleter> m_data;
  size_t m_size;

public:
  Allocator(const size_t size)
    : m_size(size), m_data(nullptr, m_data.get_deleter())
  {

  }

  const BaseType& at(const size_t index) const
  {
    return m_data[index];
  }

  BaseType& at(const size_t index)
  {
    return m_data[index];
  }

  size_t size() const
  {
    return m_size;
  }

  virtual void fill(const BaseType& value) = 0;
  virtual ~Allocator() {}
};

auto cuda_free = [](auto* ptr) { cudaFree(ptr); };

template <typename T>
class CudaAllocator : public Allocator<T, decltype(cuda_free)>
{

private:
  T* m_raw_data;

public:
  CudaAllocator(const size_t size)
    : Allocator<T, decltype(cuda_free)>(size)
  {
    m_raw_data = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_raw_data, size * sizeof(T)));
    m_data.reset(m_raw_data);
  }

  virtual void fill(const T& value) override
  {
    CUDA_SAFE_CALL(cudaMemset((void*)m_raw_data, value, this->size() * sizeof(T)));
  }

  ~CudaAllocator()
  {
    m_raw_data = nullptr;
  }
};

template <typename T>
class StandardAllocator : public Allocator<T[]>
{
private:

public:
  StandardAllocator(const size_t size)
    : Allocator<T[]>(size)
  {
    m_data.reset(new T[size]{});
  }

  virtual void fill(const T& value) override
  {
    memset(&this->m_data[0], value, this->size());
  }
};