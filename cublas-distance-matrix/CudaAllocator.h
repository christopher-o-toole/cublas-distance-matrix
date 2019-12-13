#pragma once

#include <functional>

#include "cuda_util.h"
#include "Allocator.h"

template <typename T>
class StandardAllocator;

using cuda_free = std::function<void(void*)>;

template <typename T>
class CudaAllocator : public Allocator<T, cuda_free>
{
private:
public:
  using Allocator<T, cuda_free>::Allocator;

  explicit CudaAllocator(const size_t size);
  CudaAllocator(CudaAllocator<T>&& allocator);
  CudaAllocator(const CudaAllocator<T>& allocator);
  CudaAllocator(const StandardAllocator<T>& allocator);
  CudaAllocator<T>& operator=(CudaAllocator<T> allocator);
  virtual void fill(const T& value) override;
  T* raw() { return m_raw_data; }
  const T* raw() const { return m_raw_data; }
  ~CudaAllocator();

  template <typename U>
  friend class StandardAllocator;
};