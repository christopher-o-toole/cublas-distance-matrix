#pragma once

#include "cuda_util.h"
#include "Allocator.h"
#include "StandardAllocator.h"

static auto cuda_free = [](auto* ptr) { cudaFree(ptr); };

template <typename T>
class StandardAllocator;

template <typename T>
class CudaAllocator : public Allocator<T, decltype(cuda_free)>
{
private:
public:
  using Allocator<T, decltype(cuda_free)>::Allocator;

  CudaAllocator(const size_t size);
  CudaAllocator(const CudaAllocator<T>& allocator);
  CudaAllocator(const StandardAllocator<T>& allocator);
  CudaAllocator<T>& operator=(CudaAllocator<T> allocator);
  virtual void fill(const T& value) override;
  T* raw() { return m_raw_data; }
  ~CudaAllocator();
};