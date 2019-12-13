#pragma once

#include <cstring>
#include <memory>

template <typename T>
class CudaAllocator;

template <typename T>
class StandardAllocator : public Allocator<T[]>
{
private:

public:

  explicit StandardAllocator(const size_t size);
  StandardAllocator(const StandardAllocator<T>& allocator);
  StandardAllocator(const CudaAllocator<T>& allocator);
  StandardAllocator(StandardAllocator<T>&& allocator);
  StandardAllocator<T>& operator=(StandardAllocator<T> allocator);
  virtual void fill(const T& value) override;
  T* raw() { return &m_data[0]; }
  const T* raw() const { return &m_data[0]; }
  ~StandardAllocator();

  template <typename U>
  friend class CudaAllocator;
};