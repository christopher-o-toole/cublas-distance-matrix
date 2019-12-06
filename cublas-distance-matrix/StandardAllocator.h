#pragma once

#include <cstring>

template <typename T>
class StandardAllocator : public Allocator<T[]>
{
private:

public:
  StandardAllocator(const size_t size);
  virtual void fill(const T& value) override;
};