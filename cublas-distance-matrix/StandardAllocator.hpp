#pragma once

#include "Allocator.hpp"
#include "StandardAllocator.h"

template <typename T>
StandardAllocator<T>::StandardAllocator(const size_t size)
  : Allocator<T[]>(size)
{
  m_data.reset(new T[size]{});
}

template <typename T>
void StandardAllocator<T>::fill(const T& value)
{
  memset(&this->m_data[0], value, this->size());
}