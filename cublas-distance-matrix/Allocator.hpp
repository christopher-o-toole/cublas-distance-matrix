#pragma once

#include "Allocator.h"

template<typename T, class Deleter, typename BaseType>
const BaseType& Allocator<T, Deleter, BaseType>::at(const size_t index) const
{
  return m_data[index];
}

template<typename T, class Deleter, typename BaseType>
BaseType& Allocator<T, Deleter, BaseType>::at(const size_t index)
{
  return m_data[index];
}

template<typename T, class Deleter, typename BaseType>
size_t Allocator<T, Deleter, BaseType>::size() const
{
  return m_size;
}

template<typename T, class Deleter, typename BaseType>
const BaseType* Allocator<T, Deleter, BaseType>::get() const
{
  return m_data.get();
}

template <typename U, typename UDeleter>
void swap(Allocator<U, UDeleter>& a, Allocator<U, UDeleter>& b)
{
  using std::swap;
  swap(a.m_data, b.m_data);
  swap(a.m_size, b.m_size);
}