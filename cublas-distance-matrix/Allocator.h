#pragma once

#include <memory>
#include <type_traits>

template <typename T,
  class Deleter = std::default_delete<T>,
  typename BaseType = std::remove_all_extents<T>::type>
  class Allocator
{
protected:
  std::unique_ptr<T, Deleter> m_data;
  BaseType* m_raw_data;
  size_t m_size;

public:
  Allocator(const size_t size)
    : m_size(size), 
      m_data(nullptr, m_data.get_deleter()),
      m_raw_data(nullptr)
  {

  }

  Allocator(const size_t size, const Deleter& deleter)
    : m_size(size),
      m_data(nullptr, deleter),
      m_raw_data(nullptr)
  {

  }

  Allocator(Allocator<T, Deleter>&& rvalue) noexcept :
    Allocator<T, Deleter>(rvalue.size())
  {
    this->swap(*this, rvalue);
  }

  const BaseType& at(const size_t index) const;
  BaseType& at(const size_t index);
  size_t size() const;
  const BaseType* get() const;

  virtual void fill(const BaseType& value) = 0;

  template <typename U, typename UDeleter>
  friend void swap(Allocator<U, UDeleter>& a, Allocator<U, UDeleter>& b);

  virtual ~Allocator() {}
};