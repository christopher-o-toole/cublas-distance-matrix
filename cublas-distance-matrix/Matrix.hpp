#pragma once

#include <type_traits>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <initializer_list>

#include "Allocator.hpp"
#include "util.hpp"

template<typename T, template <typename...> class Container,
         typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
class Matrix
{
private:
  size_t m_rows;
  size_t m_columns;
  Container<T> m_data;

public:
  Matrix(size_t m, size_t n, bool set_to_zero = true)
    : m_rows(m), m_columns(n), m_data(m* n)
  {
    if (set_to_zero)
    {

      m_data.fill(0);
    }
  }

  T& operator[](const std::pair<size_t, size_t>& index)
  {
    const size_t _index = index.first * rows() + index.second;
    ASSERT(_index < rows() * columns(), "index with row=" << index.first << " and column=" << index.second
      << " out of range for matrix of size " << rows() << "x" << columns());
    return m_data.at(_index);
  }

  const T& operator[](const std::pair<size_t, size_t>& index) const
  {
    return (*const_cast<Matrix<T, Container>*>(this))[index];
  }

  size_t rows() const
  {
    return m_rows;
  }

  size_t columns() const
  {
    return m_columns;
  }

  template<typename U, template <typename...> class Container>
  friend std::ostream& operator<<(std::ostream& os, const Matrix<U, Container>& rhs);
};

template<typename U, template <typename...> class Container>
std::ostream& operator<<(std::ostream& os, const Matrix<U, Container>& rhs)
{
  std::ostringstream stream;
  const size_t element_width = 8;

  auto write_value_to_stream = [&](auto value) {
    stream << std::setw(element_width) << value << " ";
  };

  for (size_t i = 0; i < rhs.rows(); ++i)
  {
    for (size_t j = 0; j < rhs.columns(); ++j)
    {
      write_value_to_stream(rhs[{i, j}]);
    }

    stream << "\n";
  }

  os << stream.str();
  return os;
}