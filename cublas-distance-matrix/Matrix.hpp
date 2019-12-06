#pragma once

#include <type_traits>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <initializer_list>

#include "cublas_util.hpp"
#include "util.hpp"

#define ENABLE_IF_NUMERIC(T) typename std::enable_if<std::is_arithmetic<T>::value, T>::type
#define ENABLE_IF_CUDA_ALLOCATOR(T, Container) std::enable_if<std::is_base_of<CudaAllocator<T>, Container<T>>::value, CudaAllocator<T>>
#define ENABLE_IF_STANDARD_ALLOCATOR(T, Container) std::enable_if<std::is_base_of<StandardAllocator<T>, Container<T>>::value, StandardAllocator<T>>

template<typename T, template <typename...> class Container,
         typename = ENABLE_IF_NUMERIC(T)>
class Matrix
{
private:
  size_t m_rows;
  size_t m_columns;
  Container<T> m_data;

public:
  template<typename U = T, typename std::enable_if<!std::is_base_of<CudaAllocator<U>, Container<U>>::value, bool>::type = true>
  Matrix(size_t m, size_t n, bool set_to_zero = true)
    : m_rows(m), m_columns(n), m_data(m*n)
  {
    std::cout << "is base of:" << std::is_base_of<CudaAllocator<T>, Container<T>>::value << std::endl;
    if (set_to_zero)
    {
      m_data.fill(0);
    }
  }
  
  template<typename U = T, typename std::enable_if<std::is_base_of<CudaAllocator<U>, Container<U>>::value, bool>::type = true>
  Matrix(size_t m, size_t n, bool set_to_zero = true)
    : m_rows(m), m_columns(n), m_data(m*n)
  {
    CUBLAS::Library::Init();

    if (set_to_zero)
    {
      m_data.fill(0);
    }

    std::cout << "cublas init" << std::endl;
    CUBLAS_SAFE_CALL(cublasSetMatrix(m, n, sizeof(T), this->m_data.get()));
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