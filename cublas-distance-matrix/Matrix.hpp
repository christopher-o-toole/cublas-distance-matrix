#pragma once

#include <type_traits>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <initializer_list>
#include <vector>

#include "cublas_util.hpp"
#include "util.hpp"

#define ENABLE_IF_NUMERIC(T) typename std::enable_if<std::is_arithmetic<T>::value, T>::type
#define ENABLE_IF_NOT_CUDA_ALLOCATOR(U, Container) typename std::enable_if<!std::is_base_of<CudaAllocator<U>, Container<U>>::value, bool>::type = true
#define ENABLE_IF_CUDA_ALLOCATOR(U, Container) typename std::enable_if<std::is_base_of<CudaAllocator<U>, Container<U>>::value, bool>::type = true

template<typename T, template <typename...> class Container,
         typename = ENABLE_IF_NUMERIC(T)>
class Matrix
{
private:
  size_t m_rows;
  size_t m_columns;
  Container<T> m_data;

public:
  Matrix()
    : m_rows(0), m_columns(0), m_data(0)
  {

  }

  template<typename U = T, ENABLE_IF_NOT_CUDA_ALLOCATOR(U, Container)>
  Matrix(size_t m, size_t n, bool set_to_zero = true)
    : m_rows(m), m_columns(n), m_data(m*n)
  {
    if (set_to_zero)
    {
      m_data.fill(0);
    }
  }

  template<typename U = T, ENABLE_IF_CUDA_ALLOCATOR(U, Container)>
  Matrix(size_t m, size_t n)
    : m_rows(m), m_columns(n), m_data(m * n)
  {
    CUBLAS::Library::Init();
    T* raw_data = new T[m * n]{ 0 };
    std::unique_ptr<T[]> data(raw_data);
    CUBLAS_SAFE_CALL(cublasSetMatrix(m, n, sizeof(T), raw_data, m, m_data.raw(), m));
  }

  template<typename U = T, ENABLE_IF_NOT_CUDA_ALLOCATOR(U, Container)>
  Matrix(const std::initializer_list<std::vector<T>>& matrix)
    : m_data(0)
  {
    m_rows = matrix.size();

    if (m_rows > 0)
    {
      m_columns = matrix.begin()->size();
    }

    ASSERT(m_rows > 0 && m_columns > 0, "Cannot create a matrix of size (" << m_rows << ", " << m_columns << ").");
    m_data = Container<T>(m_rows * m_columns);
    size_t i = 0;
    size_t j = 0;

    for (const auto& row : matrix)
    {
      ASSERT(m_columns == row.size(), "Expected a row with size " << m_columns
        << ", got a row with size " << row.size() << " instead.");

      for (const auto& element : row)
      {
        (*this)[{i, j++}] = element;
      }

      i++;
      j = 0;
    }
  }

  Matrix(const Matrix<T, Container>& matrix)
    : m_data(matrix.m_data),
      m_rows(matrix.m_rows),
      m_columns(matrix.m_columns)
  {

  }

  Matrix(Matrix<T, Container>&& matrix)
    : m_data(std::move(matrix.m_data)),
      m_rows(matrix.m_rows),
      m_columns(matrix.m_columns)
  {

  }

  template <class U, template <typename...> class OtherContainer>
  operator Matrix<U, OtherContainer>()
  {
    Matrix<U, OtherContainer> matrix(rows(), columns());
    matrix.m_data = m_data;
    return matrix;
  }

  Matrix<T, Container>& operator=(Matrix<T, Container> rhs)
  {
    swap(*this, rhs);
    return *this;
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

  template<typename U, template <typename...> class OtherContainer>
  friend std::ostream& operator<<(std::ostream& os, const Matrix<U, OtherContainer>& rhs);

  template <typename U, template <typename...> class OtherContainer>
  friend void swap(Matrix<U, OtherContainer>& lhs, Matrix<U, OtherContainer>& rhs);

  template <typename U, template <typename...> class OtherContainer, typename DefaultParam>
  friend class Matrix;
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

template <typename U, template <typename...> class Container>
void swap(Matrix<U, Container>& lhs, Matrix<U, Container>& rhs)
{
  std::swap(lhs.m_data, rhs.m_data);
  std::swap(lhs.m_rows, rhs.m_rows);
  std::swap(lhs.m_columns, rhs.m_columns);
}