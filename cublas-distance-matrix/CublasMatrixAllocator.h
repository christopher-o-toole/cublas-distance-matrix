#pragma once

#include "StandardAllocator.h"
#include "CudaAllocator.hpp"

static auto cublas_destroy = [](cublasHandle_t handle) { cublasDestroy(handle); };

template <typename T>
class CublasMatrixAllocator : public CudaAllocator<T>
{
private:
  std::unique_ptr<cublasContext*, decltype(cublas_destroy)> m_context;
  size_t m_rows;
  size_t m_columns;
public:
  using CudaAllocator<T>::CudaAllocator;
  using CudaAllocator<T>::operator=;

  CublasMatrixAllocator(StandardAllocator<T> data, const size_t rows, const size_t columns);

};