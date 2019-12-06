#pragma once

#include "CublasMatrixAllocator.h"

template <typename T>
CublasMatrixAllocator<T>::CublasMatrixAllocator(StandardAllocator<T> data, const size_t rows, const size_t columns)
  : CudaAllocator<T>(rows*columns), m_rows(rows), m_columns(columns)
{
  cublasHandle_t handle = nullptr;
  CUBLAS_SAFE_CALL(cublasCreate(&handle));
  m_context.reset(handle);

  CUBLAS_SAFE_CALL(cublasSetMatrix(rows, columns, sizeof(T), this->get()));
}