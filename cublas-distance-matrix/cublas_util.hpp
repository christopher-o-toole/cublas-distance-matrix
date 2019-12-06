#pragma once

#include "cuda_util.h"

#include <memory>

static auto cublas_destroy = [](cublasHandle_t handle) { cublasDestroy(handle); };

namespace CUBLAS
{
  class Library
  {
  private:
    static std::unique_ptr<cublasContext, decltype(cublas_destroy)> m_context;

  public:
    static void Init()
    {
      if (!Library::m_context)
      {
        cublasHandle_t handle = nullptr;
        CUBLAS_SAFE_CALL(cublasCreate(&handle));
        Library::m_context.reset(handle);
      }
    }
  };
}