#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "cublas_v2.h"

#include <stdio.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 


#define CUDA_SAFE_CALL(status) { cuda_safe_call((status), __FILE__, __LINE__); }
#define CUBLAS_SAFE_CALL(status) { cublas_safe_call((status), __FILE__, __LINE__); }

inline void cuda_safe_call(cudaError_t code, const char* file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "CUDA Error: %s (%s:%d)\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

static const char* cublasGetErrorString(cublasStatus_t error)
{
  switch (error)
  {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";

  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";

  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";

  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";

  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";

  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";

  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";

  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  }

  return "<unknown>";
}

inline void cublas_safe_call(cublasStatus_t code, const char* file, int line, bool abort = true)
{
  if (code != CUBLAS_STATUS_SUCCESS)
  {
    fprintf(stderr, "CUBLAS Error: %s (%s:%d)\n", cublasGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}