#include <iostream>

#include "cuda_util.h"
#include "Allocator.hpp"
#include "Matrix.hpp"

int main()
{
  Matrix<int, StandardAllocator> matrix(5, 5);
  Matrix<int, CudaAllocator> gpu_matrix(10, 10);
  std::cout << matrix << std::endl;
  return 0;
}
