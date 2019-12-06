#include <iostream>

#include "Allocator.hpp"
#include "StandardAllocator.hpp"
#include "CudaAllocator.hpp"
#include "Matrix.hpp"

int main()
{
  Matrix<int, StandardAllocator> matrix(5, 5);
  Matrix<int, CudaAllocator> gpu_matrix(10, 10);
  std::cout << matrix << std::endl;
  return 0;
}
