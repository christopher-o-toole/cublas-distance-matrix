#include <iostream>

#include "Allocator.hpp"
#include "StandardAllocator.hpp"
#include "CudaAllocator.hpp"
#include "Matrix.hpp"

int main()
{
  Matrix<int, StandardAllocator> matrix(5, 5);
  Matrix<int, StandardAllocator> matrix_test_init{
    {1, 0, 0},
    {0, 1, 0},
    {0, 0, 1}
  };

  Matrix<int, CudaAllocator> gpu_matrix(10, 10);
  std::cout << matrix << std::endl;
  return 0;
}
