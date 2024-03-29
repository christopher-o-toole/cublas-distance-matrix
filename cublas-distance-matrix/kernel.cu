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
  Matrix<int, StandardAllocator> test_gpu_to_cpu(gpu_matrix);
  Matrix<int, CudaAllocator> gpu_matrix_clone(gpu_matrix);

  std::cout << matrix << std::endl;
  std::cout << matrix_test_init << std::endl;
  std::cout << test_gpu_to_cpu << std::endl;

  Matrix<int, CudaAllocator> test_cpu_to_gpu(matrix_test_init);
  Matrix<int, StandardAllocator> test_gpu_to_cpu_agane(test_cpu_to_gpu);
  std::cout << test_gpu_to_cpu_agane << std::endl;

  return 0;
}
