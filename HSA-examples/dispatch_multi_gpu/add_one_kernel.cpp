#include <hip/hip_runtime.h>

extern "C" {
__global__ void add_one(uint32_t n, uint32_t* buffer) {
  auto const idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    buffer[idx] += 1;
  }
}
}
