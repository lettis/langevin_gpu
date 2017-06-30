
#include "neighbors_cuda.hpp"


namespace CUDA {
  template <typename NUM>
  __device__ void
  atomicAddReduce(NUM* result
                , NUM* value) {
    // warp-reduction
    for (unsigned int offset = 16; offset > 0; offset /= 2) {
      value += __shfl_down(value, offset);
    }
    if (threadIdx.x & 31 == 0) {
      atomicAdd(result, value);
    }
  }

} // end namespace CUDA

