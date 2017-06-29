
#include "neighbors_cuda.hpp"

#include <limits>
#include <iostream>

#include <stdio.h>

namespace CUDA {

  void
  check_error(std::string msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "CUDA error: "
                << msg << "\n"
                << cudaGetErrorString(err) << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  int
  get_num_gpus() {
    int n_gpus;
    cudaGetDeviceCount(&n_gpus);
    check_error("trying to get number of available GPUs");
    if (n_gpus == 0) {
      std::cerr << "error: no CUDA-compatible GPU(s) found."
                << std::endl
                << "       if you are sure to have one,"
                << std::endl
                << "       check your device drivers!"
                << std::endl;
      exit(EXIT_FAILURE);
    } else {
      return n_gpus;
    }
  }

  GPUSettings
  prepare_gpu(int i_gpu
            , unsigned int n_dim
            , const std::vector<std::vector<float>>& ref_coords) {
    cudaSetDevice(i_gpu);
    check_error("setting CUDA device");
    GPUSettings gpu;
    gpu.n_frames = ref_coords.size();
    gpu.id = i_gpu;
    gpu.n_dim = n_dim;
    //// reserve memory for reference point (aka 'xs')
    cudaMalloc((void**) &gpu.xs
             , sizeof(float) * n_dim);
    check_error("malloc xs");
    //// reserve memory for references
    cudaMalloc((void**) &gpu.coords
             , sizeof(float) * gpu.n_frames * gpu.n_dim);
    check_error("malloc reference coords");
    // copy reference coords in 1D array (row-major order) to GPU
    std::vector<float> tmp_coords(n_dim * gpu.n_frames);
    for (unsigned int i=0; i < gpu.n_frames; ++i) {
      for (unsigned int j=0; j < n_dim; ++j) {
        tmp_coords[i*n_dim+j] = ref_coords[i][j];
      }
    }
    cudaMemcpy(gpu.coords
             , tmp_coords.data()
             , sizeof(float) * n_dim * gpu.n_frames
             , cudaMemcpyHostToDevice);
    check_error("copying of state-splitted coordinates");
    //// allocate memory for partial (i.e. per-GPU) results
    cudaMalloc((void**) &gpu.is_neighbor
             , sizeof(char) * gpu.n_frames * (2*n_dim+1));
    check_error("malloc partial neighbor count");
    // ... and return GPU-settings
    return gpu;
  }

  void
  clear_gpu(GPUSettings gpu) {
    cudaSetDevice(gpu.id);
    check_error("setting CUDA device");
    cudaFree(gpu.xs);
    check_error("freeing memory for xs");
    cudaFree(gpu.coords);
    check_error("freeing memory for coordinates");
    cudaFree(gpu.is_neighbor);
    check_error("freeing memory for partial neighbor count");
  }

  unsigned int
  min_multiplicator(unsigned int orig
                  , unsigned int mult) {
    return (unsigned int) std::ceil(orig / ((float) mult));
  };

  __global__ void
  neighbors_krnl(float* xs
               , float* ref_coords
               , float rad2
               , float dx
               , char* is_neighbor
               , unsigned int n_rows
               , unsigned int n_cols) {
    // CUDA-specific indices for block, thread and global
    unsigned int bsize = blockDim.x;
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int gid = bid * bsize + tid;
    // locally shared memory for fast access
    // of dx-shifted xs and reference coordinates
    extern __shared__ float smem[];
    float* s_coords = (float*) &smem[0];
    float* s_xs = (float*) &smem[n_cols*BSIZE];
    // read xs to shared mem
    if (tid < n_cols) {
      float x = xs[tid];
      // unshifted xs
      s_xs[tid] = x;
      // shifted xs for numerical gradient estimation
      for (unsigned int j_shift=0; j_shift < n_cols; ++j_shift) {
        if (j_shift != tid) {
          s_xs[(2*j_shift+1)*n_cols+tid] = x;
          s_xs[(2*j_shift+2)*n_cols+tid] = x;
        } else {
          s_xs[(2*j_shift+1)*n_cols+tid] = x + dx;
          s_xs[(2*j_shift+2)*n_cols+tid] = x - dx;
        }
      }
    }
    __syncthreads();
    if (gid < n_rows) {
      unsigned int n_shifts = 2*n_cols + 1;
      // read ref coords to shared mem
      for (unsigned int j=0; j < n_cols; ++j) {
        s_coords[tid*n_cols+j] = ref_coords[gid*n_cols+j];
      }
      // check if is neighbor for different position shifts
      for (unsigned int j_shift=0; j_shift < n_shifts; ++j_shift) {
        float d2 = 0.0f;
        for (unsigned int k=0; k < n_cols; ++k) {
          float d = s_coords[tid*n_cols+k] - s_xs[j_shift*n_cols+k];
          d2 += d*d;
        }
        if (d2 <= rad2) {
          is_neighbor[gid*n_shifts + j_shift] = 1;
        } else {
          is_neighbor[gid*n_shifts + j_shift] = 0;
        }
      }
    }
  }

  //TODO: doc: order of shifts, etc.
  std::vector<char>
  neighbors(const std::vector<float>& xs
          , float rad2
          , float dx
          , const GPUSettings& gpu) {

    unsigned int n_rows = gpu.n_frames;
    unsigned int n_cols = gpu.n_dim;
    std::vector<char> is_neighbor(n_rows*(2*n_cols+1));

    unsigned int block_rng;
    unsigned int shared_mem_size;
    cudaSetDevice(gpu.id);
    check_error("set device");
    // copy current position to GPU
    cudaMemcpy(gpu.xs
             , xs.data()
             , sizeof(float) * n_cols
             , cudaMemcpyHostToDevice);
    check_error("copy reference point coordinates to device");
    // initialize is_neighbor to zero
    cudaMemset(gpu.is_neighbor
             , 0
             , sizeof(char) * n_rows * (2*n_cols+1));
    check_error("is_neighbor init");
    block_rng = min_multiplicator(n_rows, BSIZE);
    // shared mem for dx-shifted positions [n_cols * (2*n_cols + 1)]
    // and reference coordinates [n_cols * BSIZE]
    //TODO: check against max. available shared memory!
    shared_mem_size = sizeof(float) * n_cols * ((2*n_cols+1) + BSIZE);
    // kernel call
    neighbors_krnl
    <<< block_rng
      , BSIZE
      , shared_mem_size >>> (gpu.xs
                           , gpu.coords
                           , rad2
                           , dx
                           , gpu.is_neighbor
                           , n_rows
                           , n_cols);
    cudaDeviceSynchronize();
    check_error("after kernel call");
    // retrieve partial results
    cudaMemcpy(is_neighbor.data()
             , gpu.is_neighbor
             , sizeof(char) * n_rows * (2*n_cols+1)
             , cudaMemcpyDeviceToHost);
    check_error("copy neighbor estimate from device");
    return is_neighbor;
  }

} // end namespace CUDA

