
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
    GPUSettings gpu;
    gpu.n_frames = ref_coords.size();
    gpu.id = i_gpu;
    gpu.n_dim = n_dim;
    cudaSetDevice(i_gpu);
    check_error("setting CUDA device");
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

  __device__ char
  dev_is_neighbor(float* s_xs
                , float* s_coords
                , unsigned int n_cols
                , float rad2
                , unsigned int tid) {
    float d2 = 0.0f;
    for (unsigned int j=0; j < n_cols; ++j) {
      float d = s_coords[tid*n_cols+j] - s_xs[j];
      d2 += d*d;
    }
    if (d2 <= rad2) {
      return 1;
    } else {
      return 0;
    }
  }

  __global__ void
  neighbors_krnl(float* xs
               , float* ref_coords
               , float rad2
               , float dx
               , char* is_neighbor
               , unsigned int n_rows
               , unsigned int n_cols
               , unsigned int i_from
               , unsigned int i_to) {
    // CUDA-specific indices for block, thread and global
    unsigned int bsize = blockDim.x;
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int gid = bid * bsize + tid + i_from;
    // locally shared memory for fast access
    // of dx-shifted xs and reference coordinates
    extern __shared__ float smem[];
    float* s_coords = (float*) smem;
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
    if (gid < i_to) {
      unsigned int n_shifts = 2*n_cols + 1;
      // read ref coords to shared mem
      for (unsigned int j=0; j < n_cols; ++j) {
        s_coords[tid*n_cols+j] = ref_coords[gid*n_cols+j];
      }
      // check if is neighbor for different position shifts
      for (unsigned int j_shift=0; j_shift < n_shifts; ++j_shift) {
        is_neighbor[gid*n_shifts + j_shift] =
          dev_is_neighbor(&s_xs[j_shift*n_cols]
                        , &s_coords[tid*n_cols]
                        , n_cols
                        , rad2
                        , tid);
      }
    }
  }

  //TODO: doc: order of shifts, etc.
  std::vector<char>
  neighbors(const std::vector<float>& xs
          , float rad2
          , float dx
          , const std::vector<GPUSettings>& gpus) {
    int n_gpus = gpus.size();
    if (n_gpus == 0) {
      std::cerr << "error: unable to estimate free energies on GPU(s)."
                << std::endl
                << "       no GPUs have been provided."
                << std::endl;
      exit(EXIT_FAILURE);
    }
    unsigned int n_rows = gpus[0].n_frames;
    unsigned int n_cols = gpus[0].n_dim;
    unsigned int gpu_range = n_rows / n_gpus;
    int i_gpu;
    unsigned int i_from;
    unsigned int i_to;
    unsigned int block_rng;
    unsigned int shared_mem_size;
    // partial estimates: neighborhood info per GPU
    std::vector<std::vector<char>>
      is_neighbor_partial(n_gpus
                        , std::vector<char>(n_rows * (2*n_cols+1)));
    //// parallelize over available GPUs
    #pragma omp parallel for default(none)\
      private(i_gpu,i_from,i_to,block_rng,shared_mem_size)\
      firstprivate(rad2,dx,n_gpus,n_rows,n_cols,gpu_range)\
      shared(xs,gpus,is_neighbor_partial)\
      num_threads(n_gpus)\
      schedule(dynamic,1)
    for (i_gpu=0; i_gpu < n_gpus; ++i_gpu) {
      cudaSetDevice(i_gpu);
      check_error("set device");
      // set ranges for this GPU
      i_from = i_gpu * gpu_range;
      if (i_gpu == n_gpus-1) {
        i_to = n_rows;
      } else {
        i_to = (i_gpu+1) * gpu_range;
      }
      // copy current position to GPU
      cudaMemcpy(gpus[i_gpu].xs
               , xs.data()
               , sizeof(float) * n_cols
               , cudaMemcpyHostToDevice);
      check_error("copy reference point coordinates to device");
      // initialize is_neighbor to zero
      cudaMemset(gpus[i_gpu].is_neighbor
               , 0
               , sizeof(char) * n_rows * (2*n_cols+1));
      check_error("is_neighbor init");
      block_rng = min_multiplicator(i_to-i_from, BSIZE);
      // shared mem for dx-shifted positions (2*n_cols + 1)
      // and reference coordinates (n_cols * BSIZE)
      //TODO: check against max. available shared memory!
      shared_mem_size = sizeof(float) * ((2*n_cols+1) + (n_cols*BSIZE));
      // kernel call
      neighbors_krnl
      <<< block_rng
        , BSIZE
        , shared_mem_size >>> (gpus[i_gpu].xs
                             , gpus[i_gpu].coords
                             , rad2
                             , dx
                             , gpus[i_gpu].is_neighbor
                             , n_rows
                             , n_cols
                             , i_from
                             , i_to);
      cudaDeviceSynchronize();
      check_error("after kernel call");
      // retrieve partial results
      cudaMemcpy(is_neighbor_partial[i_gpu].data()
               , gpus[i_gpu].is_neighbor
               , sizeof(char) * n_rows * (2*n_cols+1)
               , cudaMemcpyDeviceToHost);
      check_error("copy neighbor estimate from device");
    }
    // combine results from all GPUs
    std::vector<char> is_neighbor = is_neighbor_partial[0];
    for (unsigned int i_gpu=1; i_gpu < n_gpus; ++i_gpu) {
      for (unsigned int i=0; i < is_neighbor_partial[i_gpu].size(); ++i) {
        if (is_neighbor_partial[i_gpu][i] == 1) {
          is_neighbor[i] = 1;
        }
      }
    }
    return is_neighbor;
  }

} // end namespace CUDA

