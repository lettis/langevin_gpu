
#include "langevin_cuda.hpp"

#include <limits>
#include <iostream>

#include <stdio.h>

namespace Langevin {
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

  void
  sync_and_check(std::string msg) {
    cudaThreadSynchronize();
    check_error(msg);
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
            , const std::vector<std::vector<float>>& ref_coords
            , const std::vector<char>& has_future
            , const std::vector<float>& fe) {
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
    //// reserve memory for reference coords
    cudaMalloc((void**) &gpu.coords
             , sizeof(float) * gpu.n_frames * gpu.n_dim);
    check_error("malloc coords");
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
    check_error("copy coords");
    //// reserve memory for reference free energies
    cudaMalloc((void**) &gpu.fe
             , sizeof(float) * gpu.n_frames);
    // copy reference free energies to GPU
    cudaMemcpy(gpu.fe
             , fe.data()
             , sizeof(float) * gpu.n_frames
             , cudaMemcpyHostToDevice);
    check_error("copy fe");
    //// allocate memory for neighborhood
    cudaMalloc((void**) &gpu.is_neighbor
             , sizeof(char) * gpu.n_frames * (2*n_dim+1));
    check_error("malloc is_neighbor");
    //// allocate memory number of neighbors
    cudaMalloc((void**) &gpu.n_neighbors
             , sizeof(unsigned int) * (2*n_dim+1));
    check_error("malloc n_neighbors");
    //// reserve memroy for futures
    cudaMalloc((void**) &gpu.has_future
             , sizeof(char) * gpu.n_frames);
    check_error("malloc has_future");
    // copy futures to GPU
    cudaMemcpy(gpu.has_future
             , has_future.data()
             , sizeof(char) * gpu.n_frames
             , cudaMemcpyHostToDevice);
    check_error("copy has_future");
//    //// allocate memory for shifted fe estimates
//    cudaMalloc((void**) &gpu.shifts_fe
//             , sizeof(float) * 2*gpu.n_dim);
//    check_error("malloc shifts_fe");
    //// allocate memory for velocity mean values
    cudaMalloc((void**) &gpu.v_means
             , sizeof(float) * 2*gpu.n_dim);
    check_error("malloc v_means");
    //// allocate memory for covariance matrix
    cudaMalloc((void**) &gpu.cov
             , sizeof(float) * gpu.n_dim*gpu.n_dim);
    check_error("malloc cov");
    // ... and return GPU-settings
    return gpu;
  }

  void
  clear_gpu(GPUSettings& gpu) {
    cudaFree(gpu.xs);
    check_error("free xs");
    cudaFree(gpu.coords);
    check_error("free coords");
    cudaFree(gpu.fe);
    check_error("free fe");
    cudaFree(gpu.is_neighbor);
    check_error("free is_neighbor");
    cudaFree(gpu.n_neighbors);
    check_error("free n_neighbors");
    cudaFree(gpu.has_future);
    check_error("free has_future");
    cudaFree(gpu.v_means);
    check_error("free v_means");
    cudaFree(gpu.cov);
    check_error("free cov");
  }

  unsigned int
  n_blocks(unsigned int n_frames
         , unsigned int block_size) {
    return (unsigned int) std::ceil(n_frames / ((float) block_size));
  };


  //// kernel functions running on GPU
  
  template <typename NUM>
  __device__ NUM
  warpReduceShfl(NUM value) {
    // warp-reduction
    for (unsigned int offset = 16; offset > 0; offset /= 2) {
      value += __shfl_down(value, offset);
    }
    return value;
  }

  template <typename NUM>
  __device__ void
  warpReduceMem(volatile NUM* field
              , unsigned int tid) {
    if (tid < 32) {
      field[tid] += field[tid+32];
      field[tid] += field[tid+16];
      field[tid] += field[tid+ 8];
      field[tid] += field[tid+ 4];
      field[tid] += field[tid+ 2];
      field[tid] += field[tid+ 1];
    }
  }


  template <typename NUM, unsigned int block_size>
  __device__ void
  pairwiseMemReduce(volatile NUM* field
                  , unsigned int tid) {
    if (block_size >= 512) {
      if (tid < 256) {
        field[tid] += field[tid+256];
      }
      __syncthreads();
    }
    if (block_size >= 256) {
      if (tid < 128) {
        field[tid] += field[tid+128];
      }
      __syncthreads();
    }
    if (block_size >= 128) {
      if (tid < 64) {
        field[tid] += field[tid+64];
      }
      __syncthreads();
    }
  }


  extern __shared__ float smem[];

  __global__ void
  neighbors_krnl(float* xs
               , float* ref_coords
               , float rad2
               , char* has_future
               , unsigned int n_frames
               , unsigned int n_dim
               , char* is_neighbor) {
    // CUDA-specific indices for block, thread and global
    unsigned int bsize = blockDim.x;
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int gid = bid * bsize + tid;
    // locally shared memory for fast access of xs and reference coordinates
    float* s_coords = (float*) &smem[0];
    float* s_xs = (float*) &smem[n_dim*BSIZE];
    // read xs to shared mem
    if (tid < n_dim) {
      s_xs[tid] = xs[tid];
    }
    __syncthreads();
    if (gid < n_frames) {
      // without history (no future, no past), don't count as neighbor!
      if ((gid == 0)
       || (gid == n_frames-1)
       || (has_future[gid] == 0)
       || (has_future[gid-1] == 0)) {
        is_neighbor[gid] = 0;
      } else {
        // read ref coords to shared mem
        for (unsigned int j=0; j < n_dim; ++j) {
          s_coords[tid*n_dim+j] = ref_coords[gid*n_dim+j];
        }
        float d2 = 0.0f;
        for (unsigned int k=0; k < n_dim && d2 < rad2; ++k) {
          float d = s_coords[tid*n_dim+k] - s_xs[k];
          d2 += d*d;
        }
        if (d2 < rad2) {
          is_neighbor[gid] = 1;
        } else {
          is_neighbor[gid] = 0;
        }
      }
    }
  }

  __global__ void
  count_neighbors_krnl(char* is_neighbor
                     , unsigned int n_frames
                     , unsigned int* n_neighbors) {
    __shared__ unsigned int smem_uint[BSIZE];
    // CUDA-specific indices for block, thread and global
    unsigned int bsize = blockDim.x;
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int gid = bid * bsize + tid;
    // count found neighbors for every reference position shift
    bool this_is_neighbor = (is_neighbor[gid] == 1);
    if (gid < n_frames-1
     && this_is_neighbor) {
      smem_uint[tid] = 1;
    } else {
      smem_uint[tid] = 0;
    }
    __syncthreads();
    //// aggregate results
    pairwiseMemReduce<unsigned int, BSIZE>(smem_uint
                                         , tid);
    warpReduceMem<unsigned int>(smem_uint
                              , tid);
    if (tid == 0) {
      atomicAdd(n_neighbors
              , smem_uint[0]);
    }
  }

  __global__ void
  v_means_krnl(char* is_neighbor
             , float* coords
             , unsigned int* n_neighbors
             , unsigned int n_frames
             , unsigned int n_dim
             , float* v_means) {
    __shared__ float v_forward[BSIZE];
    __shared__ float v_backward[BSIZE];
    // CUDA-specific indices for block, thread and global
    unsigned int bsize = blockDim.x;
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int gid = bid * bsize + tid;
    bool this_is_neighbor = (is_neighbor[gid] == 1)
                         && (0 < gid)
                         && (gid < n_frames-1);
    for (unsigned int j=0; j < n_dim; ++j) {
      // aggregate forward and backward velocities for every dimension
      if (this_is_neighbor) {
        float x = coords[gid*n_dim+j];
        float x_forward = coords[(gid+1)*n_dim+j];
        float x_backward = coords[(gid-1)*n_dim+j];
        v_forward[tid] = x_forward - x;
        v_backward[tid] = x - x_backward;
      } else {
        v_forward[tid] = 0.0f;
        v_backward[tid] = 0.0f;
      }
      //// aggregate results
      pairwiseMemReduce<float, BSIZE>(v_forward
                                    , tid);
      pairwiseMemReduce<float, BSIZE>(v_backward
                                    , tid);
      warpReduceMem(v_forward
                  , tid);
      warpReduceMem(v_backward
                  , tid);
      if (tid == 0) {
        atomicAdd(&v_means[j]
                , v_forward[0]);
        atomicAdd(&v_means[n_dim+j]
                , v_backward[0]);
      }
    }
  }

  __global__ void
  normalize_v_means_krnl(unsigned int* n_neighbors
                       , unsigned int n_dim
                       , float* v_means) {
    unsigned int j = threadIdx.x;
    if (j < n_dim) {
      v_means[j] /= (float) n_neighbors[0];
      v_means[n_dim+j] /= (float) n_neighbors[0];
    }
  }

  template <bool i_forward, bool j_forward>
  __global__ void
  cov_krnl(char* is_neighbor
         , float* coords
         , float* v_means
         , unsigned int n_frames
         , unsigned int n_dim
         , float* cov) {
    __shared__ float smem_v_i[BSIZE];
    __shared__ float smem_cov_ij[BSIZE];
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + tid;
    bool this_is_neighbor = (is_neighbor[gid] == 1)
                         && (0 < gid)
                         && (gid < n_frames-1);
    // compute local contributions to cov[i,j],
    // depending on forward or backward velocities.
    for (unsigned int i=0; i < n_dim; ++i) {
      if (this_is_neighbor) {
        if (i_forward) {
          smem_v_i[tid] = coords[(gid+1)*n_dim+i]
                        - coords[gid*n_dim+i]
                        - v_means[i];
        } else {
          smem_v_i[tid] = coords[gid*n_dim+i]
                        - coords[(gid-1)*n_dim+i]
                        - v_means[i+n_dim];
        }
      } else {
        smem_v_i[tid] = 0.0f;
      }
      __syncthreads();
      for (unsigned int j=0; j < n_dim; ++j) {

        // don't compute cov if matrix is symmetric and j > i
        if (i_forward == j_forward && j > i) {
          break;
        }

        if (this_is_neighbor) {
          float v_j;
          if (j_forward) {
            v_j = coords[(gid+1)*n_dim+j]
                - coords[gid*n_dim+j]
                - v_means[j];
          } else {
            v_j = coords[gid*n_dim+j]
                - coords[(gid-1)*n_dim+j]
                - v_means[j+n_dim];
          }
          smem_cov_ij[tid] = smem_v_i[tid] * v_j;
        } else {
          smem_cov_ij[tid] = 0.0f;
        }
        __syncthreads();
        //// aggregate results
        pairwiseMemReduce<float, BSIZE>(smem_cov_ij
                                      , tid);
        warpReduceMem<float>(smem_cov_ij
                           , tid);
        if (tid == 0) {
          atomicAdd(&cov[i*n_dim+j]
                  , smem_cov_ij[0]);
        }
      }
    }
  }


  //// kernel drivers, prepare kernels and
  //// enqueue them in GPU-queue for execution.

  void
  nq_neighbors(const std::vector<float>& xs
             , float rad2
             , GPUSettings& gpu) {
    unsigned int n_frames = gpu.n_frames;
    unsigned int n_dim = gpu.n_dim;
    // copy current position to GPU
    cudaMemcpy(gpu.xs
             , xs.data()
             , sizeof(float) * n_dim
             , cudaMemcpyHostToDevice);
    check_error("copy reference point coordinates to device");
    // shared mem for position [n_dim]
    // and reference coordinates [n_dim * BSIZE]
    unsigned int shared_mem_size = sizeof(float) * (1+BSIZE) * n_dim;
    // kernel call: find neighbors
    neighbors_krnl
    <<< n_blocks(n_frames
               , BSIZE)
      , BSIZE
      , shared_mem_size >>> (gpu.xs
                           , gpu.coords
                           , rad2
                           , gpu.has_future
                           , n_frames
                           , n_dim
                           , gpu.is_neighbor);
    check_error("kernel exec: neighbors_krnl");
    // reset neighbor count
    cudaMemsetAsync(gpu.n_neighbors
                  , 0
                  , sizeof(unsigned int));
    check_error("reset n_neighbors");
    // kernel call: count neighbors
    count_neighbors_krnl
    <<< n_blocks(n_frames
               , BSIZE)
      , BSIZE >>> (gpu.is_neighbor
                 , n_frames
                 , gpu.n_neighbors);
    check_error("kernel exec: count_neighbors_krnl"); 
  }

  void
  nq_v_means(GPUSettings& gpu) {
    // reset v_means
    cudaMemsetAsync(gpu.v_means
                  , 0
                  , sizeof(float) * 2 * gpu.n_dim);
    check_error("reset v_means");
    // kernel call: compute velocity means (both, forward and backward)
    v_means_krnl
    <<< n_blocks(gpu.n_frames
               , BSIZE)
      , BSIZE >>> (gpu.is_neighbor
                 , gpu.coords
                 , gpu.n_neighbors
                 , gpu.n_frames
                 , gpu.n_dim
                 , gpu.v_means);
    check_error("kernel exec: v_means_krnl");

    normalize_v_means_krnl
    <<< n_blocks(gpu.n_dim
               , 32)
      , 32 >>> (gpu.n_neighbors
              , gpu.n_dim
              , gpu.v_means);
    check_error("kernel exec: normalize_v_means_krnl");
  }

  void
  nq_cov(GPUSettings& gpu
       , bool i_forward
       , bool j_forward) {
    // reset cov-matrix
    cudaMemsetAsync(gpu.cov
                  , 0
                  , sizeof(float) * gpu.n_dim * gpu.n_dim);
    check_error("reset cov");

    if (i_forward
     && j_forward) {
      cov_krnl<true
             , true>
      <<< n_blocks(gpu.n_frames
                 , BSIZE)
        , BSIZE >>> (gpu.is_neighbor
                   , gpu.coords
                   , gpu.v_means
                   , gpu.n_frames
                   , gpu.n_dim
                   , gpu.cov);
    } else if (i_forward
            && (! j_forward)) {
      cov_krnl<true
             , false>
      <<< n_blocks(gpu.n_frames
                 , BSIZE)
        , BSIZE >>> (gpu.is_neighbor
                   , gpu.coords
                   , gpu.v_means
                   , gpu.n_frames
                   , gpu.n_dim
                   , gpu.cov);
    } else if ((! i_forward)
            && (! j_forward)) {
      cov_krnl<false
             , false>
      <<< n_blocks(gpu.n_frames
                 , BSIZE)
        , BSIZE >>> (gpu.is_neighbor
                   , gpu.coords
                   , gpu.v_means
                   , gpu.n_frames
                   , gpu.n_dim
                   , gpu.cov);
    }
    check_error("kernel call: cov_krnl");
  }



  //// retrieve data from GPU

  unsigned int
  get_n_neighbors(GPUSettings& gpu) {
    unsigned int n=0;
    cudaMemcpy(&n
             , gpu.n_neighbors
             , sizeof(unsigned int)
             , cudaMemcpyDeviceToHost);
    check_error("memcpy: n_neighbors from GPU");
    return n;
  }

  std::pair<std::vector<float>, std::vector<float>>
  get_v_means(GPUSettings& gpu
            , unsigned int n_neighbors) {
    std::vector<float> v_forward(gpu.n_dim);
    cudaMemcpy(v_forward.data()
             , &gpu.v_means[0]
             , sizeof(float) * gpu.n_dim
             , cudaMemcpyDeviceToHost);
    check_error("memcpy: v_means forward");
    std::vector<float> v_backward(gpu.n_dim);
    cudaMemcpy(v_forward.data()
             , &gpu.v_means[gpu.n_dim]
             , sizeof(float) * gpu.n_dim
             , cudaMemcpyDeviceToHost);
    check_error("memcpy: v_means backward");
    for (float& v: v_forward) {
      v /= (float) n_neighbors;
    }
    for (float& v: v_backward) {
      v /= (float) n_neighbors;
    }
    return {v_forward
          , v_backward};
  }

  std::vector<float>
  get_cov(GPUSettings& gpu
        , unsigned int n_neighbors) {
    std::vector<float> cov(gpu.n_dim*gpu.n_dim);
    // get raw cov-data from GPU
    cudaMemcpy(cov.data()
             , gpu.cov
             , sizeof(float) * gpu.n_dim * gpu.n_dim
             , cudaMemcpyDeviceToHost);
    check_error("memcpy: cov from GPU");
    for (unsigned int i=0; i < gpu.n_dim; ++i) {
      for (unsigned int j=0; j < gpu.n_dim; ++j) {
        cov[i*gpu.n_dim+j] /= (float) (n_neighbors-1);
      }
    }
    return cov;
  }

}} // end namespace Langevin::CUDA

