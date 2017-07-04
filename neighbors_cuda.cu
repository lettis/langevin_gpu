
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
    //// allocate memory for shifted fe estimates
    cudaMalloc((void**) &gpu.shifts_fe
             , sizeof(float) * 2*gpu.n_dim);
    check_error("malloc shifts_fe");
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
    cudaFree(gpu.shifts_fe);
    check_error("free shifts_fe");
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
               , float dx
               , char* has_future
               , unsigned int n_frames
               , unsigned int n_dim
               , char* is_neighbor) {
    // CUDA-specific indices for block, thread and global
    unsigned int bsize = blockDim.x;
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int gid = bid * bsize + tid;
    // locally shared memory for fast access
    // of dx-shifted xs and reference coordinates
    float* s_coords = (float*) &smem[0];
    float* s_xs = (float*) &smem[n_dim*BSIZE];
    // read xs to shared mem
    if (tid < n_dim) {
      float x = xs[tid];
      // unshifted xs
      s_xs[tid] = x;
      // shifted xs for numerical gradient estimation
      for (unsigned int j_shift=0; j_shift < n_dim; ++j_shift) {
        if (j_shift != tid) {
          s_xs[(2*j_shift+1)*n_dim+tid] = x;
          s_xs[(2*j_shift+2)*n_dim+tid] = x;
        } else {
          s_xs[(2*j_shift+1)*n_dim+tid] = x + dx;
          s_xs[(2*j_shift+2)*n_dim+tid] = x - dx;
        }
      }
    }
    __syncthreads();
    if (gid < n_frames) {
      unsigned int n_shifts = 2*n_dim + 1;
      // without history (no future, no past), don't count as neighbor!
      if ((gid == 0)
       || (gid == n_frames-1)
       || (has_future[gid] == 0)
       || (has_future[gid-1] == 0)) {
        for (unsigned int j_shift=0; j_shift < n_shifts; ++j_shift) {
          is_neighbor[j_shift*n_frames + gid] = 0;
        }
      } else {
        // read ref coords to shared mem
        for (unsigned int j=0; j < n_dim; ++j) {
          s_coords[tid*n_dim+j] = ref_coords[gid*n_dim+j];
        }
        // check if is neighbor for different position shifts
        for (unsigned int j_shift=0; j_shift < n_shifts; ++j_shift) {
          float d2 = 0.0f;
          for (unsigned int k=0; k < n_dim; ++k) {
            float d = s_coords[tid*n_dim+k] - s_xs[j_shift*n_dim+k];
            d2 += d*d;
          }
          if (d2 <= rad2) {
            is_neighbor[j_shift*n_frames + gid] = 1;
          } else {
            is_neighbor[j_shift*n_frames + gid] = 0;
          }
        }
      }
    }
  }

  __global__ void
  count_neighbors_krnl(char* is_neighbor
                     , unsigned int n_frames
                     , unsigned int n_shifts
                     , unsigned int* n_neighbors) {
    __shared__ unsigned int smem_uint[BSIZE];
    // CUDA-specific indices for block, thread and global
    unsigned int bsize = blockDim.x;
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int gid = bid * bsize + tid;
    for (unsigned int j_shift=0; j_shift < n_shifts; ++j_shift) {
      // count found neighbors for every reference position shift
      bool this_is_neighbor = (is_neighbor[j_shift*n_frames+gid] == 1);
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
        atomicAdd(&n_neighbors[j_shift]
                , smem_uint[0]);
      }
    }
  }

  __global__ void
  shifted_fe_sum_krnl(char* is_neighbor
                    , float* fe
                    , unsigned int n_frames
                    , unsigned int n_shifts
                    , float* shifts_fe) {
    __shared__ float smem_float[BSIZE];
    // CUDA-specific indices for block, thread and global
    unsigned int bsize = blockDim.x;
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int gid = bid * bsize + tid;
    // sum free energies for shifted reference points
    // (omit unshifted reference, sum(fe) not needed there)
    for (unsigned int j_shift=1; j_shift < n_shifts; ++j_shift) {
//TODO twice shared mem, pos/neg in one go (j < n_dim)
      bool this_is_neighbor = (is_neighbor[j_shift*n_frames+gid] == 1);
      if (gid < n_frames-1
       && this_is_neighbor) {
        smem_float[tid] = fe[gid];
      } else {
        smem_float[tid] = 0.0f;
      }
      __syncthreads();
      // reduce in shared memory
      pairwiseMemReduce<float, BSIZE>(smem_float
                                    , tid);
      // further reduce locally inside warp and aggregate
      // results from all warps in global memory
      warpReduceMem<float>(smem_float
                         , tid);
      if (tid == 0) {
        atomicAdd(&shifts_fe[j_shift-1]
                , smem_float[0]);
      }
    }
  }

  __global__ void
  v_means_krnl(char* is_neighbor
             , float* coords
             , unsigned int* n_neighbors
             , unsigned int n_frames
             , unsigned int n_dim
             , unsigned int j
             , float* v_means) {
    __shared__ float v_forward[BSIZE];
    __shared__ float v_backward[BSIZE];
    // CUDA-specific indices for block, thread and global
    unsigned int bsize = blockDim.x;
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int gid = bid * bsize + tid;
    bool this_is_neighbor = (is_neighbor[gid] == 1);
    // aggregate forward and backward velocities for every dimension
    if ((0 < gid)
     && (gid < n_frames-1)
     && this_is_neighbor) {
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
    // average velocities ...
    if (gid == 0) {
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
             , float dx
             , GPUSettings& gpu) {
    unsigned int n_rows = gpu.n_frames;
    unsigned int n_cols = gpu.n_dim;
    // copy current position to GPU
    cudaMemcpy(gpu.xs
             , xs.data()
             , sizeof(float) * n_cols
             , cudaMemcpyHostToDevice);
    check_error("copy reference point coordinates to device");
    // shared mem for dx-shifted positions [n_cols * (2*n_cols + 1)]
    // and reference coordinates [n_cols * BSIZE]
    unsigned int shared_mem_size = sizeof(float)
                                 * n_cols
                                 * ((2*n_cols+1) + BSIZE);
    // kernel call: find neighbors
    neighbors_krnl
    <<< n_blocks(n_rows
               , BSIZE)
      , BSIZE
      , shared_mem_size >>> (gpu.xs
                           , gpu.coords
                           , rad2
                           , dx
                           , gpu.has_future
                           , n_rows
                           , n_cols
                           , gpu.is_neighbor);
    check_error("kernel exec: neighbors_krnl");
    // reset neighbor count
    unsigned int n_shifts = 2*gpu.n_dim + 1;
    cudaMemsetAsync(gpu.n_neighbors
                  , 0
                  , sizeof(unsigned int) * n_shifts);
    check_error("reset n_neighbors");
    // kernel call: count neighbors
    count_neighbors_krnl
    <<< n_blocks(n_rows
               , BSIZE)
      , BSIZE >>> (gpu.is_neighbor
                 , n_rows
                 , n_shifts
                 , gpu.n_neighbors);
    check_error("kernel exec: count_neighbors_krnl"); 
  }

  void
  nq_shifted_fe_sum(GPUSettings& gpu) {
    // reset fe sums
    unsigned int n_shifts = 2*gpu.n_dim + 1;
    cudaMemset(gpu.shifts_fe
             , 0
             , sizeof(float) * (n_shifts-1));
    check_error("reset shifts_fe");
    // kernel call: compute sum of free energies for every reference shift
    shifted_fe_sum_krnl
    <<< n_blocks(gpu.n_frames
               , BSIZE)
      , BSIZE >>> (gpu.is_neighbor
                 , gpu.fe
                 , gpu.n_frames
                 , n_shifts
                 , gpu.shifts_fe);
      check_error("kernel exec: shifted_fe_sum_krnl");
  }

  void
  nq_v_means(GPUSettings& gpu) {
    // reset v_means
    cudaMemsetAsync(gpu.v_means
                  , 0
                  , sizeof(float) * 2 * gpu.n_dim);
    check_error("reset v_means");
    // kernel call: compute velocity means (both, forward and backward)
    for (unsigned int j=0; j < gpu.n_dim; ++j) {
      v_means_krnl
      <<< n_blocks(gpu.n_frames
                 , BSIZE)
        , BSIZE >>> (gpu.is_neighbor
                   , gpu.coords
                   , gpu.n_neighbors
                   , gpu.n_frames
                   , gpu.n_dim
                   , j
                   , gpu.v_means);
      check_error("kernel exec: v_means_krnl");
    }
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

  std::vector<unsigned int>
  get_n_neighbors(GPUSettings& gpu) {
    std::vector<unsigned int> n(2*gpu.n_dim + 1);
    cudaMemcpy(n.data()
             , gpu.n_neighbors
             , sizeof(unsigned int) * (2*gpu.n_dim+1)
             , cudaMemcpyDeviceToHost);
    check_error("memcpy: n_neighbors from GPU");
    return n;
  }

  std::vector<float>
  get_drift(GPUSettings& gpu
          , std::vector<unsigned int> n_neighbors
          , float dx) {
    std::vector<float> fe(2*gpu.n_dim);
    cudaMemcpy(fe.data()
             , gpu.shifts_fe
             , sizeof(float) * (2*gpu.n_dim)
             , cudaMemcpyDeviceToHost);
    check_error("memcpy: shifts_fe from GPU");
    // compute drift
    std::vector<float> drift(gpu.n_dim);
    for (unsigned int i=0; i < 2*gpu.n_dim; ++i) {
      // correctly rescale shifted free energies
      fe[i] /= (float) n_neighbors[i+1];
    }
    for (unsigned int i=0; i < gpu.n_dim; ++i) {
      drift[i] = -1.0 * (fe[2*i] - fe[2*i+1]) / 2.0 / dx;
    }
    return drift;
  }

  std::vector<float>
  get_cov(GPUSettings& gpu
        , std::vector<unsigned int> n_neighbors) {
    std::vector<float> cov(gpu.n_dim*gpu.n_dim);
    // get raw cov-data from GPU
    cudaMemcpy(cov.data()
             , gpu.cov
             , sizeof(float) * gpu.n_dim * gpu.n_dim
             , cudaMemcpyDeviceToHost);
    check_error("memcpy: cov from GPU");
    for (unsigned int i=0; i < gpu.n_dim; ++i) {
      for (unsigned int j=0; j < gpu.n_dim; ++j) {
        cov[i*gpu.n_dim+j] /= (float) (n_neighbors[0]-1);
      }
    }
    return cov;
  }

} // end namespace CUDA

