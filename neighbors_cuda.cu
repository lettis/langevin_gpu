
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
             , size(float) * gpu.n_frames);
    check_error("copy fe");
    //// allocate memory for neighborhood
    cudaMalloc((void**) &gpu.is_neighbor
             , sizeof(char) * gpu.n_frames * (2*n_dim+1));
    check_error("malloc partial neighbor count");
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
    //// allocate memory for drift
    cudaMalloc((void**) &gpu.drift
             , sizeof(float) * gpu.n_dim);
    check_error("malloc drift");
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
    cudaSetDevice(gpu.id);
    check_error("setting CUDA device");
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
    cudaFree(gpu.drift);
    check_error("free drift");
    cudaFree(gpu.v_means);
    check_error("free v_means");
    cudaFree(gpu.cov);
    check_error("free cov");
  }

  unsigned int
  min_multiplicator(unsigned int orig
                  , unsigned int mult) {
    return (unsigned int) std::ceil(orig / ((float) mult));
  };


  //// kernel functions running on GPU

  __global__ void
  neighbors_krnl(float* xs
               , float* ref_coords
               , float rad2
               , float dx
               , char* has_future
               , unsigned int n_rows
               , unsigned int n_cols
               , char* is_neighbor) {
   //TODO: col-based indices for is_neighbor
   
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
      // without history (no future, no past), don't count as neighbor!
      if ((gid == 0)
       || (gid == n_rows-1)
       || (has_future[gid] == 0)
       || (has_future[gid-1] == 0)) {
        for (unsigned int j_shift=0; j_shift < n_shifts; ++j_shift) {
          is_neighbor[gid*n_shifts + j_shift] = 0;
        }
      } else {
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
  }

  __global__ void
  count_neighbors_krnl(char* is_neighbor
                     , unsigned int n_frames
                     , unsigned int n_dim
                     , unsigned int* n_neighbors) {
    // CUDA-specific indices for block, thread and global
    unsigned int bsize = blockDim.x;
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int gid = bid * bsize + tid;
    unsigned int n_shifts = 2*n_dim + 1;
    // count found neighbors for every reference position shift
    for (unsigned int j_shift=0; j_shift < n_shifts; ++j_shift) {
      unsigned int n_neighbors_sum;
      if (gid < n_frames) {
        n_neighbors_sum = is_neighbor[gid*n_shifts + j_shift];
      } else {
        n_neighbors_sum = 0;
      }
      // reduce(add) locally inside warp and aggregate
      // results from all warps in global memory
      atomicAddReduce<unsigned int>(&n_neighbors_sum[j_shift]
                                  , &n_neighbors_sum);
    }
  }

  __global__ void
  shifted_fe_sum_krnl(char* is_neighbor
                    , float* fe
                    , unsigned int n_frames
                    , unsigned int n_dim
                    , float* shifts_fe) {
    // CUDA-specific indices for block, thread and global
    unsigned int bsize = blockDim.x;
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int gid = bid * bsize + tid;
    unsigned int n_shifts = 2*n_dim + 1;
    // sum free energies for shifted reference points
    // (omit unshifted reference, sum(fe) not needed there)
    for (unsigned int j_shift=1; j_shift < n_shifts; ++j_shift) {
      float fe_sum;
      if ((gid < n_frames)
       && (is_neighbor[gid*n_shifts+j_shift] == 1)) {
        fe_sum = fe[gid];
      } else {
        fe_sum = 0.0f;
      }
      // reduce(add) locally inside warp and aggregate
      // results from all warps in global memory
      atomicAddReduce<float>(&shifts_fe[j_shift-1]
                           , &fe_sum);
    }
  }

  __global__ void
  v_means_krnl(char* is_neighbor
             , float* coords
             , unsigned int* n_neighbors
             , unsigned int n_frames
             , unsigned int n_dim
             , float* means) {
    // CUDA-specific indices for block, thread and global
    unsigned int bsize = blockDim.x;
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int gid = bid * bsize + tid;
    // aggregate forward and backward velocities for every dimension
    for (unsigned int j=0; j < n_dim; ++j) {
      float v_forward;
      float v_backward;
      if ((gid < n_frames-1)
       && (is_neighbor[gid*n_shifts] == 1)) {
        float x = coords[gid*n_dim+j];
        float x_forward = coords[(gid+1)*gid*n_dim+j];
        float x_backward = coords[(gid-1)*n_dim+j];
        v_forward = x_forward - x;
        v_backward = x - x_backward;
      } else {
        v_forward = 0.0f;
        v_backward = 0.0f;
      }
      atomicAddReduce<float>(&v_means[j]
                           , &v_forward);
      atomicAddReduce<float>(&v_means[n_dim+j]
                           , &v_backward);
    }
    __syncthreads();
    // average velocities ...
    if (gid == 0) {
      for (unsigned int j=0; j < n_dim; ++j) {
        v_means[j] /= n_neighbors[0];
        v_means[n_dim+j] /= n_neighbors[0];
      }
    }
  }

  __global__ void
  cov_krnl(char* is_neighbor
         , float* coords
         , float* v_means
         , unsigned int n_frames
         , unsigned int n_dim
         , unsigned int i 
         , unsigned int j
         , bool i_use_forward_velocity
         , bool j_use_forward_velocity
         , float* cov) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int n_shifts = 2*n_dim+1;
    // compute local contributions to cov[i,j],
    // depending on forward or backward velocities.
    float cov_local;
    if ((gid < n_frames)
     && (is_neighbor[gid_n*n_shifts] == 1) {
      float v_i;
      float v_j;
      if (i_use_forward_velocity) {
        v_i = coords[(gid+1)*n_dim+i]
            - coords[gid*n_dim+i]
            - v_means[i];
      } else {
        v_i = coords[gid*n_dim+i]
            - coords[(gid-1)*n_dim+i]
            - v_means[i+n_dim];
      }
      if (j_use_forward_velocity) {
        v_j = coords[(gid+1)*n_dim+j]
            - coords[gid*n_dim+j]
            - v_means[j];
      } else {
        v_j = coords[gid*n_dim+j]
            - coords[(gid-1)*n_dim+j]
            - v_means[j+n_dim];
      }
      cov_local = v_i * v_j;
    } else {
      cov_local = 0.0f;
    }
    // aggregate local contributions to covariance
    atomicAddReduce(&cov[i*n_dim+j]
                  , &cov_local)
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
    block_rng = min_multiplicator(n_rows, BSIZE);
    // shared mem for dx-shifted positions [n_cols * (2*n_cols + 1)]
    // and reference coordinates [n_cols * BSIZE]
    shared_mem_size = sizeof(float) * n_cols * ((2*n_cols+1) + BSIZE);

    // kernel call: find neighbors
    neighbors_krnl
    <<< block_rng
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

    // kernel call: count neighbors
    count_neighbors_krnl
    <<< block_rng
      , BSIZE >>> (gpu.is_neighbor
                 , n_rows
                 , n_cols
                 , gpu.n_neighbors);
    check_error("kernel exec: count_neighbors_krnl"); 
  }

  void
  nq_shifted_fe_sum(GPUSettings& gpu) {
    //TODO
  }

  void
  nq_v_means(GPUSettings& gpu) {
    // TODO
  }

  void
  nq_cov(unsigned int i
       , unsigned int j
       , bool i_use_forward_velocity
       , bool j_use_forward_velocity
       , GPUSettings& gpu) {
    //TODO
  }


  //// TODO: retriever functions

  // get_n_neighbors
  // get_drift
  // get_cov
  // ...



  //TODO: don't forget to normalize data after certain steps
  //      (i.e. when retrieving the data)!

} // end namespace CUDA

