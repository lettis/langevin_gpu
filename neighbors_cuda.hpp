#pragma once

#define BSIZE 128

#include <unordered_set>
#include <unordered_map>
#include <vector>


namespace CUDA {

  struct GPUSettings {
    //// general settings / constants
    //! GPU id
    int id;
    //! # dimensions, i.e. columns
    unsigned int n_dim;
    //! # frames, i.e. rows
    unsigned int n_frames;
    //// data fields (on GPU device)
    //// these are allocated during GPU setup ('prepare_gpu')
    //// and must be freed after usage ('clear_gpu').
    //! current position to evaluate [n_dim]
    float* xs;
    //! reference coordinates [n_frames x n_dim],
    //! row-major indices (i.e.  [i,j] ident. by [i*n_cols+j])
    float* coords;
    //! reference free energies [n_frames]
    float* fe;
    //! result of neighbor search [n_frames x (2*n_cols +1)],
    //! i.e. for unshifted and shifted reference point
    char* is_neighbor;
    //! number of found neighbors (per shift)   [2*n_cols+1]
    unsigned int* n_neighbors;
    //! history of reference coordinates
    //! 0: has no future sample, i.e. end of a trajecotry chunk
    //! 1: has a future sample
    char* has_future;
    //! free energy estimates for the different shifts [2*n_dim]
    float* shifts_fe;
    //! drift == grad(fe) [n_dim]
    float* drift;
    //! means of velocity estimates (forward and backward)  [2*n_dim]
    //! (necessary for cov-mat computation)
    float* v_means;
    //! covariance matrix [n_dim x n_dim]
    float* cov;
  };

  //! Check for CUDA-related errors.
  //! Always refers to the last CUDA-API action,
  //! thus call it after(!) the related API-function.
  //! @param msg Give a helpful message to identify, where in the code
  //             the error occurred.
  void
  check_error(std::string msg="");

  int
  get_num_gpus();

  GPUSettings
  prepare_gpu(int i_gpu
            , unsigned int n_dim
            , const std::vector<std::vector<float>>& ref_coords
            , const std::vector<char>& has_future
            , const std::vector<float>& fe);

  void
  clear_gpu(GPUSettings& settings);

  //! return minimum multiplicator to fulfill: result * mult >= orig
  unsigned int
  min_multiplicator(unsigned int orig
                  , unsigned int mult);


  template <typename NUM>
  __device__ void
  atomicAddReduce(NUM* result
                , NUM* value);


  //// kernels ///////////

  __global__ void
  neighbors_krnl(float* xs
               , float* ref_coords
               , float rad2
               , float dx
               , char* has_future
               , unsigned int n_rows
               , unsigned int n_cols
               , char* is_neighbor);

  __global__ void
  count_neighbors_krnl(char* is_neighbor
                     , unsigned int n_frames
                     , unsigned int n_dim
                     , unsigned int* n_neighbors);

  __global__ void
  shifted_fe_sum_krnl(char* is_neighbor
                    , float* fe
                    , unsigned int n_frames
                    , unsigned int n_dim
                    , float* shifts_fe);

  __global__ void
  v_means_krnl(char* is_neighbor
             , float* coords
             , unsigned int* n_neighbors
             , unsigned int n_frames
             , unsigned int n_dim
             , float* means);

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
         , float* cov);


  //// kernel drivers ////////////

  // shift pattern:
  //
  //    0   0   0   0 ...
  //  +dx   0   0   0
  //  -dx   0   0   0
  //    0 +dx   0   0
  //    0 -dx   0   0
  //    0   0 +dx   0
  //    0   0 -dx   0
  //    0   0   0 +dx
  //    0   0   0 -dx
  //    .
  //    .
  //    .
  void
  nq_neighbors(const std::vector<float>& xs
             , float rad2
             , float dx
             , GPUSettings& gpu);

  void
  nq_shifted_fe_sum(GPUSettings& gpu);

  void
  nq_v_means(GPUSettings& gpu);

  void
  nq_cov(unsigned int i
       , unsigned int j
       , bool i_use_forward_velocity
       , bool j_use_forward_velocity
       , GPUSettings& gpu);

} // end namespace CUDA


// template implementations
#include "neighbors_cuda.hxx"

