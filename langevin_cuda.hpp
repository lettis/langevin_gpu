#pragma once

#define BSIZE 128

#include <unordered_set>
#include <unordered_map>
#include <vector>


namespace Langevin {
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

  //! Force thread synchronization and check for error afterwards.
  void
  sync_and_check(std::string msg);

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

  //! return minimum number of blocks needed for covering all frames
  unsigned int
  n_blocks(unsigned int n_frames
         , unsigned int block_size);


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
  nq_cov(GPUSettings& gpu
       , bool i_forward
       , bool j_forward);


  //// retrieve results from GPU //////////
  
  std::vector<unsigned int>
  get_n_neighbors(GPUSettings& gpu);

  std::vector<float>
  get_drift(GPUSettings& gpu
          , std::vector<unsigned int> n_neighbors
          , float dx);

  std::vector<float>
  get_cov(GPUSettings& gpu
        , std::vector<unsigned int> n_neighbors);

}} // end namespace Langevin::CUDA

