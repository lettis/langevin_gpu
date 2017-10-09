#pragma once

#define BSIZE 128

#include <vector>
#include <utility>
#include <string>


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
    //! # found neighbors, stored on host
    unsigned int n_neighbors;
    //! # reference states
    unsigned int n_states;
    //! squared hypersphere radius for neighbor search
    float rad2;
    //// data fields (on GPU device)
    //// these are allocated during GPU setup ('prepare_gpu')
    //// and must be freed after usage ('clear_gpu').
    //! current position to evaluate [n_dim]
    float* xs;
    //! reference coordinates [n_frames x n_dim],
    //! row-major indices (i.e.  [i,j] ident. by [i*n_cols+j])
    float* coords;
    //! reference states [n_frames]
    unsigned int* states;
    //! state count for given neighborhood (2 x [n_states], original and
    //! timeshifted)
    unsigned int* state_count;
    unsigned int* state_count_timeshift;
    //! result of neighbor search (2 x [n_frames], original and timeshifted)
    char* is_neighbor;
    char* is_neighbor_timeshift;
    //! number of found neighbors [2] (original and timeshifted)
    unsigned int* n_neighbors_dev;
    //! id to distinguish different, concatenated trajectories
    unsigned int* traj_id;
    //! means of velocity estimates (forward and backward)  [2*n_dim]
    //! (necessary for cov-mat computation and drift estimation)
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
            , float rad2
            , const std::vector<std::vector<float>>& ref_coords
            , const std::vector<char>& has_future
            , const std::vector<unsigned int>& states);

  void
  clear_gpu(GPUSettings& settings);

  //! return minimum number of blocks needed for covering all frames
  unsigned int
  n_blocks(unsigned int n_frames
         , unsigned int block_size);


  //// kernel drivers ////////////

  //! run CUDA kernels to identify neighbor frames
  void
  nq_neighbors(const std::vector<float>& xs
             , GPUSettings& gpu);

  //! shift neighbor frames in time to find followers of neighbors,
  //! i.e. retrieve dynamical information from neighborhood like
  //! transition probabilities
  void
  nq_neighbors_timeshift(unsigned int tau
                       , GPUSettings& gpu);

  //! count number of frames per state in
  //! (current or timeshifted) neighborhood
  void
  nq_count_states(GPUSettings& gpu
                , bool timeshifted);


  void
  nq_v_means(GPUSettings& gpu);

  void
  nq_cov(GPUSettings& gpu
       , bool i_forward
       , bool j_forward);


  //// retrieve results from GPU //////////
  
  unsigned int
  get_n_neighbors(GPUSettings& gpu);

  std::pair<std::vector<float>, std::vector<float>>
  get_v_means(GPUSettings& gpu);

  std::vector<float>
  get_cov(GPUSettings& gpu);

  std::vector<float>
  get_state_probs(GPUSettings& gpu
                , bool timeshifted);


}} // end namespace Langevin::CUDA

