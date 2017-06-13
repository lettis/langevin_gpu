#pragma once

#define BSIZE 128

#include <unordered_set>
#include <unordered_map>
#include <vector>


namespace CUDA {

  struct GPUSettings {
    int id;
    std::unordered_set<unsigned int> state_names;
    unsigned int n_dim;
    unsigned int n_frames;
    float* xs;
    float* coords;
    char* is_neighbor;
  };

  void
  check_error(std::string msg="");

  int
  get_num_gpus();

  GPUSettings
  prepare_gpu(int i_gpu
            , unsigned int n_dim
            , const std::vector<std::vector<float>>& ref_coords);

  void
  clear_gpu(GPUSettings settings);

  //! return minimum multiplicator to fulfill result * mult >= orig
  unsigned int
  min_multiplicator(unsigned int orig
                  , unsigned int mult);



  //TODO: doc

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
  std::vector<char>
  neighbors(const std::vector<float>& xs
          , float rad2
          , float dx
          , const std::vector<GPUSettings>& gpus);

} // end namespace CUDA

