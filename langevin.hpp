#pragma once

#include <functional>
#include <vector>

#include <Eigen/Dense>

#include "langevin_cuda.hpp"
#include "tools.hpp"

namespace Langevin {

  struct Fields {
    Eigen::VectorXf drift;
    Eigen::MatrixXf friction;
    Eigen::MatrixXf diffusion;
  };
  
  struct Frame {
    Eigen::VectorXf pos;
    Eigen::VectorXf pos_prev;
    Fields fields;
  };
  
  struct NextPos {
    Eigen::VectorXf position;
    unsigned int retries;
  };
  

  unsigned int
  update_neighbors(Eigen::VectorXf pos
                 , Langevin::CUDA::GPUSettings& gpu);
  
  // probabilities to go to a certain state
  Eigen::VectorXf
  state_transition_probabilities(Eigen::VectorXf pos
                               , unsigned int tau
                               , Langevin::CUDA::GPUSettings& gpu);

  // probabilities to be in a certain state
  Eigen::VectorXf
  state_probabilities(Eigen::VectorXf pos
                    , Langevin::CUDA::GPUSettings& gpu);

  Fields
  estimate_fields(Langevin::CUDA::GPUSettings& gpu);
  
  
  Eigen::VectorXf
  euler_integration(Frame frame
                  , Tools::Dice& rnd);
  
  Eigen::VectorXf
  propagate(Langevin::Frame frame
          , Tools::Dice& rnd
          , unsigned int min_pop
          , unsigned int& max_retries
          , Langevin::CUDA::GPUSettings& gpu);
  
  void
  write_stats_header(std::ostream& fh
                   , unsigned int n_dim
                   , std::string cmdline);
  
  void
  write_stats(std::ostream& fh
            , const Fields& dle
            , unsigned int n_neighbors
            , unsigned int retries
            , unsigned int state
            , unsigned int i_traj);

} // end Langevin::

