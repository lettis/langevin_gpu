#pragma once

#include <Eigen/Dense>

#include "langevin.hpp"
#include "msm.hpp"
#include "tools.hpp"

namespace Hybrid {

  struct Frame {
    Langevin::Frame dle;
    unsigned int state;
    unsigned int i_traj;
  };


  unsigned int
  rnd_index(std::vector<unsigned int> states
          , unsigned int state
          , Tools::Dice& rnd);


  Frame
  propagate_discrete_uncoupled(MSM::Model msm
                             , std::vector<unsigned int> ref_states
                             , std::vector<std::vector<float>> ref_coords
                             , const Frame& frame
                             , Tools::Dice& rnd
                             , unsigned int min_pop
                             , unsigned int& max_retries
                             , Langevin::CUDA::GPUSettings& gpu);

/*
  Frame
  propagate_discrete_coupled(Langevin::CUDA::GPUSettings& gpu
                           , const Eigen::MatrixXf msm
                           , const Frame& frame
                           , float c
                           , Tools::Dice& rnd);
*/


/*
  Frame
  propagate_continuous(Langevin::CUDA::GPUSettings& gpu
                     , const Eigen::MatrixXf msm
                     , const Frame& frame
                     , float c
                     , Tools::Dice& rnd);
*/

} // end namespace Hybrid::

