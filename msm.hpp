#pragma once

#include <string>
#include <functional>
#include <vector>

#include <Eigen/Dense>

#include "tools.hpp"


namespace MSM {

  struct Model {
    unsigned int n_states;
    Eigen::MatrixXf tmat;
    unsigned int tau;
  };

  Model
  load_msm(std::string fname
         , unsigned int tau
         , float rnd_seed = 0.0);

  unsigned int
  propagate(Model msm
          , unsigned int state
          , Tools::Dice& rnd);

} // end MSM::

