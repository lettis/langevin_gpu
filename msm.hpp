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
    std::vector<std::function<unsigned int()>> propagator;
  };

  Model
  load_msm(std::string fname
         , float rnd_seed = 0.0);

  unsigned int
  propagate(Model msm
          , unsigned int state);

} // end MSM::

