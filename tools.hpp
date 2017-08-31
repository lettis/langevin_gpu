#pragma once

#include <string>
#include <vector>
#include <random>

#include <Eigen/Dense>

namespace Tools {

  struct Dice {
    std::function<float()> normal;
    std::function<float()> uniform;
  };
  
  
  Dice
  initialize_dice(float seed);
  
  unsigned int
  rnd_state(Eigen::VectorXf weights
          , Dice& rnd);
  
  std::vector<char>
  read_futures(std::string fname);
  
  std::vector<unsigned int>
  read_states(std::string fname);
  
  //// misc
  
  template <typename NUM>
  bool
  is_integer_value(NUM x);
  
  template <typename NUM>
  std::vector<double>
  sum1_normalized(const std::vector<NUM>& pops);
  
  std::vector<std::pair<float, float>>
  col_min_max(const std::vector<std::vector<float>>& coords);
  
  std::string
  join_args(int argc
          , char** argv);
  
  Eigen::MatrixXf
  to_eigen_mat(std::vector<float> xs
             , bool make_symmetric=false);
  
  Eigen::VectorXf
  to_eigen_vec(std::vector<float> xs);
  
  std::vector<float>
  to_stl_vec(Eigen::VectorXf xs);
  
} // end Tools::

//// template implementations

#include "tools.hxx"

