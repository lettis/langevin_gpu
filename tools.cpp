
#include <iostream>
#include <fstream>
#include <limits>
#include <utility>

#include "tools.hpp"


namespace Tools {

  Dice
  initialize_dice(float seed) {
    Dice rnd;
    rnd.normal = std::bind(std::normal_distribution<double>(0.0, 1.0)
                         , std::mt19937(seed));
    rnd.uniform = std::bind(std::uniform_real_distribution<double>(0.0, 1.0)
                          , std::mt19937(seed));
    return rnd;
  }

  unsigned int
  rnd_state(Eigen::VectorXf weights
          , Dice& rnd) {
    // throw dice to get some random p in [0,1]
    float p = rnd.uniform();
    float p_sum = 0;
    unsigned int i = 0;
    // sum probabilities to select next state (in [1,N])
    while (p_sum <= p
        && i < weights.size()) {
      p_sum += weights(i);
      ++i;
    }
    return i;
  }
  
  std::vector<char>
  read_futures(std::string fname) {
    return read_datafile<char>(fname);
  }
  
  std::vector<unsigned int>
  read_states(std::string fname) {
    return read_datafile<unsigned int>(fname);
  }
  
  
  std::vector<std::pair<float, float>>
  col_min_max(const std::vector<std::vector<float>>& coords) {
    unsigned int nrow = coords.size();
    if (nrow == 0) {
      std::cerr << "error: no sampling. cannot compute min/max of coordinates."
                << std::endl;
      exit(EXIT_FAILURE);
    }
    unsigned int ncol = coords[0].size();
    std::vector<std::pair<float, float>>
      mm(ncol
       , {std::numeric_limits<float>::infinity()
       , -std::numeric_limits<float>::infinity()});
    for (unsigned int i=0; i < nrow; ++i) {
      for (unsigned int j=0; j < ncol; ++j) {
        float ref_val = coords[i][j];
        if (ref_val < mm[j].first) {
          mm[j].first = ref_val;
        }
        if (ref_val > mm[j].second) {
          mm[j].second = ref_val;
        }
      }
    }
    return mm;
  }
  
  
  std::string
  join_args(int argc
          , char** argv) {
    std::string s = "";
    for (int i=0; i < argc; ++i) {
      s += " " + std::string(argv[i]);
    }
    return s;
  }
  
  
  
  Eigen::MatrixXf
  to_eigen_mat(std::vector<float> xs
             , bool make_symmetric) {
    unsigned int n = std::sqrt(xs.size());
    Eigen::MatrixXf mat(n, n);
    for (unsigned int i=0; i < n; ++i) {
      if (make_symmetric) {
        for (unsigned int j=0; j <= i; ++j) {
          mat(i,j) = xs[i*n+j];
          mat(j,i) = mat(i,j);
        }
      } else {
        for (unsigned int j=0; j < n; ++j) {
          mat(i,j) = xs[i*n+j];
        }
      }
    }
    return mat;
  }
  
  Eigen::VectorXf
  to_eigen_vec(std::vector<float> xs) {
    unsigned int n = xs.size();
    Eigen::VectorXf vec(n);
    for (unsigned int i=0; i < n; ++i) {
      vec(i) = xs[i];
    }
    return vec;
  }
  
  std::vector<float>
  to_stl_vec(Eigen::VectorXf xs) {
    unsigned int n = xs.size();
    std::vector<float> vec(n);
    for (unsigned int i=0; i < n; ++i) {
      vec[i] = xs(i);
    }
    return vec;
  }

} // end Tools::

