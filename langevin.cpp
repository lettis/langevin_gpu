
#include <iostream>
#include <algorithm>

#include "langevin.hpp"

namespace Langevin {

std::vector<float>
propagate(std::vector<float> position
        , std::vector<float> prev_position
        , Eigen::VectorXf drift
        , Eigen::MatrixXf diffusion
        , Eigen::MatrixXf friction
        , std::function<float()>& rnd) {
  unsigned int n_dim = position.size();
  Eigen::VectorXf pos_new(n_dim);
  Eigen::VectorXf pos(n_dim);
  Eigen::VectorXf pos_prev(n_dim);
  Eigen::VectorXf noise(n_dim);
  // initialize vectors
  for (unsigned int i=0; i < n_dim; ++i) {
    pos(i) = position[i];
    pos_prev(i) = prev_position[i];
    noise(i) = rnd();
  }
  // compute new position from dLE
  pos_new = pos + drift - friction*(pos-pos_prev) + diffusion*noise;
  std::vector<float> new_position(n_dim);
  for (unsigned int i=0; i < n_dim; ++i) {
    new_position[i] = pos_new(i);
  }
  return new_position;
}

void
write_stats_header(std::ostream& fh
                 , unsigned int n_dim
                 , std::string cmdline) { 
  fh << "# " << cmdline << std::endl;
  fh << "#";
  for (unsigned int i=0; i < n_dim; ++i) {
    fh << " f_" << i+1;
  }
  for (unsigned int i=0; i < n_dim; ++i) {
    for (unsigned int j=0; j < n_dim; ++j) {
      fh << " g_" << i+1 << "_" << j+1;
    }
  }
  for (unsigned int i=0; i < n_dim; ++i) {
    for (unsigned int j=0; j <= i; ++j) {
      fh << " K_" << i+1 << "_" << j+1;
    }
  }
  fh << " pop n_retries" << std::endl;
}

void
write_stats(std::ostream& fh
          , const Eigen::VectorXf& f
          , const Eigen::MatrixXf& gamma
          , const Eigen::MatrixXf& kappa
          , unsigned int n_neighbors
          , unsigned int retries) {
  unsigned int n_dim = f.size();
  // write drift
  for (unsigned int i=0; i < n_dim; ++i) {
    fh << " " << f(i);
  }
  // write friction
  for (unsigned int i=0; i < n_dim; ++i) {
    for (unsigned int j=0; j < n_dim; ++j) {
      fh << " " << gamma(i,j);
    }
  }
  // write diffusion (lower triangle of matrix)
  for (unsigned int i=0; i < n_dim; ++i) {
    for (unsigned int j=0; j <= i; ++j) {
      fh << " " << kappa(i,j);
    }
  }
  // write neighbor populations
  fh << " " << n_neighbors;
  // propagation retries
  fh << " " << retries << std::endl;
}

} // end Langevin::

