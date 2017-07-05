#pragma once

#include <functional>
#include <vector>

#include <Eigen/Dense>

namespace Langevin {

std::vector<float>
propagate(std::vector<float> position
        , std::vector<float> prev_position
        , Eigen::VectorXf drift
        , Eigen::MatrixXf diffusion
        , Eigen::MatrixXf friction
        , std::function<float()>& rnd);

void
write_stats_header(std::ostream& fh
                 , unsigned int n_dim
                 , std::string cmdline);

void
write_stats(std::ostream& fh
          , const Eigen::VectorXf& f
          , const Eigen::MatrixXf& gamma
          , const Eigen::MatrixXf& kappa
          , unsigned int n_neighbors
          , unsigned int retries);

} // end Langevin::
