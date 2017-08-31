
#include <iostream>
#include <algorithm>

#include "langevin.hpp"
#include "tools.hpp"

namespace Langevin {

  unsigned int
  update_neighbors(Eigen::VectorXf pos
                 , Langevin::CUDA::GPUSettings& gpu) {
    Langevin::CUDA::nq_neighbors(Tools::to_stl_vec(pos)
                               , gpu);
    return Langevin::CUDA::get_n_neighbors(gpu);
  }


  Eigen::VectorXf
  state_transition_probabilities(Eigen::VectorXf pos
                               , unsigned int tau
                               , Langevin::CUDA::GPUSettings& gpu) {
    //TODO




  }


  Fields
  estimate_fields(Langevin::CUDA::GPUSettings& gpu) {
    Fields dle;
    // compute local velocities (forward and backward)
    // for cov-matrix and drift estimation
    Langevin::CUDA::nq_v_means(gpu);
    //// covariance matrices with forward and backward velocities:
    //// first enqueue ('nq_..') kernel for computation, then retrieve
    //// results.
    // forward, backward
    Langevin::CUDA::nq_cov(gpu
                         , true
                         , false);
    Eigen::MatrixXf cov_fwd_bwd = Tools::to_eigen_mat(
        Langevin::CUDA::get_cov(gpu));
    // backward, backward
    Langevin::CUDA::nq_cov(gpu
                         , false
                         , false);
    Eigen::MatrixXf cov_bwd_bwd = Tools::to_eigen_mat(
        Langevin::CUDA::get_cov(gpu)
      , true);
    // forward, forward
    Langevin::CUDA::nq_cov(gpu
                         , true
                         , true);
    Eigen::MatrixXf cov_fwd_fwd = Tools::to_eigen_mat(
        Langevin::CUDA::get_cov(gpu)
      , true);
  
  
    // friction
    dle.friction = -1.0 * (cov_fwd_bwd * cov_bwd_bwd.inverse());
    // drift
    auto v_means = Langevin::CUDA::get_v_means(gpu);
    dle.drift = Tools::to_eigen_vec(v_means.first)
              + dle.friction * Tools::to_eigen_vec(v_means.second);
    // noise amplitude (i.e. diffusion) ...
    dle.diffusion = cov_fwd_fwd
                         - dle.friction
                           * cov_bwd_bwd
                           * dle.friction.transpose();
    // ... from Cholesky decomposition
    dle.diffusion = Eigen::LLT<Eigen::MatrixXf>(dle.diffusion).matrixL();
    return dle;
  }
  
  
  Eigen::VectorXf
  euler_integration(Langevin::Frame frame
                  , Tools::Dice& rnd) {
    unsigned int n_dim = frame.pos.size();
    Eigen::VectorXf pos_new(n_dim);
    Eigen::VectorXf noise(n_dim);
    // initialize noise
    for (unsigned int i=0; i < n_dim; ++i) {
      noise(i) = rnd.normal();
    }
    // compute new position from dLE
    pos_new = frame.pos
            + frame.fields.drift
            - frame.fields.friction*(frame.pos-frame.pos_prev)
            + frame.fields.diffusion*noise;
    return pos_new;
  }
  
  
  Eigen::VectorXf
  propagate(Langevin::Frame frame
          , Tools::Dice& rnd
          , unsigned int min_pop
          , unsigned int& max_retries
          , Langevin::CUDA::GPUSettings& gpu) {
    Eigen::VectorXf pos_new;
    bool propagation_failed = true;
    for (; max_retries != 0; --max_retries) {
      // make step to new position
      pos_new = euler_integration(frame
                                , rnd);
      // update neighborhood at new position
      unsigned int n_neighbors = update_neighbors(pos_new
                                                , gpu);
      // check: enough neighbors found to further integrate Langevin?
      if (n_neighbors >= min_pop) {
        propagation_failed = false;
        break;
      }
    }
    if (propagation_failed) {
      pos_new = {};
    }
    return pos_new;
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
    fh << " pop n_retries state" << std::endl;
  }
  
  void
  write_stats(std::ostream& fh
            , const Fields& dle
            , unsigned int n_neighbors
            , unsigned int retries
            , unsigned int state) {
    unsigned int n_dim = dle.drift.size();
    // write drift
    for (unsigned int i=0; i < n_dim; ++i) {
      fh << " " << dle.drift(i);
    }
    // write friction
    for (unsigned int i=0; i < n_dim; ++i) {
      for (unsigned int j=0; j < n_dim; ++j) {
        fh << " " << dle.friction(i,j);
      }
    }
    // write diffusion (lower triangle of matrix)
    for (unsigned int i=0; i < n_dim; ++i) {
      for (unsigned int j=0; j <= i; ++j) {
        fh << " " << dle.diffusion(i,j);
      }
    }
    // write neighbor populations
    fh << " " << n_neighbors;
    // propagation retries
    fh << " " << retries;
    // current state (0, if no state assigned)
    fh << " " << state << std::endl;
  }

} // end Langevin::

