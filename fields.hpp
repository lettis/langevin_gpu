#pragma once

#include <functional>

#include <Eigen/Dense>

#include "neighbors_cuda.hpp"

//! @returns ids of neighbors around reference points
std::vector<std::vector<unsigned int>>
neighbors(std::vector<float> ref_point
        , float rad2
        , float dx
        , CUDA::GPUSettings& gpu);

//! remove all frames which have either no past (i.e. are first in trajectory)
//! or have no future (i.e. are last in trajectory).
//! @returns frame ids which have a history.
std::vector<unsigned int>
remove_all_without_history(std::vector<unsigned int> neighbor_ids
                         , std::vector<unsigned int> has_future);

//! compute drift field by numerical differentiation
//! as gradient of free energy.
//! Local free energy landscape (along different shifts) is defined by the
//! given neighbor ids.
//! @returns Eigen::VectorXf with dimensionality of the given space encoding
//!          the drift.
Eigen::VectorXf
drift(std::vector<std::vector<unsigned int>> neighbor_ids
    , std::vector<float> fe
    , float dx);

//! compute covariance matrix of velocities.
//! is_forward_velocity == TRUE corresponds to v_fwd = x_{n+1} - x_n,
//! is_forward_velocity == FALSE corresponds to v_bwd = x_n - x_{n-1}.
//! @returns Symmetric Eigen::MatrixXf of covariances cov_{ij}.
template<bool is_forward_velocity_1
       , bool is_forward_velocity_2>
Eigen::MatrixXf
covariance(std::vector<unsigned int> neighbor_ids
         , const std::vector<std::vector<float>>& ref_coords);

//! compute standard covariance matrix over given data set
//! @returns Symmetric Eigen::MatrixXf of covariances cov_{ij} with
//!          dimensionality of given space.
Eigen::MatrixXf
covariance(std::vector<std::vector<float>> v1
         , std::vector<std::vector<float>> v2);

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

//// template implementations
#include "fields.hxx"

