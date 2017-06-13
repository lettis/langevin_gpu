
#include <algorithm>

#include "fields.hpp"

//TODO: doc: order of shifts, etc.
std::vector<std::vector<unsigned int>>
neighbors(std::vector<float> ref_point
        , float rad2
        , float dx
        , std::vector<CUDA::GPUSettings>& gpus) {
  std::vector<char> neighbor_matrix = CUDA::neighbors(ref_point
                                                    , rad2
                                                    , dx
                                                    , gpus);
  unsigned int n_frames = gpus[0].n_frames;
  unsigned int n_dim = gpus[0].n_dim;
  unsigned int n_shifts = 2*n_dim + 1;
  std::vector<std::vector<unsigned int>> neighbors(n_shifts);
  for (unsigned int j=0; j < n_shifts; ++j) {
    for (unsigned int i=0; i < n_frames; ++i) {
      if (neighbor_matrix[i*n_shifts + j] == 1) {
        neighbors[j].push_back(i);
      }
    }
  }
  return neighbors;
}

std::vector<unsigned int>
remove_all_without_history(std::vector<unsigned int> neighbor_ids
                         , std::vector<unsigned int> has_future) {
  unsigned int n_frames = has_future.size();
  auto frame_has_no_history = [&] (unsigned int i) -> bool {
    return (i == 0)
        || (i == n_frames-1)
        || (has_future[i] == 0)
        || (has_future[i-1] == 0);
  };
  neighbor_ids.erase(std::remove_if(neighbor_ids.begin()
                                  , neighbor_ids.end()
                                  , frame_has_no_history)
                   , neighbor_ids.end());
  return neighbor_ids;
}

Eigen::VectorXf
drift(std::vector<std::vector<unsigned int>> neighbor_ids
    , std::vector<float> fe
    , float dx) {
  unsigned int n_dim = (neighbor_ids.size()-1) / 2;
  Eigen::VectorXf drift(n_dim);
  // helper function to compute average free energy for given shift
  auto free_energy = [&] (unsigned int i_shift) -> float {
    float sum_fe = 0.0f;
    for (unsigned int i_neighbor: neighbor_ids[i_shift]) {
      sum_fe += fe[i_neighbor];
    }
    return sum_fe / neighbor_ids[i_shift].size();
  };
  // compute FE-gradient
  for (unsigned int i=1; i <= n_dim; ++i) {
    drift(i) = (free_energy(2*i-1) - free_energy(2*i)) / 2 / dx;
  }
  return drift;
}

Eigen::MatrixXf
covariance(std::vector<std::vector<float>> v1
         , std::vector<std::vector<float>> v2) {
  unsigned int n_frames = v1.size();
  unsigned int n_dim = v1[0].size();
  Eigen::MatrixXf cov = Eigen::MatrixXf::Zero(n_dim
                                            , n_dim);
  // compute averages beforehand and use <(X-<X>).(Y-<Y>)>
  // instead of <XY> - <X><Y>, because the second variant
  // --while formally correct-- is prone to floating-point errors.
  std::vector<float> mu1(n_dim, 0);
  std::vector<float> mu2(n_dim, 0);
  for (unsigned int k=0; k < n_frames; ++k) {
    for (unsigned int j=0; j < n_dim; ++j) {
      mu1[j] += v1[k][j];
      mu2[j] += v2[k][j];
    }
  }
  for (float& mu: mu1) {
    mu /= n_frames;
  }
  for (float& mu: mu2) {
    mu /= n_frames;
  }
  // compute covariances
  for (unsigned int k=0; k < n_frames; ++k) {
    for (unsigned int i=0; i < n_dim; ++i) {
      for (unsigned int j=0; j < n_dim; ++j) {
        cov(i,j) += (v1[k][i] - mu1[i]) * (v2[k][j] - mu2[j]);
      }
    }
  }
  cov = cov / (n_frames-1.0);
  // return cov-matrix with enforced symmetry
  // (to counter numerical instabilities in the data)
  return 0.5 * (cov + cov.transpose());
}

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
      fh << " k_" << i+1 << "_" << j+1;
    }
  }
  fh << " pop" << std::endl;
}

void
write_stats(std::ostream& fh
          , const Eigen::VectorXf& f
          , const Eigen::MatrixXf& gamma
          , const Eigen::MatrixXf& kappa
          , unsigned int n_neighbors) {
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
  fh << n_neighbors << std::endl;
}

