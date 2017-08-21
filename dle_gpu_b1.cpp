/*
Copyright (c) 2017, Florian Sittel (www.lettis.net)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <unordered_set>

#include <omp.h>
#include <boost/program_options.hpp>
#include <Eigen/Cholesky>

#include "coords_file/coords_file.hpp"

#include "tools.hpp"
#include "langevin.hpp"
#include "langevin_cuda.hpp"


int main(int argc, char* argv[]) {
  namespace b_po = boost::program_options;
  b_po::variables_map args;
  b_po::options_description desc(std::string(argv[1]).append(
    "\n\n"
    "TODO: description\n"
    "\n"
    "options"));
  desc.add_options()
    ("help,h", b_po::bool_switch()->default_value(false),
     "show this help.")
    // required inputs
    ("input,i", b_po::value<std::string>()->required(),
     "input (required): coordinates.")
    ("future,f", b_po::value<std::string>()->required(),
     "input (required): defines 'futures' of frames, i.e. has follower or not")
    ("free-energies,F", b_po::value<std::string>()->required(),
     "input (required): reference per-frame free energies.")
    ("length,L", b_po::value<unsigned int>()->required(),
     "input (required): length of simulated trajectory")
    ("radius,r", b_po::value<float>()->required(),
     "input (required): radius for probability integration.")
    // options
//    ("dx,x", b_po::value<float>()->default_value(0.0),
//     "input           : dx per dimension of numerical"
//     " differentiation (default: compute from radius)")
    ("seed,I", b_po::value<float>()->default_value(0.0),
     "input           : set seed for random number generator"
     " (default: 0, i.e. generate seed)")
    ("minpop,P", b_po::value<unsigned int>()->default_value(200),
     "input           : min. number of neighbors for gradient estimation."
                      " default: 200")
    ("output,o", b_po::value<std::string>()->default_value(""),
     "output:           the sampled coordinates"
     " default: stdout.")
    ("stats,s", b_po::value<std::string>()->default_value(""),
     "output:           stats like field estimates, neighbor-populations, etc")
//    ("temperature,T", b_po::value<unsigned int>()->default_value(300),
//     "                  temperature for mass-correction of drift"
//                      " (default: 300)")
    ("dry-run", b_po::bool_switch()->default_value(false),
     "                  run a 'dry' run for testing purposes: do not propagate"
                      " new trajectory, but take positions from input and"
                      " estimate fields")
    ("igpu", b_po::value<int>()->default_value(0),
     "                  index of GPU to use (default: 0)")
    ("verbose,v", b_po::bool_switch()->default_value(false),
     "                  give verbose output.")
  ;
  // parse cmd arguments           
  try {
    b_po::store(b_po::command_line_parser(argc, argv)
                  .options(desc)
                  .run()
              , args);
    b_po::notify(args);
  } catch (b_po::error& e) {
    if ( ! args["help"].as<bool>()) {
      std::cerr << "\nerror parsing arguments:\n\n"
                << e.what()
                << "\n\n"
                << std::endl;
    }
    std::cerr << desc << std::endl;
    return EXIT_FAILURE;
  }
  //TODO: secure against missing options,
  //      exception handling of cmd args
  if (args["help"].as<bool>()) {
    std::cout << desc << std::endl;
    return EXIT_SUCCESS;
  }
  // various parameters
  int i_gpu = args["igpu"].as<int>();
  float radius = args["radius"].as<float>();
  float rad2 = radius * radius;
//  float dx = args["dx"].as<float>();
//  if (dx == 0.0) {
//    //TODO better default dx estimate?
//    dx = 0.5 * radius;
//  }
//  unsigned int T = args["temperature"].as<unsigned int>();
  unsigned int propagation_length = args["length"].as<unsigned int>();
  unsigned int min_pop = args["minpop"].as<unsigned int>();
  unsigned int max_propagation_retries = 100;
  bool is_dry_run = args["dry-run"].as<bool>();
  // random number generator
  float rnd_seed = args["seed"].as<float>();
  std::function<float()>
    rnd = std::bind(std::normal_distribution<double>(0.0, 1.0)
                  , std::mt19937(rnd_seed));
  // input (coordinates)
  std::string fname_in = args["input"].as<std::string>();
  std::vector<std::vector<float>> ref_coords;
  unsigned int n_dim = 0;
  {
    CoordsFile::FilePointer fh = CoordsFile::open(fname_in
                                                , "r");
    while ( ! fh->eof()) {
      std::vector<float> buf = fh->next();
      if (buf.size() > 0) {
        ref_coords.push_back(buf);
      }
    }
    bool no_ref_coords = false;
    if (ref_coords.size() == 0) {
      no_ref_coords = true;
    } else {
      n_dim = ref_coords[0].size();
    }
    if (no_ref_coords
     || n_dim == 0) {
      std::cerr << "error: empty reference coordinates file" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  // input (free energies)
  std::vector<float> fe = read_fe(args["free-energies"].as<std::string>());
  // input (futures)
  std::vector<char> has_future =
    read_futures(args["future"].as<std::string>());
  // prepare output file (or stdout)
  std::string fname_out = args["output"].as<std::string>();
  CoordsFile::FilePointer fh_out = CoordsFile::open(fname_out, "w");
  // prepare stats-output (fields, etc)
  std::ofstream fh_stats;
  std::string fname_stats = args["stats"].as<std::string>();
  if (fname_stats != "") {
    fh_stats.open(fname_stats);
    Langevin::write_stats_header(fh_stats
                               , n_dim
                               , join_args(argc, argv));
  }
  // GPU setup
  Langevin::CUDA::GPUSettings gpu_settings;
  if (i_gpu < Langevin::CUDA::get_num_gpus()) {
    gpu_settings = Langevin::CUDA::prepare_gpu(i_gpu
                                             , n_dim
                                             , ref_coords
                                             , has_future
                                             , fe);
  } else {
    std::cerr << "error: no CUDA-enabled GPU with index "
              << i_gpu
              << " found"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  // initial coordinate: last of input
  std::vector<float> position;
  if (is_dry_run) {
    // dry run: recreate fields for given data
    position = ref_coords[0];
  } else {
    // the real thing: propagate trajectory starting at end of input
    position = ref_coords.back();
  }
  std::vector<float> prev_position = position;
  // find nearest neighbors at initial position
  Langevin::CUDA::nq_neighbors(position
                             , rad2
                             , gpu_settings);
  unsigned int n_neighbors = Langevin::CUDA::get_n_neighbors(gpu_settings);
  //// integrate Langevin dynamics
  for (unsigned int i_frame=0; i_frame < propagation_length; ++i_frame) {
    // compute local velocities (forward and backward)
    // for cov-matrix and drift estimation
    Langevin::CUDA::nq_v_means(gpu_settings);
    //// covariance matrices with forward and backward velocities:
    //// first enqueue ('nq_..') kernel for computation, then retrieve
    //// results.
    // forward, backward
    Langevin::CUDA::nq_cov(gpu_settings
                         , true
                         , false);
    Eigen::MatrixXf cov_fwd_bwd = to_eigen_mat(
        Langevin::CUDA::get_cov(gpu_settings
                              , n_neighbors));
    // backward, backward
    Langevin::CUDA::nq_cov(gpu_settings
                         , false
                         , false);
    Eigen::MatrixXf cov_bwd_bwd = to_eigen_mat(
        Langevin::CUDA::get_cov(gpu_settings
                              , n_neighbors)
      , true);
    // forward, forward
    Langevin::CUDA::nq_cov(gpu_settings
                         , true
                         , true);
    Eigen::MatrixXf cov_fwd_fwd = to_eigen_mat(
        Langevin::CUDA::get_cov(gpu_settings
                              , n_neighbors)
      , true);
    // friction
    Eigen::MatrixXf gamma = -1.0 * (cov_fwd_bwd * cov_bwd_bwd.inverse());
    // drift
    auto v_means = Langevin::CUDA::get_v_means(gpu_settings
                                             , n_neighbors);
    Eigen::VectorXf f = to_eigen_vec(v_means.first)
                      + gamma * to_eigen_vec(v_means.second);
    // noise amplitude (i.e. diffusion) ...
    Eigen::MatrixXf kappa = cov_fwd_fwd
                          - gamma * cov_bwd_bwd * gamma.transpose();
    // ... from Cholesky decomposition
    kappa = Eigen::LLT<Eigen::MatrixXf>(kappa).matrixL();
    // Euler propagation -> new position
    std::vector<float> new_position;
    unsigned int retries = 0;
    if (is_dry_run) {
      new_position = ref_coords[i_frame];
      // find neighbors at new position
      Langevin::CUDA::nq_neighbors(new_position
                                 , rad2
                                 , gpu_settings);
      n_neighbors = Langevin::CUDA::get_n_neighbors(gpu_settings);
    } else {
      bool propagation_failed = true;
      for (retries=0; retries <= max_propagation_retries; ++retries) {
        new_position = Langevin::propagate(position
                                         , prev_position
                                         , f
                                         , kappa
                                         , gamma
                                         , rnd);
        // find neighbors at new position
        Langevin::CUDA::nq_neighbors(new_position
                                   , rad2
                                   , gpu_settings);
        n_neighbors = Langevin::CUDA::get_n_neighbors(gpu_settings);
        // check: enough neighbors found to further integrate Langevin?
        if (n_neighbors >= min_pop) {
          propagation_failed = false;
          break;
        }
      }
      if (propagation_failed) {
        std::cerr << "error: unable to propagate to a low-energy region after "
                  << max_propagation_retries
                  << " retries. stopping."
                  << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    prev_position = position;
    position = new_position;
    // output: position
    fh_out->write(new_position);
    // output: stats
    Langevin::write_stats(fh_stats
                        , f
                        , gamma
                        , kappa
                        , n_neighbors
                        , retries);
  }
  // cleanup
  Langevin::CUDA::clear_gpu(gpu_settings);
  return EXIT_SUCCESS;
}

