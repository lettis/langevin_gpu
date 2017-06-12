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
#include "tools_io.hpp"
#include "neighbors_cuda.hpp"
#include "fields.hpp"

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
    ("dx,x", b_po::value<float>()->default_value(0.0),
     "input           : dx per dimension of numerical"
     " differentiation (default: compute from radius)")
    ("radius,r", b_po::value<float>()->required(),
     "input (required): radius for probability integration.")
    ("seed,I", b_po::value<float>()->default_value(0.0),
     "input           : set seed for random number generator"
     " (default: 0, i.e. generate seed)")
    // options
    ("output,o", b_po::value<std::string>()->default_value(""),
     "output:           the sampled coordinates"
     " default: stdout.")
    ("length,L", b_po::value<unsigned int>()->required(),
     "output:           length of simulated trajectory")
    ("temperature,T", b_po::value<unsigned int>()->default_value(300),
     "                  temperature for mass-correction of drift"
                      " (default: 300)")
    ("verbose,v", b_po::bool_switch()->default_value(false),
     "                  give verbose output.")
    ("nthreads,n", b_po::value<int>()->default_value(0),
     "                  number of OpenMP threads. default: 0; i.e. use"
     " OMP_NUM_THREADS env-variable.")
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
  // setup OpenMP
  int n_threads = 0;
  if (args.count("nthreads")) {
    n_threads = args["nthreads"].as<int>();
  }
  if (n_threads > 0) {
    omp_set_num_threads(n_threads);
  }
  // various parameters
  float radius = args["radius"].as<float>();
  float rad2 = radius * radius;
  float dx = args["dx"].as<float>();
  if (dx == 0.0) {
    //TODO better default dx estimate?
    dx = 0.5 * radius;
  }
  unsigned int T = args["temperature"].as<unsigned int>();
  unsigned int propagation_length = args["length"].as<unsigned int>();
  // random number generator
  float rnd_seed = args["seed"].as<float>();
  std::function<float()>
    rnd = std::bind(std::normal_distribution<double>(0.0, 1.0)
                  , std::mt19937(rnd_seed));
  // input (coordinates)
  std::string fname_out = args["output"].as<std::string>();
  std::vector<std::vector<float>> ref_coords;
  unsigned int n_dim = 0;
  {
    CoordsFile::FilePointer fh =
      CoordsFile::open(args["coords"].as<std::string>()
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
  std::vector<unsigned int> has_future =
    read_states(args["future"].as<std::string>());
  // prepare output file (or stdout)
  bool use_stdout;
  CoordsFile::FilePointer fh_out;
  if (fname_out == "") {
    use_stdout = true;
  } else {
    use_stdout = false;
    fh_out = CoordsFile::open(fname_out, "w");
  }
  // GPU setup
  unsigned int n_gpus = CUDA::get_num_gpus();
  std::vector<CUDA::GPUSettings> gpu_settings(n_gpus);
  for (unsigned int i_gpu=0; i_gpu < n_gpus; ++i_gpu) {
    gpu_settings[i_gpu] = CUDA::prepare_gpu(i_gpu
                                          , n_dim
                                          , ref_coords);
  }
  // initial coordinate: last of input
  std::vector<float> position = ref_coords.back();
  std::vector<float> prev_position = position;
  // sampling loop (langevin propagation):
  for (unsigned int i_frame=0; i_frame < propagation_length; ++i_frame) {
    std::vector<std::vector<unsigned int>> neighbor_ids =
      neighbors(position
              , rad2
              , dx
              , gpu_settings);
    // remove frames without previous or following neighbor
    for (std::vector<unsigned int>& neighborhood: neighbor_ids) {
      neighborhood = remove_all_without_history(neighborhood
                                              , has_future);
    }
    // compute drift as gradient of free energies
    Eigen::VectorXf f_est = drift(neighbor_ids
                                , fe
                                , dx);
    // covariance matrices with forward and backward velocities
    Eigen::MatrixXf cov_fwd_bwd = covariance<true, false>(neighbor_ids[0]
                                                        , ref_coords);
    Eigen::MatrixXf cov_bwd_bwd = covariance<false, false>(neighbor_ids[0]
                                                         , ref_coords);
    Eigen::MatrixXf cov_fwd_fwd = covariance<true, true>(neighbor_ids[0]
                                                       , ref_coords);
    // friction
    Eigen::MatrixXf gamma = -1.0 * (cov_fwd_bwd * cov_bwd_bwd.inverse());
    // noise amplitude ...
    Eigen::MatrixXf kappa = cov_fwd_fwd
                          - gamma * cov_bwd_bwd * gamma.transpose();
    // ... from Cholesky decomposition
    kappa = Eigen::LLT<Eigen::MatrixXf>(kappa).matrixL();
    // mass correction for drift from
    // m_ii = K_ii^2 / (2kT (gamma_ii + 1))  and
    // kT = 38/300 T
    Eigen::MatrixXf m_inv = Eigen::MatrixXf::Zero(n_dim
                                                , n_dim);
    for (unsigned int j=0; j < n_dim; ++j) {
      m_inv(j,j) = (gamma(j,j)+1.0) / (kappa(j,j)*kappa(j,j));
    }
    m_inv = 19.0/75.0 * T * m_inv;
    f_est = m_inv * f_est;
    // Euler propagation -> new position
    std::vector<float> new_position = propagate(position
                                              , prev_position
                                              , f_est
                                              , gamma
                                              , kappa
                                              , rnd);
    prev_position = position;
    position = new_position;

    //TODO: output
    //output(new_position)
    //output(fields)
  }

  for (unsigned int i_gpu=0; i_gpu < n_gpus; ++i_gpu) {
    clear_gpu(gpu_settings[i_gpu]);
  }

  return EXIT_SUCCESS;
}

