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
#include "msm.hpp"
#include "hybrid_msm_dle.hpp"


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
    ("states,S", b_po::value<std::string>()->required(),
     "input (required): assigned state of frames (must be 1 to N)")
    ("tmat,t", b_po::value<std::string>()->required(),
     "input (required): MSM transition probability (row-normalized,"
     " i.e. T_{ij} encodes transition from i to j.")
    ("length,L", b_po::value<unsigned int>()->required(),
     "input (required): length of simulated trajectory")
    ("radius,r", b_po::value<float>()->required(),
     "input (required): radius for probability integration.")
    // options
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
//  unsigned int T = args["temperature"].as<unsigned int>();
  unsigned int propagation_length = args["length"].as<unsigned int>();
  unsigned int min_pop = args["minpop"].as<unsigned int>();
  unsigned int max_dle_retries = 100;
  // random number generator
  float rnd_seed = args["seed"].as<float>();
  Tools::Dice rnd = Tools::initialize_dice(rnd_seed);
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
  // input (futures)
  std::vector<char> has_future =
    Tools::read_futures(args["future"].as<std::string>());
  // input (states)
  std::vector<unsigned int> states =
    Tools::read_states(args["states"].as<std::string>());
  // input (msm)
  MSM::Model msm = MSM::load_msm(args["tmat"].as<std::string>()
                               , rnd_seed);
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
                               , Tools::join_args(argc, argv));
  }
  // GPU setup
  Langevin::CUDA::GPUSettings gpu;
  if (i_gpu < Langevin::CUDA::get_num_gpus()) {
    gpu = Langevin::CUDA::prepare_gpu(i_gpu
                                    , n_dim
                                    , rad2
                                    , ref_coords
                                    , has_future
                                    , states);
  } else {
    std::cerr << "error: no CUDA-enabled GPU with index "
              << i_gpu
              << " found"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  // sampling (start at end of input)
  Hybrid::Frame frame;
  frame.dle.pos = Tools::to_eigen_vec(ref_coords.back());
  frame.dle.pos_prev = frame.dle.pos;
  frame.state = states.back();
  frame.i_traj = 1;
  for (unsigned int i_frame=0; i_frame < propagation_length; ++i_frame) {
    unsigned int retries = max_dle_retries;
    // new frame from uncoupled model
    frame = Hybrid::propagate_discrete_uncoupled(msm
                                               , states
                                               , ref_coords
                                               , frame
                                               , rnd
                                               , min_pop
                                               , retries
                                               , gpu);
    retries = max_dle_retries - retries;
    // output: position
    fh_out->write(Tools::to_stl_vec(frame.dle.pos));
    // output: stats
    Langevin::write_stats(fh_stats
                        , frame.dle.fields
                        , gpu.n_neighbors
                        , retries);
  }
  // cleanup
  Langevin::CUDA::clear_gpu(gpu);
  return EXIT_SUCCESS;
}

