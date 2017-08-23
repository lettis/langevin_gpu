
#include <fstream>
#include <cmath>
#include <random>

#include "msm.hpp"

namespace MSM {

  Model
  load_msm(std::string fname
         , float rnd_seed) {
    // load transition matrix data
    std::vector<float> buf = Tools::read_datafile<float>(fname);
    unsigned int dim = std::sqrt(buf.size());
    if (dim*dim != buf.size()) {
      std::cerr << "error: transition matrix is not square!" << std::endl;
      exit(EXIT_FAILURE);
    }
    // convert data into matrix and initialize
    // weighted distribution for MSM propagation
    Eigen::MatrixXf tmat(dim, dim);
    std::vector<std::function<unsigned int()>> propagator(dim);
    for (unsigned int i=0; i < dim; ++i) {
      std::vector<float> weights(dim);
      for (unsigned int j=0; j < dim; ++j) {
        weights[j] = buf[i*dim+j];
        tmat(i,j) = weights[j];
      }
      propagator[i] = std::bind(
          std::discrete_distribution<unsigned int>(weights.begin()
                                                 , weights.end())
        , std::mt19937(rnd_seed));
    }
    return {dim, tmat, propagator};
  }

  unsigned int
  propagate(Model msm
          , unsigned int state) {
    // states are in [1,N], propagator works in [0,N-1]
    return msm.propagator[--state]() + 1;
  }

} // end MSM::



