
#include <fstream>
#include <cmath>
#include <random>

#include "msm.hpp"

namespace MSM {

  Model
  load_msm(std::string fname
         , unsigned int tau) {
    // load transition matrix data
    std::vector<float> buf = Tools::read_datafile<float>(fname);
    unsigned int dim = std::sqrt(buf.size());
    if (dim*dim != buf.size()) {
      std::cerr << "error: transition matrix is not square!" << std::endl;
      exit(EXIT_FAILURE);
    }
    // convert data into matrix
    Eigen::MatrixXf tmat(dim, dim);
    for (unsigned int i=0; i < dim; ++i) {
      for (unsigned int j=0; j < dim; ++j) {
        tmat(i,j) =  buf[i*dim+j];
      }
    }
    return {dim, tmat, tau};
  }

  unsigned int
  propagate(Model msm
          , unsigned int state
          , Tools::Dice& rnd) {
    return Tools::rnd_state(msm.tmat.row(state-1)
                          , rnd);
  }

} // end MSM::

