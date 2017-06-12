
#include <iostream>
#include <fstream>
#include <limits>
#include <utility>

#include "tools.hpp"

std::vector<unsigned int>
read_states(std::string fname) {
  return _read_col<unsigned int>(fname);
}

std::vector<unsigned int>
read_pops(std::string fname) {
  return _read_col<unsigned int>(fname);
}

std::vector<float>
read_fe(std::string fname) {
  return _read_col<float>(fname);
}


std::vector<std::pair<float, float>>
col_min_max(const std::vector<std::vector<float>>& coords) {
  unsigned int nrow = coords.size();
  if (nrow == 0) {
    std::cerr << "error: no sampling. cannot compute min/max of coordinates." << std::endl;
    exit(EXIT_FAILURE);
  }
  unsigned int ncol = coords[0].size();
  std::vector<std::pair<float, float>> mm(ncol
                                        , {std::numeric_limits<float>::infinity()
                                         , -std::numeric_limits<float>::infinity()});
  for (unsigned int i=0; i < nrow; ++i) {
    for (unsigned int j=0; j < ncol; ++j) {
      float ref_val = coords[i][j];
      if (ref_val < mm[j].first) {
        mm[j].first = ref_val;
      }
      if (ref_val > mm[j].second) {
        mm[j].second = ref_val;
      }
    }
  }
  return mm;
}



