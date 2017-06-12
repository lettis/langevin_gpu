#pragma once

#include <string>
#include <vector>


std::vector<unsigned int>
read_states(std::string fname);

std::vector<unsigned int>
read_pops(std::string fname);

std::vector<float>
read_fe(std::string fname);

//// misc

template <typename NUM>
std::vector<double>
sum1_normalized(const std::vector<NUM>& pops);

std::vector<std::pair<float, float>>
col_min_max(const std::vector<std::vector<float>>& coords);


//// template implementations

#include "tools.hxx"

