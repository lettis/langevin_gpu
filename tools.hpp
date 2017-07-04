#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>


std::vector<char>
read_futures(std::string fname);

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

std::string
join_args(int argc
        , char** argv);


Eigen::MatrixXf
to_eigen_mat(std::vector<float> xs
           , bool make_symmetric=false);


Eigen::VectorXf
to_eigen_vec(std::vector<float> xs);


//// template implementations

#include "tools.hxx"

