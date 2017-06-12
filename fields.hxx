
template<bool is_forward_velocity_1
       , bool is_forward_velocity_2>
Eigen::MatrixXf
covariance(std::vector<unsigned int> neighbor_ids
         , const std::vector<std::vector<float>>& ref_coords) {
  unsigned int n_frames = neighbor_ids.size();
  unsigned int n_dim = ref_coords[0].size();
  std::vector<std::vector<float>> v1(n_frames);
  std::vector<std::vector<float>> v2(n_frames);
  // coordinate difference  v[i1] - v[i2]
  auto diff = [&](unsigned int i1
                , unsigned int i2) -> std::vector<float> {
    std::vector<float> v_diff = ref_coords[i1];
    for (unsigned int j=0; j < n_dim; ++j) {
      v_diff[j] -= ref_coords[i2][j];
    }
    return v_diff;
  };
  // compute forward or backward velocities from coordinates
  for (unsigned int i=0; i < n_frames; ++i) {
    if (is_forward_velocity_1) {
      v1[i] = diff(neighbor_ids[i]+1
                 , neighbor_ids[i]);
    } else {
      v1[i] = diff(neighbor_ids[i]
                 , neighbor_ids[i]-1);
    }
    if (is_forward_velocity_2) {
      v2[i] = diff(neighbor_ids[i]+1
                 , neighbor_ids[i]);
    } else {
      v2[i] = diff(neighbor_ids[i]
                 , neighbor_ids[i]-1);
    }
  }
  // compute covariance over velocities
  return covariance(v1
                  , v2);
}


