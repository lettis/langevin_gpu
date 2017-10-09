
#include <algorithm>

#include "hybrid_msm_dle.hpp"


namespace Hybrid {

  unsigned int
  rnd_index(std::vector<unsigned int> states
          , unsigned int selected_state
          , Tools::Dice& rnd) {
    // construct index list (base 1)
    std::vector<unsigned int> indices(states.size());
    std::iota(indices.begin()
            , indices.end()
            , 0);
    // remove indices of unselected states
    indices.erase(std::remove_if(indices.begin()
                               , indices.end()
                               , [selected_state](unsigned int s) -> bool {
                                   return (s != selected_state);
                                 })
                , indices.end());
    if (indices.size() == 0) {
      std::cerr << "error: cannot pick index for state "
                << selected_state
                << ". no such state in trajectory!"
                << std::endl;
      exit(EXIT_FAILURE);
    }
    // select random index
    unsigned int i = ((unsigned int) rnd.uniform() * (indices.size()-1));
    return indices[i];
  }

  Hybrid::Frame
  propagate_discrete_uncoupled(MSM::Model msm
                             , std::vector<unsigned int> ref_states
                             , std::vector<std::vector<float>> ref_coords
                             , const Hybrid::Frame& frame
                             , Tools::Dice& rnd
                             , unsigned int min_pop
                             , unsigned int& max_retries
                             , Langevin::CUDA::GPUSettings& gpu) {
    Hybrid::Frame next_frame;
    next_frame.dle = frame.dle;
    next_frame.dle.pos_prev = next_frame.dle.pos;
    // next state from MCMC sampling
    next_frame.state = MSM::propagate(msm
                                    , frame.state
                                    , rnd);
    if (frame.state == next_frame.state) {
      //// state unchanged: dLE propagation
      Langevin::update_neighbors(frame.dle.pos
                               , gpu);
      // fields from neighborhood
      next_frame.dle.fields = Langevin::estimate_fields(gpu);
      // propagate dLE dynamics for new position
      next_frame.dle.pos = Langevin::propagate(
          next_frame.dle
        , rnd
        , min_pop
        , max_retries
        , gpu);
      if (next_frame.dle.pos.size() == 0) {
        // propagation failed: random position; count as new trajectory
        unsigned int i = rnd_index(ref_states
                                 , next_frame.state
                                 , rnd);
        next_frame.dle.pos = Tools::to_eigen_vec(ref_coords[i]);
        next_frame.i_traj++;
      }
    } else {
      // MSM jumped to new state; select coords randomly from reference data
      unsigned int i = rnd_index(ref_states
                               , next_frame.state
                               , rnd);
      next_frame.dle.pos = Tools::to_eigen_vec(ref_coords[i]);
    }
    return next_frame;
  }

  Hybrid::Frame
  propagate_discrete_coupled(float c
                           , MSM::Model msm
                           , std::vector<unsigned int> ref_states
                           , std::vector<std::vector<float>> ref_coords
                           , const Hybrid::Frame& frame
                           , Tools::Dice& rnd
                           , unsigned int min_pop
                           , unsigned int& max_retries
                           , Langevin::CUDA::GPUSettings& gpu) {
    Hybrid::Frame next_frame = frame;
    next_frame.dle.pos_prev = frame.dle.pos;
    // transition probabilities from MSM and dLE
    Eigen::VectorXf p_msm = msm.tmat.row(frame.state-1);
    Eigen::VectorXf p_dle =
      Langevin::state_transition_probabilities(frame.dle.pos
                                             , msm.tau
                                             , gpu);
    // next state from MCMC sampling of combined transition probabilities
    next_frame.state = Tools::rnd_state(c*p_msm + (1-c)*p_dle
                                      , rnd);
    if (frame.state == next_frame.state) {
      //// state unchanged: dLE propagation
      Langevin::update_neighbors(frame.dle.pos
                               , gpu);
      // fields from neighborhood
      next_frame.dle.fields = Langevin::estimate_fields(gpu);
      // propagate dLE dynamics for new position
      next_frame.dle.pos = Langevin::propagate(
          next_frame.dle
        , rnd
        , min_pop
        , max_retries
        , gpu);
      if (next_frame.dle.pos.size() == 0) {
        // propagation failed: random position; count as new trajectory
        unsigned int i = rnd_index(ref_states
                                 , next_frame.state
                                 , rnd);
        next_frame.dle.pos = Tools::to_eigen_vec(ref_coords[i]);
        next_frame.i_traj++;
      }
    } else {
      // MSM jumped to new state; select coords randomly from reference data
      unsigned int i = rnd_index(ref_states
                               , next_frame.state
                               , rnd);
      next_frame.dle.pos = Tools::to_eigen_vec(ref_coords[i]);
    }
    return next_frame;
  }

  Hybrid::Frame
  propagate_continuous(float c
                     , MSM::Model msm
                     , std::vector<unsigned int> ref_states
                     , std::vector<std::vector<float>> ref_coords
                     , const Hybrid::Frame& frame
                     , Tools::Dice& rnd
                     , unsigned int min_pop
                     , unsigned int& max_retries
                     , Langevin::CUDA::GPUSettings& gpu) {
    // get transition probablities
    Eigen::VectorXf p_msm = msm.tmat.row(frame.state-1);
    Eigen::VectorXf p_dle =
      Langevin::state_transition_probabilities(frame.dle.pos
                                             , msm.tau
                                             , gpu);
    Eigen::VectorXf p = c*p_msm + (1-c)*p_dle;
    // run dLE propagations
    bool propagation_succeeded = false;
    Hybrid::Frame next_frame = frame;
    next_frame.dle.pos_prev = frame.dle.pos;
    // update neighbor list / fields for dLE propagation
    Langevin::update_neighbors(frame.dle.pos
                             , gpu);
    next_frame.dle.fields = Langevin::estimate_fields(gpu);
    while ( ! propagation_succeeded) {
      // propagate to next candidate position
      next_frame.dle.pos = Langevin::propagate(
          next_frame.dle
        , rnd
        , min_pop
        , max_retries
        , gpu);
      if (next_frame.dle.pos.size() == 0) {
        // abort, nothing worked; get new random position
        break;
      } else {
        // estimate most likely state at new position
        Eigen::VectorXf state_probs =
          Langevin::state_probabilities(next_frame.dle.pos
                                      , gpu);
        // state id == index (in base 1) of max. probability
        state_probs.maxCoeff(&next_frame.state);
        ++next_frame.state;
        // accept/decline propagation based on Monte Carlo decision
        propagation_succeeded = (rnd.uniform() < p(next_frame.state));
      }
    }
    // randomly chosen new frame (in same state) if not succeeded
    if ( ! propagation_succeeded) {
      next_frame.state = frame.state;
      unsigned int i = rnd_index(ref_states
                               , next_frame.state
                               , rnd);
      next_frame.dle.pos = Tools::to_eigen_vec(ref_coords[i]);
    }
    return next_frame;
  }

  Hybrid::Frame
  propagate_dle(std::vector<unsigned int> ref_states
              , std::vector<std::vector<float>> ref_coords
              , const Hybrid::Frame& frame
              , Tools::Dice& rnd
              , unsigned int min_pop
              , unsigned int& max_retries
              , Langevin::CUDA::GPUSettings& gpu) {
    // run dLE propagations
    Hybrid::Frame next_frame = frame;
    next_frame.dle.pos_prev = frame.dle.pos;
    // update neighbor list / fields for dLE propagation
    Langevin::update_neighbors(frame.dle.pos
                             , gpu);
    next_frame.dle.fields = Langevin::estimate_fields(gpu);
    // propagate to next candidate position
    next_frame.dle.pos = Langevin::propagate(
        next_frame.dle
      , rnd
      , min_pop
      , max_retries
      , gpu);
    if (next_frame.dle.pos.size() == 0) {
      // propagation failed: randomly choose new frame in same state
      next_frame.state = frame.state;
      unsigned int i = rnd_index(ref_states
                               , next_frame.state
                               , rnd);
      next_frame.dle.pos = Tools::to_eigen_vec(ref_coords[i]);
    } else {
      // propagation succeeded: get most likely state from environment
      Eigen::VectorXf state_probs =
        Langevin::state_probabilities(next_frame.dle.pos
                                    , gpu);
      // state id == index (in base 1) of max. probability
      state_probs.maxCoeff(&next_frame.state);
      ++next_frame.state;
    }
    return next_frame;
  }

} // end namespace Hybrid::

