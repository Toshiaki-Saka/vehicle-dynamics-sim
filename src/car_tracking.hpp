#pragma once
#include "types.hpp"
#include <Eigen/Dense>

// Pure Pursuit path tracking
TrackingHistory simulate_pure_pursuit(
    const Path& path,
    const State3D& initial_state,
    double v, double dt, int n_steps,
    double wheelbase     = 2.7,
    double lookahead_gain = 0.6,
    double lookahead_min  = 4.0
);

// Stanley path tracking (front-wheel reference, explicit use of lateral deviation)
TrackingHistory simulate_stanley(
    const Path& path,
    const State3D& initial_state,
    double v, double dt, int n_steps,
    double wheelbase = 2.7,
    double k_gain    = 2.5
);

// LQR path tracking (the K gain is computed with Python scipy CARE and passed as an argument)
TrackingHistory simulate_lqr(
    const Path& path,
    const State3D& initial_state,
    double v, double dt, int n_steps,
    const Eigen::Matrix<double, 1, 4>& K,
    double wheelbase = 2.7
);
