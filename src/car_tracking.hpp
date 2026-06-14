#pragma once
#include "types.hpp"
#include <Eigen/Dense>

// Pure Pursuit 経路追従
TrackingHistory simulate_pure_pursuit(
    const Path& path,
    const State3D& initial_state,
    double v, double dt, int n_steps,
    double wheelbase     = 2.7,
    double lookahead_gain = 0.6,
    double lookahead_min  = 4.0
);

// Stanley 経路追従 (前輪基準・横偏差陽的使用)
TrackingHistory simulate_stanley(
    const Path& path,
    const State3D& initial_state,
    double v, double dt, int n_steps,
    double wheelbase = 2.7,
    double k_gain    = 2.5
);

// LQR 経路追従 (K ゲインは Python scipy CARE で計算し引数として渡す)
TrackingHistory simulate_lqr(
    const Path& path,
    const State3D& initial_state,
    double v, double dt, int n_steps,
    const Eigen::Matrix<double, 1, 4>& K,
    double wheelbase = 2.7
);
