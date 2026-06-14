#pragma once
#include <vector>

struct State3D {
    double x{0.0}, y{0.0}, psi{0.0};
};

struct PathPoint {
    double x{0.0}, y{0.0}, yaw{0.0}, curvature{0.0};
};

using Path = std::vector<PathPoint>;

struct TrackingHistory {
    std::vector<double> x, y, psi, delta;
    double rms_error{0.0};
    double max_steer_deg{0.0};
};

struct CarDynResult {
    std::vector<double> time, vy, yaw_rate;
    double Kv{0.0}, vx{0.0};
    double yaw_gain_theory{0.0}, yaw_gain_sim{0.0};
};

struct AircraftResult {
    std::vector<double> time, u_pert, w, q, theta;
    std::vector<double> eigvals_real, eigvals_imag;
};

struct ShipResult {
    std::vector<double> time, psi, delta, x_pos, y_pos;
    double psi_target{0.0};
    double settle_time{-1.0};
};
