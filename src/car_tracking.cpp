#include "car_tracking.hpp"
#include "path_utils.hpp"
#include <cmath>
#include <numbers>
#include <algorithm>

namespace {

constexpr double DEG35_RAD = 35.0 * std::numbers::pi / 180.0;

// Kinematic bicycle model, one-step update
State3D bicycle_step(const State3D& s, double v, double delta, double L, double dt) {
    return {
        .x   = s.x   + v * std::cos(s.psi) * dt,
        .y   = s.y   + v * std::sin(s.psi) * dt,
        .psi = s.psi + (v / L) * std::tan(delta) * dt
    };
}

// Determine RMS deviation and maximum steering from the steering history and return the history
TrackingHistory finalize(std::vector<double>&& x, std::vector<double>&& y,
                          std::vector<double>&& psi, std::vector<double>&& delta,
                          const Path& path)
{
    TrackingHistory h;
    h.x = std::move(x); h.y = std::move(y);
    h.psi = std::move(psi); h.delta = std::move(delta);

    std::vector<State3D> states;
    states.reserve(h.x.size());
    for (size_t i = 0; i < h.x.size(); ++i)
        states.push_back({h.x[i], h.y[i], h.psi[i]});
    h.rms_error = evaluate_rms(states, path);

    double max_abs = 0.0;
    for (double d : h.delta) max_abs = std::max(max_abs, std::abs(d));
    h.max_steer_deg = max_abs * 180.0 / std::numbers::pi;
    return h;
}

} // namespace

// ── Pure Pursuit ─────────────────────────────────────────────────────
TrackingHistory simulate_pure_pursuit(
    const Path& path, const State3D& init,
    double v, double dt, int n_steps,
    double L, double k, double Ld_min)
{
    int n_path = static_cast<int>(path.size());
    State3D state = init;
    int prev_idx = 0;

    std::vector<double> xs, ys, psis, deltas;
    xs.reserve(n_steps); ys.reserve(n_steps);
    psis.reserve(n_steps); deltas.reserve(n_steps);

    for (int i = 0; i < n_steps; ++i) {
        xs.push_back(state.x); ys.push_back(state.y); psis.push_back(state.psi);

        double Ld = std::max(k * v, Ld_min);
        prev_idx = find_nearest_in_window(state.x, state.y, path, prev_idx);

        // Search forward along the path for a point at least the look-ahead distance away
        int target_idx = prev_idx;
        for (int j = 0; j < n_path; ++j) {
            int cand = (prev_idx + j) % n_path;
            double dx = path[cand].x - state.x;
            double dy = path[cand].y - state.y;
            if (std::hypot(dx, dy) >= Ld) { target_idx = cand; break; }
        }

        double alpha = normalize_angle(
            std::atan2(path[target_idx].y - state.y,
                       path[target_idx].x - state.x) - state.psi);
        double delta = std::clamp(std::atan2(2.0 * L * std::sin(alpha), Ld),
                                  -DEG35_RAD, DEG35_RAD);
        deltas.push_back(delta);
        state = bicycle_step(state, v, delta, L, dt);
    }
    return finalize(std::move(xs), std::move(ys), std::move(psis),
                    std::move(deltas), path);
}

// ── Stanley ──────────────────────────────────────────────────────────
TrackingHistory simulate_stanley(
    const Path& path, const State3D& init,
    double v, double dt, int n_steps,
    double L, double k)
{
    State3D state = init;
    int prev_idx = 0;

    std::vector<double> xs, ys, psis, deltas;
    xs.reserve(n_steps); ys.reserve(n_steps);
    psis.reserve(n_steps); deltas.reserve(n_steps);

    for (int i = 0; i < n_steps; ++i) {
        xs.push_back(state.x); ys.push_back(state.y); psis.push_back(state.psi);

        // Front-wheel position
        double fx = state.x + L * std::cos(state.psi);
        double fy = state.y + L * std::sin(state.psi);
        prev_idx = find_nearest_in_window(fx, fy, path, prev_idx);

        double yaw_p = path[prev_idx].yaw;
        double dx = path[prev_idx].x - fx;
        double dy = path[prev_idx].y - fy;
        // Projection onto the path's left normal -> signed lateral deviation
        double e_y  = dx * (-std::sin(yaw_p)) + dy * std::cos(yaw_p);
        double e_psi = normalize_angle(yaw_p - state.psi);

        double delta = std::clamp(
            e_psi + std::atan2(k * e_y, std::max(v, 0.1)),
            -DEG35_RAD, DEG35_RAD);
        deltas.push_back(delta);
        state = bicycle_step(state, v, delta, L, dt);
    }
    return finalize(std::move(xs), std::move(ys), std::move(psis),
                    std::move(deltas), path);
}

// ── LQR ──────────────────────────────────────────────────────────────
// The K gain is received as a value computed externally (Python scipy CARE).
// State x = (e_y, ė_y, e_ψ, ė_ψ), where ė_y and ė_ψ are approximated as 0 (purely geometric tracking).
TrackingHistory simulate_lqr(
    const Path& path, const State3D& init,
    double v, double dt, int n_steps,
    const Eigen::Matrix<double, 1, 4>& K,
    double L)
{
    State3D state = init;
    int prev_idx = 0;

    std::vector<double> xs, ys, psis, deltas;
    xs.reserve(n_steps); ys.reserve(n_steps);
    psis.reserve(n_steps); deltas.reserve(n_steps);

    for (int i = 0; i < n_steps; ++i) {
        xs.push_back(state.x); ys.push_back(state.y); psis.push_back(state.psi);

        prev_idx = find_nearest_in_window(state.x, state.y, path, prev_idx);

        double nx = -std::sin(path[prev_idx].yaw);
        double ny =  std::cos(path[prev_idx].yaw);
        double e_y  = (state.x - path[prev_idx].x) * nx
                    + (state.y - path[prev_idx].y) * ny;
        double e_psi = normalize_angle(state.psi - path[prev_idx].yaw);

        Eigen::Vector4d x_state(e_y, 0.0, e_psi, 0.0);
        double delta_fb = -(K * x_state)(0);
        double delta_ff = L * path[prev_idx].curvature;
        double delta = std::clamp(delta_fb + delta_ff, -DEG35_RAD, DEG35_RAD);
        deltas.push_back(delta);
        state = bicycle_step(state, v, delta, L, dt);
    }
    return finalize(std::move(xs), std::move(ys), std::move(psis),
                    std::move(deltas), path);
}
