#include "ship.hpp"
#include "ode_solver.hpp"
#include <numbers>
#include <cmath>
#include <algorithm>

ShipResult simulate_ship_nomoto(
    double K_nomoto, double T_nomoto,
    double psi_target_deg, double Kp, double Kd,
    double v_ship, double t_end, int n_steps)
{
    using std::numbers::pi;
    double psi_target = psi_target_deg * pi / 180.0;
    double delta_max  = 35.0 * pi / 180.0;

    // ── Nomoto 1次モデル: T·ψ̈ + ψ̇ = K·δ ─────────────────────────
    // 状態 x = [ψ, ψ̇]^T
    // PD 制御: δ = clip(Kp·e - Kd·ψ̇, ±δ_max)
    OdeFunc<2> f = [&](const Eigen::Vector2d& x) -> Eigen::Vector2d {
        double psi     = x(0);
        double psi_dot = x(1);
        double e        = psi_target - psi;
        double delta    = std::clamp(Kp * e - Kd * psi_dot, -delta_max, delta_max);
        double psi_ddot = (K_nomoto * delta - psi_dot) / T_nomoto;
        return Eigen::Vector2d(psi_dot, psi_ddot);
    };

    std::vector<double> t(n_steps);
    for (int i = 0; i < n_steps; ++i)
        t[i] = t_end * i / (n_steps - 1);

    auto states = integrate_rk4<2>(f, Eigen::Vector2d::Zero(), t);

    ShipResult res;
    res.psi_target = psi_target;
    res.time.reserve(n_steps);
    res.psi.reserve(n_steps);
    res.delta.reserve(n_steps);
    res.x_pos.reserve(n_steps);
    res.y_pos.reserve(n_steps);

    double tol = 1.5 * pi / 180.0;

    for (int i = 0; i < n_steps; ++i) {
        double psi_v   = states[i](0);
        double psi_dot = states[i](1);
        double e       = psi_target - psi_v;
        double delta   = std::clamp(Kp * e - Kd * psi_dot, -delta_max, delta_max);
        res.time.push_back(t[i]);
        res.psi.push_back(psi_v);
        res.delta.push_back(delta);
    }

    // 船体軌跡(縦速度 v_ship 一定)
    res.x_pos.push_back(0.0);
    res.y_pos.push_back(0.0);
    for (int i = 1; i < n_steps; ++i) {
        double dt = t[i] - t[i - 1];
        res.x_pos.push_back(res.x_pos.back() + v_ship * std::cos(res.psi[i - 1]) * dt);
        res.y_pos.push_back(res.y_pos.back() + v_ship * std::sin(res.psi[i - 1]) * dt);
    }

    // 整定時間(±1.5度以内で以後も逸脱しない最初の時刻)
    res.settle_time = -1.0;
    for (int i = 0; i < n_steps; ++i) {
        if (std::abs(res.psi[i] - psi_target) < tol) {
            bool settled = true;
            for (int j = i; j < n_steps; ++j)
                if (std::abs(res.psi[j] - psi_target) >= tol) { settled = false; break; }
            if (settled) { res.settle_time = t[i]; break; }
        }
    }

    return res;
}
