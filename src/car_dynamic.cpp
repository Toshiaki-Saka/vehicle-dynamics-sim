#include "car_dynamic.hpp"
#include "ode_solver.hpp"
#include <numbers>
#include <cmath>

CarDynResult simulate_car_dynamic(
    double m, double Iz, double lf, double lr,
    double Cf, double Cr, double vx,
    double delta_step_deg, double t_end, int n_steps)
{
    using std::numbers::pi;

    // ── State-space model construction ──────────────────────────
    // State x = [v_y (lateral velocity), ψ̇ (yaw rate)]^T
    // Newton-Euler equations linearized with input u = δ (steering angle)
    // From Rajamani "Vehicle Dynamics and Control" Ch.2
    double a11 = -(Cf + Cr) / (m * vx);
    double a12 = -(vx + (lf * Cf - lr * Cr) / (m * vx));
    double a21 = -(lf * Cf - lr * Cr) / (Iz * vx);
    double a22 = -(lf * lf * Cf + lr * lr * Cr) / (Iz * vx);

    Eigen::Matrix2d A;
    A << a11, a12,
         a21, a22;
    Eigen::Vector2d B(Cf / m, lf * Cf / Iz);

    double delta = delta_step_deg * pi / 180.0;

    // ẋ = Ax + B·δ (constant step input)
    OdeFunc<2> f = [&](const Eigen::Vector2d& x) -> Eigen::Vector2d {
        return A * x + B * delta;
    };

    // Time vector
    std::vector<double> t(n_steps);
    for (int i = 0; i < n_steps; ++i)
        t[i] = t_end * i / (n_steps - 1);

    auto states = integrate_rk4<2>(f, Eigen::Vector2d::Zero(), t);

    CarDynResult res;
    res.time.reserve(n_steps);
    res.vy.reserve(n_steps);
    res.yaw_rate.reserve(n_steps);
    res.vx = vx;

    for (int i = 0; i < n_steps; ++i) {
        res.time.push_back(t[i]);
        res.vy.push_back(states[i](0));
        res.yaw_rate.push_back(states[i](1));
    }

    // Stability factor and steady-state yaw gain
    double L = lf + lr;
    res.Kv = (lr * m) / (2.0 * L * Cf) - (lf * m) / (2.0 * L * Cr);
    res.yaw_gain_theory = vx / (L * (1.0 + res.Kv * vx * vx));
    res.yaw_gain_sim    = res.yaw_rate.back() / delta;

    return res;
}
