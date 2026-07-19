#include "aircraft.hpp"
#include "ode_solver.hpp"
#include <numbers>
#include <cmath>
#include <Eigen/Eigenvalues>

AircraftResult simulate_aircraft_longitudinal(
    double t_end, int n_steps, double delta_e_deg)
{
    // ── Boeing 747 longitudinal linear model ────────────────────────
    // State x = [Δu (m/s), w (m/s), q (rad/s), θ (rad)]^T
    // Flight condition: altitude 12,200 m, V = 235 m/s (M ~ 0.8)
    // Reference: representative parameters from Etkin & Reid "Dynamics of Flight"
    Eigen::Matrix4d A;
    A << -0.00643,   0.0263,    0.0,   -9.81,
         -0.0941,   -0.624,  235.0,    0.0,
         -0.000222, -0.00153, -0.668,   0.0,
          0.0,       0.0,     1.0,     0.0;

    Eigen::Vector4d B(0.0, -32.7, -2.08, 0.0);

    // Eigenvalue analysis -> identify short-period and phugoid modes
    Eigen::EigenSolver<Eigen::Matrix4d> es(A);
    auto eigvals = es.eigenvalues();

    double delta_e = delta_e_deg * std::numbers::pi / 180.0;

    // ẋ = Ax + B·δ_e (constant step input)
    OdeFunc<4> f = [&](const Eigen::Vector4d& x) -> Eigen::Vector4d {
        return A * x + B * delta_e;
    };

    std::vector<double> t(n_steps);
    for (int i = 0; i < n_steps; ++i)
        t[i] = t_end * i / (n_steps - 1);

    auto states = integrate_rk4<4>(f, Eigen::Vector4d::Zero(), t);

    AircraftResult res;
    res.time.reserve(n_steps);
    res.u_pert.reserve(n_steps); res.w.reserve(n_steps);
    res.q.reserve(n_steps);      res.theta.reserve(n_steps);

    for (int i = 0; i < n_steps; ++i) {
        res.time.push_back(t[i]);
        res.u_pert.push_back(states[i](0));
        res.w.push_back(states[i](1));
        res.q.push_back(states[i](2));
        res.theta.push_back(states[i](3));
    }

    for (int i = 0; i < 4; ++i) {
        res.eigvals_real.push_back(eigvals(i).real());
        res.eigvals_imag.push_back(eigvals(i).imag());
    }

    return res;
}
