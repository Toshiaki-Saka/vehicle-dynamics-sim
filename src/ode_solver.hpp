#pragma once
#include <functional>
#include <vector>
#include <Eigen/Dense>

// Time-invariant RK4: f(x) -> dx/dt
template<int N>
using OdeFunc = std::function<Eigen::Matrix<double, N, 1>(const Eigen::Matrix<double, N, 1>&)>;

template<int N>
Eigen::Matrix<double, N, 1> rk4_step(
    const OdeFunc<N>& f,
    const Eigen::Matrix<double, N, 1>& x,
    double dt)
{
    auto k1 = f(x);
    auto k2 = f(x + 0.5 * dt * k1);
    auto k3 = f(x + 0.5 * dt * k2);
    auto k4 = f(x + dt * k3);
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

template<int N>
std::vector<Eigen::Matrix<double, N, 1>> integrate_rk4(
    const OdeFunc<N>& f,
    const Eigen::Matrix<double, N, 1>& x0,
    const std::vector<double>& t)
{
    std::vector<Eigen::Matrix<double, N, 1>> result;
    result.reserve(t.size());
    result.push_back(x0);
    for (size_t i = 1; i < t.size(); ++i) {
        double dt = t[i] - t[i - 1];
        result.push_back(rk4_step<N>(f, result.back(), dt));
    }
    return result;
}
