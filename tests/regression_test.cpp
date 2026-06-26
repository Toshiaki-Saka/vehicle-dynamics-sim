// regression_test.cpp — numerical regression guard for the C++ core.
//
// Runs the same simulations as src/main.cpp and asserts that the headline
// numbers stay within tolerance of the committed baseline (examples/results.json).
// Exits non-zero on any violation so `ctest` turns a regression into a red build.
//
// Baselines were produced by this repository's own binary and match
// examples/results.json exactly.

#include <cmath>
#include <cstdio>
#include <numbers>
#include <vector>

#include <Eigen/Dense>

#include "types.hpp"
#include "path_utils.hpp"
#include "car_tracking.hpp"
#include "car_dynamic.hpp"
#include "aircraft.hpp"
#include "ship.hpp"

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
}

// Assert |value - expected| <= tol.
void close(double value, double expected, double tol, const char* what) {
    const bool ok = std::fabs(value - expected) <= tol;
    if (ok) {
        std::printf("  [PASS] %s (%.6f ~ %.6f)\n", what, value, expected);
    } else {
        std::printf("  [FAIL] %s: got %.6f, expected %.6f +/- %.1e\n",
                    what, value, expected, tol);
        ++g_failures;
    }
}

}  // namespace

int main() {
    std::printf("=== Vehicle Dynamics regression test ===\n");

    // ---- Scenario identical to src/main.cpp -------------------------------
    const Path path = make_ellipse_path(50.0, 30.0, 400);
    const State3D init{50.0, 2.0, std::numbers::pi / 2.0};
    const double v = 8.0, dt = 0.05;
    const int n_steps = static_cast<int>(50.0 / dt);  // 1000

    // LQR gain as designed by python/compute_lqr_gain.py (examples/lqr_k.json).
    Eigen::Matrix<double, 1, 4> K;
    K << 3.162277660168389, 0.491582874020104,
         3.206724109337935, 0.3258385885750341;

    // ---- 1. Car path tracking --------------------------------------------
    std::printf("\n[1] Car path tracking RMS / max steer\n");
    const auto pp  = simulate_pure_pursuit(path, init, v, dt, n_steps);
    const auto st  = simulate_stanley     (path, init, v, dt, n_steps);
    const auto lqr = simulate_lqr         (path, init, v, dt, n_steps, K);

    close(pp.rms_error,      0.195306, 1e-3, "pure_pursuit RMS");
    close(pp.max_steer_deg, 16.669025, 1e-2, "pure_pursuit max steer");
    close(st.rms_error,      0.202910, 1e-3, "stanley RMS");
    close(st.max_steer_deg, 25.354575, 1e-2, "stanley max steer");
    close(lqr.rms_error,     0.191009, 1e-3, "lqr RMS");
    close(lqr.max_steer_deg, 35.0,     1e-2, "lqr max steer (clamped)");

    // ---- 2. Car dynamic 2-DOF --------------------------------------------
    std::printf("\n[2] Car dynamic understeer gradient\n");
    const auto car = simulate_car_dynamic();
    check(car.Kv > 0.0, "Kv > 0 (understeer)");
    close(car.Kv, 0.00104167, 1e-5, "Kv value");

    // ---- 3. Aircraft longitudinal modes ----------------------------------
    std::printf("\n[3] Aircraft longitudinal eigenvalues\n");
    const auto ac = simulate_aircraft_longitudinal();
    int stable = 0, pairs = 0;
    double wn_short = 0.0, wn_phugoid = 1e9;
    for (std::size_t i = 0; i < ac.eigvals_real.size(); ++i) {
        const double re = ac.eigvals_real[i], im = ac.eigvals_imag[i];
        if (re < 0.0) ++stable;
        if (im > 1e-6) {  // one representative per conjugate pair
            ++pairs;
            const double wn = std::hypot(re, im);
            wn_short   = std::max(wn_short, wn);
            wn_phugoid = std::min(wn_phugoid, wn);
        }
    }
    check(stable == 4,  "all four eigenvalues stable (Re < 0)");
    check(pairs == 2,   "two oscillatory modes (short-period + phugoid)");
    // Short-period is the fast mode, phugoid the slow one.
    check(wn_short > 0.5 && wn_short < 1.2,
          "short-period natural frequency in [0.5, 1.2] rad/s");
    check(wn_phugoid > 0.002 && wn_phugoid < 0.05,
          "phugoid natural frequency in [0.002, 0.05] rad/s");
    check(wn_short > wn_phugoid * 10.0,
          "short-period clearly faster than phugoid");

    // ---- 4. Ship Nomoto settling -----------------------------------------
    std::printf("\n[4] Ship Nomoto settle time\n");
    const auto ship = simulate_ship_nomoto();
    check(ship.settle_time > 0.0, "ship settled within simulation time");
    close(ship.settle_time, 124.84, 1.0, "ship settle time");

    // ---- Verdict ----------------------------------------------------------
    std::printf("\n%s (%d failure(s))\n",
                g_failures ? "REGRESSION TEST FAILED" : "ALL REGRESSION CHECKS PASSED",
                g_failures);
    return g_failures ? 1 : 0;
}
