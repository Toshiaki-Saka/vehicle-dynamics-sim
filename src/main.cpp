#include <iostream>
#include <fstream>
#include <string>
#include <numbers>
#include <nlohmann/json.hpp>
#include <Eigen/Dense>

#include "types.hpp"
#include "path_utils.hpp"
#include "car_tracking.hpp"
#include "car_dynamic.hpp"
#include "aircraft.hpp"
#include "ship.hpp"

using json = nlohmann::json;

static json tracking_to_json(const TrackingHistory& h) {
    json j;
    j["x"]            = h.x;
    j["y"]            = h.y;
    j["psi"]          = h.psi;
    j["delta"]        = h.delta;
    j["rms_error"]    = h.rms_error;
    j["max_steer_deg"] = h.max_steer_deg;
    return j;
}

int main(int argc, char* argv[]) {
    std::string lqr_path    = "lqr_k.json";
    std::string output_path = "results.json";
    if (argc >= 2) lqr_path    = argv[1];
    if (argc >= 3) output_path = argv[2];

    std::cout << "=== Vehicle Dynamics Simulation (C++20) ===\n\n";

    // ── Load LQR gain ─────────────────────────────────────────────
    Eigen::Matrix<double, 1, 4> K = Eigen::Matrix<double, 1, 4>::Zero();
    {
        std::ifstream f(lqr_path);
        if (f.is_open()) {
            auto j = json::parse(f);
            auto kv = j["K"].get<std::vector<double>>();
            for (int i = 0; i < 4; ++i) K(0, i) = kv[i];
            std::cout << "LQR K loaded from " << lqr_path << "\n"
                      << "  K = [" << K << "]\n\n";
        } else {
            std::cerr << "Warning: " << lqr_path << " not found. LQR uses zero gain.\n\n";
        }
    }

    // ── 1. Car path tracking ──────────────────────────────────────
    std::cout << "[1/4] Car path tracking (Pure Pursuit / Stanley / LQR)...\n";
    auto path = make_ellipse_path(50.0, 30.0, 400);
    State3D init{50.0, 2.0, std::numbers::pi / 2.0};
    constexpr double v = 8.0, dt = 0.05;
    constexpr int n_steps = static_cast<int>(50.0 / dt);  // 1000 steps

    std::vector<double> car_time(n_steps);
    for (int i = 0; i < n_steps; ++i) car_time[i] = i * dt;

    auto pp  = simulate_pure_pursuit(path, init, v, dt, n_steps);
    auto st  = simulate_stanley     (path, init, v, dt, n_steps);
    auto lqr = simulate_lqr         (path, init, v, dt, n_steps, K);

    std::cout << "  Pure Pursuit : RMS=" << pp.rms_error
              << " m, max steer=" << pp.max_steer_deg << " deg\n";
    std::cout << "  Stanley      : RMS=" << st.rms_error
              << " m, max steer=" << st.max_steer_deg << " deg\n";
    std::cout << "  LQR          : RMS=" << lqr.rms_error
              << " m, max steer=" << lqr.max_steer_deg << " deg\n\n";

    // ── 2. Car dynamics ───────────────────────────────────────────
    std::cout << "[2/4] Car dynamic 2-DOF (step response)...\n";
    auto car_dyn = simulate_car_dynamic();
    std::string us_str = car_dyn.Kv > 0 ? "understeer"
                        : car_dyn.Kv < 0 ? "oversteer" : "neutral";
    std::cout << "  Kv=" << car_dyn.Kv << " (" << us_str << ")\n";
    std::cout << "  Yaw gain: theory=" << car_dyn.yaw_gain_theory
              << "  sim=" << car_dyn.yaw_gain_sim << "\n\n";

    // ── 3. Aircraft longitudinal motion ───────────────────────────
    std::cout << "[3/4] Aircraft longitudinal model...\n";
    auto aircraft = simulate_aircraft_longitudinal();
    std::cout << "  Eigenvalues (complex pairs):\n";
    for (int i = 0; i < 4; ++i) {
        double re = aircraft.eigvals_real[i], im = aircraft.eigvals_imag[i];
        if (im > 1e-6) {
            double wn     = std::hypot(re, im);
            double zeta   = -re / wn;
            double period = 2.0 * std::numbers::pi / im;
            std::cout << "    " << re << " +/- " << im << "j"
                      << "  wn=" << wn << "  zeta=" << zeta
                      << "  T=" << period << " s\n";
        }
    }
    std::cout << "\n";

    // ── 4. Ship Nomoto ────────────────────────────────────────────
    std::cout << "[4/4] Ship Nomoto model...\n";
    auto ship = simulate_ship_nomoto();
    if (ship.settle_time > 0)
        std::cout << "  Settle time: " << ship.settle_time << " s\n\n";
    else
        std::cout << "  Did not settle within simulation time.\n\n";

    // ── JSON output ───────────────────────────────────────────────
    std::cout << "Writing results to " << output_path << "...\n";
    json out;

    out["car_time"] = car_time;

    {
        json p;
        for (const auto& pt : path) {
            p["x"].push_back(pt.x);
            p["y"].push_back(pt.y);
            p["yaw"].push_back(pt.yaw);
            p["curvature"].push_back(pt.curvature);
        }
        out["car_path"] = p;
    }

    out["pure_pursuit"] = tracking_to_json(pp);
    out["stanley"]      = tracking_to_json(st);
    out["lqr"]          = tracking_to_json(lqr);

    {
        json j;
        j["time"]             = car_dyn.time;
        j["vy"]               = car_dyn.vy;
        j["yaw_rate"]         = car_dyn.yaw_rate;
        j["Kv"]               = car_dyn.Kv;
        j["vx"]               = car_dyn.vx;
        j["yaw_gain_theory"]  = car_dyn.yaw_gain_theory;
        j["yaw_gain_sim"]     = car_dyn.yaw_gain_sim;
        out["car_dynamic"] = j;
    }

    {
        json j;
        j["time"]        = aircraft.time;
        j["u_pert"]      = aircraft.u_pert;
        j["w"]           = aircraft.w;
        j["q"]           = aircraft.q;
        j["theta"]       = aircraft.theta;
        j["eigvals_real"] = aircraft.eigvals_real;
        j["eigvals_imag"] = aircraft.eigvals_imag;
        out["aircraft"] = j;
    }

    {
        json j;
        j["time"]        = ship.time;
        j["psi"]         = ship.psi;
        j["delta"]       = ship.delta;
        j["x"]           = ship.x_pos;
        j["y"]           = ship.y_pos;
        j["psi_target"]  = ship.psi_target;
        j["settle_time"] = ship.settle_time;
        out["ship"] = j;
    }

    std::ofstream ofs(output_path);
    ofs << out.dump();
    std::cout << "Done.\n";
    return 0;
}
