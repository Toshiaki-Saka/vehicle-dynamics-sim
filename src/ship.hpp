#pragma once
#include "types.hpp"

// Course-change maneuvering simulation using the Nomoto first-order model + PD control
ShipResult simulate_ship_nomoto(
    double K_nomoto       = 0.18,   // rudder-effectiveness gain [1/s]
    double T_nomoto       = 50.0,   // time constant [s]
    double psi_target_deg = 30.0,   // target heading [deg]
    double Kp             = 2.0,    // PD proportional gain
    double Kd             = 8.0,    // PD derivative gain
    double v_ship         = 8.0,    // ship speed [m/s] (about 16 knots)
    double t_end          = 600.0,
    int    n_steps        = 3000
);
