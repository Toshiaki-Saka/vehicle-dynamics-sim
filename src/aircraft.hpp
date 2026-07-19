#pragma once
#include "types.hpp"

// Compute the elevator step response of the Boeing 747 longitudinal 4-DOF
// linear model with RK4. Also returns the eigenvalues (short-period and
// phugoid modes).
AircraftResult simulate_aircraft_longitudinal(
    double t_end        = 1500.0,  // simulation time [s]
    int    n_steps      = 8000,
    double delta_e_deg  = -1.0    // elevator input [deg] (nose-up direction)
);
