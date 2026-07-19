#pragma once
#include "types.hpp"

// Compute the step response of the linear 2-DOF dynamic bicycle model
// (lateral velocity and yaw rate) with RK4. Returns results equivalent to
// Python's scipy.signal.lti + step.
CarDynResult simulate_car_dynamic(
    double m             = 1500.0,   // vehicle mass [kg]
    double Iz            = 2500.0,   // yaw inertia [kg m^2]
    double lf            = 1.2,      // CG to front axle [m]
    double lr            = 1.5,      // CG to rear axle [m]
    double Cf            = 80000.0,  // front-wheel cornering stiffness [N/rad]
    double Cr            = 80000.0,  // rear-wheel cornering stiffness [N/rad]
    double vx            = 20.0,     // longitudinal velocity [m/s]
    double delta_step_deg = 1.0,     // step steering angle [deg]
    double t_end         = 5.0,
    int    n_steps       = 1000
);
