#pragma once
#include "types.hpp"

// Generate an elliptical path (each point has tangent heading and curvature)
Path make_ellipse_path(double a = 50.0, double b = 30.0, int n = 400);

// Return the nearest-neighbor index within a window starting from prev_idx
int find_nearest_in_window(double x, double y, const Path& path,
                            int prev_idx, int window = 50);

// Compute the RMS of the shortest distance from each state to the path
double evaluate_rms(const std::vector<State3D>& history, const Path& path);

// Normalize an angle to [-pi, pi]
double normalize_angle(double a);
