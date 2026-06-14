#include "path_utils.hpp"
#include <cmath>
#include <numbers>
#include <limits>
#include <algorithm>

Path make_ellipse_path(double a, double b, int n) {
    using std::numbers::pi;
    Path path;
    path.reserve(n);
    for (int i = 0; i < n; ++i) {
        double t   = 2.0 * pi * i / n;
        double x   =  a * std::cos(t);
        double y   =  b * std::sin(t);
        double dx  = -a * std::sin(t);
        double dy  =  b * std::cos(t);
        double yaw = std::atan2(dy, dx);
        double ddx = -a * std::cos(t);
        double ddy = -b * std::sin(t);
        double denom = std::pow(dx * dx + dy * dy, 1.5);
        double curvature = (dx * ddy - dy * ddx) / denom;
        path.push_back({.x = x, .y = y, .yaw = yaw, .curvature = curvature});
    }
    return path;
}

int find_nearest_in_window(double x, double y, const Path& path,
                             int prev_idx, int window)
{
    int n = static_cast<int>(path.size());
    int best_idx = prev_idx;
    double best_d2 = std::numeric_limits<double>::max();
    for (int k = 0; k < window; ++k) {
        int idx = (prev_idx + k) % n;
        double dx = path[idx].x - x;
        double dy = path[idx].y - y;
        double d2 = dx * dx + dy * dy;
        if (d2 < best_d2) {
            best_d2 = d2;
            best_idx = idx;
        }
    }
    return best_idx;
}

double evaluate_rms(const std::vector<State3D>& history, const Path& path) {
    double sum_sq = 0.0;
    for (const auto& s : history) {
        double min_d2 = std::numeric_limits<double>::max();
        for (const auto& p : path) {
            double dx = p.x - s.x;
            double dy = p.y - s.y;
            min_d2 = std::min(min_d2, dx * dx + dy * dy);
        }
        sum_sq += min_d2;
    }
    return std::sqrt(sum_sq / history.size());
}

double normalize_angle(double a) {
    using std::numbers::pi;
    a = std::fmod(a + pi, 2.0 * pi);
    if (a < 0.0) a += 2.0 * pi;
    return a - pi;
}
