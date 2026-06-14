#pragma once
#include "types.hpp"

// Nomoto 1次モデル + PD 制御による変針操船シミュレーション
ShipResult simulate_ship_nomoto(
    double K_nomoto       = 0.18,   // 舵効きゲイン [1/s]
    double T_nomoto       = 50.0,   // 時定数 [s]
    double psi_target_deg = 30.0,   // 目標方位 [deg]
    double Kp             = 2.0,    // PD 比例ゲイン
    double Kd             = 8.0,    // PD 微分ゲイン
    double v_ship         = 8.0,    // 船速 [m/s] (約16ノット)
    double t_end          = 600.0,
    int    n_steps        = 3000
);
