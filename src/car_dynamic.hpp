#pragma once
#include "types.hpp"

// 線形2自由度動力学自転車モデル(横速度・ヨーレート)のステップ応答を RK4 で計算。
// Python の scipy.signal.lti + step と等価な結果を返す。
CarDynResult simulate_car_dynamic(
    double m             = 1500.0,   // 車両質量 [kg]
    double Iz            = 2500.0,   // ヨー慣性 [kg m^2]
    double lf            = 1.2,      // 重心〜前車軸 [m]
    double lr            = 1.5,      // 重心〜後車軸 [m]
    double Cf            = 80000.0,  // 前輪コーナリングスティフネス [N/rad]
    double Cr            = 80000.0,  // 後輪コーナリングスティフネス [N/rad]
    double vx            = 20.0,     // 縦速度 [m/s]
    double delta_step_deg = 1.0,     // ステップ操舵角 [deg]
    double t_end         = 5.0,
    int    n_steps       = 1000
);
