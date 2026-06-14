#pragma once
#include "types.hpp"

// Boeing 747 縦運動 4 自由度線形モデルのエレベータステップ応答を RK4 で計算。
// 固有値(短周期・フゴイドモード)も同時に返す。
AircraftResult simulate_aircraft_longitudinal(
    double t_end        = 1500.0,  // シミュレーション時間 [s]
    int    n_steps      = 8000,
    double delta_e_deg  = -1.0    // エレベータ入力 [deg] (機首上げ方向)
);
