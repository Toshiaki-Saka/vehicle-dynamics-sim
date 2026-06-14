#pragma once
#include "types.hpp"

// 楕円経路を生成(各点の接線方位・曲率付き)
Path make_ellipse_path(double a = 50.0, double b = 30.0, int n = 400);

// prev_idx を起点とするウィンドウ内の最近傍インデックスを返す
int find_nearest_in_window(double x, double y, const Path& path,
                            int prev_idx, int window = 50);

// 各状態から経路への最短距離の RMS を計算
double evaluate_rms(const std::vector<State3D>& history, const Path& path);

// 角度を [-pi, pi] に正規化
double normalize_angle(double a);
