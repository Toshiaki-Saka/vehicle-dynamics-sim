"""
MPC 経路追従シミュレーション (Python に残す部分)
==============================================
scipy.optimize.minimize (SLSQP) を用いる制約付き最適制御。
C++ 側で実装すると外部 QP ソルバが必要になるため Python に残す。
"""

import sys
import json
import numpy as np
from scipy.optimize import minimize


# ── 楕円経路生成(C++ 側と同一パラメータ) ────────────────────────
def _make_ellipse_path(a=50.0, b=30.0, n=400):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    path_xy = np.column_stack([a * np.cos(t), b * np.sin(t)])
    dx, dy  = -a * np.sin(t), b * np.cos(t)
    yaw     = np.arctan2(dy, dx)
    return path_xy, yaw


def _evaluate_rms(states, path_xy):
    errors = []
    for s in states:
        d = np.hypot(path_xy[:, 0] - s[0], path_xy[:, 1] - s[1])
        errors.append(d.min())
    return float(np.sqrt(np.mean(np.array(errors) ** 2)))


# ── MPC コントローラ ─────────────────────────────────────────────
class MPCPathTracker:
    def __init__(self, wheelbase=2.7, horizon=15, dt_ctrl=0.1,
                 delta_max=np.deg2rad(35), delta_rate_max=np.deg2rad(40)):
        self.L              = wheelbase
        self.N              = horizon
        self.dt_ctrl        = dt_ctrl
        self.delta_max      = delta_max
        self.delta_rate_max = delta_rate_max
        self.prev_delta     = 0.0
        self.prev_idx       = 0
        self.warm_start     = None

    def _find_nearest(self, x, y, path_xy, window=80):
        n = len(path_xy)
        idx_range = np.arange(self.prev_idx, self.prev_idx + window) % n
        d2 = (path_xy[idx_range, 0] - x)**2 + (path_xy[idx_range, 1] - y)**2
        return idx_range[np.argmin(d2)]

    def _predict_cost(self, delta_seq, state, path_xy, path_yaw, v):
        x, y, psi = state
        prev_psi  = psi
        cost      = 0.0
        local_idx = self.prev_idx
        n_path    = len(path_xy)
        for k in range(self.N):
            x = x + v * np.cos(psi) * self.dt_ctrl
            y = y + v * np.sin(psi) * self.dt_ctrl
            psi_new = psi + (v / self.L) * np.tan(delta_seq[k]) * self.dt_ctrl
            psi_dot = (psi_new - prev_psi) / self.dt_ctrl
            prev_psi = psi
            psi = psi_new

            window   = np.arange(local_idx, local_idx + 20) % n_path
            d2       = (path_xy[window, 0] - x)**2 + (path_xy[window, 1] - y)**2
            local_idx = window[np.argmin(d2)]

            dx   = path_xy[local_idx, 0] - x
            dy_  = path_xy[local_idx, 1] - y
            e_y  = dx * (-np.sin(path_yaw[local_idx])) + dy_ * np.cos(path_yaw[local_idx])
            e_psi = (psi - path_yaw[local_idx] + np.pi) % (2 * np.pi) - np.pi

            cost += 30.0 * e_y**2
            cost += 30.0 * e_psi**2
            cost +=  5.0 * psi_dot**2
            cost +=  1.0 * delta_seq[k]**2
            prev_d = self.prev_delta if k == 0 else delta_seq[k - 1]
            cost +=  3.0 * (delta_seq[k] - prev_d)**2
        return cost

    def step(self, state, path_xy, path_yaw, v, dt):
        x, y, _ = state
        self.prev_idx = self._find_nearest(x, y, path_xy)

        delta0 = (np.concatenate([self.warm_start[1:], [self.warm_start[-1]]])
                  if self.warm_start is not None
                  else np.full(self.N, self.prev_delta))

        bounds = [(-self.delta_max, self.delta_max)] * self.N
        result = minimize(
            self._predict_cost, delta0,
            args=(state, path_xy, path_yaw, v),
            method='SLSQP', bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-5}
        )
        self.warm_start = result.x
        delta_opt = float(np.clip(
            result.x[0],
            self.prev_delta - self.delta_rate_max * dt,
            self.prev_delta + self.delta_rate_max * dt,
        ))
        self.prev_delta = delta_opt

        x, y, psi = state
        return (np.array([
            x   + v * np.cos(psi) * dt,
            y   + v * np.sin(psi) * dt,
            psi + (v / self.L) * np.tan(delta_opt) * dt,
        ]), delta_opt)


# ── メインシミュレーション ────────────────────────────────────────
def run_mpc_simulation(horizon=15, dt_mpc=0.1, t_end=50.0,
                       v=8.0, n_path=400):
    path_xy, path_yaw = _make_ellipse_path(50.0, 30.0, n_path)
    initial_state = np.array([50.0, 2.0, np.pi / 2])
    n_mpc = int(t_end / dt_mpc)
    time_arr = np.arange(n_mpc) * dt_mpc

    mpc = MPCPathTracker(wheelbase=2.7, horizon=horizon, dt_ctrl=dt_mpc)
    state = initial_state.copy()
    hist_s = np.zeros((n_mpc, 3))
    hist_d = np.zeros(n_mpc)

    for i in range(n_mpc):
        hist_s[i] = state
        state, d  = mpc.step(state, path_xy, path_yaw, v, dt_mpc)
        hist_d[i] = d
        if (i + 1) % 50 == 0:
            print(f"  MPC step {i+1}/{n_mpc}", flush=True)

    rms = _evaluate_rms(hist_s, path_xy)
    return {
        "time":          time_arr.tolist(),
        "x":             hist_s[:, 0].tolist(),
        "y":             hist_s[:, 1].tolist(),
        "psi":           hist_s[:, 2].tolist(),
        "delta":         hist_d.tolist(),
        "rms_error":     rms,
        "max_steer_deg": float(np.rad2deg(np.abs(hist_d).max())),
    }


if __name__ == "__main__":
    output_path = sys.argv[1] if len(sys.argv) > 1 else "mpc_results.json"
    print("MPC simulation started...")
    result = run_mpc_simulation()
    with open(output_path, "w") as f:
        json.dump(result, f)
    print(f"MPC RMS={result['rms_error']:.3f} m, "
          f"max steer={result['max_steer_deg']:.2f} deg")
    print(f"Written: {output_path}")
