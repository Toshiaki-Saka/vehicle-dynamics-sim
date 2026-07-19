"""
====================================================================
Integrated Motion Simulation of Cars, Aircraft, and Ships
====================================================================

This script simulates, in a unified framework, the motion models of the
three transportation systems covered in the preceding discussions, and
visualizes the results.

Included models:
  1. Car      : Kinematic bicycle model + Pure Pursuit path tracking
  2. Car      : Linear dynamic 2-DOF model (steering step response)
  3. Aircraft : Linear longitudinal motion model (short-period & phugoid modes)
  4. Ship     : Nomoto first-order model (course-change maneuvering)

Each model can be run independently, and all results are visualized together
at the end.

How to run:
    python vehicle_dynamics_simulation.py

Dependencies: numpy, matplotlib, scipy
====================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.integrate import odeint
from scipy.signal import lti, step

# Output directory: same location as the script (fall back to the current directory)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

# Font setup for Japanese display (automatic fallback depending on the environment)
def _setup_japanese_font():
    """Configure a Japanese font. Try the available candidates in order."""
    candidates = [
        'Noto Sans CJK JP', 'IPAexGothic', 'IPAGothic',
        'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'TakaoGothic',
        'VL Gothic', 'Source Han Sans JP', 'DejaVu Sans'
    ]
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            rcParams['font.family'] = name
            return name
    return 'DejaVu Sans'

_font_used = _setup_japanese_font()
rcParams['axes.unicode_minus'] = False  # Prevent the minus sign from becoming garbled


def print_introduction():
    """Explain the theoretical background of the whole script before the simulation starts."""
    print("\n" + "▓" * 70)
    print("  Integrated Motion Simulation of Cars, Aircraft, and Ships")
    print("▓" * 70)
    print()
    print("[Introduction] Why the Newton-Euler form rather than the Lagrangian equations")
    print("-" * 70)
    print("  The motion models of transportation systems are, almost without exception,")
    print("  written in Newton-Euler form rather than in the Lagrangian equations,")
    print("  the standard form of analytical mechanics. The reasons:")
    print()
    print("  1. Nonholonomic constraints")
    print("     A car's tires carry the 'no side-slip' constraint, but this is a")
    print("     constraint on velocities that cannot be integrated into a relation")
    print("     among coordinates. Handling it in the Lagrangian form requires the")
    print("     Lagrange-d'Alembert equations with undetermined multipliers, and the")
    print("     formulation abruptly becomes cumbersome.")
    print()
    print("  2. Non-conservative forces dominate")
    print("     Tire friction, aerodynamic drag, fluid drag, thrust, and other")
    print("     non-conservative forces determine the motion. In the L = T - V")
    print("     framework they are merely pushed onto the right-hand side as external")
    print("     force terms, and the appeal of the Lagrangian form (the elegance of")
    print("     motion emerging from symmetry and energy) is lost.")
    print()
    print("  3. Affinity with a control-oriented view")
    print("     Modern control theory demands the state-space representation")
    print("     dx/dt = f(x,u). Building the state equations from Newton-Euler is")
    print("     straightforward, whereas building them from the Lagrangian adds the")
    print("     extra step of solving M(q)q̈ for q̈.")
    print()
    print("  4. For multibody systems, Kane's method is superior")
    print("     For a full vehicle including suspension (typically 14 DOF), the")
    print("     partial derivatives of energy explode. Kane's method (a partial-")
    print("     velocity-based d'Alembert principle) is used inside commercial")
    print("     software such as CarSim.")
    print()
    print("  -> It is not that the Lagrangian form theoretically 'cannot be written';")
    print("     rather, in this problem domain it cedes its role to other tools")
    print("     (Newton-Euler, Kane, state space) — that is the accurate picture.")
    print()
    print("[Structure of this script]")
    print("-" * 70)
    print("  (1) Car - kinematic bicycle + Pure Pursuit (geometric path tracking)")
    print("  (2) Car - dynamic 2-DOF model (linear tire + step response)")
    print("  (3) Aircraft - longitudinal 4-DOF (short-period & phugoid modes)")
    print("  (4) Ship - Nomoto first-order model (course-change maneuvering + LOS guidance)")
    print()
    print("  At the start of each simulation, the theoretical background, model")
    print("  equations, and design philosophy are explained in detail.")
    print("▓" * 70)


# ====================================================================
# 1. Car: kinematic bicycle model + comparison of 4 control laws
#    (Pure Pursuit, Stanley, LQR, MPC)
# ====================================================================
class StanleyController:
    """
    Stanley control: a geometric control law developed by the Stanford team for the
    2005 DARPA Grand Challenge. It is referenced at the front wheel and uses the
    lateral error explicitly, so recovery is fast.

        δ = e_ψ + arctan(k · e_y / v_x)

    e_y is the (signed) lateral error from the front-wheel position to the nearest
    path point; e_ψ is the difference between the path tangent direction and the
    vehicle yaw angle.
    """

    def __init__(self, wheelbase=2.7, k_gain=2.5):
        self.L = wheelbase
        self.k = k_gain  # Lateral-error gain

    def step(self, state, target_path, path_yaw, v, dt, prev_idx=0):
        x, y, psi = state
        # Transform to the front-wheel position
        fx = x + self.L * np.cos(psi)
        fy = y + self.L * np.sin(psi)

        # Find the nearest-point index via a local search (supports a looping path)
        n_path = len(target_path)
        search_window = 50
        idx_range = np.arange(prev_idx, prev_idx + search_window) % n_path
        d2 = (target_path[idx_range, 0] - fx) ** 2 + (target_path[idx_range, 1] - fy) ** 2
        nearest = idx_range[np.argmin(d2)]

        # Lateral error (signed): project the vector from the front wheel to the
        # nearest path point onto the left normal of the path tangent
        # (-sin ψ_path, cos ψ_path).
        # Convention: e_y > 0 when the vehicle is on the right of the path, e_y < 0
        # when on the left. The Stanley law δ = e_ψ + arctan(k·e_y/v) is stable with
        # this sign convention.
        dx = target_path[nearest, 0] - fx
        dy = target_path[nearest, 1] - fy
        e_y = dx * (-np.sin(path_yaw[nearest])) + dy * np.cos(path_yaw[nearest])

        # Yaw error
        e_psi = path_yaw[nearest] - psi
        e_psi = (e_psi + np.pi) % (2 * np.pi) - np.pi

        # Stanley control law
        delta = e_psi + np.arctan2(self.k * e_y, max(v, 0.1))
        delta = np.clip(delta, -np.deg2rad(35), np.deg2rad(35))

        # Update the state with the kinematic bicycle model
        x_new = x + v * np.cos(psi) * dt
        y_new = y + v * np.sin(psi) * dt
        psi_new = psi + (v / self.L) * np.tan(delta) * dt

        return np.array([x_new, y_new, psi_new]), delta, nearest


class LQRPathTracker:
    """
    LQR controller for a 4th-order system built from a linear 2-DOF model plus the
    path deviation.

    State x = (e_y, ė_y, e_ψ, ė_ψ), input u = δ.
    K is obtained from the analytical solution of the Riccati equation
    (scipy.linalg.solve_continuous_are).
    """

    def __init__(self, wheelbase=2.7, m=1500.0, Iz=2500.0,
                 lf=1.2, lr=1.5, Cf=80000.0, Cr=80000.0, vx=8.0,
                 Q=None, R=None):
        self.L = wheelbase
        self.vx = vx
        self.Cf = Cf; self.Cr = Cr
        self.lf = lf; self.lr = lr
        self.m = m;   self.Iz = Iz

        # Continuous-time state-space model of the 4th-order system including path deviation
        # Rajamani "Vehicle Dynamics and Control" Eq.(2.45)-(2.49)
        A = np.array([
            [0, 1, 0, 0],
            [0, -(2*Cf + 2*Cr)/(m*vx),  (2*Cf + 2*Cr)/m, (-2*Cf*lf + 2*Cr*lr)/(m*vx)],
            [0, 0, 0, 1],
            [0, -(2*Cf*lf - 2*Cr*lr)/(Iz*vx), (2*Cf*lf - 2*Cr*lr)/Iz,
             -(2*Cf*lf**2 + 2*Cr*lr**2)/(Iz*vx)]
        ])
        B = np.array([[0], [2*Cf/m], [0], [2*Cf*lf/Iz]])

        if Q is None:
            Q = np.diag([10.0, 0.5, 10.0, 0.5])
        if R is None:
            R = np.array([[1.0]])

        from scipy.linalg import solve_continuous_are
        P = solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.solve(R, B.T @ P)  # K = R^{-1} B^T P
        self.A = A; self.B = B

    def step(self, state, target_path, path_yaw, path_curvature, v, dt, prev_idx=0):
        x, y, psi = state
        # Nearest point
        n_path = len(target_path)
        search_window = 50
        idx_range = np.arange(prev_idx, prev_idx + search_window) % n_path
        d2 = (target_path[idx_range, 0] - x)**2 + (target_path[idx_range, 1] - y)**2
        nearest = idx_range[np.argmin(d2)]

        # Compute the path deviation
        dx = x - target_path[nearest, 0]
        dy = y - target_path[nearest, 1]
        nx = -np.sin(path_yaw[nearest])
        ny =  np.cos(path_yaw[nearest])
        e_y = dx * nx + dy * ny
        e_psi = psi - path_yaw[nearest]
        e_psi = (e_psi + np.pi) % (2 * np.pi) - np.pi

        # 4th-order state (e_y_dot and e_psi_dot are reconstructed approximately)
        # Strictly, lateral velocity and yaw rate are needed, but here they are
        # approximated by 0 for simplicity
        x_state = np.array([e_y, 0.0, e_psi, 0.0])

        # Steering command = -K x + feedforward (curvature correction)
        delta_fb = -float((self.K @ x_state).item())
        # Feedforward: Ackermann steering for the curvature
        delta_ff = self.L * path_curvature[nearest]
        delta = delta_fb + delta_ff
        delta = np.clip(delta, -np.deg2rad(35), np.deg2rad(35))

        # State update
        x_new = x + v * np.cos(psi) * dt
        y_new = y + v * np.sin(psi) * dt
        psi_new = psi + (v / self.L) * np.tan(delta) * dt

        return np.array([x_new, y_new, psi_new]), delta, nearest


class MPCPathTracker:
    """
    Constrained linear MPC controller.

    The prediction model is the kinematic bicycle.
    Constraints: |δ| ≤ δ_max, |Δδ| ≤ Δδ_max.
    Solver: quadratic programming (scipy.optimize.minimize, SLSQP).

    Being able to handle constraints explicitly is the fundamental difference from
    LQR. In this implementation, the scheme extends Pure Pursuit with multi-point
    look-ahead so that several look-ahead points within the prediction horizon are
    tracked, imposing the constraints explicitly.
    """

    def __init__(self, wheelbase=2.7, horizon=10, dt_ctrl=0.1,
                 delta_max=np.deg2rad(35), delta_rate_max=np.deg2rad(40)):
        self.L = wheelbase
        self.N = horizon
        self.dt_ctrl = dt_ctrl
        self.delta_max = delta_max
        self.delta_rate_max = delta_rate_max
        self.prev_delta = 0.0
        self.prev_idx = 0
        self.warm_start = None

    def _find_nearest(self, x, y, target_path, search_window=80):
        """Find the nearest point via a windowed local search (monotonic progression)."""
        n_path = len(target_path)
        idx_range = np.arange(self.prev_idx, self.prev_idx + search_window) % n_path
        d2 = (target_path[idx_range, 0] - x)**2 + (target_path[idx_range, 1] - y)**2
        nearest = idx_range[np.argmin(d2)]
        return nearest

    def _predict_cost(self, delta_seq, state, target_path, path_yaw, v):
        """Evaluate the cost over the horizon."""
        x, y, psi = state
        prev_psi = psi
        cost = 0.0
        local_idx = self.prev_idx
        n_path = len(target_path)
        for k in range(self.N):
            # One-step prediction
            x = x + v * np.cos(psi) * self.dt_ctrl
            y = y + v * np.sin(psi) * self.dt_ctrl
            psi_new = psi + (v / self.L) * np.tan(delta_seq[k]) * self.dt_ctrl
            psi_dot = (psi_new - prev_psi) / self.dt_ctrl
            prev_psi = psi
            psi = psi_new

            # Nearest point on the path via local search
            window = np.arange(local_idx, local_idx + 20) % n_path
            d2 = (target_path[window, 0] - x)**2 + (target_path[window, 1] - y)**2
            local_idx = window[np.argmin(d2)]

            # Path deviation
            dx = target_path[local_idx, 0] - x
            dy = target_path[local_idx, 1] - y
            e_y = dx * (-np.sin(path_yaw[local_idx])) + dy * np.cos(path_yaw[local_idx])
            e_psi = (psi - path_yaw[local_idx] + np.pi) % (2 * np.pi) - np.pi

            # Cost function (balances lateral error, yaw error, yaw rate, and steering penalty)
            cost += 30.0 * e_y**2          # Lateral error
            cost += 30.0 * e_psi**2        # Yaw error (if the heading matches, tracking follows naturally)
            cost += 5.0 * psi_dot**2       # Yaw-rate suppression (prevents oscillation)
            cost += 1.0 * delta_seq[k]**2  # Steering penalty
            if k > 0:
                cost += 3.0 * (delta_seq[k] - delta_seq[k-1])**2
            else:
                cost += 3.0 * (delta_seq[k] - self.prev_delta)**2
        return cost

    def step(self, state, target_path, path_yaw, v, dt):
        from scipy.optimize import minimize
        x, y, psi = state
        # Update the nearest point for the current state
        self.prev_idx = self._find_nearest(x, y, target_path)

        # Warm start
        if self.warm_start is not None:
            delta0 = np.concatenate([self.warm_start[1:], [self.warm_start[-1]]])
        else:
            delta0 = np.full(self.N, self.prev_delta)

        bounds = [(-self.delta_max, self.delta_max)] * self.N

        result = minimize(
            self._predict_cost, delta0,
            args=(state, target_path, path_yaw, v),
            method='SLSQP', bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-5}
        )
        self.warm_start = result.x
        delta_opt = result.x[0]
        # Hard constraint: steering rate
        delta_opt = np.clip(delta_opt,
                            self.prev_delta - self.delta_rate_max * dt,
                            self.prev_delta + self.delta_rate_max * dt)
        self.prev_delta = delta_opt

        # State update (on the actual-vehicle plant side)
        x, y, psi = state
        x_new = x + v * np.cos(psi) * dt
        y_new = y + v * np.sin(psi) * dt
        psi_new = psi + (v / self.L) * np.tan(delta_opt) * dt
        return np.array([x_new, y_new, psi_new]), delta_opt


class KinematicBicyclePurePursuit:
    """
    Track a path with Pure Pursuit using a rear-wheel-referenced kinematic bicycle model.

    State : (x, y, psi)
    Input : speed v, steering angle delta (determined by Pure Pursuit)
    """

    def __init__(self, wheelbase=2.7, lookahead_gain=0.5, lookahead_min=2.0):
        self.L = wheelbase            # Wheelbase [m] (passenger-car class)
        self.k = lookahead_gain       # Speed coefficient of the look-ahead distance
        self.Ld_min = lookahead_min   # Minimum look-ahead distance [m]

    def step(self, state, target_path, v, dt, prev_idx=0):
        """One-step update. target_path is an array of (x_path, y_path).
        prev_idx is the path index followed on the previous step (supports a looping path)."""
        x, y, psi = state
        Ld = max(self.k * v, self.Ld_min)

        # First find the nearest point (local search around prev_idx to prevent backtracking)
        n_path = len(target_path)
        search_window = 50
        idx_range = np.arange(prev_idx, prev_idx + search_window) % n_path
        local_dx = target_path[idx_range, 0] - x
        local_dy = target_path[idx_range, 1] - y
        local_dist = np.hypot(local_dx, local_dy)
        nearest_in_window = idx_range[np.argmin(local_dist)]

        # Advance along the path from the nearest point to find a point at least the look-ahead distance away
        target_idx = nearest_in_window
        for k in range(n_path):
            cand = (nearest_in_window + k) % n_path
            d = np.hypot(target_path[cand, 0] - x, target_path[cand, 1] - y)
            if d >= Ld:
                target_idx = cand
                break

        gx, gy = target_path[target_idx]

        # Angle to the target point alpha in the vehicle coordinate frame
        alpha = np.arctan2(gy - y, gx - x) - psi
        # Normalize alpha to [-pi, pi]
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi

        # Pure Pursuit steering law
        delta = np.arctan2(2.0 * self.L * np.sin(alpha), Ld)
        # Physical steering-angle limit (±35 degrees)
        delta = np.clip(delta, -np.deg2rad(35), np.deg2rad(35))

        # Discretization of the kinematic bicycle model (Euler method)
        x_new = x + v * np.cos(psi) * dt
        y_new = y + v * np.sin(psi) * dt
        psi_new = psi + (v / self.L) * np.tan(delta) * dt

        return np.array([x_new, y_new, psi_new]), delta, nearest_in_window


def _make_ellipse_path(a=50.0, b=30.0, n=400):
    """Generate an elliptical path together with the path tangent heading and curvature at each point."""
    t = np.linspace(0, 2 * np.pi, n)
    path = np.column_stack([a * np.cos(t), b * np.sin(t)])
    # Path tangent direction (direction of travel)
    dx = -a * np.sin(t)
    dy =  b * np.cos(t)
    yaw = np.arctan2(dy, dx)
    # Curvature κ = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
    ddx = -a * np.cos(t)
    ddy = -b * np.sin(t)
    curvature = (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5
    return path, yaw, curvature


def _evaluate_rms(history_state, path):
    errors = []
    for s in history_state:
        d = np.hypot(path[:, 0] - s[0], path[:, 1] - s[1])
        errors.append(d.min())
    return np.sqrt(np.mean(np.array(errors) ** 2))


def simulate_car_pure_pursuit():
    """Compare the four path-tracking control laws (Pure Pursuit, Stanley, LQR, MPC) on the same path."""
    print("\n" + "=" * 70)
    print("[Simulation 1] Car: comparison of four path-tracking control laws")
    print("=" * 70)
    print()
    print("- Theoretical background: kinematic bicycle model and Ackermann geometry")
    print("-" * 70)
    print("  In the low-speed regime the vehicle is approximated as a 'bicycle'. The")
    print("  left and right wheels are lumped to the center, and under the rear-wheel-")
    print("  referenced, no-side-slip constraint, three state equations are obtained:")
    print()
    print("       ẋ = v cos ψ")
    print("       ẏ = v sin ψ")
    print("       ψ̇ = (v/L) tan δ        <- the Ackermann geometry itself")
    print()
    print("  Here L is the wheelbase and δ the front-wheel steering angle. The third")
    print("  equation follows directly from the Ackermann condition tan δ = L/R, in")
    print("  which the left and right front wheels share a common turning center.")
    print("  In a real vehicle the left/right steering angles are geometrically")
    print("  approximated by the steering mechanism (tie rods) so as to satisfy")
    print("  cot δ_o - cot δ_i = w/L.")
    print()
    print("- The four path-tracking control laws")
    print("-" * 70)
    print("  (1) Pure Pursuit (classical geometric control, rear-wheel referenced)")
    print("       δ = arctan(2 L sin α / Ld)")
    print("       A staple since the Stanford DARPA Grand Challenge era.")
    print("       α is the angle to the target point in the vehicle frame; Ld is the look-ahead distance.")
    print("       One line to implement, minimal computation, tuning of Ld only.")
    print()
    print("  (2) Stanley (classical geometric control, front-wheel referenced)")
    print("       δ = e_ψ + arctan(k · e_y / v_x)")
    print("       The Stanford entry that won the 2005 DARPA Grand Challenge.")
    print("       It uses the lateral error e_y explicitly, so path recovery is fast.")
    print()
    print("  (3) LQR (optimal control, unconstrained, analytical solution)")
    print("       Minimize J = ∫(xᵀQx + uᵀRu)dt; from the solution of the Riccati")
    print("       equation obtain u = -Kx + δ_ff (curvature feedforward).")
    print("       Handles the 4th-order state x = (e_y, ė_y, e_ψ, ė_ψ) explicitly.")
    print()
    print("  (4) MPC (optimal control, constrained, look-ahead)")
    print("       Finite-horizon optimization can handle steering-rate and steering-angle limits explicitly.")
    print("       This implementation uses 10-step look-ahead + an SLSQP solver.")
    print("       The mainstream for path tracking in LKAS and autonomous driving.")
    print()
    print("- Simulation setup")
    print("-" * 70)
    print("  Path  : elliptical circuit (major axis 50 m, minor axis 30 m)")
    print("  Speed : 8.0 m/s, start position: offset +2 m laterally from the path")
    print("  All control laws are compared with the same path, initial condition, speed, and vehicle parameters.")
    print()

    # Share the path and initial condition
    path, path_yaw, path_curvature = _make_ellipse_path(a=50.0, b=30.0, n=400)
    initial_state = np.array([50.0, 2.0, np.pi / 2])  # offset +2 m laterally
    v = 8.0
    dt = 0.05
    T_end = 50.0
    n_steps = int(T_end / dt)
    time = np.arange(n_steps) * dt

    results = {}

    # (1) Pure Pursuit
    print("  (1) Running Pure Pursuit...", end=' ')
    car_pp = KinematicBicyclePurePursuit(wheelbase=2.7, lookahead_gain=0.6, lookahead_min=4.0)
    state = initial_state.copy()
    hist_s = np.zeros((n_steps, 3)); hist_d = np.zeros(n_steps); idx = 0
    for i in range(n_steps):
        hist_s[i] = state
        state, d, idx = car_pp.step(state, path, v, dt, idx)
        hist_d[i] = d
    rms = _evaluate_rms(hist_s, path)
    results['pure_pursuit'] = {'state': hist_s, 'delta': hist_d, 'rms': rms}
    print(f"done. RMS error = {rms:.3f} m, max steering = {np.rad2deg(np.abs(hist_d).max()):.2f}°")

    # (2) Stanley
    print("  (2) Running Stanley      ...", end=' ')
    car_st = StanleyController(wheelbase=2.7, k_gain=2.5)
    state = initial_state.copy()
    hist_s = np.zeros((n_steps, 3)); hist_d = np.zeros(n_steps); idx = 0
    for i in range(n_steps):
        hist_s[i] = state
        state, d, idx = car_st.step(state, path, path_yaw, v, dt, idx)
        hist_d[i] = d
    rms = _evaluate_rms(hist_s, path)
    results['stanley'] = {'state': hist_s, 'delta': hist_d, 'rms': rms}
    print(f"done. RMS error = {rms:.3f} m, max steering = {np.rad2deg(np.abs(hist_d).max()):.2f}°")

    # (3) LQR
    print("  (3) Running LQR          ...", end=' ')
    car_lqr = LQRPathTracker(wheelbase=2.7, vx=v)
    state = initial_state.copy()
    hist_s = np.zeros((n_steps, 3)); hist_d = np.zeros(n_steps); idx = 0
    for i in range(n_steps):
        hist_s[i] = state
        state, d, idx = car_lqr.step(state, path, path_yaw, path_curvature, v, dt, idx)
        hist_d[i] = d
    rms = _evaluate_rms(hist_s, path)
    results['lqr'] = {'state': hist_s, 'delta': hist_d, 'rms': rms,
                      'gain': car_lqr.K}
    print(f"done. RMS error = {rms:.3f} m, max steering = {np.rad2deg(np.abs(hist_d).max()):.2f}°")

    # (4) MPC (computationally heavy, so sample with a coarser control period)
    print("  (4) Running MPC          ...", end=' ')
    car_mpc = MPCPathTracker(wheelbase=2.7, horizon=15, dt_ctrl=0.1)
    dt_mpc = 0.1  # MPC updates at a 100 ms period (realistic computational load)
    n_mpc = int(T_end / dt_mpc)
    state = initial_state.copy()
    hist_s = np.zeros((n_mpc, 3)); hist_d = np.zeros(n_mpc)
    for i in range(n_mpc):
        hist_s[i] = state
        state, d = car_mpc.step(state, path, path_yaw, v, dt_mpc)
        hist_d[i] = d
    rms = _evaluate_rms(hist_s, path)
    results['mpc'] = {'state': hist_s, 'delta': hist_d, 'rms': rms,
                      'time': np.arange(n_mpc) * dt_mpc}
    print(f"done. RMS error = {rms:.3f} m, max steering = {np.rad2deg(np.abs(hist_d).max()):.2f}°")

    print()
    print("- Simulation results summary")
    print("-" * 70)
    print(f"  {'Control law':<15} {'RMS err [m]':>12} {'Max steer [°]':>15} {'Characteristics'}")
    print(f"  {'-'*15} {'-'*12} {'-'*15} {'-'*30}")
    print(f"  {'Pure Pursuit':<15} {results['pure_pursuit']['rms']:>12.3f}"
          f" {np.rad2deg(np.abs(results['pure_pursuit']['delta']).max()):>15.2f}"
          f"   Look-ahead arc tracking")
    print(f"  {'Stanley':<15} {results['stanley']['rms']:>12.3f}"
          f" {np.rad2deg(np.abs(results['stanley']['delta']).max()):>15.2f}"
          f"   Fast recovery using lateral error explicitly")
    print(f"  {'LQR':<15} {results['lqr']['rms']:>12.3f}"
          f" {np.rad2deg(np.abs(results['lqr']['delta']).max()):>15.2f}"
          f"   Riccati analytical solution + FF correction")
    print(f"  {'MPC':<15} {results['mpc']['rms']:>12.3f}"
          f" {np.rad2deg(np.abs(results['mpc']['delta']).max()):>15.2f}"
          f"   Constrained look-ahead optimization")
    print()
    print(f"  -> Differences in the characteristics of each control law:")
    print(f"    Pure Pursuit tends to cut to the inside due to the look-ahead lag")
    print(f"    Stanley recovers lateral error quickly, with the smallest steady-state tracking error")
    print(f"    LQR produces continuous, smooth steering and settles onto the path in steady state")
    print(f"    MPC handles constraints explicitly, so the steering rate stays gentle")

    return {
        'time': time,
        'path': path,
        'results': results,
        'state': results['pure_pursuit']['state'],   # for compatibility
        'delta': results['pure_pursuit']['delta'],   # for compatibility
        'rms_error': results['pure_pursuit']['rms'], # for compatibility
    }


# ====================================================================
# 2. Car: dynamic 2-DOF model (steering step response)
# ====================================================================
def simulate_car_dynamic_bicycle():
    """
    Compute the response to a step steering input with a linear dynamic bicycle model.
    State x = (lateral velocity v_y, yaw rate psi_dot), input u = steering angle delta.
    """
    print("\n" + "=" * 70)
    print("[Simulation 2] Car: dynamic 2-DOF model step response")
    print("=" * 70)
    print()
    print("- Theoretical background: why the kinematic model is not enough")
    print("-" * 70)
    print("  In the low-speed regime the tires do not side-slip, and the Ackermann")
    print("  geometry was sufficient. In the high-speed regime (high lateral-")
    print("  acceleration regime), however, the tires take a slip angle α and")
    print("  generate a lateral force F_y = -C_α · α (linear region).")
    print("  This lateral force makes the vehicle drift sideways, and the phenomenon")
    print("  of a delayed yaw response appears.")
    print()
    print("- Linear 2-DOF model (the standard form of vehicle dynamics)")
    print("-" * 70)
    print("  Obtained by linearizing Newton-Euler (constant longitudinal speed vx, small-angle approximation):")
    print()
    print("    m(v̇_y + vx·ψ̇) = F_yf cos δ + F_yr")
    print("    Iz·ψ̈           = ℓf·F_yf cos δ - ℓr·F_yr")
    print()
    print("  With state x = (v_y, ψ̇)ᵀ and input u = δ, this becomes a linear time-invariant system ẋ = Ax + Bu.")
    print("  This is the basic form of vehicle motion control (Rajamani textbook, Ch.2).")
    print()
    print("- The Pacejka Magic Formula and the friction circle (considerations at the limit)")
    print("-" * 70)
    print("  Real tire lateral force saturates. The empirical Pacejka Magic Formula:")
    print()
    print("    F_y = D sin[ C arctan{ Bα - E(Bα - arctan(Bα)) } ]")
    print()
    print("  At small slip angles F_y ≈ -C_α·α (linear), but it decreases after the peak.")
    print("  Furthermore, the longitudinal and lateral forces must satisfy the friction-")
    print("  circle constraint √(F_x² + F_y²) ≤ μF_z (the same budget is shared between")
    print("  longitudinal and lateral). This is why accelerating too much mid-corner")
    print("  causes understeer, and applying the brakes causes tuck-in.")
    print()
    print("- Stability factor Kv and under/oversteer")
    print("-" * 70)
    print("  Yaw gain in steady cornering: ψ̇/δ = vx / [L(1 + Kv·vx²)]")
    print("    Kv = (ℓr·m)/(2L·Cf) - (ℓf·m)/(2L·Cr)")
    print()
    print("  Kv > 0 : understeer (harder to turn at high speed) - safe for a passenger car")
    print("  Kv = 0 : neutral")
    print("  Kv < 0 : oversteer (tucks in at high speed) - prone to spinning, dangerous")
    print()
    print("- Path-tracking control: LQR and MPC")
    print("-" * 70)
    print("  With a 4th-order system that adds the path deviation (e_y, e_ψ) to this linear model:")
    print("    LQR: minimize J = ∫(xᵀQx + uᵀRu)dt; analytical solution via the Riccati equation")
    print("    MPC: constrained finite-horizon optimization, solving a QP each step")
    print("         |δ| ≤ δ_max, |Δδ| ≤ Δδ_max, |α| ≤ α_lin, etc.")
    print()
    print("  LQR is light to compute and offers a stability guarantee; MPC handles")
    print("  constraints and look-ahead (curvature preview). Commercial autonomous")
    print("  driving (LKAS, ACC) is built from combinations of these.")
    print()
    print("- Simulation setup")
    print("-" * 70)
    print("  State : lateral velocity v_y, yaw rate ψ_dot")
    print("  Input : step steering 1.0 degree")
    print("  Speed : 20 m/s (about 72 km/h)")
    print()

    # Parameters of a typical sedan
    m  = 1500.0     # Vehicle mass [kg]
    Iz = 2500.0     # Yaw inertia [kg m^2]
    lf = 1.2        # Distance from CG to front axle [m]
    lr = 1.5        # Distance from CG to rear axle [m]
    Cf = 80000.0    # Front cornering stiffness [N/rad]
    Cr = 80000.0    # Rear cornering stiffness [N/rad]
    vx = 20.0       # Longitudinal speed [m/s] (about 72 km/h)

    # A, B matrices derived in the previous discussion
    a11 = -(Cf + Cr) / (m * vx)
    a12 = -(vx + (lf * Cf - lr * Cr) / (m * vx))
    a21 = -(lf * Cf - lr * Cr) / (Iz * vx)
    a22 = -(lf * lf * Cf + lr * lr * Cr) / (Iz * vx)
    A = np.array([[a11, a12], [a21, a22]])
    B = np.array([[Cf / m], [lf * Cf / Iz]])
    C = np.eye(2)
    D = np.zeros((2, 1))

    # Compute the step response
    sys = lti(A, B, C, D)
    t = np.linspace(0, 5.0, 1000)
    delta_step = np.deg2rad(1.0)
    t_out, y_out = step(sys, T=t)
    y_out = y_out * delta_step  # convert to a 1-degree step

    # Compute the stability factor and understeer gradient (for reference)
    Kv = (lr * m) / (2 * (lf + lr) * Cf) - (lf * m) / (2 * (lf + lr) * Cr)

    # Theoretical value of the steady-state yaw gain
    L = lf + lr
    yaw_gain_theory = vx / (L * (1 + Kv * vx ** 2))
    yaw_steady = y_out[-1, 1]
    yaw_gain_sim = yaw_steady / delta_step

    print(f"- Simulation results")
    print("-" * 70)
    judge = 'understeer' if Kv > 0 else 'oversteer' if Kv < 0 else 'neutral'
    print(f"  Stability factor Kv = {Kv:.5f}  ({judge})")
    print(f"  Steady-state yaw rate  = {np.rad2deg(yaw_steady):.3f} deg/s (for 1 degree of steering)")
    print(f"  Yaw gain: theory {yaw_gain_theory:.3f} vs simulation {yaw_gain_sim:.3f}")
    print(f"  -> A fast response in which the yaw rate settles in about 0.1 to 0.3 s (this is characteristic of a car)")
    print(f"  -> ESC intervenes by watching the error between this 'intended yaw rate' and the actual yaw rate")
    print(f"     (oversteer -> brake the outer front wheel, understeer -> brake the inner rear wheel)")

    return {
        'time': t_out,
        'vy': y_out[:, 0],
        'yaw_rate': y_out[:, 1],
        'Kv': Kv,
        'vx': vx,
    }


# ====================================================================
# 3. Aircraft: linear longitudinal motion model (short-period & phugoid)
# ====================================================================
def simulate_aircraft_longitudinal():
    """
    Compute the elevator step response with a linear model of the aircraft's longitudinal motion.
    """
    print("\n" + "=" * 70)
    print("[Simulation 3] Aircraft: linear longitudinal motion model")
    print("=" * 70)
    print()
    print("- Theoretical background: full 6-DOF rigid-body motion")
    print("-" * 70)
    print("  An aircraft undergoes free-space motion unconstrained by the ground, so")
    print("  unlike a car it has no nonholonomic constraints. Instead, the 6-DOF")
    print("  rigid-body motion is written in Newton-Euler form in the body frame")
    print("  (with Coriolis terms):")
    print()
    print("    m(u̇ + qw - rv) = Fx - mg sin θ")
    print("    m(v̇ + ru - pw) = Fy + mg cos θ sin φ")
    print("    m(ẇ + pv - qu) = Fz + mg cos θ cos φ")
    print("    Ix·ṗ - (Iy-Iz)qr = L     (roll:  controlled by aileron δ_a)")
    print("    Iy·q̇ - (Iz-Ix)rp = M     (pitch: controlled by elevator δ_e)")
    print("    Iz·ṙ - (Ix-Iy)pq = N     (yaw:   controlled by rudder   δ_r)")
    print()
    print("  The external forces (F) and moments (L,M,N) are determined by aerodynamics:")
    print("    F = (1/2)ρV²·S·C_*(α, β, q, δ_a, δ_e, δ_r, ...)")
    print("  The coefficients C_* are identified from wind tunnel, CFD, and flight tests.")
    print()
    print("- Separation of longitudinal and lateral/directional motion")
    print("-" * 70)
    print("  Because of the symmetric airframe shape, linearization separates the motion into two:")
    print()
    print("    Longitudinal : (Δu, w, q, θ)  <- operated by elevator δ_e")
    print("                   oscillation of pitch, speed, altitude, angle of attack")
    print("    Lateral      : (β, p, r, φ)   <- operated by aileron δ_a + rudder δ_r")
    print("                   coupling of roll, yaw, sideslip, and bank angle")
    print()
    print("  This simulation treats the longitudinal motion (Boeing 747 cruise condition).")
    print()
    print("- The two natural modes of longitudinal motion")
    print("-" * 70)
    print("  In the eigenvalue analysis of the linear system ẋ = Ax + Bu, two complex-conjugate pairs always appear:")
    print()
    print("    Short-period mode:")
    print("      period 1-5 s, heavily damped, fast oscillation of pitch angle and angle of attack")
    print("      directly tied to the pilot's handling feel; a basic index of airframe stability")
    print()
    print("    Phugoid mode:")
    print("      period 30-100 s, lightly damped, long-period oscillation exchanging speed and altitude")
    print("      corresponds to the exchange between potential and kinetic energy")
    print("      that 'gentle rise-and-fall' motion of an airplane")
    print()
    print("  For a car, 'natural modes' almost never become an issue, but for an")
    print("  aircraft they are a central design concern. Stability augmentation systems")
    print("  (SAS), control augmentation systems (CAS), and autopilots shape these modes.")
    print()
    print("- Contrast with the car")
    print("-" * 70)
    print("  Car      : planar 3-DOF (or 14), nonholonomic constraints, tire-friction dominated")
    print("             response time 0.1 s to a few s, natural modes not particularly an issue")
    print("  Aircraft : 6-DOF, free space, aerodynamics dominated")
    print("             response time a few s to minutes, natural modes are a design target")
    print()
    print("  Both share the structure of Newton-Euler + an external-force model.")
    print()
    print("- Simulation setup")
    print("-" * 70)
    print("  Airframe model : Boeing 747 class 4-DOF longitudinal motion (Etkin & Reid textbook)")
    print("  Flight condition : cruise (altitude 12,200 m, V = 235 m/s, M ≈ 0.8)")
    print("  State          : (Δu, w, q, θ)")
    print("  Input          : elevator step -1.0 degree (nose-up direction)")
    print()

    # Boeing 747 cruise condition (altitude 12200 m, speed 235 m/s ≈ M0.8)
    # Representative values from the Etkin & Reid "Dynamics of Flight" textbook (slightly adapted)
    # State order: [u (m/s), w (m/s), q (rad/s), theta (rad)]
    # In the aviation industry it is standard to use the vertical velocity component w
    # instead of the angle of attack alpha
    A = np.array([
        [-0.00643,   0.0263,    0.0,      -9.81  ],
        [-0.0941,   -0.624,   235.0,       0.0   ],
        [-0.000222, -0.00153, -0.668,      0.0   ],
        [ 0.0,       0.0,      1.0,        0.0   ]
    ])
    B = np.array([
        [ 0.0    ],
        [-32.7   ],
        [-2.08   ],
        [ 0.0    ]
    ])
    C = np.eye(4)
    D = np.zeros((4, 1))

    # Eigenvalue analysis (short-period & phugoid modes)
    eigvals = np.linalg.eigvals(A)
    print(f"- Simulation results")
    print("-" * 70)
    print(f"  Eigenvalues of the system:")
    for ev in eigvals:
        if abs(ev.imag) > 1e-6:
            wn = abs(ev)
            zeta = -ev.real / wn
            T_period = 2 * np.pi / abs(ev.imag) if abs(ev.imag) > 1e-6 else float('inf')
            print(f"     {ev.real:+.4f} ± {abs(ev.imag):.4f}j  "
                  f"(ωn={wn:.3f} rad/s, ζ={zeta:.3f}, period={T_period:.1f}s)")

    # Mode classification: short period -> short-period, long period -> phugoid
    complex_pairs = [ev for ev in eigvals if ev.imag > 1e-6]
    if len(complex_pairs) >= 2:
        sorted_pairs = sorted(complex_pairs, key=lambda e: abs(e.imag), reverse=True)
        sp = sorted_pairs[0]; ph = sorted_pairs[1]
        print(f"  -> Short-period mode : period {2*np.pi/abs(sp.imag):.2f} s (fast, heavily damped)")
        print(f"  -> Phugoid mode      : period {2*np.pi/abs(ph.imag):.2f} s (slow, lightly damped)")
        print(f"  -> Period ratio about {abs(ph.imag)/abs(sp.imag) if False else 2*np.pi/abs(ph.imag) / (2*np.pi/abs(sp.imag)):.0f} times.")
        print(f"     The short period damps out in a few seconds, while the phugoid oscillates slowly")
        print(f"     over a long period, so in the plot we observe both a short and a long time axis.")

    # Step response (elevator -1 degree -> nose up)
    sys = lti(A, B, C, D)
    t = np.linspace(0, 1500.0, 8000)  # long enough to see the phugoid's long period (about 850 s)
    t_out, y_out = step(sys, T=t)
    y_out = y_out * np.deg2rad(-1.0)

    print(f"  1500-second simulation complete")

    return {
        'time': t_out,
        'u_pert': y_out[:, 0],
        'w': y_out[:, 1],
        'q': y_out[:, 2],
        'theta': y_out[:, 3],
        'eigvals': eigvals,
    }


# ====================================================================
# 4. Ship: course-change maneuvering with the Nomoto first-order model
# ====================================================================
def simulate_ship_nomoto():
    """
    Nomoto first-order model: T*ψ_ddot + ψ_dot = K * δ
    """
    print("\n" + "=" * 70)
    print("[Simulation 4] Ship: course change with the Nomoto first-order model")
    print("=" * 70)
    print()
    print("- Theoretical background: Fossen's standard model")
    print("-" * 70)
    print("  A ship is also 6-DOF, but there is traditional naming:")
    print("    Surge (fore-aft) / Sway (port-starboard) / Heave (up-down)")
    print("    Roll / Pitch / Yaw (heading)")
    print()
    print("  Thor I. Fossen's (NTNU) standard model is the industry standard:")
    print()
    print("    M·ν̇ + C(ν)ν + D(ν)ν + g(η) = τ + τ_wind + τ_wave")
    print()
    print("    η : position and attitude in the inertial frame (x, y, z, φ, θ, ψ)ᵀ")
    print("    ν : velocity and angular velocity in the body frame (u, v, w, p, q, r)ᵀ")
    print("    M = M_RB + M_A : rigid-body inertia + added mass")
    print("    D(ν)         : fluid drag (linear + quadratic)")
    print("    g(η)         : restoring forces and moments")
    print()
    print("- Major differences from cars and aircraft")
    print("-" * 70)
    print("  1. Added mass")
    print("     Accelerating a body underwater also accelerates the surrounding water")
    print("     -> an apparent increase in mass. In aviation the air density is small")
    print("     and it can be neglected, but in water M_A becomes the same order as")
    print("     M_RB. This is one reason the ship's response is slow.")
    print()
    print("  2. Quadratic nature of fluid drag")
    print("     Tire friction is widely handled with a linear approximation, but fluid")
    print("     drag is dominated by the square of the speed. A term D_q ∝ |ν| is essential.")
    print()
    print("  3. Restoring forces")
    print("     From the relative position of the center of buoyancy B and the center of")
    print("     gravity G, restoring moments arise in roll and pitch. GM_T (transverse")
    print("     metacentric height) and GM_L (longitudinal metacentric height) are the")
    print("     basic quantities that determine stability, corresponding to a car's CG")
    print("     height and track ratio.")
    print()
    print("  4. Slow maneuverability and underactuation")
    print("     Against 6 DOF there is only a propeller + rudder (+ thrusters). The")
    print("     response time is also orders of magnitude slower (on the order of tens of seconds to minutes).")
    print()
    print("- The Nomoto first-order model (a practical form for course-change control)")
    print("-" * 70)
    print("  A simplified model widely used for course-keeping and course-change control:")
    print()
    print("    T·ψ̈ + ψ̇ = K·δ")
    print()
    print("    T : time constant (on the order of seconds to minutes, depending on ship size)")
    print("    K : rudder-effectiveness gain (steady turning gain per rudder angle)")
    print("    δ : rudder angle")
    print()
    print("  A staple for ship path tracking is this model + Line-of-Sight (LOS)")
    print("  guidance + PID/LQR. In this simulation, a course change to a target")
    print("  heading of 30 degrees is done with PD control.")
    print()
    print("- Comparison of the response time scales of the three transportation systems")
    print("-" * 70)
    print("  Car yaw response       : 0.1-1 s")
    print("  Aircraft short period  : 1-5 s")
    print("  Aircraft phugoid       : 30-100 s (about 850 s in this case)")
    print("  Ship course change     : tens of seconds to minutes")
    print()
    print("  While all handle the same rigid-body motion, the time scale varies over")
    print("  three orders of magnitude due to differences in medium and inertia.")
    print()
    print("- Simulation setup")
    print("-" * 70)
    print("  Ship type : mid-size container ship (about 200 m class)")
    print("  T = 50 s, K = 0.18 (1/s)")
    print("  Target heading : 30 degrees, PD control + rudder-angle limit ±35 degrees")
    print()

    # Representative ship parameters
    K_nomoto = 0.18    # Rudder-effectiveness gain [1/s] (heading response gain)
    T_nomoto = 50.0    # Time constant [s] (large due to the ship's inertia)

    # Course change to a target heading of 30 degrees with course-keeping PD control
    psi_target = np.deg2rad(30.0)
    Kp = 2.0
    Kd = 8.0
    delta_max = np.deg2rad(35.0)  # Physical rudder-angle limit

    def dynamics(state, t):
        psi, psi_dot = state
        # Compute the rudder-angle command with PD control
        e = psi_target - psi
        delta_cmd = Kp * e - Kd * psi_dot
        delta = np.clip(delta_cmd, -delta_max, delta_max)
        # Express the Nomoto first-order equation as a system of first-order ODEs
        psi_ddot = (K_nomoto * delta - psi_dot) / T_nomoto
        return [psi_dot, psi_ddot]

    t = np.linspace(0, 600.0, 3000)  # 10-minute simulation
    sol = odeint(dynamics, [0.0, 0.0], t)
    psi_hist = sol[:, 0]
    psi_dot_hist = sol[:, 1]

    # Reconstruct the rudder-angle history
    delta_hist = np.zeros_like(t)
    for i, ti in enumerate(t):
        e = psi_target - psi_hist[i]
        delta_cmd = Kp * e - Kd * psi_dot_hist[i]
        delta_hist[i] = np.clip(delta_cmd, -delta_max, delta_max)

    # Evaluate the settling time (the time it enters within ±5% of the target)
    tol = np.deg2rad(1.5)
    settled = np.where(np.abs(psi_hist - psi_target) < tol)[0]
    print(f"- Simulation results")
    print("-" * 70)
    if len(settled) > 0:
        t_settle = t[settled[0]]
        for idx in settled:
            if np.all(np.abs(psi_hist[idx:] - psi_target) < tol):
                t_settle = t[idx]
                break
        print(f"  Settling time (convergence within ±1.5 degrees): {t_settle:.1f} s")
    else:
        print(f"  Did not settle (did not converge within the time)")
    print(f"  Nomoto time constant T = {T_nomoto} s, rudder-effectiveness gain K = {K_nomoto}")
    print(f"  Maximum rudder angle   = {np.rad2deg(np.abs(delta_hist).max()):.1f} degrees")
    print(f"  -> Rudder saturates (35 degrees) at the initial motion, then settles while decaying.")
    print(f"  -> Against a car's yaw response (<1 s), the ship takes more than 2 minutes.")
    print(f"     The effect of added mass, fluid damping, and the large hull inertia.")

    # Generate the hull trajectory (assuming a constant longitudinal speed of 8 m/s)
    v_ship = 8.0  # about 16 knots
    x_ship = np.zeros_like(t)
    y_ship = np.zeros_like(t)
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        x_ship[i] = x_ship[i-1] + v_ship * np.cos(psi_hist[i-1]) * dt
        y_ship[i] = y_ship[i-1] + v_ship * np.sin(psi_hist[i-1]) * dt

    return {
        'time': t,
        'psi': psi_hist,
        'psi_target': psi_target,
        'delta': delta_hist,
        'x': x_ship,
        'y': y_ship,
    }


# ====================================================================
# Visualization
# ====================================================================
def visualize_all(car_pp, car_dyn, aircraft, ship):
    """Visualize the four simulation results together in a single figure."""
    print("\n" + "=" * 66)
    print("[Visualization] Plotting all simulation results")
    print("=" * 66)
    print(f"  Font used: {_font_used}")

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle('Integrated Motion Simulation Results of Cars, Aircraft, and Ships',
                 fontsize=16, fontweight='bold', y=0.995)

    # ---- (1) Car: trajectory comparison of the 4 control laws ----
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(car_pp['path'][:, 0], car_pp['path'][:, 1],
             'k--', lw=1.2, label='Reference path', alpha=0.5)
    colors = {'pure_pursuit': '#1f77b4', 'stanley': '#2ca02c',
              'lqr': '#d62728', 'mpc': '#9467bd'}
    labels = {'pure_pursuit': 'Pure Pursuit', 'stanley': 'Stanley',
              'lqr': 'LQR', 'mpc': 'MPC'}
    for key in ['pure_pursuit', 'stanley', 'lqr', 'mpc']:
        s = car_pp['results'][key]['state']
        rms = car_pp['results'][key]['rms']
        ax1.plot(s[:, 0], s[:, 1], color=colors[key], lw=1.5,
                 label=f'{labels[key]} (RMS={rms:.2f}m)')
    ax1.plot(car_pp['results']['pure_pursuit']['state'][0, 0],
             car_pp['results']['pure_pursuit']['state'][0, 1],
             'k*', ms=10, label='Start point')
    ax1.set_xlabel('X coordinate [m]')
    ax1.set_ylabel('Y coordinate [m]')
    ax1.set_title('(1) Car: trajectory comparison of 4 control laws\n(Pure Pursuit / Stanley / LQR / MPC)')
    ax1.legend(loc='best', fontsize=8)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # ---- (2) Car: steering command history of the 4 control laws ----
    ax2 = plt.subplot(2, 3, 2)
    for key in ['pure_pursuit', 'stanley', 'lqr']:
        d = car_pp['results'][key]['delta']
        ax2.plot(car_pp['time'], np.rad2deg(d), color=colors[key],
                 lw=1.2, label=labels[key], alpha=0.85)
    # MPC has a different time axis
    mpc_time = car_pp['results']['mpc']['time']
    mpc_delta = car_pp['results']['mpc']['delta']
    ax2.plot(mpc_time, np.rad2deg(mpc_delta), color=colors['mpc'],
             lw=1.5, label=labels['mpc'], alpha=0.85)
    ax2.axhline(0, color='k', lw=0.5)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Steering angle [deg]')
    ax2.set_title('(2) Car: steering command comparison of 4 control laws\n(MPC control period 100 ms)')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ---- (3) Car dynamics: step response ----
    ax3 = plt.subplot(2, 3, 3)
    ax3a = ax3
    ax3b = ax3.twinx()
    l1 = ax3a.plot(car_dyn['time'], car_dyn['vy'],
                    'b-', lw=1.5, label='Lateral velocity v_y [m/s]')
    l2 = ax3b.plot(car_dyn['time'], np.rad2deg(car_dyn['yaw_rate']),
                    'r-', lw=1.5, label='Yaw rate [deg/s]')
    ax3a.set_xlabel('Time [s]')
    ax3a.set_ylabel('Lateral velocity [m/s]', color='b')
    ax3b.set_ylabel('Yaw rate [deg/s]', color='r')
    ax3a.tick_params(axis='y', labelcolor='b')
    ax3b.tick_params(axis='y', labelcolor='r')
    ax3a.set_title(f'(3) Car: dynamic 2-DOF model\nstep steering response (v={car_dyn["vx"]} m/s)')
    lines = l1 + l2
    ax3a.legend(lines, [l.get_label() for l in lines], loc='best', fontsize=9)
    ax3a.grid(True, alpha=0.3)

    # ---- (4) Aircraft: short-period mode (short time) ----
    ax4 = plt.subplot(2, 3, 4)
    mask_short = aircraft['time'] <= 15.0
    # Displayed approximately as angle of attack α ≈ w / V_inf (V_inf = 235 m/s cruise speed)
    alpha_approx = aircraft['w'] / 235.0
    ax4.plot(aircraft['time'][mask_short],
             np.rad2deg(alpha_approx[mask_short]),
             'b-', lw=1.5, label='Angle of attack α ≈ w/V [deg]')
    ax4.plot(aircraft['time'][mask_short],
             np.rad2deg(aircraft['q'][mask_short]),
             'r-', lw=1.5, label='Pitch rate q [deg/s]')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Response quantity')
    ax4.set_title('(4) Aircraft: short-period mode\n(elevator -1 degree step, 15 s)')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # ---- (5) Aircraft: phugoid mode (long time) ----
    ax5 = plt.subplot(2, 3, 5)
    ax5a = ax5
    ax5b = ax5.twinx()
    l1 = ax5a.plot(aircraft['time'], aircraft['u_pert'],
                    'b-', lw=1.5, label='Speed perturbation Δu [m/s]')
    l2 = ax5b.plot(aircraft['time'], np.rad2deg(aircraft['theta']),
                    'g-', lw=1.5, label='Pitch angle θ [deg]')
    ax5a.set_xlabel('Time [s]')
    ax5a.set_ylabel('Speed perturbation [m/s]', color='b')
    ax5b.set_ylabel('Pitch angle [deg]', color='g')
    ax5a.tick_params(axis='y', labelcolor='b')
    ax5b.tick_params(axis='y', labelcolor='g')
    ax5a.set_title('(5) Aircraft: phugoid mode\n(same response, long-period oscillation over 1500 s)')
    lines = l1 + l2
    ax5a.legend(lines, [l.get_label() for l in lines], loc='best', fontsize=9)
    ax5a.grid(True, alpha=0.3)

    # ---- (6) Ship: course-change simulation ----
    ax6 = plt.subplot(2, 3, 6)
    ax6a = ax6
    ax6b = ax6.twinx()
    l1 = ax6a.plot(ship['time'], np.rad2deg(ship['psi']),
                    'b-', lw=1.8, label='Heading ψ [deg]')
    l_target = ax6a.axhline(np.rad2deg(ship['psi_target']),
                             color='k', ls='--', lw=1.0, label='Target heading')
    l2 = ax6b.plot(ship['time'], np.rad2deg(ship['delta']),
                    'r-', lw=1.0, alpha=0.7, label='Rudder angle δ [deg]')
    ax6a.set_xlabel('Time [s]')
    ax6a.set_ylabel('Heading [deg]', color='b')
    ax6b.set_ylabel('Rudder angle [deg]', color='r')
    ax6a.tick_params(axis='y', labelcolor='b')
    ax6b.tick_params(axis='y', labelcolor='r')
    ax6a.set_title('(6) Ship: course change with the Nomoto model\n(target 30 degrees, 10 minutes)')
    lines = [l1[0], l_target, l2[0]]
    ax6a.legend(lines, [l.get_label() for l in lines], loc='best', fontsize=9)
    ax6a.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'vehicle_dynamics_results.png')
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    print(f"  -> Saved the results plot: {output_path}")
    plt.close()

    # ---- Also output the ship trajectory in a separate figure (an overly elongated trajectory is easier to read this way) ----
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ship['x'], ship['y'], 'b-', lw=1.8, label='Ship trajectory')
    ax.plot(ship['x'][0], ship['y'][0], 'go', ms=10, label='Departure point')
    ax.plot(ship['x'][-1], ship['y'][-1], 'r^', ms=10, label='Arrival point')
    # Direction of the target heading
    L_arrow = 1500
    ax.plot([0, L_arrow * np.cos(ship['psi_target'])],
            [0, L_arrow * np.sin(ship['psi_target'])],
            'k--', lw=1.0, alpha=0.5, label='Target heading')
    ax.set_xlabel('Eastward distance [m]')
    ax.set_ylabel('Northward distance [m]')
    ax.set_title('Ship track (course-change maneuvering to a target heading of 30 degrees)')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    output_path2 = os.path.join(OUTPUT_DIR, 'vehicle_dynamics_ship_track.png')
    plt.savefig(output_path2, dpi=130, bbox_inches='tight')
    print(f"  -> Saved the ship-trajectory plot: {output_path2}")
    plt.close()

    return output_path, output_path2


# ====================================================================
# Main
# ====================================================================
def print_conclusion():
    """A summary after all simulations finish, giving an overview of the whole discussion."""
    print("\n" + "=" * 70)
    print("[Summary] Motion control of transportation systems - an overview")
    print("=" * 70)
    print()
    print("- 1. The spectrum of dynamics formulations")
    print("-" * 70)
    print("  Newton-Euler   : best for single rigid bodies, bicycle models, etc. Directly ties to the state equations.")
    print("  Lagrange       : nonholonomic constraints are handled with undetermined multipliers, but cumbersome.")
    print("                   Little appeal for systems dominated by non-conservative forces.")
    print("  Kane's method  : avoids equation explosion for multibody systems (14-DOF full vehicle, etc.).")
    print("                   Adopted inside commercial software such as CarSim.")
    print()
    print("  -> For cars, aircraft, and ships alike, Newton-Euler is central;")
    print("     multibody uses Kane, and the Lagrangian sits in between with little role to play.")
    print()
    print("- 2. Tire, aerodynamic, and hydrodynamic force models")
    print("-" * 70)
    print("  Car      : linear tire F_y = -C_α·α (linear region)")
    print("             Pacejka Magic Formula (the whole region including saturation)")
    print("             friction circle √(F_x² + F_y²) ≤ μF_z (longitudinal/lateral force distribution constraint)")
    print("  Aircraft : dynamic pressure q̄ = (1/2)ρV² × wing area × aerodynamic coefficient C_*(α, β, ...)")
    print("             the coefficients are functions of angle of attack, control-surface angle, and angular rate. Identified by wind tunnel/CFD.")
    print("  Ship     : added mass (negligible in air, same order in water)")
    print("             quadratic drag, restoring forces, wave/wind disturbances")
    print()
    print("- 3. The hierarchy of path tracking and attitude control")
    print("-" * 70)
    print("  Classical geometric control : Pure Pursuit (rear-wheel referenced), Stanley (front-wheel referenced)")
    print("                 minimal computation; still in active use for parking assist and low-speed robotics")
    print("  Optimal control : LQR (unconstrained, analytical solution), MPC (constrained, look-ahead)")
    print("                 the mainstream for path tracking in LKAS, ACC, and autonomous driving")
    print("  For ships    : Nomoto model + LOS guidance + PID/MPC")
    print("  For aircraft : SAS/CAS + autopilot + gain scheduling")
    print()
    print("- 4. State estimation and sensor fusion")
    print("-" * 70)
    print("  EKF/UKF        : estimate states that cannot be measured directly (body slip angle β, lateral velocity v_y,")
    print("                   road friction μ, etc.) from IMU + GNSS + encoders")
    print("  Factor graph   : iSAM2 and the like, globally optimizing while retaining past observations")
    print("                   the modern mainstream for autonomous-driving SLAM")
    print("  INS/GNSS integration : the hierarchy of Loosely / Tightly / Deep Coupling")
    print()
    print("- 5. The chassis-control hierarchy (car)")
    print("-" * 70)
    print("  ABS  : keeps near the peak-friction slip ratio κ ≈ 0.1-0.2")
    print("  TCS  : slip control on the drive side")
    print("  ESC  : intervenes based on the error between the intended and actual yaw rate")
    print("         oversteer -> brake the outer front wheel, understeer -> brake the inner rear wheel")
    print("         standard on almost all passenger cars since 2012; fatal accidents cut by 30-50%")
    print("  TV   : torque vectoring (independent left/right wheel torque distribution)")
    print("         a good match for the independent motors of a BEV")
    print("  Integrated control: optimally distributes all actuators via Control Allocation")
    print()
    print("- 6. The spectrum of learning-based methods")
    print("-" * 70)
    print("  Fully model-based ── Pure Pursuit / LQR / MPC")
    print("       ↓")
    print("  GP-MPC / Neural-MPC ── learn only the tire model")
    print("       ↓")
    print("  Residual Learning ── learn a correction term for the control law")
    print("       ↓")
    print("  RL/IL with a safety filter ── guarantee safety with a CBF QP")
    print("       ↓")
    print("  Fully end-to-end ── PilotNet, Tesla FSD, Wayve")
    print()
    print("  The further down, the higher the weight on learning: the performance")
    print("  ceiling is higher, but safety assurance and certification get harder.")
    print("  Commercial autonomous driving is the upper half; the research frontier is the lower half.")
    print()
    print("- 7. Multi-vehicle cooperation (V2X)")
    print("-" * 70)
    print("  V2V / V2I / V2N / V2P  <- communicate via DSRC or C-V2X (5G NR-V2X)")
    print("  CACC (platooning)         : feedforward of the leader's acceleration, time gap 0.6-1.0 s")
    print("  String stability          : the condition that prevents amplification of accel/decel toward followers")
    print("                              ordinary ACC does not satisfy it; CACC can")
    print("  Intersection coordination : centralized MILP / decentralized auctions")
    print()
    print("- 8. The essential differences among the three transportation systems")
    print("-" * 70)
    print("  Car      : nonholonomic constraints + road-friction dominated")
    print("             response time 0.1-1 s, natural modes not an issue")
    print()
    print("  Aircraft : free space + aerodynamics (low density) + separation of longitudinal/lateral modes")
    print("             short period a few s, phugoid tens of seconds to minutes")
    print("             shaping the natural modes is central to control design")
    print()
    print("  Ship     : free space + hydrodynamics (high density) + added mass + restoring forces")
    print("             response time tens of seconds to minutes, strongly nonlinear")
    print()
    print("  Common : write the rigid-body dynamics in Newton-Euler form and add an")
    print("           external-force model suited to the medium and constraints. The")
    print("           Lagrangian form never comes to the center of implementation in any of these fields.")
    print()
    print("=" * 70)


def main():
    print_introduction()

    # Run each simulation
    car_pp   = simulate_car_pure_pursuit()
    car_dyn  = simulate_car_dynamic_bicycle()
    aircraft = simulate_aircraft_longitudinal()
    ship     = simulate_ship_nomoto()

    # Visualization
    paths = visualize_all(car_pp, car_dyn, aircraft, ship)

    # Summary
    print_conclusion()

    print()
    print("Simulation complete. Output files:")
    for p in paths:
        print(f"  - {p}")
    print("▓" * 70 + "\n")


if __name__ == '__main__':
    main()
