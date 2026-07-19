"""
A small contrasting example: Lagrangian simulation of a 2-link robot arm
======================================================================
The main body of this repository (car / aircraft / ship) formulates all
equations of motion in Newton-Euler form. As explained in
docs_en/why_newton_euler.md, that is the natural choice for vehicles governed
by nonholonomic constraints and non-conservative forces.

So for what kind of system does the Lagrangian form become the "natural
choice"? The classic example is the robot arm. As a minimal example for
contrast, this script formulates the equations of motion of a 2-link planar arm
with the Lagrangian method and simulates its free motion (falling like a
pendulum under gravity).

Why is the Lagrangian method straightforward here:
  - The joint constraints are holonomic (relations between coordinates). Once the
    generalized coordinates q = (θ1, θ2) are chosen, the constraints vanish and no
    Lagrange multipliers are needed.
  - Apart from the drive torque, gravity (a conservative force) dominates. This
    fits directly into the L = T - V framework.
  - From the single scalar function L, the equations of motion follow through
    purely mechanical partial differentiation.

The equations of motion take the standard form
M(q) q̈ + C(q, q̇) q̇ + g(q) = τ.
We solve this for q̈, cast it into state-space form, and integrate with RK4.
(You can also observe the difference: for vehicles, Newton-Euler connects
 directly to the state equations from the start, whereas here an extra step of
 inverting M is required.)

How to run:
    python python/lagrangian_arm.py

Dependencies: numpy, matplotlib
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ── Physical parameters ──────────────────────────────────────────
M1, M2 = 1.0, 1.0      # mass of each link [kg]
L1, L2 = 1.0, 1.0      # length of each link [m]
LC1, LC2 = 0.5, 0.5    # center-of-mass position of each link (from the base) [m]
I1, I2 = M1 * L1**2 / 12.0, M2 * L2**2 / 12.0   # inertia about the center of mass [kg m^2]
G = 9.81               # gravitational acceleration [m/s^2]


def manipulator_matrices(q):
    """
    Return M(q), C(q,q̇)·q̇, and g(q) for the 2-link arm as derived with the
    Lagrangian method.

    Computing
        d/dt(∂L/∂q̇) - ∂L/∂q = τ
    from the Lagrangian L = T - V yields the following closed-form expressions
    for the planar 2-link arm (for the derivation see any robotics textbook,
    e.g. Spong et al. "Robot Modeling and Control", Chapter 7).
    """
    q1, q2 = q
    c2 = np.cos(q2)

    # Mass (inertia) matrix M(q)
    m11 = (I1 + I2 + M1 * LC1**2 + M2 * (L1**2 + LC2**2 + 2 * L1 * LC2 * c2))
    m12 = I2 + M2 * (LC2**2 + L1 * LC2 * c2)
    m22 = I2 + M2 * LC2**2
    M = np.array([[m11, m12],
                  [m12, m22]])
    return M


def manipulator_rhs(q, dq):
    """Return the Coriolis/centrifugal term C(q,q̇)q̇ and the gravity term g(q) together."""
    q1, q2 = q
    dq1, dq2 = dq
    s2 = np.sin(q2)
    h = M2 * L1 * LC2 * s2

    # Coriolis/centrifugal term
    coriolis = np.array([
        -h * dq2 * (2 * dq1 + dq2),
         h * dq1 * dq1,
    ])

    # Gravity term g(q)
    g1 = ((M1 * LC1 + M2 * L1) * G * np.cos(q1)
          + M2 * LC2 * G * np.cos(q1 + q2))
    g2 = M2 * LC2 * G * np.cos(q1 + q2)
    gravity = np.array([g1, g2])

    return coriolis, gravity


def dynamics(state, tau=np.zeros(2)):
    """
    Return the time derivative of the state x = [q1, q2, dq1, dq2].

    Solve the equations of motion  M(q) q̈ + C q̇ + g = τ  for q̈:
        q̈ = M^{-1} (τ - C q̇ - g)
    Whereas Newton-Euler connects directly to the state equations from the start,
    the Lagrangian form requires this extra step of taking M^{-1}.
    """
    q  = state[:2]
    dq = state[2:]
    M = manipulator_matrices(q)
    coriolis, gravity = manipulator_rhs(q, dq)
    ddq = np.linalg.solve(M, tau - coriolis - gravity)
    return np.concatenate([dq, ddq])


def rk4_step(state, dt):
    k1 = dynamics(state)
    k2 = dynamics(state + 0.5 * dt * k1)
    k3 = dynamics(state + 0.5 * dt * k2)
    k4 = dynamics(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def total_energy(state):
    """Total mechanical energy T + V. Should be conserved when the drive torque is zero."""
    q  = state[:2]
    dq = state[2:]
    M = manipulator_matrices(q)
    T = 0.5 * dq @ M @ dq

    q1, q2 = q
    # Height of each link's center of mass (joint 1 at the origin, x-axis horizontal reference)
    y1 = LC1 * np.sin(q1)
    y2 = L1 * np.sin(q1) + LC2 * np.sin(q1 + q2)
    # Take the potential-energy reference at the lowest point where all links point
    # straight down, offsetting so that V is always >= 0 (to make the conserved
    # quantity easier to read).
    v_offset = M1 * G * LC1 + M2 * G * (L1 + LC2)
    V = M1 * G * y1 + M2 * G * y2 + v_offset
    return T + V


def simulate(t_end=10.0, dt=0.002):
    """Free motion under gravity (pendulum-like fall from the initial pose)."""
    n = int(t_end / dt)
    # Initial pose: release from rest with the first link horizontal and the second
    # link bent by 90 degrees. Thereafter it moves like a double pendulum under
    # gravity alone.
    state = np.array([0.0, np.pi / 2.0, 0.0, 0.0])

    t_hist  = np.zeros(n)
    q_hist  = np.zeros((n, 2))
    e_hist  = np.zeros(n)

    for i in range(n):
        t_hist[i] = i * dt
        q_hist[i] = state[:2]
        e_hist[i] = total_energy(state)
        state = rk4_step(state, dt)

    return t_hist, q_hist, e_hist


def main():
    print("=" * 66)
    print(" 2-link robot arm - Lagrangian simulation (for contrast)")
    print("=" * 66)
    print()
    print(" Whereas the main vehicle models use the Newton-Euler form, the")
    print(" Lagrangian method is more natural for a robot arm. As a concrete")
    print(" example, we solve a 2-link arm in free motion under gravity from")
    print(" M(q)q̈ + Cq̇ + g = τ.")
    print()

    t, q, e = simulate()

    # Express the fluctuation range of the total energy as a relative ratio to the
    # mean energy held by the system. Since the drive torque is zero and only
    # gravity (a conservative force) acts, it is theoretically constant -- the
    # fluctuation is just the numerical error of RK4.
    e_drift = e.max() - e.min()
    e_mean = float(np.mean(e))
    print(f" Simulation time      : {t[-1]:.1f} s ({len(t)} steps)")
    print(f" Total energy (mean)   : {e_mean:.4f} J")
    print(f" Energy fluctuation    : {e_drift:.3e} J "
          f"({e_drift / e_mean * 100:.4f} % of the mean)")
    print("   -> Since the drive torque is zero and gravity is conservative, the")
    print("      total energy is theoretically constant. The fluctuation is only")
    print("      the numerical error of RK4, which is tiny. This also confirms that")
    print("      the Lagrangian form handles conservative systems cleanly.")
    print()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, os.pardir, "lagrangian_arm_result.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(t, np.rad2deg(q[:, 0]), label=r"$\theta_1$ (joint 1)")
    ax1.plot(t, np.rad2deg(q[:, 1]), label=r"$\theta_2$ (joint 2)")
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("joint angle [deg]")
    ax1.set_title("2-link arm: free motion under gravity")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, e, color="tab:red")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("total energy  T + V  [J]")
    ax2.set_title("Energy conservation check (RK4)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.abspath(out_path), dpi=120)
    print(f" Saved plot: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
