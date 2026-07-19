"""
Choosing the Right Dynamics Formulation — Animation Demo
==========================================================
The point is not "which one is correct" but "choose the formulation
that suits the application", shown visually with three side-by-side
animations.

Left   : Robot arm (Lagrangian method)      M(q)q̈ + Cq̇ + g = τ
Center : Car path tracking (Newton-Euler)   ẋ = Ax + Bu (Pure Pursuit)
Right  : Ship course change (Newton-Euler)  Tψ̈ + ψ̇ = Kδ (Nomoto)

How to run:
    python python/animation_demo.py

Output:
    animation_demo.gif  (saved to the project root)

Dependencies: numpy, matplotlib, Pillow
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import rcParams

# ── Font configuration ──────────────────────────────────────────────────────
def _setup_font():
    candidates = [
        'Yu Gothic', 'Meiryo', 'Noto Sans CJK JP', 'IPAexGothic',
        'Hiragino Sans', 'TakaoGothic', 'DejaVu Sans'
    ]
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            rcParams['font.family'] = name
            return name
    return 'DejaVu Sans'

_setup_font()
rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


# ══════════════════════════════════════════════════════════════════════
# 1. Robot arm — Lagrangian method
# ══════════════════════════════════════════════════════════════════════
# Parameters
_M1, _M2 = 1.0, 1.0
_L1, _L2 = 1.0, 1.0
_LC1, _LC2 = 0.5, 0.5
_I1 = _M1 * _L1**2 / 12.0
_I2 = _M2 * _L2**2 / 12.0
_G  = 9.81


def _arm_mass_matrix(q):
    c2 = np.cos(q[1])
    m11 = (_I1 + _I2 + _M1*_LC1**2
           + _M2*(_L1**2 + _LC2**2 + 2*_L1*_LC2*c2))
    m12 = _I2 + _M2*(_LC2**2 + _L1*_LC2*c2)
    m22 = _I2 + _M2*_LC2**2
    return np.array([[m11, m12], [m12, m22]])


def _arm_rhs(q, dq):
    s2 = np.sin(q[1])
    h  = _M2 * _L1 * _LC2 * s2
    cor = np.array([-h*dq[1]*(2*dq[0]+dq[1]), h*dq[0]**2])
    g1  = ((_M1*_LC1 + _M2*_L1)*_G*np.cos(q[0])
           + _M2*_LC2*_G*np.cos(q[0]+q[1]))
    g2  = _M2*_LC2*_G*np.cos(q[0]+q[1])
    return cor, np.array([g1, g2])


def _arm_deriv(state):
    q, dq = state[:2], state[2:]
    M  = _arm_mass_matrix(q)
    cor, grav = _arm_rhs(q, dq)
    ddq = np.linalg.solve(M, -cor - grav)
    return np.concatenate([dq, ddq])


def _arm_energy(state):
    q, dq = state[:2], state[2:]
    M = _arm_mass_matrix(q)
    T = 0.5 * dq @ M @ dq
    y1 = _LC1 * np.sin(q[0])
    y2 = _L1*np.sin(q[0]) + _LC2*np.sin(q[0]+q[1])
    offset = _M1*_G*_LC1 + _M2*_G*(_L1+_LC2)
    V = _M1*_G*y1 + _M2*_G*y2 + offset
    return T + V


def simulate_arm(t_end=8.0, dt=0.005):
    n = int(t_end / dt)
    state = np.array([0.0, np.pi/2, 0.0, 0.0])
    q_hist = np.zeros((n, 2))
    e_hist = np.zeros(n)
    t_hist = np.arange(n) * dt
    for i in range(n):
        q_hist[i] = state[:2]
        e_hist[i] = _arm_energy(state)
        k1 = _arm_deriv(state)
        k2 = _arm_deriv(state + 0.5*dt*k1)
        k3 = _arm_deriv(state + 0.5*dt*k2)
        k4 = _arm_deriv(state + dt*k3)
        state = state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    return t_hist, q_hist, e_hist


def arm_fk(q):
    """Forward kinematics: joint angles -> link endpoint coordinates"""
    x1 = _L1 * np.cos(q[0])
    y1 = _L1 * np.sin(q[0])
    x2 = x1 + _L2 * np.cos(q[0] + q[1])
    y2 = y1 + _L2 * np.sin(q[0] + q[1])
    return (x1, y1), (x2, y2)


# ══════════════════════════════════════════════════════════════════════
# 2. Car path tracking — Newton-Euler (Pure Pursuit)
# ══════════════════════════════════════════════════════════════════════
def _make_ellipse_path(a=14.0, b=8.0, n=400):
    th = np.linspace(0, 2*np.pi, n, endpoint=False)
    px = a * np.cos(th)
    py = b * np.sin(th)
    dyaw = np.arctan2(np.gradient(py), np.gradient(px))
    return np.stack([px, py], axis=1), dyaw


def simulate_car(t_end=30.0, dt=0.05):
    path, pyaw = _make_ellipse_path()
    n_path = len(path)
    Ld  = 3.0          # look-ahead distance [m]
    L   = 2.7          # wheelbase [m]
    v   = 5.0          # speed [m/s]

    state = np.array([path[0,0], path[0,1] + 1.5, pyaw[0]])  # 1.5 m lateral offset
    n_steps = int(t_end / dt)
    hist = np.zeros((n_steps, 3))
    delta_hist = np.zeros(n_steps)
    prev_idx = 0

    for i in range(n_steps):
        hist[i] = state
        x, y, psi = state

        # Pure Pursuit: find the look-ahead point
        dists = np.sqrt((path[:,0]-x)**2 + (path[:,1]-y)**2)
        # Search forward from the current nearest point
        idx_range = np.arange(prev_idx, prev_idx + n_path//2) % n_path
        d = dists[idx_range]
        ahead = idx_range[d >= Ld]
        if len(ahead) == 0:
            ahead_idx = (prev_idx + 5) % n_path
        else:
            ahead_idx = ahead[0]
        prev_idx = idx_range[np.argmin(d)]

        tx, ty = path[ahead_idx]
        # Target direction in the vehicle body frame
        alpha = np.arctan2(ty - y, tx - x) - psi
        alpha = (alpha + np.pi) % (2*np.pi) - np.pi
        dist = np.hypot(tx - x, ty - y)
        if dist < 1e-3:
            dist = 1e-3
        delta = np.arctan2(2*L*np.sin(alpha), dist)
        delta = np.clip(delta, -np.deg2rad(35), np.deg2rad(35))
        delta_hist[i] = delta

        # Kinematic bicycle model (Euler)
        state[0] += v * np.cos(psi) * dt
        state[1] += v * np.sin(psi) * dt
        state[2] += (v / L) * np.tan(delta) * dt

    t_hist = np.arange(n_steps) * dt
    return t_hist, hist, delta_hist, path, pyaw


def car_corners(x, y, psi, car_l=2.0, car_w=1.0):
    """Return the coordinates of the vehicle's four corners (for matplotlib Polygon)"""
    corners = np.array([
        [ car_l/2,  car_w/2],
        [-car_l/2,  car_w/2],
        [-car_l/2, -car_w/2],
        [ car_l/2, -car_w/2],
    ])
    c, s = np.cos(psi), np.sin(psi)
    R = np.array([[c, -s], [s, c]])
    rotated = corners @ R.T
    return rotated + np.array([x, y])


# ══════════════════════════════════════════════════════════════════════
# 3. Ship course change — Newton-Euler (Nomoto first-order model)
# ══════════════════════════════════════════════════════════════════════
def simulate_ship(t_end=180.0, dt=0.1):
    K_nom  = 0.18    # Rudder effectiveness gain
    T_nom  = 50.0    # Time constant [s]
    psi_tgt = np.deg2rad(60.0)
    Kp, Kd = 2.0, 8.0
    v_ship = 5.0     # [m/s]  ship speed
    delta_max = np.deg2rad(35)

    n = int(t_end / dt)
    psi = 0.0
    dpsi = 0.0
    x, y = 0.0, 0.0

    psi_hist  = np.zeros(n)
    delta_hist = np.zeros(n)
    x_hist, y_hist = np.zeros(n), np.zeros(n)
    t_hist = np.arange(n) * dt

    for i in range(n):
        psi_hist[i]  = psi
        x_hist[i]    = x
        y_hist[i]    = y

        err  = psi_tgt - psi
        err  = (err + np.pi) % (2*np.pi) - np.pi
        delta = Kp*err - Kd*dpsi
        delta = np.clip(delta, -delta_max, delta_max)
        delta_hist[i] = delta

        # Nomoto: T·ψ̈ + ψ̇ = K·δ  → ψ̈ = (K·δ - ψ̇) / T
        ddpsi = (K_nom * delta - dpsi) / T_nom
        dpsi += ddpsi * dt
        psi  += dpsi  * dt

        x += v_ship * np.cos(psi) * dt
        y += v_ship * np.sin(psi) * dt

    return t_hist, psi_hist, delta_hist, x_hist, y_hist, psi_tgt


def ship_polygon(x, y, psi, length=8.0, width=2.5):
    """Represent the hull as a pentagon"""
    pts = np.array([
        [ length*0.5,       0.0      ],   # Bow
        [ length*0.2,  width*0.5    ],
        [-length*0.5,  width*0.5    ],
        [-length*0.5, -width*0.5    ],
        [ length*0.2, -width*0.5    ],
    ])
    c, s = np.cos(psi), np.sin(psi)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + np.array([x, y])


# ══════════════════════════════════════════════════════════════════════
# Main: run simulations -> generate animation
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print(" Choosing the Right Dynamics Formulation — Animation Demo")
    print("=" * 60)
    print(" Computing simulation data...")

    t_arm, q_arm, e_arm      = simulate_arm(t_end=8.0, dt=0.005)
    t_car, xy_car, d_car, path_xy, _ = simulate_car(t_end=30.0, dt=0.05)
    t_ship, psi_ship, d_ship, xs, ys, psi_tgt = simulate_ship(t_end=180.0, dt=0.1)

    print(f"  Robot arm : {len(t_arm)} steps")
    print(f"  Car       : {len(t_car)} steps")
    print(f"  Ship      : {len(t_ship)} steps")

    # Resample all data to a common frame count for the animation
    N_FRAMES = 120
    arm_idx  = np.linspace(0, len(t_arm)-1,  N_FRAMES, dtype=int)
    car_idx  = np.linspace(0, len(t_car)-1,  N_FRAMES, dtype=int)
    ship_idx = np.linspace(0, len(t_ship)-1, N_FRAMES, dtype=int)

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 6))
    fig.patch.set_facecolor('#0d1117')

    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[3, 1],
        hspace=0.35, wspace=0.35,
        left=0.05, right=0.97, top=0.88, bottom=0.08
    )
    ax_arm   = fig.add_subplot(gs[0, 0])
    ax_car   = fig.add_subplot(gs[0, 1])
    ax_ship  = fig.add_subplot(gs[0, 2])
    ax_e     = fig.add_subplot(gs[1, 0])   # Energy
    ax_dcur  = fig.add_subplot(gs[1, 1])   # Steering angle
    ax_psi   = fig.add_subplot(gs[1, 2])   # Heading

    DARK   = '#0d1117'
    PANEL  = '#161b22'
    TEXT   = '#e6edf3'
    GREEN  = '#3fb950'
    BLUE   = '#58a6ff'
    ORANGE = '#ff7b72'
    PURPLE = '#bc8cff'
    YELLOW = '#e3b341'

    for ax in [ax_arm, ax_car, ax_ship, ax_e, ax_dcur, ax_psi]:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#30363d')

    # Title
    fig.text(0.5, 0.97,
             'Choosing the Right Dynamics Formulation — Different applications call for different equations',
             ha='center', va='top', color=TEXT, fontsize=13, fontweight='bold')

    # Heading and equation label for each panel
    labels = [
        ('Robot arm',        'Lagrangian method', r'$M(q)\ddot{q}+C\dot{q}+g=\tau$',  GREEN),
        ('Car path tracking', 'Newton-Euler method', r'$\dot{x}=Ax+Bu$ (Pure Pursuit)',    BLUE),
        ('Ship course change', 'Newton-Euler method', r'$T\ddot{\psi}+\dot{\psi}=K\delta$  (Nomoto)', ORANGE),
    ]
    for i, (title, method, eq, col) in enumerate(labels):
        col_x = 0.175 + i * 0.31
        fig.text(col_x, 0.925, title,
                 ha='center', color=col, fontsize=11, fontweight='bold')
        fig.text(col_x, 0.895, method,
                 ha='center', color=TEXT, fontsize=8.5, alpha=0.75)
        fig.text(col_x, 0.87, eq,
                 ha='center', color=col, fontsize=8, alpha=0.9)

    # ── Arm drawing initialization ─────────────────────────────────────────────
    ax_arm.set_xlim(-2.3, 2.3)
    ax_arm.set_ylim(-2.3, 2.3)
    ax_arm.set_aspect('equal')
    ax_arm.set_xlabel('x [m]', color=TEXT, fontsize=8)
    ax_arm.set_ylabel('y [m]', color=TEXT, fontsize=8)
    ax_arm.axhline(0, color='#30363d', lw=0.5)
    ax_arm.axvline(0, color='#30363d', lw=0.5)
    # Trajectory (tip of link 2)
    tip2_x = [_L1*np.cos(q_arm[arm_idx[0],0]) + _L2*np.cos(q_arm[arm_idx[0],0]+q_arm[arm_idx[0],1])]
    tip2_y = [_L1*np.sin(q_arm[arm_idx[0],0]) + _L2*np.sin(q_arm[arm_idx[0],0]+q_arm[arm_idx[0],1])]
    (arm_trace,) = ax_arm.plot([], [], color=GREEN, lw=0.8, alpha=0.35)
    (link1,) = ax_arm.plot([], [], color=GREEN,  lw=4, solid_capstyle='round')
    (link2,) = ax_arm.plot([], [], color=PURPLE, lw=4, solid_capstyle='round')
    joint0 = ax_arm.plot(0, 0, 'o', color=TEXT, ms=8, zorder=5)[0]
    joint1 = ax_arm.plot([], [], 'o', color=GREEN,  ms=6, zorder=5)[0]
    joint2 = ax_arm.plot([], [], 'o', color=PURPLE, ms=6, zorder=5)[0]
    # Legend
    ax_arm.legend(
        [mpatches.Patch(color=GREEN), mpatches.Patch(color=PURPLE)],
        ['Link 1', 'Link 2'],
        loc='upper right', fontsize=7,
        facecolor=PANEL, edgecolor='#30363d', labelcolor=TEXT
    )

    # ── Car tracking drawing initialization ──────────────────────────────────────────────
    ax_car.set_xlim(-18, 18)
    ax_car.set_ylim(-12, 12)
    ax_car.set_aspect('equal')
    ax_car.set_xlabel('x [m]', color=TEXT, fontsize=8)
    ax_car.set_ylabel('y [m]', color=TEXT, fontsize=8)
    ax_car.plot(path_xy[:,0], path_xy[:,1],
                '--', color='#30363d', lw=1.2, label='Reference path')
    (car_trace,) = ax_car.plot([], [], color=BLUE, lw=0.8, alpha=0.4)
    car_patch = plt.Polygon(
        car_corners(xy_car[0,0], xy_car[0,1], xy_car[0,2]),
        closed=True, facecolor=BLUE, edgecolor='white', lw=0.8, zorder=5
    )
    ax_car.add_patch(car_patch)
    car_arrow = ax_car.annotate(
        '', xy=(0,0), xytext=(0,0),
        arrowprops=dict(arrowstyle='->', color=YELLOW, lw=1.5)
    )
    rms_text = ax_car.text(
        0.03, 0.97, '', transform=ax_car.transAxes,
        color=TEXT, fontsize=7.5, va='top'
    )

    # ── Ship drawing initialization ────────────────────────────────────────────────
    ax_ship.set_xlim(-5, 900)
    ax_ship.set_ylim(-5, 900)
    ax_ship.set_aspect('equal')
    ax_ship.set_xlabel('x [m]', color=TEXT, fontsize=8)
    ax_ship.set_ylabel('y [m]', color=TEXT, fontsize=8)
    # Target heading line
    L_line = 600
    ax_ship.plot(
        [0, L_line*np.cos(psi_tgt)],
        [0, L_line*np.sin(psi_tgt)],
        '--', color='#30363d', lw=1.0, label=f'Target heading {np.rad2deg(psi_tgt):.0f}°'
    )
    ax_ship.legend(loc='lower right', fontsize=7,
                   facecolor=PANEL, edgecolor='#30363d', labelcolor=TEXT)
    (ship_trace,) = ax_ship.plot([], [], color=ORANGE, lw=0.8, alpha=0.5)
    ship_patch = plt.Polygon(
        ship_polygon(xs[0], ys[0], psi_ship[0]),
        closed=True, facecolor=ORANGE, edgecolor='white', lw=0.8, zorder=5
    )
    ax_ship.add_patch(ship_patch)
    psi_text = ax_ship.text(
        0.03, 0.97, '', transform=ax_ship.transAxes,
        color=TEXT, fontsize=7.5, va='top'
    )

    # ── Bottom-row subplot initialization ───────────────────────────────────────
    e_ref = e_arm[0]

    ax_e.set_xlim(0, t_arm[-1])
    ax_e.set_ylim(e_ref - 0.005, e_ref + 0.015)
    ax_e.set_xlabel('Time [s]', color=TEXT, fontsize=7)
    ax_e.set_ylabel('T+V [J]', color=TEXT, fontsize=7)
    ax_e.set_title('Energy conservation (no drive torque)', color=TEXT, fontsize=7.5)
    ax_e.axhline(e_ref, color='#30363d', lw=1, ls='--')
    (line_e,) = ax_e.plot([], [], color=GREEN, lw=1.2)
    e_text = ax_e.text(
        0.02, 0.92, '', transform=ax_e.transAxes,
        color=GREEN, fontsize=7, va='top'
    )

    ax_dcur.set_xlim(0, t_car[-1])
    ax_dcur.set_ylim(-40, 40)
    ax_dcur.set_xlabel('Time [s]', color=TEXT, fontsize=7)
    ax_dcur.set_ylabel('Steering angle [°]', color=TEXT, fontsize=7)
    ax_dcur.set_title('Front-wheel steering angle', color=TEXT, fontsize=7.5)
    ax_dcur.axhline(0, color='#30363d', lw=0.8)
    (line_d,) = ax_dcur.plot([], [], color=BLUE, lw=1.2)

    ax_psi.set_xlim(0, t_ship[-1])
    ax_psi.set_ylim(-5, np.rad2deg(psi_tgt) + 10)
    ax_psi.set_xlabel('Time [s]', color=TEXT, fontsize=7)
    ax_psi.set_ylabel('Heading [°]', color=TEXT, fontsize=7)
    ax_psi.set_title('Ship heading', color=TEXT, fontsize=7.5)
    ax_psi.axhline(np.rad2deg(psi_tgt), color='#30363d', lw=1, ls='--',
                   label=f'Target {np.rad2deg(psi_tgt):.0f}°')
    ax_psi.legend(fontsize=7, facecolor=PANEL, edgecolor='#30363d', labelcolor=TEXT)
    (line_psi,) = ax_psi.plot([], [], color=ORANGE, lw=1.2)

    # Progress bar
    prog_bar = fig.text(0.5, 0.015, '', ha='center', color=TEXT, fontsize=7)

    # ── Animation update function ────────────────────────────────────────
    def update(frame):
        ai = arm_idx[frame]
        ci = car_idx[frame]
        si = ship_idx[frame]

        # --- Robot arm ---
        q = q_arm[ai]
        p1, p2 = arm_fk(q)
        link1.set_data([0, p1[0]], [0, p1[1]])
        link2.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        joint1.set_data([p1[0]], [p1[1]])
        joint2.set_data([p2[0]], [p2[1]])
        # Tip trajectory
        tip_xi = (_L1*np.cos(q_arm[arm_idx[:frame+1],0])
                  + _L2*np.cos(q_arm[arm_idx[:frame+1],0] + q_arm[arm_idx[:frame+1],1]))
        tip_yi = (_L1*np.sin(q_arm[arm_idx[:frame+1],0])
                  + _L2*np.sin(q_arm[arm_idx[:frame+1],0] + q_arm[arm_idx[:frame+1],1]))
        arm_trace.set_data(tip_xi, tip_yi)

        # --- Car ---
        cx, cy, cpsi = xy_car[ci]
        corners = car_corners(cx, cy, cpsi)
        car_patch.set_xy(corners)
        car_trace.set_data(xy_car[:ci+1, 0], xy_car[:ci+1, 1])
        # Heading arrow
        arr_len = 2.5
        car_arrow.xy     = (cx + arr_len*np.cos(cpsi), cy + arr_len*np.sin(cpsi))
        car_arrow.xytext = (cx, cy)
        # RMS deviation text
        if ci > 0:
            dists_to_path = np.min(
                np.sqrt((xy_car[:ci+1,0:1] - path_xy[:,0])**2
                      + (xy_car[:ci+1,1:2] - path_xy[:,1])**2),
                axis=1
            )
            rms = np.sqrt(np.mean(dists_to_path**2))
            rms_text.set_text(f'RMS deviation: {rms:.3f} m\nSteering: {np.rad2deg(d_car[ci]):.1f}°')

        # --- Ship ---
        ship_patch.set_xy(ship_polygon(xs[si], ys[si], psi_ship[si]))
        ship_trace.set_data(xs[:si+1], ys[:si+1])
        psi_deg = np.rad2deg(psi_ship[si])
        tgt_deg = np.rad2deg(psi_tgt)
        psi_text.set_text(
            f'Heading: {psi_deg:.1f}°\nTarget: {tgt_deg:.0f}°\n'
            f'Error: {tgt_deg-psi_deg:.1f}°\nTime: {t_ship[si]:.0f} s'
        )

        # --- Bottom-row plots ---
        # Energy
        line_e.set_data(t_arm[arm_idx[:frame+1]], e_arm[arm_idx[:frame+1]])
        drift = abs(e_arm[ai] - e_ref)
        e_text.set_text(f'Drift: {drift:.2e} J\n({drift/e_ref*100:.5f}%)')

        # Steering angle
        line_d.set_data(t_car[car_idx[:frame+1]], np.rad2deg(d_car[car_idx[:frame+1]]))

        # Heading
        line_psi.set_data(t_ship[ship_idx[:frame+1]], np.rad2deg(psi_ship[ship_idx[:frame+1]]))

        # Progress
        prog_bar.set_text(
            f'Frame {frame+1}/{N_FRAMES}  |  '
            f'Arm: {t_arm[ai]:.1f}s  |  '
            f'Car: {t_car[ci]:.1f}s  |  '
            f'Ship: {t_ship[si]:.0f}s'
        )

        return (link1, link2, joint1, joint2, arm_trace,
                car_patch, car_trace, rms_text,
                ship_patch, ship_trace, psi_text,
                line_e, e_text, line_d, line_psi, prog_bar)

    print(" Generating animation... (this may take a while)")
    ani = FuncAnimation(
        fig, update,
        frames=N_FRAMES,
        interval=50,
        blit=False
    )

    out_path = os.path.join(OUTPUT_DIR, 'animation_demo.gif')
    writer = PillowWriter(fps=20)
    ani.save(out_path, writer=writer, dpi=80)
    plt.close(fig)

    out_abs = os.path.abspath(out_path)
    print(f"\n Saved: {out_abs}")
    print()
    print(" ┌──────────────────────────────────────────────────────┐")
    print(" │  Left    Robot arm         — Lagrangian method       │")
    print(" │       M(q)q̈ + Cq̇ + g = τ                           │")
    print(" │       Conservative system -> energy conserved as theory predicts │")
    print(" │                                                      │")
    print(" │  Center  Car path tracking — Newton-Euler method     │")
    print(" │       ẋ = Ax + Bu  (Pure Pursuit)                   │")
    print(" │       Nonholonomic constraint -> maps directly to state equation │")
    print(" │                                                      │")
    print(" │  Right   Ship course change — Newton-Euler method    │")
    print(" │       Tψ̈ + ψ̇ = Kδ  (Nomoto)                       │")
    print(" │       Added mass / fluid drag -> response time of tens of seconds to minutes │")
    print(" └──────────────────────────────────────────────────────┘")


if __name__ == '__main__':
    main()
