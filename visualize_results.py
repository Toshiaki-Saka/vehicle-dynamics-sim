"""
Visualization of C++ simulation results
=================================
Load the results.json produced by vehicle_dynamics.exe and generate plots.

Usage:
    python visualize_results.py [results.json] [output_prefix]

When arguments are omitted:
    Input   : results.json (current directory)
    Output  : cpp_results.png / cpp_results_ship_track.png
"""

import sys
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


def _setup_japanese_font():
    candidates = [
        'Noto Sans CJK JP', 'IPAexGothic', 'IPAGothic',
        'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'TakaoGothic',
        'VL Gothic', 'Source Han Sans JP', 'DejaVu Sans',
    ]
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            rcParams['font.family'] = name
            return name
    return 'DejaVu Sans'


_font_used = _setup_japanese_font()
rcParams['axes.unicode_minus'] = False


def load_results(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def plot_all(data: dict, out_prefix: str) -> list[str]:
    car_time  = np.array(data['car_time'])
    path_x    = np.array(data['car_path']['x'])
    path_y    = np.array(data['car_path']['y'])

    pp  = {k: np.array(v) if isinstance(v, list) else v for k, v in data['pure_pursuit'].items()}
    st  = {k: np.array(v) if isinstance(v, list) else v for k, v in data['stanley'].items()}
    lqr = {k: np.array(v) if isinstance(v, list) else v for k, v in data['lqr'].items()}

    dyn = {k: np.array(v) if isinstance(v, list) else v for k, v in data['car_dynamic'].items()}
    air = {k: np.array(v) if isinstance(v, list) else v for k, v in data['aircraft'].items()}
    ship = {k: np.array(v) if isinstance(v, list) else v for k, v in data['ship'].items()}

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle('Vehicle Dynamics Simulation Results (C++)',
                 fontsize=16, fontweight='bold', y=0.995)

    colors = {'pure_pursuit': '#1f77b4', 'stanley': '#2ca02c', 'lqr': '#d62728'}
    labels = {'pure_pursuit': 'Pure Pursuit', 'stanley': 'Stanley', 'lqr': 'LQR'}

    # ── (1) Trajectory comparison ───────────────────────────────────
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(path_x, path_y, 'k--', lw=1.2, label='Reference path', alpha=0.5)
    for key, hist in [('pure_pursuit', pp), ('stanley', st), ('lqr', lqr)]:
        rms = hist['rms_error']
        ax1.plot(hist['x'], hist['y'], color=colors[key], lw=1.5,
                 label=f"{labels[key]} (RMS={rms:.2f}m)")
    ax1.plot(pp['x'][0], pp['y'][0], 'k*', ms=10, label='Start point')
    ax1.set_xlabel('X coordinate [m]')
    ax1.set_ylabel('Y coordinate [m]')
    ax1.set_title('(1) Car: trajectory comparison of 3 control laws\n(Pure Pursuit / Stanley / LQR)')
    ax1.legend(loc='best', fontsize=8)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # ── (2) Steering command ────────────────────────────────────────
    ax2 = fig.add_subplot(2, 3, 2)
    for key, hist in [('pure_pursuit', pp), ('stanley', st), ('lqr', lqr)]:
        ax2.plot(car_time, np.rad2deg(hist['delta']),
                 color=colors[key], lw=1.2, label=labels[key], alpha=0.85)
    ax2.axhline(0, color='k', lw=0.5)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Steering angle [deg]')
    ax2.set_title('(2) Car: steering command comparison')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── (3) Dynamics step response ───────────────────────────────────
    ax3a = fig.add_subplot(2, 3, 3)
    ax3b = ax3a.twinx()
    l1 = ax3a.plot(dyn['time'], dyn['vy'], 'b-', lw=1.5, label='Lateral velocity v_y [m/s]')
    l2 = ax3b.plot(dyn['time'], np.rad2deg(dyn['yaw_rate']),
                   'r-', lw=1.5, label='Yaw rate [deg/s]')
    ax3a.set_xlabel('Time [s]')
    ax3a.set_ylabel('Lateral velocity [m/s]', color='b')
    ax3b.set_ylabel('Yaw rate [deg/s]', color='r')
    ax3a.tick_params(axis='y', labelcolor='b')
    ax3b.tick_params(axis='y', labelcolor='r')
    ax3a.set_title(f'(3) Car: 2-DOF dynamics model\nstep response (v={dyn["vx"]:.0f} m/s)')
    ax3a.legend(l1 + l2, [l.get_label() for l in l1 + l2], loc='best', fontsize=9)
    ax3a.grid(True, alpha=0.3)

    # ── (4) Short-period mode ─────────────────────────────────────────
    ax4 = fig.add_subplot(2, 3, 4)
    mask = air['time'] <= 15.0
    alpha_approx = air['w'] / 235.0
    ax4.plot(air['time'][mask], np.rad2deg(alpha_approx[mask]),
             'b-', lw=1.5, label='Angle of attack α ≈ w/V [deg]')
    ax4.plot(air['time'][mask], np.rad2deg(air['q'][mask]),
             'r-', lw=1.5, label='Pitch rate q [deg/s]')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Response')
    ax4.set_title('(4) Aircraft: short-period mode\n(elevator -1 deg step, 15 s)')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # ── (5) Phugoid mode ──────────────────────────────────────────────
    ax5a = fig.add_subplot(2, 3, 5)
    ax5b = ax5a.twinx()
    l1 = ax5a.plot(air['time'], air['u_pert'], 'b-', lw=1.5, label='Velocity perturbation Δu [m/s]')
    l2 = ax5b.plot(air['time'], np.rad2deg(air['theta']),
                   'g-', lw=1.5, label='Pitch angle θ [deg]')
    ax5a.set_xlabel('Time [s]')
    ax5a.set_ylabel('Velocity perturbation [m/s]', color='b')
    ax5b.set_ylabel('Pitch angle [deg]', color='g')
    ax5a.tick_params(axis='y', labelcolor='b')
    ax5b.tick_params(axis='y', labelcolor='g')
    ax5a.set_title('(5) Aircraft: phugoid mode\n(same response, long-period oscillation)')
    ax5a.legend(l1 + l2, [l.get_label() for l in l1 + l2], loc='best', fontsize=9)
    ax5a.grid(True, alpha=0.3)

    # ── (6) Ship course change ─────────────────────────────────────────
    ax6a = fig.add_subplot(2, 3, 6)
    ax6b = ax6a.twinx()
    l1 = ax6a.plot(ship['time'], np.rad2deg(ship['psi']),
                   'b-', lw=1.8, label='Heading ψ [deg]')
    l_t = ax6a.axhline(np.rad2deg(float(ship['psi_target'])),
                       color='k', ls='--', lw=1.0, label='Target heading')
    l2 = ax6b.plot(ship['time'], np.rad2deg(ship['delta']),
                   'r-', lw=1.0, alpha=0.7, label='Rudder angle δ [deg]')
    ax6a.set_xlabel('Time [s]')
    ax6a.set_ylabel('Heading [deg]', color='b')
    ax6b.set_ylabel('Rudder angle [deg]', color='r')
    ax6a.tick_params(axis='y', labelcolor='b')
    ax6b.tick_params(axis='y', labelcolor='r')
    settle = float(ship['settle_time'])
    title_settle = f'settling: {settle:.0f}s' if settle > 0 else 'not settled'
    ax6a.set_title(f'(6) Ship: course change with the Nomoto model\n(target 30 deg, {title_settle})')
    lines = [l1[0], l_t, l2[0]]
    ax6a.legend(lines, [l.get_label() for l in lines], loc='best', fontsize=9)
    ax6a.grid(True, alpha=0.3)

    plt.tight_layout()
    out1 = f'{out_prefix}.png'
    plt.savefig(out1, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {out1}')

    # ── Ship track ─────────────────────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ship['x'], ship['y'], 'b-', lw=1.8, label='Ship track')
    ax.plot(float(ship['x'][0]), float(ship['y'][0]), 'go', ms=10, label='Departure point')
    ax.plot(float(ship['x'][-1]), float(ship['y'][-1]), 'r^', ms=10, label='Arrival point')
    psi_t = float(ship['psi_target'])
    L_arrow = 1500
    ax.plot([0, L_arrow * np.cos(psi_t)], [0, L_arrow * np.sin(psi_t)],
            'k--', lw=1.0, alpha=0.5, label='Target heading')
    ax.set_xlabel('Eastward distance [m]')
    ax.set_ylabel('Northward distance [m]')
    ax.set_title('Ship track (course-change maneuver to a target heading of 30 deg)')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    out2 = f'{out_prefix}_ship_track.png'
    plt.savefig(out2, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {out2}')

    return [out1, out2]


def print_summary(data: dict):
    print('\n--- Simulation results summary ---')
    for key, label in [('pure_pursuit', 'Pure Pursuit'),
                        ('stanley',      'Stanley     '),
                        ('lqr',          'LQR         ')]:
        r = data[key]
        print(f'  {label}: RMS={r["rms_error"]:.3f} m, '
              f'max steering={r["max_steer_deg"]:.2f} deg')

    d = data['car_dynamic']
    judge = ('understeer' if d['Kv'] > 0 else
             'oversteer' if d['Kv'] < 0 else 'neutral')
    print(f'  Kv={d["Kv"]:.5f} ({judge}), '
          f'yaw gain theory={d["yaw_gain_theory"]:.3f} measured={d["yaw_gain_sim"]:.3f}')

    s = data['ship']
    settle = s['settle_time']
    print(f'  Ship settling time: {settle:.1f} s' if settle > 0 else '  Ship: not settled')
    print()


def main():
    results_path = sys.argv[1] if len(sys.argv) > 1 else 'results.json'
    out_prefix   = sys.argv[2] if len(sys.argv) > 2 else 'cpp_results'

    # Align the output destination with the directory of results.json
    base_dir = os.path.dirname(os.path.abspath(results_path))
    out_prefix = os.path.join(base_dir, os.path.basename(out_prefix))

    if not os.path.exists(results_path):
        print(f'Error: {results_path} not found.')
        print('Please run the following first:')
        print('  python python\\compute_lqr_gain.py lqr_k.json')
        print('  .\\build\\Release\\vehicle_dynamics.exe lqr_k.json results.json')
        sys.exit(1)

    print(f'Loading: {results_path}')
    data = load_results(results_path)

    print_summary(data)

    print(f'Generating plots (font: {_font_used})...')
    saved = plot_all(data, out_prefix)

    print('\nDone. Output files:')
    for p in saved:
        print(f'  {p}')


if __name__ == '__main__':
    main()
