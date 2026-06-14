"""
C++ シミュレーション結果の可視化
=================================
vehicle_dynamics.exe が出力した results.json を読み込んでグラフを生成する。

使い方:
    python visualize_results.py [results.json] [output_prefix]

引数省略時:
    入力  : results.json (カレントディレクトリ)
    出力  : cpp_results.png / cpp_results_ship_track.png
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

    # ── (1) 軌跡比較 ────────────────────────────────────────────────
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(path_x, path_y, 'k--', lw=1.2, label='目標経路', alpha=0.5)
    for key, hist in [('pure_pursuit', pp), ('stanley', st), ('lqr', lqr)]:
        rms = hist['rms_error']
        ax1.plot(hist['x'], hist['y'], color=colors[key], lw=1.5,
                 label=f"{labels[key]} (RMS={rms:.2f}m)")
    ax1.plot(pp['x'][0], pp['y'][0], 'k*', ms=10, label='開始点')
    ax1.set_xlabel('X 座標 [m]')
    ax1.set_ylabel('Y 座標 [m]')
    ax1.set_title('① 自動車:3制御則の軌跡比較\n(Pure Pursuit / Stanley / LQR)')
    ax1.legend(loc='best', fontsize=8)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # ── (2) 操舵指令 ────────────────────────────────────────────────
    ax2 = fig.add_subplot(2, 3, 2)
    for key, hist in [('pure_pursuit', pp), ('stanley', st), ('lqr', lqr)]:
        ax2.plot(car_time, np.rad2deg(hist['delta']),
                 color=colors[key], lw=1.2, label=labels[key], alpha=0.85)
    ax2.axhline(0, color='k', lw=0.5)
    ax2.set_xlabel('時間 [s]')
    ax2.set_ylabel('操舵角 [度]')
    ax2.set_title('② 自動車:操舵指令比較')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── (3) 動力学 ステップ応答 ──────────────────────────────────────
    ax3a = fig.add_subplot(2, 3, 3)
    ax3b = ax3a.twinx()
    l1 = ax3a.plot(dyn['time'], dyn['vy'], 'b-', lw=1.5, label='横速度 v_y [m/s]')
    l2 = ax3b.plot(dyn['time'], np.rad2deg(dyn['yaw_rate']),
                   'r-', lw=1.5, label='ヨーレート [度/s]')
    ax3a.set_xlabel('時間 [s]')
    ax3a.set_ylabel('横速度 [m/s]', color='b')
    ax3b.set_ylabel('ヨーレート [度/s]', color='r')
    ax3a.tick_params(axis='y', labelcolor='b')
    ax3b.tick_params(axis='y', labelcolor='r')
    ax3a.set_title(f'③ 自動車:動力学2自由度モデル\nステップ応答(v={dyn["vx"]:.0f} m/s)')
    ax3a.legend(l1 + l2, [l.get_label() for l in l1 + l2], loc='best', fontsize=9)
    ax3a.grid(True, alpha=0.3)

    # ── (4) 短周期モード ──────────────────────────────────────────────
    ax4 = fig.add_subplot(2, 3, 4)
    mask = air['time'] <= 15.0
    alpha_approx = air['w'] / 235.0
    ax4.plot(air['time'][mask], np.rad2deg(alpha_approx[mask]),
             'b-', lw=1.5, label='迎角 α ≈ w/V [度]')
    ax4.plot(air['time'][mask], np.rad2deg(air['q'][mask]),
             'r-', lw=1.5, label='ピッチレート q [度/s]')
    ax4.set_xlabel('時間 [s]')
    ax4.set_ylabel('応答量')
    ax4.set_title('④ 航空機:短周期モード\n(エレベータ -1度ステップ・15秒)')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # ── (5) フゴイドモード ────────────────────────────────────────────
    ax5a = fig.add_subplot(2, 3, 5)
    ax5b = ax5a.twinx()
    l1 = ax5a.plot(air['time'], air['u_pert'], 'b-', lw=1.5, label='速度摂動 Δu [m/s]')
    l2 = ax5b.plot(air['time'], np.rad2deg(air['theta']),
                   'g-', lw=1.5, label='ピッチ角 θ [度]')
    ax5a.set_xlabel('時間 [s]')
    ax5a.set_ylabel('速度摂動 [m/s]', color='b')
    ax5b.set_ylabel('ピッチ角 [度]', color='g')
    ax5a.tick_params(axis='y', labelcolor='b')
    ax5b.tick_params(axis='y', labelcolor='g')
    ax5a.set_title('⑤ 航空機:フゴイドモード\n(同応答・長周期振動)')
    ax5a.legend(l1 + l2, [l.get_label() for l in l1 + l2], loc='best', fontsize=9)
    ax5a.grid(True, alpha=0.3)

    # ── (6) 船舶 変針 ──────────────────────────────────────────────────
    ax6a = fig.add_subplot(2, 3, 6)
    ax6b = ax6a.twinx()
    l1 = ax6a.plot(ship['time'], np.rad2deg(ship['psi']),
                   'b-', lw=1.8, label='船首方位 ψ [度]')
    l_t = ax6a.axhline(np.rad2deg(float(ship['psi_target'])),
                       color='k', ls='--', lw=1.0, label='目標方位')
    l2 = ax6b.plot(ship['time'], np.rad2deg(ship['delta']),
                   'r-', lw=1.0, alpha=0.7, label='舵角 δ [度]')
    ax6a.set_xlabel('時間 [s]')
    ax6a.set_ylabel('船首方位 [度]', color='b')
    ax6b.set_ylabel('舵角 [度]', color='r')
    ax6a.tick_params(axis='y', labelcolor='b')
    ax6b.tick_params(axis='y', labelcolor='r')
    settle = float(ship['settle_time'])
    title_settle = f'整定:{settle:.0f}s' if settle > 0 else '未整定'
    ax6a.set_title(f'⑥ 船舶:Nomoto モデルによる変針\n(目標30度・{title_settle})')
    lines = [l1[0], l_t, l2[0]]
    ax6a.legend(lines, [l.get_label() for l in lines], loc='best', fontsize=9)
    ax6a.grid(True, alpha=0.3)

    plt.tight_layout()
    out1 = f'{out_prefix}.png'
    plt.savefig(out1, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {out1}')

    # ── 船舶軌跡 ───────────────────────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ship['x'], ship['y'], 'b-', lw=1.8, label='船舶軌跡')
    ax.plot(float(ship['x'][0]), float(ship['y'][0]), 'go', ms=10, label='出発点')
    ax.plot(float(ship['x'][-1]), float(ship['y'][-1]), 'r^', ms=10, label='到達点')
    psi_t = float(ship['psi_target'])
    L_arrow = 1500
    ax.plot([0, L_arrow * np.cos(psi_t)], [0, L_arrow * np.sin(psi_t)],
            'k--', lw=1.0, alpha=0.5, label='目標方位')
    ax.set_xlabel('東向き距離 [m]')
    ax.set_ylabel('北向き距離 [m]')
    ax.set_title('船舶の航跡(目標方位30度への変針操船)')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    out2 = f'{out_prefix}_ship_track.png'
    plt.savefig(out2, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {out2}')

    return [out1, out2]


def print_summary(data: dict):
    print('\n--- シミュレーション結果サマリ ---')
    for key, label in [('pure_pursuit', 'Pure Pursuit'),
                        ('stanley',      'Stanley     '),
                        ('lqr',          'LQR         ')]:
        r = data[key]
        print(f'  {label}: RMS={r["rms_error"]:.3f} m, '
              f'最大操舵={r["max_steer_deg"]:.2f} 度')

    d = data['car_dynamic']
    judge = ('アンダーステア' if d['Kv'] > 0 else
             'オーバーステア' if d['Kv'] < 0 else 'ニュートラル')
    print(f'  Kv={d["Kv"]:.5f} ({judge}), '
          f'ヨーゲイン 理論={d["yaw_gain_theory"]:.3f} 実測={d["yaw_gain_sim"]:.3f}')

    s = data['ship']
    settle = s['settle_time']
    print(f'  船舶整定時間: {settle:.1f} s' if settle > 0 else '  船舶: 未整定')
    print()


def main():
    results_path = sys.argv[1] if len(sys.argv) > 1 else 'results.json'
    out_prefix   = sys.argv[2] if len(sys.argv) > 2 else 'cpp_results'

    # 出力先をresults.jsonと同じディレクトリに揃える
    base_dir = os.path.dirname(os.path.abspath(results_path))
    out_prefix = os.path.join(base_dir, os.path.basename(out_prefix))

    if not os.path.exists(results_path):
        print(f'Error: {results_path} が見つかりません。')
        print('先に以下を実行してください:')
        print('  python python\\compute_lqr_gain.py lqr_k.json')
        print('  .\\build\\Release\\vehicle_dynamics.exe lqr_k.json results.json')
        sys.exit(1)

    print(f'読み込み: {results_path}')
    data = load_results(results_path)

    print_summary(data)

    print(f'グラフ生成中 (フォント: {_font_used})...')
    saved = plot_all(data, out_prefix)

    print('\n完了。出力ファイル:')
    for p in saved:
        print(f'  {p}')


if __name__ == '__main__':
    main()
