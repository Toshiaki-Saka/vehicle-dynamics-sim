"""
対比用ミニ例: 2リンクロボットアームのラグランジュ法シミュレーション
======================================================================
本リポジトリ本体(自動車・航空機・船舶)は、すべて Newton-Euler 形式で
運動方程式を立てています。docs/why_newton_euler.md で説明したとおり、
非ホロノミック拘束と非保存力が支配する輸送機関では、それが素直だからです。

では、ラグランジュ形式が「自然な選択」になるのはどういう系か。
その典型がロボットアームです。このスクリプトは、対比のための最小例として
2リンク平面アームの運動方程式をラグランジュ法で立て、自由運動(重力下で
振り子のように落ちる様子)をシミュレーションします。

なぜここではラグランジュ法が素直なのか:
  - 関節の拘束はホロノミック(座標どうしの関係)。一般化座標 q = (θ1, θ2)
    を選んだ時点で拘束が消え、未定乗数が要らない。
  - 駆動トルク以外は重力(保存力)が主。L = T - V の枠組みにそのまま乗る。
  - スカラ関数 L ひとつから、機械的な偏微分だけで運動方程式が出る。

運動方程式は標準形  M(q) q̈ + C(q, q̇) q̇ + g(q) = τ  になります。
これを q̈ について解いて状態空間に落とし、RK4 で積分します。
(輸送機関では Newton-Euler が最初から状態方程式に直結するのに対し、
 ここでは M の逆行列を取る一手間が入る ── という違いも観察できます。)

実行方法:
    python python/lagrangian_arm.py

依存ライブラリ: numpy, matplotlib
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ── 物理パラメータ ───────────────────────────────────────────────
M1, M2 = 1.0, 1.0      # 各リンク質量 [kg]
L1, L2 = 1.0, 1.0      # 各リンク長 [m]
LC1, LC2 = 0.5, 0.5    # 各リンク重心位置(根元から)[m]
I1, I2 = M1 * L1**2 / 12.0, M2 * L2**2 / 12.0   # 重心まわり慣性 [kg m^2]
G = 9.81               # 重力加速度 [m/s^2]


def manipulator_matrices(q):
    """
    ラグランジュ法で導いた 2リンクアームの M(q), C(q,q̇)·q̇, g(q) を返す。

    ラグランジアン L = T - V から
        d/dt(∂L/∂q̇) - ∂L/∂q = τ
    を計算すると、平面2リンクアームについては以下の閉じた式になる
    (導出は任意の robotics 教科書、例: Spong et al. "Robot Modeling
     and Control" 第7章 を参照)。
    """
    q1, q2 = q
    c2 = np.cos(q2)

    # 質量(慣性)行列 M(q)
    m11 = (I1 + I2 + M1 * LC1**2 + M2 * (L1**2 + LC2**2 + 2 * L1 * LC2 * c2))
    m12 = I2 + M2 * (LC2**2 + L1 * LC2 * c2)
    m22 = I2 + M2 * LC2**2
    M = np.array([[m11, m12],
                  [m12, m22]])
    return M


def manipulator_rhs(q, dq):
    """コリオリ・遠心項 C(q,q̇)q̇ と重力項 g(q) をまとめて返す。"""
    q1, q2 = q
    dq1, dq2 = dq
    s2 = np.sin(q2)
    h = M2 * L1 * LC2 * s2

    # コリオリ・遠心項
    coriolis = np.array([
        -h * dq2 * (2 * dq1 + dq2),
         h * dq1 * dq1,
    ])

    # 重力項 g(q)
    g1 = ((M1 * LC1 + M2 * L1) * G * np.cos(q1)
          + M2 * LC2 * G * np.cos(q1 + q2))
    g2 = M2 * LC2 * G * np.cos(q1 + q2)
    gravity = np.array([g1, g2])

    return coriolis, gravity


def dynamics(state, tau=np.zeros(2)):
    """
    状態 x = [q1, q2, dq1, dq2] の時間微分を返す。

    運動方程式  M(q) q̈ + C q̇ + g = τ  を q̈ について解く:
        q̈ = M^{-1} (τ - C q̇ - g)
    Newton-Euler なら最初から状態方程式に直結するのに対し、
    ラグランジュ形式ではこの M^{-1} を取る一手間が入る。
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
    """全力学的エネルギー T + V。駆動トルク0なら保存されるはず。"""
    q  = state[:2]
    dq = state[2:]
    M = manipulator_matrices(q)
    T = 0.5 * dq @ M @ dq

    q1, q2 = q
    # 各リンク重心高さ(関節1を原点、x 軸水平基準)
    y1 = LC1 * np.sin(q1)
    y2 = L1 * np.sin(q1) + LC2 * np.sin(q1 + q2)
    # 位置エネルギーの基準点を「全リンクが真下を向いた最下点」に取り、
    # V が常に 0 以上になるようオフセットする(保存量を見やすくするため)。
    v_offset = M1 * G * LC1 + M2 * G * (L1 + LC2)
    V = M1 * G * y1 + M2 * G * y2 + v_offset
    return T + V


def simulate(t_end=10.0, dt=0.002):
    """重力下の自由運動(初期姿勢から落下する振り子的な動き)。"""
    n = int(t_end / dt)
    # 初期姿勢: 第1リンクを水平、第2リンクを90度曲げた状態から、
    # 静止状態でリリース。あとは重力だけで二重振り子的に運動する。
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
    print(" 2リンクロボットアーム ― ラグランジュ法シミュレーション(対比用)")
    print("=" * 66)
    print()
    print(" 本体の輸送機関モデルが Newton-Euler 形式なのに対し、ロボット")
    print(" アームではラグランジュ法が素直になる。その実例として、重力下で")
    print(" 自由運動する2リンクアームを M(q)q̈ + Cq̇ + g = τ から解く。")
    print()

    t, q, e = simulate()

    # 全エネルギーの変動幅を、系が保持するエネルギーの平均値に対する
    # 相対比で示す。駆動トルク0かつ重力(保存力)のみなので、理論上は
    # 一定 ── 変動は RK4 の数値誤差ぶんだけ。
    e_drift = e.max() - e.min()
    e_mean = float(np.mean(e))
    print(f" シミュレーション時間 : {t[-1]:.1f} s ({len(t)} ステップ)")
    print(f" 全エネルギー(平均)  : {e_mean:.4f} J")
    print(f" エネルギー変動幅     : {e_drift:.3e} J "
          f"(平均値の {e_drift / e_mean * 100:.4f} %)")
    print("   → 駆動トルク0かつ重力は保存力なので、全エネルギーは理論上")
    print("      一定。変動は RK4 の数値誤差ぶんだけで、ごく僅か。これは")
    print("      ラグランジュ形式が保存系を素直に扱えることの確認でもある。")
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
    print(f" グラフを保存しました: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
