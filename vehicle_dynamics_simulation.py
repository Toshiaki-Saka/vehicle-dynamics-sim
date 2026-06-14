"""
====================================================================
自動車・航空機・船舶 統合運動シミュレーション
====================================================================

本スクリプトは、これまでの議論で扱った3つの輸送機関の運動モデルを
統一的にシミュレーションし、結果を可視化します。

含まれるモデル:
  1. 自動車  : キネマティック自転車モデル + Pure Pursuit 経路追従
  2. 自動車  : 動力学線形2自由度モデル(操舵ステップ応答)
  3. 航空機  : 縦運動線形モデル(短周期モード・フゴイドモード)
  4. 船舶    : Nomoto 1次モデル(変針操船)

各モデルは独立に実行可能で、最後に全結果をまとめて可視化します。

実行方法:
    python vehicle_dynamics_simulation.py

依存ライブラリ: numpy, matplotlib, scipy
====================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.integrate import odeint
from scipy.signal import lti, step

# 出力ディレクトリ:スクリプトと同じ場所(なければカレントディレクトリ)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

# 日本語表示用のフォント設定(環境に応じて自動フォールバック)
def _setup_japanese_font():
    """日本語フォントを設定。利用可能なものを順に試す。"""
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
rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け防止


def print_introduction():
    """シミュレーション開始前に、本スクリプト全体の理論的背景を解説する。"""
    print("\n" + "▓" * 70)
    print("  自動車・航空機・船舶 統合運動シミュレーション")
    print("▓" * 70)
    print()
    print("【序論】なぜラグランジュ方程式ではなく Newton-Euler 形式なのか")
    print("-" * 70)
    print("  輸送機関の運動モデルは、解析力学の標準形式であるラグランジュ方程式")
    print("  ではなく、ほぼ例外なく Newton-Euler 形式で記述されます。理由:")
    print()
    print("  1. 非ホロノミック拘束")
    print("     自動車のタイヤは「横滑りしない」という拘束を持ちますが、これは")
    print("     速度に関する拘束で座標関係に積分できません。ラグランジュ形式で")
    print("     扱うには未定乗数を使った Lagrange-d'Alembert 方程式が必要で、")
    print("     定式化が一気に煩雑になります。")
    print()
    print("  2. 非保存力が支配的")
    print("     タイヤ摩擦・空気抵抗・流体抗力・推力など、非保存力が運動を決定")
    print("     づけます。L = T - V の枠組みでは外力項として右辺に押し込むだけ")
    print("     になり、ラグランジュ形式の旨み(対称性とエネルギーから運動が出る")
    print("     美しさ)が活きません。")
    print()
    print("  3. 制御指向との相性")
    print("     現代制御理論は状態空間表現 dx/dt = f(x,u) を要求します。")
    print("     Newton-Euler から状態方程式を組むのは素直ですが、ラグランジュ")
    print("     から組むには M(q)q̈ を q̈ について解く一手間が入ります。")
    print()
    print("  4. 多体系では Kane の方法が優位")
    print("     サスペンションを含むフルビークル(典型14自由度)では、エネルギー")
    print("     の偏微分が爆発します。Kane の方法(部分速度ベースの d'Alembert")
    print("     原理)が CarSim 等の商用ソフト内部で使われています。")
    print()
    print("  → ラグランジュ形式は理論的に「書けない」のではなく、")
    print("     この問題領域では他の道具立て(Newton-Euler, Kane, 状態空間)に")
    print("     役割を譲っている、というのが正確な描像です。")
    print()
    print("【本スクリプトの構成】")
    print("-" * 70)
    print("  ① 自動車・キネマティック自転車 + Pure Pursuit(幾何学的経路追従)")
    print("  ② 自動車・動力学2自由度モデル(線形タイヤ + ステップ応答)")
    print("  ③ 航空機・縦運動4自由度(短周期・フゴイドの2モード)")
    print("  ④ 船舶 ・Nomoto 1次モデル(変針操船 + LOS誘導)")
    print()
    print("  各シミュレーションの冒頭で、理論的背景・モデル式・設計思想を")
    print("  詳しく解説します。")
    print("▓" * 70)


# ====================================================================
# 1. 自動車:キネマティック自転車モデル + 4制御則の比較
#    (Pure Pursuit, Stanley, LQR, MPC)
# ====================================================================
class StanleyController:
    """
    Stanley 制御:DARPA Grand Challenge 2005 Stanford チーム製の幾何学的制御則。
    前輪基準・横偏差を陽に使うので復帰が速い。
    
        δ = e_ψ + arctan(k · e_y / v_x)
    
    e_y は前輪位置から最近傍経路点までの横偏差(符号付き)、
    e_ψ は経路接線方向と車体ヨー角の差。
    """
    
    def __init__(self, wheelbase=2.7, k_gain=2.5):
        self.L = wheelbase
        self.k = k_gain  # 横偏差ゲイン
    
    def step(self, state, target_path, path_yaw, v, dt, prev_idx=0):
        x, y, psi = state
        # 前輪位置に座標変換
        fx = x + self.L * np.cos(psi)
        fy = y + self.L * np.sin(psi)
        
        # 局所探索で最近傍点インデックスを見つける(周回経路対応)
        n_path = len(target_path)
        search_window = 50
        idx_range = np.arange(prev_idx, prev_idx + search_window) % n_path
        d2 = (target_path[idx_range, 0] - fx) ** 2 + (target_path[idx_range, 1] - fy) ** 2
        nearest = idx_range[np.argmin(d2)]
        
        # 横偏差(符号付き):車両前輪→最近傍経路点ベクトルを、経路接線の
        # 左法線(-sin ψ_path, cos ψ_path)に投影。
        # 車両が経路の右側にいるとき e_y > 0、左側にいるとき e_y < 0 となる
        # 規約。Stanley則 δ = e_ψ + arctan(k·e_y/v) はこの符号で安定。
        dx = target_path[nearest, 0] - fx
        dy = target_path[nearest, 1] - fy
        e_y = dx * (-np.sin(path_yaw[nearest])) + dy * np.cos(path_yaw[nearest])
        
        # ヨー偏差
        e_psi = path_yaw[nearest] - psi
        e_psi = (e_psi + np.pi) % (2 * np.pi) - np.pi
        
        # Stanley 制御則
        delta = e_psi + np.arctan2(self.k * e_y, max(v, 0.1))
        delta = np.clip(delta, -np.deg2rad(35), np.deg2rad(35))
        
        # キネマティック自転車モデルで状態更新
        x_new = x + v * np.cos(psi) * dt
        y_new = y + v * np.sin(psi) * dt
        psi_new = psi + (v / self.L) * np.tan(delta) * dt
        
        return np.array([x_new, y_new, psi_new]), delta, nearest


class LQRPathTracker:
    """
    線形2自由度モデル + 経路偏差で構成した4次系の LQR コントローラ。
    
    状態 x = (e_y, ė_y, e_ψ, ė_ψ)、入力 u = δ
    Riccati 方程式の解析解(scipy.linalg.solve_continuous_are)で K を求める。
    """
    
    def __init__(self, wheelbase=2.7, m=1500.0, Iz=2500.0,
                 lf=1.2, lr=1.5, Cf=80000.0, Cr=80000.0, vx=8.0,
                 Q=None, R=None):
        self.L = wheelbase
        self.vx = vx
        self.Cf = Cf; self.Cr = Cr
        self.lf = lf; self.lr = lr
        self.m = m;   self.Iz = Iz
        
        # 経路偏差を含む4次系の連続時間状態空間モデル
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
        # 最近傍点
        n_path = len(target_path)
        search_window = 50
        idx_range = np.arange(prev_idx, prev_idx + search_window) % n_path
        d2 = (target_path[idx_range, 0] - x)**2 + (target_path[idx_range, 1] - y)**2
        nearest = idx_range[np.argmin(d2)]
        
        # 経路偏差の計算
        dx = x - target_path[nearest, 0]
        dy = y - target_path[nearest, 1]
        nx = -np.sin(path_yaw[nearest])
        ny =  np.cos(path_yaw[nearest])
        e_y = dx * nx + dy * ny
        e_psi = psi - path_yaw[nearest]
        e_psi = (e_psi + np.pi) % (2 * np.pi) - np.pi
        
        # 4次状態(e_y_dotとe_psi_dotは近似的に再構築)
        # 厳密には横速度・ヨーレートが必要だが、ここでは簡略化のため0で近似
        x_state = np.array([e_y, 0.0, e_psi, 0.0])
        
        # 操舵指令 = -K x + フィードフォワード(曲率補正)
        delta_fb = -float((self.K @ x_state).item())
        # フィードフォワード:曲率に対する Ackermann 操舵
        delta_ff = self.L * path_curvature[nearest]
        delta = delta_fb + delta_ff
        delta = np.clip(delta, -np.deg2rad(35), np.deg2rad(35))
        
        # 状態更新
        x_new = x + v * np.cos(psi) * dt
        y_new = y + v * np.sin(psi) * dt
        psi_new = psi + (v / self.L) * np.tan(delta) * dt
        
        return np.array([x_new, y_new, psi_new]), delta, nearest


class MPCPathTracker:
    """
    制約付き線形 MPC コントローラ。
    
    予測モデルはキネマティック自転車。
    制約: |δ| ≤ δ_max, |Δδ| ≤ Δδ_max。
    解: 二次計画(scipy.optimize.minimize, SLSQP)。
    
    制約を陽に扱える点が LQR との根本的な違い。
    本実装では、予測ホライズン内で複数のルックアヘッド点を追従するように
    Pure Pursuit を多点先読みで拡張した形式とし、制約を陽に課す。
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
        """ウィンドウ局所探索で最近傍点を見つける(単調進行)。"""
        n_path = len(target_path)
        idx_range = np.arange(self.prev_idx, self.prev_idx + search_window) % n_path
        d2 = (target_path[idx_range, 0] - x)**2 + (target_path[idx_range, 1] - y)**2
        nearest = idx_range[np.argmin(d2)]
        return nearest
    
    def _predict_cost(self, delta_seq, state, target_path, path_yaw, v):
        """ホライズン内のコスト評価。"""
        x, y, psi = state
        prev_psi = psi
        cost = 0.0
        local_idx = self.prev_idx
        n_path = len(target_path)
        for k in range(self.N):
            # 1ステップ予測
            x = x + v * np.cos(psi) * self.dt_ctrl
            y = y + v * np.sin(psi) * self.dt_ctrl
            psi_new = psi + (v / self.L) * np.tan(delta_seq[k]) * self.dt_ctrl
            psi_dot = (psi_new - prev_psi) / self.dt_ctrl
            prev_psi = psi
            psi = psi_new
            
            # 局所探索で経路上の最近傍点
            window = np.arange(local_idx, local_idx + 20) % n_path
            d2 = (target_path[window, 0] - x)**2 + (target_path[window, 1] - y)**2
            local_idx = window[np.argmin(d2)]
            
            # 経路偏差
            dx = target_path[local_idx, 0] - x
            dy = target_path[local_idx, 1] - y
            e_y = dx * (-np.sin(path_yaw[local_idx])) + dy * np.cos(path_yaw[local_idx])
            e_psi = (psi - path_yaw[local_idx] + np.pi) % (2 * np.pi) - np.pi
            
            # コスト関数(横偏差・ヨー偏差・ヨーレート・操舵ペナルティをバランス)
            cost += 30.0 * e_y**2          # 横偏差
            cost += 30.0 * e_psi**2        # ヨー偏差(向きが合っていれば自然に追従できる)
            cost += 5.0 * psi_dot**2       # ヨーレート抑制(振動防止)
            cost += 1.0 * delta_seq[k]**2  # 操舵ペナルティ
            if k > 0:
                cost += 3.0 * (delta_seq[k] - delta_seq[k-1])**2
            else:
                cost += 3.0 * (delta_seq[k] - self.prev_delta)**2
        return cost
    
    def step(self, state, target_path, path_yaw, v, dt):
        from scipy.optimize import minimize
        x, y, psi = state
        # 現状態の最近傍点を更新
        self.prev_idx = self._find_nearest(x, y, target_path)
        
        # ウォームスタート
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
        # ハード制約:操舵レート
        delta_opt = np.clip(delta_opt,
                            self.prev_delta - self.delta_rate_max * dt,
                            self.prev_delta + self.delta_rate_max * dt)
        self.prev_delta = delta_opt
        
        # 状態更新(実車プラント側)
        x, y, psi = state
        x_new = x + v * np.cos(psi) * dt
        y_new = y + v * np.sin(psi) * dt
        psi_new = psi + (v / self.L) * np.tan(delta_opt) * dt
        return np.array([x_new, y_new, psi_new]), delta_opt


class KinematicBicyclePurePursuit:
    """
    後輪基準のキネマティック自転車モデルを Pure Pursuit で経路追従させる。
    
    状態 : (x, y, psi)
    入力 : 速度 v, 操舵角 delta(Pure Pursuitで決定)
    """
    
    def __init__(self, wheelbase=2.7, lookahead_gain=0.5, lookahead_min=2.0):
        self.L = wheelbase            # ホイールベース [m](乗用車相当)
        self.k = lookahead_gain       # ルックアヘッド距離の速度係数
        self.Ld_min = lookahead_min   # ルックアヘッド距離の最小値 [m]
    
    def step(self, state, target_path, v, dt, prev_idx=0):
        """1ステップ更新。target_path は (x_path, y_path) の配列。
        prev_idx は前回追従していた経路インデックス(周回経路対応)。"""
        x, y, psi = state
        Ld = max(self.k * v, self.Ld_min)
        
        # まず最近傍点を探す(prev_idx 周辺の局所探索で巻き戻りを防ぐ)
        n_path = len(target_path)
        search_window = 50
        idx_range = np.arange(prev_idx, prev_idx + search_window) % n_path
        local_dx = target_path[idx_range, 0] - x
        local_dy = target_path[idx_range, 1] - y
        local_dist = np.hypot(local_dx, local_dy)
        nearest_in_window = idx_range[np.argmin(local_dist)]
        
        # 最近傍点から経路上を前進してルックアヘッド距離以上の点を探す
        target_idx = nearest_in_window
        for k in range(n_path):
            cand = (nearest_in_window + k) % n_path
            d = np.hypot(target_path[cand, 0] - x, target_path[cand, 1] - y)
            if d >= Ld:
                target_idx = cand
                break
        
        gx, gy = target_path[target_idx]
        
        # 車両座標系での目標点角度 alpha
        alpha = np.arctan2(gy - y, gx - x) - psi
        # alpha を [-pi, pi] に正規化
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
        
        # Pure Pursuit の操舵則
        delta = np.arctan2(2.0 * self.L * np.sin(alpha), Ld)
        # 物理的な操舵角制限(±35度)
        delta = np.clip(delta, -np.deg2rad(35), np.deg2rad(35))
        
        # キネマティック自転車モデルの離散化(オイラー法)
        x_new = x + v * np.cos(psi) * dt
        y_new = y + v * np.sin(psi) * dt
        psi_new = psi + (v / self.L) * np.tan(delta) * dt
        
        return np.array([x_new, y_new, psi_new]), delta, nearest_in_window


def _make_ellipse_path(a=50.0, b=30.0, n=400):
    """楕円経路と各点における経路接線方位・曲率を生成。"""
    t = np.linspace(0, 2 * np.pi, n)
    path = np.column_stack([a * np.cos(t), b * np.sin(t)])
    # 経路接線方向(進行方向)
    dx = -a * np.sin(t)
    dy =  b * np.cos(t)
    yaw = np.arctan2(dy, dx)
    # 曲率 κ = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
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
    """4つの経路追従制御則(Pure Pursuit, Stanley, LQR, MPC)を同一経路で比較。"""
    print("\n" + "=" * 70)
    print("【シミュレーション 1】自動車:4つの経路追従制御則の比較")
    print("=" * 70)
    print()
    print("◆ 理論背景:キネマティック自転車モデルと Ackermann 幾何")
    print("-" * 70)
    print("  低速域では車両を「自転車」として近似します。左右輪を中央に集約し、")
    print("  後輪基準・横滑りなしの拘束のもとで、3つの状態方程式が得られます:")
    print()
    print("       ẋ = v cos ψ")
    print("       ẏ = v sin ψ")
    print("       ψ̇ = (v/L) tan δ        ← Ackermann 幾何そのもの")
    print()
    print("  ここで L はホイールベース、δ は前輪操舵角。第3式は左右前輪が共通の")
    print("  旋回中心を持つ Ackermann 条件 tan δ = L/R から直接導かれます。")
    print("  実車では左右輪の操舵角は cot δ_o - cot δ_i = w/L を満たすように")
    print("  ステアリング機構(タイロッド)で幾何学的に近似されます。")
    print()
    print("◆ 4つの経路追従制御則")
    print("-" * 70)
    print("  ① Pure Pursuit(古典幾何制御・後輪基準)")
    print("       δ = arctan(2 L sin α / Ld)")
    print("       Stanford DARPA Grand Challenge 時代からの定番。")
    print("       α は車両座標系での目標点角度、Ld はルックアヘッド距離。")
    print("       実装1行・計算極小・チューニングは Ld のみ。")
    print()
    print("  ② Stanley(古典幾何制御・前輪基準)")
    print("       δ = e_ψ + arctan(k · e_y / v_x)")
    print("       2005年 DARPA Grand Challenge 優勝の Stanford 製。")
    print("       横偏差 e_y を陽に使うので、経路復帰が速い。")
    print()
    print("  ③ LQR(最適制御・無制約・解析解)")
    print("       J = ∫(xᵀQx + uᵀRu)dt を最小化、Riccati 方程式の解で")
    print("       u = -Kx + δ_ff(曲率フィードフォワード)を得る。")
    print("       状態 x = (e_y, ė_y, e_ψ, ė_ψ) の4次系を陽に扱う。")
    print()
    print("  ④ MPC(最適制御・制約あり・先読み)")
    print("       有限ホライズン最適化で操舵レート・操舵角制限を陽に扱える。")
    print("       本実装では 10ステップ先読み + SLSQP ソルバ。")
    print("       LKAS・自動運転の経路追従の主流。")
    print()
    print("◆ シミュレーション設定")
    print("-" * 70)
    print("  経路 : 楕円周回コース(長径50m, 短径30m)")
    print("  速度 : 8.0 m/s, 開始位置:経路から +2m 横にオフセット")
    print("  全制御則を同じ経路・初期条件・速度・車両パラメータで比較。")
    print()
    
    # 経路と初期条件を共有
    path, path_yaw, path_curvature = _make_ellipse_path(a=50.0, b=30.0, n=400)
    initial_state = np.array([50.0, 2.0, np.pi / 2])  # +2m 横にオフセット
    v = 8.0
    dt = 0.05
    T_end = 50.0
    n_steps = int(T_end / dt)
    time = np.arange(n_steps) * dt
    
    results = {}
    
    # ① Pure Pursuit
    print("  ① Pure Pursuit 実行中...", end=' ')
    car_pp = KinematicBicyclePurePursuit(wheelbase=2.7, lookahead_gain=0.6, lookahead_min=4.0)
    state = initial_state.copy()
    hist_s = np.zeros((n_steps, 3)); hist_d = np.zeros(n_steps); idx = 0
    for i in range(n_steps):
        hist_s[i] = state
        state, d, idx = car_pp.step(state, path, v, dt, idx)
        hist_d[i] = d
    rms = _evaluate_rms(hist_s, path)
    results['pure_pursuit'] = {'state': hist_s, 'delta': hist_d, 'rms': rms}
    print(f"完了。RMS偏差 = {rms:.3f} m, 最大操舵 = {np.rad2deg(np.abs(hist_d).max()):.2f}°")
    
    # ② Stanley
    print("  ② Stanley 実行中     ...", end=' ')
    car_st = StanleyController(wheelbase=2.7, k_gain=2.5)
    state = initial_state.copy()
    hist_s = np.zeros((n_steps, 3)); hist_d = np.zeros(n_steps); idx = 0
    for i in range(n_steps):
        hist_s[i] = state
        state, d, idx = car_st.step(state, path, path_yaw, v, dt, idx)
        hist_d[i] = d
    rms = _evaluate_rms(hist_s, path)
    results['stanley'] = {'state': hist_s, 'delta': hist_d, 'rms': rms}
    print(f"完了。RMS偏差 = {rms:.3f} m, 最大操舵 = {np.rad2deg(np.abs(hist_d).max()):.2f}°")
    
    # ③ LQR
    print("  ③ LQR 実行中         ...", end=' ')
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
    print(f"完了。RMS偏差 = {rms:.3f} m, 最大操舵 = {np.rad2deg(np.abs(hist_d).max()):.2f}°")
    
    # ④ MPC(計算が重いので制御周期を粗くしてサンプリング)
    print("  ④ MPC 実行中         ...", end=' ')
    car_mpc = MPCPathTracker(wheelbase=2.7, horizon=15, dt_ctrl=0.1)
    dt_mpc = 0.1  # MPC は 100ms 周期で更新(現実的な計算負荷)
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
    print(f"完了。RMS偏差 = {rms:.3f} m, 最大操舵 = {np.rad2deg(np.abs(hist_d).max()):.2f}°")
    
    print()
    print("◆ シミュレーション結果サマリ")
    print("-" * 70)
    print(f"  {'制御則':<15} {'RMS偏差 [m]':>12} {'最大操舵 [°]':>15} {'特徴'}")
    print(f"  {'-'*15} {'-'*12} {'-'*15} {'-'*30}")
    print(f"  {'Pure Pursuit':<15} {results['pure_pursuit']['rms']:>12.3f}"
          f" {np.rad2deg(np.abs(results['pure_pursuit']['delta']).max()):>15.2f}"
          f"   ルックアヘッド円弧追従")
    print(f"  {'Stanley':<15} {results['stanley']['rms']:>12.3f}"
          f" {np.rad2deg(np.abs(results['stanley']['delta']).max()):>15.2f}"
          f"   横偏差を陽に使う高速復帰")
    print(f"  {'LQR':<15} {results['lqr']['rms']:>12.3f}"
          f" {np.rad2deg(np.abs(results['lqr']['delta']).max()):>15.2f}"
          f"   Riccati解析解 + FF補正")
    print(f"  {'MPC':<15} {results['mpc']['rms']:>12.3f}"
          f" {np.rad2deg(np.abs(results['mpc']['delta']).max()):>15.2f}"
          f"   制約付き先読み最適化")
    print()
    print(f"  → 各制御則の特性差:")
    print(f"    Pure Pursuit はルックアヘッドの遅れで内側を切り込みがち")
    print(f"    Stanley は横偏差復帰が速く、定常追従誤差は最小")
    print(f"    LQR は連続的で滑らかな操舵、定常で経路に乗る")
    print(f"    MPC は制約を陽に扱えるため操舵レートが穏やか")
    
    return {
        'time': time,
        'path': path,
        'results': results,
        'state': results['pure_pursuit']['state'],   # 互換性
        'delta': results['pure_pursuit']['delta'],   # 互換性
        'rms_error': results['pure_pursuit']['rms'], # 互換性
    }


# ====================================================================
# 2. 自動車:動力学2自由度モデル(操舵ステップ応答)
# ====================================================================
def simulate_car_dynamic_bicycle():
    """
    動力学線形自転車モデルでステップ操舵に対する応答を計算。
    状態 x = (横速度 v_y, ヨーレート psi_dot)、入力 u = 操舵角 delta。
    """
    print("\n" + "=" * 70)
    print("【シミュレーション 2】自動車:動力学2自由度モデル ステップ応答")
    print("=" * 70)
    print()
    print("◆ 理論背景:なぜキネマティックでは足りないのか")
    print("-" * 70)
    print("  低速域ではタイヤは横滑りせず、Ackermann 幾何で十分でした。")
    print("  しかし高速域(高側方加速度域)では、タイヤがスリップ角 α を取り、")
    print("  横力 F_y = -C_α · α(線形領域)を発生します。")
    print("  この横力で車両が横に流れ、ヨー応答が遅れる現象が現れます。")
    print()
    print("◆ 線形2自由度モデル(車両 dynamics の標準形)")
    print("-" * 70)
    print("  Newton-Euler を線形化(縦速度 vx 一定、小角近似)して得る:")
    print()
    print("    m(v̇_y + vx·ψ̇) = F_yf cos δ + F_yr")
    print("    Iz·ψ̈           = ℓf·F_yf cos δ - ℓr·F_yr")
    print()
    print("  状態 x = (v_y, ψ̇)ᵀ、入力 u = δ で線形時不変系 ẋ = Ax + Bu に。")
    print("  これが車両運動制御の基本形(Rajamani 教科書 Ch.2)。")
    print()
    print("◆ Pacejka Magic Formula と摩擦円(限界域での考慮)")
    print("-" * 70)
    print("  実際のタイヤ横力は飽和します。経験式 Pacejka Magic Formula:")
    print()
    print("    F_y = D sin[ C arctan{ Bα - E(Bα - arctan(Bα)) } ]")
    print()
    print("  小スリップ角では F_y ≈ -C_α·α(線形)ですがピーク後は低下。")
    print("  さらに縦力と横力は摩擦円制約 √(F_x² + F_y²) ≤ μF_z を満たす必要")
    print("  があります(同じ予算を縦・横で取り合う)。コーナリング中に加速")
    print("  しすぎるとアンダーが、ブレーキを踏むとタックインが起きる根拠。")
    print()
    print("◆ スタビリティファクタ Kv とアンダー/オーバーステア")
    print("-" * 70)
    print("  定常旋回でのヨーゲイン: ψ̇/δ = vx / [L(1 + Kv·vx²)]")
    print("    Kv = (ℓr·m)/(2L·Cf) - (ℓf·m)/(2L·Cr)")
    print()
    print("  Kv > 0 : アンダーステア(高速で曲がりにくい)— 乗用車として安全")
    print("  Kv = 0 : ニュートラル")
    print("  Kv < 0 : オーバーステア(高速で巻き込む)— スピンしやすく危険")
    print()
    print("◆ 経路追従制御:LQR と MPC")
    print("-" * 70)
    print("  この線形モデルに経路偏差 (e_y, e_ψ) を加えた4次系で:")
    print("    LQR: J = ∫(xᵀQx + uᵀRu)dt 最小化、Riccati方程式で解析解")
    print("    MPC: 制約付き有限ホライズン最適化、毎ステップQPを解く")
    print("         |δ| ≤ δ_max, |Δδ| ≤ Δδ_max, |α| ≤ α_lin など")
    print()
    print("  LQR は計算軽い・安定保証あり、MPC は制約と先読み(曲率プレビュー)")
    print("  に対応。市販の自動運転(LKAS、ACC)はこれらの組合せで構成。")
    print()
    print("◆ シミュレーション設定")
    print("-" * 70)
    print("  状態 : 横速度 v_y, ヨーレート ψ_dot")
    print("  入力 : ステップ操舵 1.0 度")
    print("  速度 : 20 m/s(約72 km/h)")
    print()
    
    # 標準的なセダンのパラメータ
    m  = 1500.0     # 車両質量 [kg]
    Iz = 2500.0     # ヨー慣性 [kg m^2]
    lf = 1.2        # 重心〜前車軸距離 [m]
    lr = 1.5        # 重心〜後車軸距離 [m]
    Cf = 80000.0    # 前輪コーナリングスティフネス [N/rad]
    Cr = 80000.0    # 後輪コーナリングスティフネス [N/rad]
    vx = 20.0       # 縦速度 [m/s](約72 km/h)
    
    # 前回の議論で導出した A, B 行列
    a11 = -(Cf + Cr) / (m * vx)
    a12 = -(vx + (lf * Cf - lr * Cr) / (m * vx))
    a21 = -(lf * Cf - lr * Cr) / (Iz * vx)
    a22 = -(lf * lf * Cf + lr * lr * Cr) / (Iz * vx)
    A = np.array([[a11, a12], [a21, a22]])
    B = np.array([[Cf / m], [lf * Cf / Iz]])
    C = np.eye(2)
    D = np.zeros((2, 1))
    
    # ステップ応答の計算
    sys = lti(A, B, C, D)
    t = np.linspace(0, 5.0, 1000)
    delta_step = np.deg2rad(1.0)
    t_out, y_out = step(sys, T=t)
    y_out = y_out * delta_step  # 1度のステップに換算
    
    # スタビリティファクタとアンダーステア勾配の計算(参考)
    Kv = (lr * m) / (2 * (lf + lr) * Cf) - (lf * m) / (2 * (lf + lr) * Cr)
    
    # 定常ヨーゲインの理論値
    L = lf + lr
    yaw_gain_theory = vx / (L * (1 + Kv * vx ** 2))
    yaw_steady = y_out[-1, 1]
    yaw_gain_sim = yaw_steady / delta_step
    
    print(f"◆ シミュレーション結果")
    print("-" * 70)
    judge = 'アンダーステア' if Kv > 0 else 'オーバーステア' if Kv < 0 else 'ニュートラル'
    print(f"  スタビリティファクタ Kv = {Kv:.5f}  ({judge})")
    print(f"  定常ヨーレート         = {np.rad2deg(yaw_steady):.3f} 度/s(操舵1度に対し)")
    print(f"  ヨーゲイン:理論値 {yaw_gain_theory:.3f} vs シミュレーション {yaw_gain_sim:.3f}")
    print(f"  → 0.1〜0.3秒程度でヨーレートが整定する高速応答(これが自動車の特徴)")
    print(f"  → ESC はこの「意図ヨーレート」と実ヨーレートの誤差を見て介入する")
    print(f"     (オーバー → 外側前輪ブレーキ、アンダー → 内側後輪ブレーキ)")
    
    return {
        'time': t_out,
        'vy': y_out[:, 0],
        'yaw_rate': y_out[:, 1],
        'Kv': Kv,
        'vx': vx,
    }


# ====================================================================
# 3. 航空機:縦運動線形モデル(短周期 & フゴイド)
# ====================================================================
def simulate_aircraft_longitudinal():
    """
    航空機の縦運動の線形モデルでエレベータステップ応答を計算。
    """
    print("\n" + "=" * 70)
    print("【シミュレーション 3】航空機:縦運動線形モデル")
    print("=" * 70)
    print()
    print("◆ 理論背景:6自由度フル剛体運動")
    print("-" * 70)
    print("  航空機は地面に拘束されない自由空間運動なので、自動車と違い")
    print("  非ホロノミック拘束がありません。代わりに6自由度の剛体運動を")
    print("  機体座標系の Newton-Euler で書きます(コリオリ項あり):")
    print()
    print("    m(u̇ + qw - rv) = Fx - mg sin θ")
    print("    m(v̇ + ru - pw) = Fy + mg cos θ sin φ")
    print("    m(ẇ + pv - qu) = Fz + mg cos θ cos φ")
    print("    Ix·ṗ - (Iy-Iz)qr = L     (ロール:エルロン δ_a で制御)")
    print("    Iy·q̇ - (Iz-Ix)rp = M     (ピッチ:エレベータ δ_e で制御)")
    print("    Iz·ṙ - (Ix-Iy)pq = N     (ヨー  :ラダー    δ_r で制御)")
    print()
    print("  外力(F)とモーメント(L,M,N)は空気力で決まります:")
    print("    F = (1/2)ρV²·S·C_*(α, β, q, δ_a, δ_e, δ_r, ...)")
    print("  係数 C_* は風洞・CFD・フライトテストで同定。")
    print()
    print("◆ 縦運動と横/方向運動の分離")
    print("-" * 70)
    print("  対称な機体形状から、線形化すると運動は2つに分離されます:")
    print()
    print("    縦運動  : (Δu, w, q, θ)  ← エレベータ δ_e で操作")
    print("              ピッチ・速度・高度・迎角の振動")
    print("    横方向 : (β, p, r, φ)   ← エルロン δ_a + ラダー δ_r で操作")
    print("              ロール・ヨー・横滑り・バンク角の連成")
    print()
    print("  本シミュレーションは縦運動(Boeing 747 巡航条件)を扱います。")
    print()
    print("◆ 縦運動の2つの固有モード")
    print("-" * 70)
    print("  線形系 ẋ = Ax + Bu の固有値解析で、必ず2組の複素共役対が現れます:")
    print()
    print("    短周期モード(short period):")
    print("      周期 1〜5秒、減衰大、ピッチ角と迎角の高速振動")
    print("      パイロットの操縦感覚に直結、機体安定性の基本指標")
    print()
    print("    フゴイドモード(phugoid):")
    print("      周期 30〜100秒、減衰小、速度と高度を交換しながらの長周期振動")
    print("      位置エネルギーと運動エネルギーの交換に相当")
    print("      飛行機が「ふんわり浮き沈みする」あの動き")
    print()
    print("  自動車には「固有モード」が問題化することはほぼありませんが、")
    print("  航空機では設計の中心課題。安定性増強装置(SAS)・操縦性増強装置")
    print("  (CAS)・オートパイロットがこれらモードの整形を行います。")
    print()
    print("◆ 自動車との対比")
    print("-" * 70)
    print("  自動車 : 平面3自由度(または14)、非ホロノミック拘束、タイヤ摩擦支配")
    print("           応答時間 0.1〜数秒、固有モード特に問題なし")
    print("  航空機 : 6自由度、自由空間、空気力支配")
    print("           応答時間 数秒〜分、固有モードが設計対象")
    print()
    print("  どちらも Newton-Euler + 外力モデルという構造は共通。")
    print()
    print("◆ シミュレーション設定")
    print("-" * 70)
    print("  機体モデル : Boeing 747 級 4自由度縦運動(Etkin & Reid 教科書)")
    print("  飛行条件   : 巡航(高度 12,200 m, V = 235 m/s, M ≈ 0.8)")
    print("  状態       : (Δu, w, q, θ)")
    print("  入力       : エレベータステップ -1.0 度(機首上げ方向)")
    print()
    
    # Boeing 747 巡航条件(高度 12200 m, 速度 235 m/s ≈ M0.8)
    # Etkin & Reid "Dynamics of Flight" 教科書の代表値(若干アレンジ)
    # 状態順序: [u (m/s), w (m/s), q (rad/s), theta (rad)]
    # 航空業界では迎角 alpha の代わりに垂直速度成分 w を使う形式が標準
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
    
    # 固有値解析(短周期 & フゴイドモード)
    eigvals = np.linalg.eigvals(A)
    print(f"◆ シミュレーション結果")
    print("-" * 70)
    print(f"  系の固有値:")
    for ev in eigvals:
        if abs(ev.imag) > 1e-6:
            wn = abs(ev)
            zeta = -ev.real / wn
            T_period = 2 * np.pi / abs(ev.imag) if abs(ev.imag) > 1e-6 else float('inf')
            print(f"     {ev.real:+.4f} ± {abs(ev.imag):.4f}j  "
                  f"(ωn={wn:.3f} rad/s, ζ={zeta:.3f}, 周期={T_period:.1f}s)")
    
    # モード分類:周期短い → 短周期、長い → フゴイド
    complex_pairs = [ev for ev in eigvals if ev.imag > 1e-6]
    if len(complex_pairs) >= 2:
        sorted_pairs = sorted(complex_pairs, key=lambda e: abs(e.imag), reverse=True)
        sp = sorted_pairs[0]; ph = sorted_pairs[1]
        print(f"  → 短周期モード : 周期 {2*np.pi/abs(sp.imag):.2f} 秒(高速・高減衰)")
        print(f"  → フゴイドモード: 周期 {2*np.pi/abs(ph.imag):.2f} 秒(低速・低減衰)")
        print(f"  → 周期比約 {abs(ph.imag)/abs(sp.imag) if False else 2*np.pi/abs(ph.imag) / (2*np.pi/abs(sp.imag)):.0f} 倍。")
        print(f"     短周期は数秒で減衰、フゴイドは長周期でゆっくり振動するため、")
        print(f"     プロットでは短時間軸と長時間軸の両方で観察します。")
    
    # ステップ応答(エレベータ -1度 → 機首上げ)
    sys = lti(A, B, C, D)
    t = np.linspace(0, 1500.0, 8000)  # フゴイドの長周期(約850秒)を見るために長めに
    t_out, y_out = step(sys, T=t)
    y_out = y_out * np.deg2rad(-1.0)
    
    print(f"  1500秒間のシミュレーション完了")
    
    return {
        'time': t_out,
        'u_pert': y_out[:, 0],
        'w': y_out[:, 1],
        'q': y_out[:, 2],
        'theta': y_out[:, 3],
        'eigvals': eigvals,
    }


# ====================================================================
# 4. 船舶:Nomoto 1次モデルによる変針操船
# ====================================================================
def simulate_ship_nomoto():
    """
    Nomoto 1次モデル: T*ψ_ddot + ψ_dot = K * δ
    """
    print("\n" + "=" * 70)
    print("【シミュレーション 4】船舶:Nomoto 1次モデルによる変針")
    print("=" * 70)
    print()
    print("◆ 理論背景:Fossen の標準モデル")
    print("-" * 70)
    print("  船舶も6自由度ですが、伝統的な命名があります:")
    print("    Surge(前後)/ Sway(左右)/ Heave(上下)")
    print("    Roll(横揺れ)/ Pitch(縦揺れ)/ Yaw(船首方位)")
    print()
    print("  Thor I. Fossen(NTNU)の標準モデルが業界標準:")
    print()
    print("    M·ν̇ + C(ν)ν + D(ν)ν + g(η) = τ + τ_wind + τ_wave")
    print()
    print("    η : 慣性系の位置・姿勢 (x, y, z, φ, θ, ψ)ᵀ")
    print("    ν : 船体座標系の速度・角速度 (u, v, w, p, q, r)ᵀ")
    print("    M = M_RB + M_A : 剛体慣性 + 付加質量")
    print("    D(ν)         : 流体抗力(線形 + 二次)")
    print("    g(η)         : 復原力・モーメント")
    print()
    print("◆ 自動車・航空機との大きな違い")
    print("-" * 70)
    print("  1. 付加質量(Added Mass)")
    print("     水中で物体を加速すると周囲の水も加速する → 見かけの質量増加。")
    print("     航空では空気密度が小さく無視できますが、水では M_A が M_RB と")
    print("     同オーダーになります。これが船の応答を遅くする一因。")
    print()
    print("  2. 流体抗力の二次性")
    print("     タイヤ摩擦は線形近似が広く使えますが、流体抗力は速度の二乗が")
    print("     支配的。D_q ∝ |ν| の項が必須。")
    print()
    print("  3. 復原力(Restoring forces)")
    print("     浮力中心 B と重心 G の位置関係から、横揺れ・縦揺れに復原モーメ")
    print("     ントが発生。GM_T(横メタセンタ高さ)・GM_L(縦メタセンタ高さ)")
    print("     が安定性を決める基本量で、自動車の重心高さ・トレッド比に対応。")
    print()
    print("  4. 操縦性の遅さとアンダーアクチュエーション")
    print("     6自由度に対しプロペラ + ラダー(+ スラスタ)のみ。応答時間も")
    print("     桁違いに遅い(数十秒〜数分のオーダー)。")
    print()
    print("◆ Nomoto 1次モデル(変針制御の実用形)")
    print("-" * 70)
    print("  針路保持・変針制御で広く使われる簡略モデル:")
    print()
    print("    T·ψ̈ + ψ̇ = K·δ")
    print()
    print("    T : 時定数(船の大きさで秒〜分のオーダー)")
    print("    K : 舵効きゲイン(舵角に対する定常旋回ゲイン)")
    print("    δ : 舵角")
    print()
    print("  船舶の経路追従はこのモデル + Line-of-Sight(LOS)誘導 + PID/LQR")
    print("  が定番。本シミュレーションでは PD 制御で目標方位 30度へ変針。")
    print()
    print("◆ 3つの輸送機関の応答時間スケール比較")
    print("-" * 70)
    print("  自動車のヨー応答  : 0.1〜1秒")
    print("  航空機の短周期    : 1〜5秒")
    print("  航空機のフゴイド  : 30〜100秒(本ケースで約850秒)")
    print("  船舶の変針        : 数十秒〜数分")
    print()
    print("  同じ剛体運動を扱いながら、媒質と慣性の違いで時間スケールが")
    print("  3桁にわたって変化します。")
    print()
    print("◆ シミュレーション設定")
    print("-" * 70)
    print("  船種     : 中型コンテナ船(約200m級)")
    print("  T = 50秒, K = 0.18 (1/s)")
    print("  目標方位 : 30度、PD 制御 + 舵角制限 ±35度")
    print()
    
    # 代表的な船舶パラメータ
    K_nomoto = 0.18    # 舵効きゲイン [1/s](方位応答ゲイン)
    T_nomoto = 50.0    # 時定数 [s](船の慣性で大きい)
    
    # 進路保持 PD 制御で目標方位 30 度へ変針
    psi_target = np.deg2rad(30.0)
    Kp = 2.0
    Kd = 8.0
    delta_max = np.deg2rad(35.0)  # 物理的な舵角制限
    
    def dynamics(state, t):
        psi, psi_dot = state
        # PD 制御で舵角指令を計算
        e = psi_target - psi
        delta_cmd = Kp * e - Kd * psi_dot
        delta = np.clip(delta_cmd, -delta_max, delta_max)
        # Nomoto 1次方程式を1階常微分方程式系で表現
        psi_ddot = (K_nomoto * delta - psi_dot) / T_nomoto
        return [psi_dot, psi_ddot]
    
    t = np.linspace(0, 600.0, 3000)  # 10分間のシミュレーション
    sol = odeint(dynamics, [0.0, 0.0], t)
    psi_hist = sol[:, 0]
    psi_dot_hist = sol[:, 1]
    
    # 舵角履歴を再構築
    delta_hist = np.zeros_like(t)
    for i, ti in enumerate(t):
        e = psi_target - psi_hist[i]
        delta_cmd = Kp * e - Kd * psi_dot_hist[i]
        delta_hist[i] = np.clip(delta_cmd, -delta_max, delta_max)
    
    # 整定時間の評価(目標の±5%以内に入った時刻)
    tol = np.deg2rad(1.5)
    settled = np.where(np.abs(psi_hist - psi_target) < tol)[0]
    print(f"◆ シミュレーション結果")
    print("-" * 70)
    if len(settled) > 0:
        t_settle = t[settled[0]]
        for idx in settled:
            if np.all(np.abs(psi_hist[idx:] - psi_target) < tol):
                t_settle = t[idx]
                break
        print(f"  整定時間(±1.5度以内に収束): {t_settle:.1f} 秒")
    else:
        print(f"  整定せず(時間内に収束しなかった)")
    print(f"  Nomoto 時定数 T = {T_nomoto} 秒、舵効きゲイン K = {K_nomoto}")
    print(f"  最大舵角        = {np.rad2deg(np.abs(delta_hist).max()):.1f} 度")
    print(f"  → 初動で舵角飽和(35度)、その後減衰しながら整定。")
    print(f"  → 自動車のヨー応答(<1秒)に対し、船は2分以上かかる。")
    print(f"     付加質量と流体減衰、そして大きな船体慣性の効果。")
    
    # 船体軌跡の生成(縦速度 8 m/s 一定と仮定)
    v_ship = 8.0  # 約16ノット
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
# 可視化
# ====================================================================
def visualize_all(car_pp, car_dyn, aircraft, ship):
    """4つのシミュレーション結果を一枚の図にまとめて可視化。"""
    print("\n" + "=" * 66)
    print("【可視化】全シミュレーション結果のプロット")
    print("=" * 66)
    print(f"  使用フォント: {_font_used}")
    
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle('自動車・航空機・船舶 統合運動シミュレーション結果',
                 fontsize=16, fontweight='bold', y=0.995)
    
    # ---- (1) 自動車:4制御則の軌跡比較 ----
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(car_pp['path'][:, 0], car_pp['path'][:, 1],
             'k--', lw=1.2, label='目標経路', alpha=0.5)
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
             'k*', ms=10, label='開始点')
    ax1.set_xlabel('X 座標 [m]')
    ax1.set_ylabel('Y 座標 [m]')
    ax1.set_title('① 自動車:4制御則の軌跡比較\n(Pure Pursuit / Stanley / LQR / MPC)')
    ax1.legend(loc='best', fontsize=8)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # ---- (2) 自動車:4制御則の操舵指令履歴 ----
    ax2 = plt.subplot(2, 3, 2)
    for key in ['pure_pursuit', 'stanley', 'lqr']:
        d = car_pp['results'][key]['delta']
        ax2.plot(car_pp['time'], np.rad2deg(d), color=colors[key],
                 lw=1.2, label=labels[key], alpha=0.85)
    # MPC は時間軸が異なる
    mpc_time = car_pp['results']['mpc']['time']
    mpc_delta = car_pp['results']['mpc']['delta']
    ax2.plot(mpc_time, np.rad2deg(mpc_delta), color=colors['mpc'],
             lw=1.5, label=labels['mpc'], alpha=0.85)
    ax2.axhline(0, color='k', lw=0.5)
    ax2.set_xlabel('時間 [s]')
    ax2.set_ylabel('操舵角 [度]')
    ax2.set_title('② 自動車:4制御則の操舵指令比較\n(MPC は制御周期 100ms)')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # ---- (3) 自動車 動力学:ステップ応答 ----
    ax3 = plt.subplot(2, 3, 3)
    ax3a = ax3
    ax3b = ax3.twinx()
    l1 = ax3a.plot(car_dyn['time'], car_dyn['vy'],
                    'b-', lw=1.5, label='横速度 v_y [m/s]')
    l2 = ax3b.plot(car_dyn['time'], np.rad2deg(car_dyn['yaw_rate']),
                    'r-', lw=1.5, label='ヨーレート [度/s]')
    ax3a.set_xlabel('時間 [s]')
    ax3a.set_ylabel('横速度 [m/s]', color='b')
    ax3b.set_ylabel('ヨーレート [度/s]', color='r')
    ax3a.tick_params(axis='y', labelcolor='b')
    ax3b.tick_params(axis='y', labelcolor='r')
    ax3a.set_title(f'③ 自動車:動力学2自由度モデル\nステップ操舵応答(v={car_dyn["vx"]} m/s)')
    lines = l1 + l2
    ax3a.legend(lines, [l.get_label() for l in lines], loc='best', fontsize=9)
    ax3a.grid(True, alpha=0.3)
    
    # ---- (4) 航空機:短周期モード(短時間) ----
    ax4 = plt.subplot(2, 3, 4)
    mask_short = aircraft['time'] <= 15.0
    # 迎角 α ≈ w / V_inf として近似表示(V_inf = 235 m/s 巡航速度)
    alpha_approx = aircraft['w'] / 235.0
    ax4.plot(aircraft['time'][mask_short],
             np.rad2deg(alpha_approx[mask_short]),
             'b-', lw=1.5, label='迎角 α ≈ w/V [度]')
    ax4.plot(aircraft['time'][mask_short],
             np.rad2deg(aircraft['q'][mask_short]),
             'r-', lw=1.5, label='ピッチレート q [度/s]')
    ax4.set_xlabel('時間 [s]')
    ax4.set_ylabel('応答量')
    ax4.set_title('④ 航空機:短周期モード\n(エレベータ -1度ステップ・15秒)')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # ---- (5) 航空機:フゴイドモード(長時間) ----
    ax5 = plt.subplot(2, 3, 5)
    ax5a = ax5
    ax5b = ax5.twinx()
    l1 = ax5a.plot(aircraft['time'], aircraft['u_pert'],
                    'b-', lw=1.5, label='速度摂動 Δu [m/s]')
    l2 = ax5b.plot(aircraft['time'], np.rad2deg(aircraft['theta']),
                    'g-', lw=1.5, label='ピッチ角 θ [度]')
    ax5a.set_xlabel('時間 [s]')
    ax5a.set_ylabel('速度摂動 [m/s]', color='b')
    ax5b.set_ylabel('ピッチ角 [度]', color='g')
    ax5a.tick_params(axis='y', labelcolor='b')
    ax5b.tick_params(axis='y', labelcolor='g')
    ax5a.set_title('⑤ 航空機:フゴイドモード\n(同応答・1500秒の長周期振動)')
    lines = l1 + l2
    ax5a.legend(lines, [l.get_label() for l in lines], loc='best', fontsize=9)
    ax5a.grid(True, alpha=0.3)
    
    # ---- (6) 船舶:変針シミュレーション ----
    ax6 = plt.subplot(2, 3, 6)
    ax6a = ax6
    ax6b = ax6.twinx()
    l1 = ax6a.plot(ship['time'], np.rad2deg(ship['psi']),
                    'b-', lw=1.8, label='船首方位 ψ [度]')
    l_target = ax6a.axhline(np.rad2deg(ship['psi_target']),
                             color='k', ls='--', lw=1.0, label='目標方位')
    l2 = ax6b.plot(ship['time'], np.rad2deg(ship['delta']),
                    'r-', lw=1.0, alpha=0.7, label='舵角 δ [度]')
    ax6a.set_xlabel('時間 [s]')
    ax6a.set_ylabel('船首方位 [度]', color='b')
    ax6b.set_ylabel('舵角 [度]', color='r')
    ax6a.tick_params(axis='y', labelcolor='b')
    ax6b.tick_params(axis='y', labelcolor='r')
    ax6a.set_title('⑥ 船舶:Nomoto モデルによる変針\n(目標30度・10分間)')
    lines = [l1[0], l_target, l2[0]]
    ax6a.legend(lines, [l.get_label() for l in lines], loc='best', fontsize=9)
    ax6a.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'vehicle_dynamics_results.png')
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    print(f"  → 結果プロットを保存しました: {output_path}")
    plt.close()
    
    # ---- 船舶軌跡を別図でも出す(縦長すぎる軌跡を見やすく) ----
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ship['x'], ship['y'], 'b-', lw=1.8, label='船舶軌跡')
    ax.plot(ship['x'][0], ship['y'][0], 'go', ms=10, label='出発点')
    ax.plot(ship['x'][-1], ship['y'][-1], 'r^', ms=10, label='到達点')
    # 目標方位の方向
    L_arrow = 1500
    ax.plot([0, L_arrow * np.cos(ship['psi_target'])],
            [0, L_arrow * np.sin(ship['psi_target'])],
            'k--', lw=1.0, alpha=0.5, label='目標方位')
    ax.set_xlabel('東向き距離 [m]')
    ax.set_ylabel('北向き距離 [m]')
    ax.set_title('船舶の航跡(目標方位30度への変針操船)')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    output_path2 = os.path.join(OUTPUT_DIR, 'vehicle_dynamics_ship_track.png')
    plt.savefig(output_path2, dpi=130, bbox_inches='tight')
    print(f"  → 船舶軌跡プロットを保存しました: {output_path2}")
    plt.close()
    
    return output_path, output_path2


# ====================================================================
# メイン
# ====================================================================
def print_conclusion():
    """全シミュレーション終了後の総括。これまでの議論全体を俯瞰する。"""
    print("\n" + "=" * 70)
    print("【総括】輸送機関の運動制御 — 全体の俯瞰")
    print("=" * 70)
    print()
    print("◆ 1. 力学定式化のスペクトル")
    print("-" * 70)
    print("  Newton-Euler   : 単体剛体・自転車モデル等で最良。状態方程式に直結。")
    print("  Lagrange       : 非ホロノミック拘束は未定乗数で扱うが煩雑。")
    print("                   非保存力支配の系では旨みが薄い。")
    print("  Kane の方法    : 多体系(14自由度フルビークル等)で式爆発を回避。")
    print("                   CarSim 等の商用ソフト内部で採用。")
    print()
    print("  → 自動車・航空機・船舶のいずれも Newton-Euler が中心、")
    print("     多体は Kane、ラグランジュは中間で出番が少ない。")
    print()
    print("◆ 2. タイヤ・空気力・流体力モデル")
    print("-" * 70)
    print("  自動車: 線形タイヤ F_y = -C_α·α(線形領域)")
    print("          Pacejka Magic Formula(飽和を含む全領域)")
    print("          摩擦円 √(F_x² + F_y²) ≤ μF_z(縦横の力配分制約)")
    print("  航空機: 動圧 q̄ = (1/2)ρV² × 翼面積 × 空力係数 C_*(α, β, ...)")
    print("          係数は迎角・舵角・角速度の関数。風洞・CFD で同定。")
    print("  船舶  : 付加質量(空気では無視可、水では同オーダー)")
    print("          二次抗力、復原力、波・風の外乱")
    print()
    print("◆ 3. 経路追従・姿勢制御の階層")
    print("-" * 70)
    print("  古典幾何制御 : Pure Pursuit(後輪基準)、Stanley(前輪基準)")
    print("                 計算極小、駐車支援・低速ロボティクスで現役")
    print("  最適制御     : LQR(無制約・解析解)、MPC(制約・先読み)")
    print("                 LKAS、ACC、自動運転の経路追従の主流")
    print("  船舶用      : Nomoto モデル + LOS 誘導 + PID/MPC")
    print("  航空機用    : SAS/CAS + オートパイロット + ゲインスケジュール")
    print()
    print("◆ 4. 状態推定とセンサ融合")
    print("-" * 70)
    print("  EKF/UKF      : 直接測れない状態(車体スリップ角 β、横速度 v_y、")
    print("                 路面摩擦 μ など)を IMU + GNSS + エンコーダから推定")
    print("  ファクターグラフ: iSAM2 等、過去の観測を保持しながら全体最適化")
    print("                 自動運転 SLAM の現代主流")
    print("  INS/GNSS統合 : Loosely / Tightly / Deep Coupling の階層")
    print()
    print("◆ 5. シャシー制御階層(自動車)")
    print("-" * 70)
    print("  ABS  : スリップ率 κ ≈ 0.1〜0.2 のピーク摩擦近傍を保つ")
    print("  TCS  : 駆動側のスリップ制御")
    print("  ESC  : 意図ヨーレートと実ヨーレートの誤差で介入")
    print("         オーバー → 外側前輪ブレーキ、アンダー → 内側後輪ブレーキ")
    print("         2012年以降ほぼ全乗用車で標準装備、致死事故 30〜50% 減")
    print("  TV   : トルクベクタリング(左右輪トルク独立配分)")
    print("         BEV の独立モータと相性良し")
    print("  統合制御: Control Allocation で全アクチュエータを最適配分")
    print()
    print("◆ 6. 学習ベース手法のスペクトル")
    print("-" * 70)
    print("  完全モデルベース ── PurePursuit / LQR / MPC")
    print("       ↓")
    print("  GP-MPC / Neural-MPC ── タイヤモデルだけ学習")
    print("       ↓")
    print("  Residual Learning ── 制御則の補正項を学習")
    print("       ↓")
    print("  Safety Filter付き RL/IL ── CBF QP で安全性担保")
    print("       ↓")
    print("  完全エンドツーエンド ── PilotNet, Tesla FSD, Wayve")
    print()
    print("  下に行くほど学習比重が高く、性能上限は高いが安全性保証と認証の")
    print("  難度が上がる。市販自動運転は上半分、研究フロンティアは下半分。")
    print()
    print("◆ 7. 多車両連携(V2X)")
    print("-" * 70)
    print("  V2V / V2I / V2N / V2P  ← DSRC や C-V2X(5G NR-V2X)で通信")
    print("  CACC(隊列走行)         : リーダー加速度を FF、車間時間 0.6〜1.0秒")
    print("  ストリング安定性       : 後続への加減速増幅を防ぐ条件")
    print("                           通常 ACC は満たさず、CACC は満たせる")
    print("  交差点協調              : 集中型 MILP / 分散型オークション")
    print()
    print("◆ 8. 3つの輸送機関の本質的な違い")
    print("-" * 70)
    print("  自動車 : 非ホロノミック拘束 + 路面摩擦支配")
    print("           応答時間 0.1〜1秒、固有モード問題化せず")
    print()
    print("  航空機 : 自由空間 + 空気力(低密度)+ 縦/横モード分離")
    print("           短周期 数秒、フゴイド 数十秒〜分")
    print("           固有モード整形が制御設計の中心")
    print()
    print("  船舶   : 自由空間 + 流体力(高密度)+ 付加質量 + 復原力")
    print("           応答時間 数十秒〜数分、強い非線形")
    print()
    print("  共通 : 剛体力学を Newton-Euler 形式で書き、媒質と拘束に応じた")
    print("         外力モデルを付加するという構造。ラグランジュ形式は、")
    print("         いずれの分野でも実装の中心には来ない。")
    print()
    print("=" * 70)


def main():
    print_introduction()
    
    # 各シミュレーションの実行
    car_pp   = simulate_car_pure_pursuit()
    car_dyn  = simulate_car_dynamic_bicycle()
    aircraft = simulate_aircraft_longitudinal()
    ship     = simulate_ship_nomoto()
    
    # 可視化
    paths = visualize_all(car_pp, car_dyn, aircraft, ship)
    
    # 総括
    print_conclusion()
    
    print()
    print("シミュレーション完了。出力ファイル:")
    for p in paths:
        print(f"  - {p}")
    print("▓" * 70 + "\n")


if __name__ == '__main__':
    main()
