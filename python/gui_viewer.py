"""
Vehicle Dynamics Simulation - Result Viewer GUI
=================================================
Displays C++ simulation results and Python MPC results in a single view.

How to launch:
    python python/gui_viewer.py
"""

import os
import sys
import json
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import rcParams, font_manager
# Automatic Japanese-font configuration
def _setup_jp_font():
    candidates = ["Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP",
                  "IPAexGothic", "IPAGothic", "TakaoGothic", "DejaVu Sans"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            rcParams["font.family"] = name
            return
_setup_jp_font()
rcParams["axes.unicode_minus"] = False
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ── Path configuration ─────────────────────────────────────────────────────
PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR   = os.path.dirname(os.path.abspath(__file__))
LQR_K_PATH   = os.path.join(PROJECT_DIR, "lqr_k.json")
RESULTS_PATH = os.path.join(PROJECT_DIR, "results.json")
MPC_PATH     = os.path.join(PROJECT_DIR, "mpc_results.json")

COLORS = {
    "pure_pursuit": "#1f77b4",
    "stanley":      "#2ca02c",
    "lqr":          "#d62728",
    "mpc":          "#9467bd",
}
LABELS = {
    "pure_pursuit": "Pure Pursuit",
    "stanley":      "Stanley",
    "lqr":          "LQR",
    "mpc":          "MPC",
}


def find_exe():
    """Automatically locate the built executable."""
    candidates = [
        os.path.join(PROJECT_DIR, "build", "Release", "vehicle_dynamics.exe"),
        os.path.join(PROJECT_DIR, "build", "Debug",   "vehicle_dynamics.exe"),
        os.path.join(PROJECT_DIR, "build", "vehicle_dynamics.exe"),
        os.path.join(PROJECT_DIR, "build", "vehicle_dynamics"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


# ═══════════════════════════════════════════════════════════════════
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Vehicle Dynamics Simulation Viewer")
        self.geometry("1500x900")
        self.minsize(1100, 700)

        self.exe_var     = tk.StringVar(value=find_exe())
        self.results     = None
        self.mpc_results = None

        self._build_ui()

        # Auto-load existing results if present
        if os.path.exists(RESULTS_PATH):
            self.after(200, self._load_and_plot)

    # ── UI construction ────────────────────────────────────────────────────
    def _build_ui(self):
        # Left panel (fixed width)
        left = ttk.Frame(self, width=260, padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        ttk.Label(left, text="Vehicle Dynamics Viewer",
                  font=("", 11, "bold")).pack(pady=(4, 12))

        # Executable path
        ttk.Label(left, text="C++ executable:").pack(anchor=tk.W)
        exe_row = ttk.Frame(left)
        exe_row.pack(fill=tk.X, pady=2)
        ttk.Entry(exe_row, textvariable=self.exe_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(exe_row, text="…", width=3,
                   command=self._browse_exe).pack(side=tk.RIGHT)

        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=8)

        # Action buttons
        ttk.Button(left, text="▶  Run all simulations",
                   command=self._run_all).pack(fill=tk.X, pady=3)
        ttk.Button(left, text="📂  Load existing results",
                   command=self._load_and_plot).pack(fill=tk.X, pady=3)

        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=8)

        # Progress bar
        self.progress = ttk.Progressbar(left, mode="indeterminate", length=200)
        self.progress.pack(fill=tk.X, pady=4)

        # Log
        ttk.Label(left, text="Status log:").pack(anchor=tk.W)
        self.log_box = scrolledtext.ScrolledText(
            left, height=20, width=30, state=tk.DISABLED,
            font=("Consolas", 8))
        self.log_box.pack(fill=tk.BOTH, expand=True, pady=4)

        ttk.Label(left,
                  text="* MPC uses the Python scipy optimizer\n"
                       "  LQR gain uses scipy CARE (Python)\n"
                       "  everything else uses C++20 + Eigen",
                  foreground="gray", font=("", 8)).pack(pady=4)

        # Right panel: tabs
        self.nb = ttk.Notebook(self)
        self.nb.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.tab_frames = {}
        for key, name in [
            ("path",     "① Car path tracking"),
            ("dynamic",  "② Car dynamics"),
            ("aircraft", "③ Aircraft longitudinal motion"),
            ("ship",     "④ Ship Nomoto"),
        ]:
            f = ttk.Frame(self.nb)
            self.nb.add(f, text=name)
            self.tab_frames[key] = f

    # ── Utilities ──────────────────────────────────────────────
    def _browse_exe(self):
        p = filedialog.askopenfilename(
            title="Select the C++ executable",
            filetypes=[("Executable", "*.exe"), ("All files", "*.*")])
        if p:
            self.exe_var.set(p)

    def _log(self, msg: str):
        self.log_box.config(state=tk.NORMAL)
        self.log_box.insert(tk.END, msg + "\n")
        self.log_box.see(tk.END)
        self.log_box.config(state=tk.DISABLED)
        self.update_idletasks()

    def _clear_tab(self, key: str):
        for w in self.tab_frames[key].winfo_children():
            w.destroy()

    def _embed_figure(self, key: str, fig: Figure):
        """Embed a Figure into a tab."""
        canvas = FigureCanvasTkAgg(fig, master=self.tab_frames[key])
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, self.tab_frames[key])
        toolbar.update()

    # ── Execution ───────────────────────────────────────────────────────
    def _run_all(self):
        t = threading.Thread(target=self._run_all_thread, daemon=True)
        t.start()

    def _run_all_thread(self):
        self.progress.start(10)
        try:
            # Step 1: LQR gain (Python scipy CARE)
            self._log("Step 1: Computing LQR gain (scipy CARE)...")
            ret = subprocess.run(
                [sys.executable,
                 os.path.join(PYTHON_DIR, "compute_lqr_gain.py"),
                 LQR_K_PATH],
                capture_output=True, text=True, cwd=PROJECT_DIR)
            if ret.returncode != 0:
                self._log(f"  ERROR: {ret.stderr.strip()}")
                return
            self._log(f"  {ret.stdout.strip()}")

            # Step 2: C++ simulation
            exe = self.exe_var.get()
            self._log(f"\nStep 2: Running C++ simulation...")
            if not os.path.exists(exe):
                self._log(f"  Executable not found:\n  {exe}")
                self._log("  Please build it first:")
                self._log("    cd build && cmake --build . --config Release")
                return
            ret = subprocess.run(
                [exe, LQR_K_PATH, RESULTS_PATH],
                capture_output=True, text=True, cwd=PROJECT_DIR)
            if ret.returncode != 0:
                self._log(f"  ERROR: {ret.stderr.strip()}")
                return
            for line in ret.stdout.strip().splitlines():
                self._log(f"  {line}")

            # Step 3: MPC (Python scipy optimizer)
            self._log("\nStep 3: MPC simulation (scipy SLSQP)...")
            self._log("  (this takes tens of seconds)")
            ret = subprocess.run(
                [sys.executable,
                 os.path.join(PYTHON_DIR, "run_mpc.py"),
                 MPC_PATH],
                capture_output=True, text=True, cwd=PROJECT_DIR)
            if ret.returncode != 0:
                self._log(f"  MPC ERROR: {ret.stderr.strip()}")
            else:
                for line in ret.stdout.strip().splitlines():
                    self._log(f"  {line}")

            # Step 4: Update plots
            self._log("\nStep 4: Updating plots...")
            self.after(0, self._load_and_plot)

        finally:
            self.progress.stop()

    # ── Load results & plot ──────────────────────────────────────
    def _load_and_plot(self):
        if not os.path.exists(RESULTS_PATH):
            self._log("results.json not found. Please run the simulation first.")
            return
        with open(RESULTS_PATH, encoding="utf-8") as f:
            self.results = json.load(f)
        self.mpc_results = None
        if os.path.exists(MPC_PATH):
            with open(MPC_PATH, encoding="utf-8") as f:
                self.mpc_results = json.load(f)

        self._plot_path_tracking()
        self._plot_car_dynamic()
        self._plot_aircraft()
        self._plot_ship()
        self._log("Plots updated.")

    # ── Plot: path tracking ─────────────────────────────────────────
    def _plot_path_tracking(self):
        self._clear_tab("path")
        r = self.results
        fig = Figure(figsize=(14, 5.5), dpi=95)

        # Trajectory
        ax1 = fig.add_subplot(1, 2, 1)
        px, py = r["car_path"]["x"], r["car_path"]["y"]
        ax1.plot(px, py, "k--", lw=1.2, alpha=0.5, label="Reference path")
        for key in ("pure_pursuit", "stanley", "lqr"):
            d = r[key]
            ax1.plot(d["x"], d["y"], color=COLORS[key], lw=1.5,
                     label=f"{LABELS[key]} (RMS={d['rms_error']:.3f}m)")
        if self.mpc_results:
            d = self.mpc_results
            ax1.plot(d["x"], d["y"], color=COLORS["mpc"], lw=1.5,
                     label=f"MPC (RMS={d['rms_error']:.3f}m)")
        ax1.plot(r["pure_pursuit"]["x"][0], r["pure_pursuit"]["y"][0],
                 "k*", ms=12, label="Start point", zorder=5)
        ax1.set_xlabel("X [m]"); ax1.set_ylabel("Y [m]")
        ax1.set_title("Trajectory comparison (Pure Pursuit / Stanley / LQR / MPC)")
        ax1.legend(fontsize=8); ax1.set_aspect("equal"); ax1.grid(alpha=0.3)

        # Steering angle
        ax2 = fig.add_subplot(1, 2, 2)
        t = r["car_time"]
        for key in ("pure_pursuit", "stanley", "lqr"):
            ax2.plot(t, np.rad2deg(r[key]["delta"]),
                     color=COLORS[key], lw=1.2, label=LABELS[key], alpha=0.85)
        if self.mpc_results:
            ax2.plot(self.mpc_results["time"],
                     np.rad2deg(self.mpc_results["delta"]),
                     color=COLORS["mpc"], lw=1.5, label="MPC", alpha=0.85)
        ax2.axhline(0, color="k", lw=0.5)
        ax2.set_xlabel("Time [s]"); ax2.set_ylabel("Steering angle [deg]")
        ax2.set_title("Steering angle history")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

        fig.tight_layout()
        self._embed_figure("path", fig)

    # ── Plot: dynamic bicycle ──────────────────────────────────────
    def _plot_car_dynamic(self):
        self._clear_tab("dynamic")
        d = self.results["car_dynamic"]
        fig = Figure(figsize=(10, 5), dpi=95)
        ax = fig.add_subplot(1, 1, 1)
        ax2 = ax.twinx()

        l1 = ax.plot(d["time"], d["vy"], "b-", lw=1.5, label="Lateral velocity vy [m/s]")
        l2 = ax2.plot(d["time"], np.rad2deg(d["yaw_rate"]), "r-", lw=1.5,
                      label="Yaw rate [deg/s]")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Lateral velocity [m/s]", color="b")
        ax2.set_ylabel("Yaw rate [deg/s]", color="r")
        ax.tick_params(axis="y", labelcolor="b")
        ax2.tick_params(axis="y", labelcolor="r")
        us = ("understeer" if d["Kv"] > 0
              else "oversteer" if d["Kv"] < 0 else "neutral")
        ax.set_title(
            f"Dynamic 2-DOF model step response (vx={d['vx']} m/s)\n"
            f"Kv={d['Kv']:.5f} ({us})  "
            f"Yaw gain: theory={d['yaw_gain_theory']:.3f}  "
            f"sim={d['yaw_gain_sim']:.3f}")
        lines = l1 + l2
        ax.legend(lines, [l.get_label() for l in lines], loc="best")
        ax.grid(alpha=0.3)

        fig.tight_layout()
        self._embed_figure("dynamic", fig)

    # ── Plot: aircraft ───────────────────────────────────────────
    def _plot_aircraft(self):
        self._clear_tab("aircraft")
        d = self.results["aircraft"]
        t = np.array(d["time"])
        fig = Figure(figsize=(14, 5.5), dpi=95)

        # Short-period mode (0-15 s)
        ax1 = fig.add_subplot(1, 2, 1)
        mask = t <= 15.0
        alpha_approx = np.array(d["w"]) / 235.0
        ax1.plot(t[mask], np.rad2deg(alpha_approx[mask]),
                 "b-", lw=1.5, label="Angle of attack α≈w/V [deg]")
        ax1.plot(t[mask], np.rad2deg(np.array(d["q"])[mask]),
                 "r-", lw=1.5, label="Pitch rate q [deg/s]")
        ax1.set_xlabel("Time [s]"); ax1.set_ylabel("Response")
        ax1.set_title("Short-period mode (elevator -1° step)")
        ax1.legend(); ax1.grid(alpha=0.3)

        # Phugoid mode (full range -> downsampled)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2b = ax2.twinx()
        stride = max(1, len(t) // 2000)
        ts = t[::stride]
        us_arr = np.array(d["u_pert"])[::stride]
        th_arr = np.array(d["theta"])[::stride]
        l1 = ax2.plot(ts, us_arr,          "b-", lw=1.5, label="Velocity perturbation Δu [m/s]")
        l2 = ax2b.plot(ts, np.rad2deg(th_arr), "g-", lw=1.5, label="Pitch angle θ [deg]")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Velocity perturbation [m/s]", color="b")
        ax2b.set_ylabel("Pitch angle [deg]", color="g")
        ax2.tick_params(axis="y", labelcolor="b")
        ax2b.tick_params(axis="y", labelcolor="g")
        ax2.set_title("Phugoid mode (long-period oscillation ~1500s)")
        lines = l1 + l2
        ax2.legend(lines, [l.get_label() for l in lines]); ax2.grid(alpha=0.3)

        # Annotate the title with eigenvalue information
        ev_info = []
        eig_r = d["eigvals_real"]
        eig_i = d["eigvals_imag"]
        pairs_seen = set()
        for i in range(4):
            if abs(eig_i[i]) > 1e-6 and eig_i[i] > 0:
                wn = (eig_r[i]**2 + eig_i[i]**2) ** 0.5
                zeta = -eig_r[i] / wn
                period = 2 * np.pi / abs(eig_i[i])
                ev_info.append(f"T={period:.1f}s ζ={zeta:.3f}")
        if ev_info:
            fig.suptitle("Eigenmodes: " + "  /  ".join(ev_info), fontsize=9, y=1.0)

        fig.tight_layout()
        self._embed_figure("aircraft", fig)

    # ── Plot: ship ─────────────────────────────────────────────
    def _plot_ship(self):
        self._clear_tab("ship")
        d = self.results["ship"]
        fig = Figure(figsize=(14, 5.5), dpi=95)

        # Heading / rudder angle vs. time
        ax1 = fig.add_subplot(1, 2, 1)
        ax1b = ax1.twinx()
        l1 = ax1.plot(d["time"], np.rad2deg(d["psi"]),
                      "b-", lw=1.8, label="Heading ψ [deg]")
        lt = ax1.axhline(np.rad2deg(d["psi_target"]),
                         color="k", ls="--", lw=1.2, label="Target heading")
        l2 = ax1b.plot(d["time"], np.rad2deg(d["delta"]),
                       "r-", lw=1.0, alpha=0.7, label="Rudder angle δ [deg]")
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Heading [deg]", color="b")
        ax1b.set_ylabel("Rudder angle [deg]", color="r")
        ax1.tick_params(axis="y", labelcolor="b")
        ax1b.tick_params(axis="y", labelcolor="r")
        settle = d.get("settle_time", -1)
        settle_str = f"{settle:.1f} s" if settle > 0 else "not settled"
        ax1.set_title(f"Nomoto first-order model course change\nSettling time: {settle_str}")
        lines = [l1[0], lt, l2[0]]
        ax1.legend(lines, [l.get_label() for l in lines]); ax1.grid(alpha=0.3)

        # Track
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(d["x"], d["y"], "b-", lw=1.8, label="Ship trajectory")
        ax2.plot(d["x"][0],  d["y"][0],  "go", ms=10, label="Departure point", zorder=5)
        ax2.plot(d["x"][-1], d["y"][-1], "r^", ms=10, label="Arrival point", zorder=5)
        L = 1500
        tgt = d["psi_target"]
        ax2.plot([0, L * np.cos(tgt)], [0, L * np.sin(tgt)],
                 "k--", lw=1.0, alpha=0.5, label="Target heading line")
        ax2.set_xlabel("East [m]"); ax2.set_ylabel("North [m]")
        ax2.set_title("Ship track")
        ax2.legend(); ax2.set_aspect("equal"); ax2.grid(alpha=0.3)

        fig.tight_layout()
        self._embed_figure("ship", fig)


# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = App()
    app.mainloop()
