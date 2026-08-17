"""Parameter-consistency guard between the C++ core and the Python MPC.

Every model is integrated in C++ (`src/`) except MPC, which needs a constrained
optimiser and therefore stays in `python/run_mpc.py`. That one exception has to
drive the *same* scenario as the C++ control laws -- same ellipse, same start,
same speed, same wheelbase -- or the MPC curve plotted next to Pure Pursuit /
Stanley / LQR is comparing two different problems.

This test pins the canonical values and asserts both sources still declare them.
It is a source-level drift detector: no build, no heavy dependencies (stdlib
only), so it runs anywhere. Comparison ignores whitespace so it is insensitive
to formatting.

Deliberately *not* checked: the integration step. The C++ core runs at
dt = 0.05 s while the MPC recomputes at dt_ctrl = 0.1 s, because re-solving the
SLSQP problem every 50 ms is neither realistic nor necessary. That difference is
a modelling decision, not drift.

Run with: ``python -m pytest tests/test_param_consistency.py`` or directly
``python tests/test_param_consistency.py``.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PY_SOURCE = ROOT / "python" / "run_mpc.py"


def _squash(text: str) -> str:
    """Drop all whitespace so the check is formatting-insensitive."""
    return "".join(text.split())


# C++ values may live in main.cpp or in the headers' default arguments.
CPP = _squash(
    (ROOT / "src" / "main.cpp").read_text(encoding="utf-8")
    + "".join(p.read_text(encoding="utf-8") for p in (ROOT / "src").glob("*.hpp"))
)
PY = _squash(PY_SOURCE.read_text(encoding="utf-8"))

# (description, fragment expected in C++, fragment expected in Python)
SHARED = [
    ("ellipse path a/b/n",
     "make_ellipse_path(50.0,30.0,400)",
     "_make_ellipse_path(a=50.0,b=30.0,n=400)"),
    ("initial state x0/y0",
     "init{50.0,2.0,",
     "np.array([50.0,2.0,np.pi/2])"),
    ("forward speed v = 8.0 m/s",
     "v=8.0",
     "v=8.0"),
    ("wheelbase 2.7 m",
     "wheelbase=2.7",
     "wheelbase=2.7"),
]


def test_shared_parameters_match():
    problems = []
    rel = PY_SOURCE.relative_to(ROOT).as_posix()
    for desc, cpp_frag, py_frag in SHARED:
        if _squash(cpp_frag) not in CPP:
            problems.append(f"C++ source missing {desc!r}: {cpp_frag!r}")
        if _squash(py_frag) not in PY:
            problems.append(f"Python {rel} missing {desc!r}: {py_frag!r}")
    assert not problems, ("Parameter drift between the C++ core and the Python MPC:\n"
                          + "\n".join(problems))


def test_no_duplicate_simulator_in_python():
    """The narrated front-end must not grow its own integrator again.

    vehicle_dynamics_simulation.py used to re-implement all four models in
    Python. It now reads results.json; if a controller class or an ODE solve
    reappears there, this catches it.
    """
    front_end = (ROOT / "vehicle_dynamics_simulation.py").read_text(encoding="utf-8")
    banned = ["class StanleyController", "class LQRPathTracker",
              "class MPCPathTracker", "class KinematicBicyclePurePursuit",
              "odeint(", "from scipy"]
    found = [b for b in banned if b in front_end]
    assert not found, (
        "vehicle_dynamics_simulation.py should only narrate and plot the C++ "
        "results, but it contains: " + ", ".join(found))


if __name__ == "__main__":
    test_shared_parameters_match()
    test_no_duplicate_simulator_in_python()
    print("Parameter consistency OK: the C++ core and the Python MPC share the "
          "same scenario, and the front-end integrates nothing.")
