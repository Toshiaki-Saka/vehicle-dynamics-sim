"""Parameter-consistency guard between the C++ and Python simulations.

The C++ (`src/`) and Python (`vehicle_dynamics_simulation.py`) implementations
duplicate the shared scenario parameters (ellipse path, initial state, speed,
time step, wheelbase). If one side is edited without the other, the two
simulations silently diverge.

This test pins the canonical values and asserts both sources still declare
them. It is a source-level drift detector: no build, no heavy dependencies
(stdlib only), so it runs anywhere. Comparison ignores whitespace so it is
insensitive to formatting.

Run with: ``python -m pytest tests/test_param_consistency.py`` or directly
``python tests/test_param_consistency.py``.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _squash(text: str) -> str:
    """Drop all whitespace so the check is formatting-insensitive."""
    return "".join(text.split())


# C++ values may live in main.cpp or in the headers' default arguments.
CPP = _squash(
    (ROOT / "src" / "main.cpp").read_text(encoding="utf-8")
    + "".join(p.read_text(encoding="utf-8") for p in (ROOT / "src").glob("*.hpp"))
)
PY = _squash((ROOT / "vehicle_dynamics_simulation.py").read_text(encoding="utf-8"))

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
    ("time step dt = 0.05 s",
     "dt=0.05",
     "dt=0.05"),
    ("wheelbase 2.7 m",
     "wheelbase=2.7",
     "wheelbase=2.7"),
]


def test_shared_parameters_match():
    problems = []
    for desc, cpp_frag, py_frag in SHARED:
        if _squash(cpp_frag) not in CPP:
            problems.append(f"C++ source missing {desc!r}: {cpp_frag!r}")
        if _squash(py_frag) not in PY:
            problems.append(
                f"Python vehicle_dynamics_simulation.py missing {desc!r}: {py_frag!r}"
            )
    assert not problems, "Parameter drift between C++ and Python:\n" + "\n".join(problems)


if __name__ == "__main__":
    test_shared_parameters_match()
    print("Parameter consistency OK: C++ and Python share the same scenario.")
