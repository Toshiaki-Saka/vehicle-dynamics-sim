# The right modeling approach depends on what you are trying to represent

> **The one thing this document wants to convey**
>
> **There is no single "correct" set of equations of motion. The structure of what you want to represent determines the modeling approach that fits.**
>
> Even when dealing with the same "rigid-body motion," whether the target is
> a car, a robot arm, an aircraft, or a ship, **the formulation that writes down
> naturally is completely different**. This project exists to demonstrate this one
> point with "code that runs."

---

## 1. The claim: the method follows the target

When you open mechanics textbooks by field, you notice something curious.

- **Robotics** textbooks invariably derive the equations of motion using the **Lagrangian equations**.
- **Automotive, aircraft, and ship** textbooks, almost without exception, write them in the **Newton–Euler formulation**.
- Commercial software for **full vehicles (14 degrees of freedom including suspension)** uses **Kane's method** internally.

This is not a matter of style or preference. **The physical structure of the target forces the choice of the method that fits.**

| Target to represent | Governing structure | Modeling approach that fits |
|---|---|---|
| Robot arm | Holonomic constraint + conservative force (gravity) | **Lagrangian method** |
| Automobile | Nonholonomic constraint (tires) + non-conservative force (friction) | **Newton–Euler** |
| Aircraft | Free space + aerodynamic force (non-conservative) + eigenmodes | **Newton–Euler** |
| Ship | High-density medium + added mass + restoring force | **Newton–Euler (Fossen formulation)** |
| Full vehicle (multibody) | Large-scale connected rigid bodies | **Kane's method** |

**Even though it is all the same "rigid-body dynamics," when the target changes the tool that fits changes.**
This is the central claim of this project.

---

## 2. Why there is no "single correct method"

The important point is that **every method is physically equally correct**.
Describing a car with the Lagrangian method, or a robot arm with Newton–Euler,
is in principle possible. And yet people change their method for each target
because **what differs is not "can it be solved" but "can it be written naturally."**

The merit of a method is decided by the following three structures.

### (1) Type of constraint — holonomic or nonholonomic

- **Holonomic constraint** (a relation among coordinates, e.g., a robot joint)
  → Choosing generalized coordinates makes the constraint vanish. **The Lagrangian method is natural.**
- **Nonholonomic constraint** (a constraint on velocity, e.g., no tire side-slip)
  → It cannot be eliminated by a coordinate transformation, and the Lagrangian approach requires undetermined multipliers.
  **With Newton–Euler, you simply write it as an external force — "the force the tire produces."**

### (2) Nature of the forces — conservative or non-conservative

- **Conservative forces dominate** (gravity, elastic force)
  → They fit into a potential $V$, and the Lagrangian $L = T - V$ comes into its own.
- **Non-conservative forces dominate** (tire friction, air resistance, fluid drag, thrust)
  → They do not fit within the $L$ framework and are merely pushed into the generalized forces on the right-hand side.
  The elegance of the Lagrangian approach vanishes, and **the effort is no different from Newton–Euler.**

### (3) Final objective — analysis or control

- If **control design** is the objective, the state equation $\dot{x} = f(x, u)$ is required.
  → Newton–Euler connects directly to a first-order state equation.
  The Lagrangian approach comes out in the second-order form $M(q)\ddot{q} + C\dot{q} + g = \tau$,
  adding the extra step of taking $M^{-1}$.

Run the target through these three questions, and the method that fits determines itself.

---

## 3. Four worked examples this repository demonstrates

Each simulation in this project is a **demonstration** of this claim.

### Automobile — nonholonomic constraint → Newton–Euler

The tire's "no side-slip" constraint is a constraint on velocity (nonholonomic).
The Lagrangian approach requires undetermined multipliers, but with Newton–Euler you simply write the lateral force as an external force.
The control laws (Pure Pursuit / Stanley / LQR / MPC) all connect directly to the state equation.
→ [`src/car_dynamic.cpp`](../src/car_dynamic.cpp), [`src/car_tracking.cpp`](../src/car_tracking.cpp)

### Aircraft — free space + eigenmodes → Newton–Euler

Free-space motion with no ground constraint. Aerodynamic forces (non-conservative) dominate,
and the **eigenmodes** — the short-period mode (about 10 seconds) and the phugoid mode (about 14 minutes) —
become the primary subject of control design. They can be separated clearly from the force balance.
→ [`src/aircraft.cpp`](../src/aircraft.cpp)

### Ship — high-density medium + added mass → Newton–Euler (Fossen formulation)

The medium is dense, and the **added mass** (the effect that the surrounding water also moves during acceleration)
acts on the same order as the hull inertia. The response is slow, on the order of tens of seconds to minutes.
The Fossen standard form is a direct extension of Newton–Euler and is easy to grasp with physical intuition.
→ [`src/ship.cpp`](../src/ship.cpp)

### Robot arm (for contrast) — holonomic constraint + conservative force → Lagrangian

The opposite pole of the three above. The joint constraint is holonomic, and gravity is a conservative force.
The equations of motion come out mechanically from just the partial derivatives of the scalar function $L = T - V$.
That total energy is conserved with no drive torque can also be confirmed numerically.
→ [`python/lagrangian_arm.py`](../python/lagrangian_arm.py)

**Only by placing all four side by side can you truly feel that "the target selects the method."**
This is why this single repository houses three modes of transportation alongside a robot arm.

---

## 4. From target to method — a guide for judgment

When modeling a new target, ask about the target's structure in the following order.

```
What do you want to represent?
│
├─ Is it a multibody system (connected rigid bodies with 6+ DOF)?
│   └─ Yes → Kane's method / multibody analysis software
│
├─ Is the constraint nonholonomic (a velocity constraint)?
│   └─ Yes → Newton–Euler (the constraint can be written as an external force)
│
├─ Do non-conservative forces (friction, drag, thrust) dominate?
│   └─ Yes → Newton–Euler (the Lagrangian advantage does not come into play)
│
├─ Is control design the final objective?
│   └─ Yes → Newton–Euler (connects directly to the state equation)
│
└─ Holonomic constraint + conservative forces dominate, and analysis is the objective
    └─ Yes → Lagrangian method (the scalar L = T - V comes into its own)
```

**Do not fix the method first and force the target into it. Observe the target, and make the method follow its structure.**

---

## 5. Further reading

This document focuses on the essence of the claim. For the details of each formulation's
derivation, simulation results, and code correspondence, see the following.

| Document | Contents |
|---|---|
| [`equation_selection_guide.md`](equation_selection_guide.md) | Detailed explanation of the derivation and appropriate use of each formulation, including results (detailed version) |
| [`why_newton_euler.md`](why_newton_euler.md) | The reasons (four points) why Newton–Euler is used for modes of transportation |

---

## References

- R. Rajamani, *Vehicle Dynamics and Control*, Springer. — Automobile (Newton–Euler)
- B. Etkin and L. D. Reid, *Dynamics of Flight: Stability and Control*, Wiley. — Aircraft
- T. I. Fossen, *Handbook of Marine Craft Hydrodynamics and Motion Control*, Wiley. — Ship
- M. W. Spong et al., *Robot Modeling and Control*, Wiley. — Robot arm (Lagrangian)
- T. R. Kane and D. A. Levinson, *Dynamics: Theory and Applications*, McGraw-Hill. — Multibody systems (Kane)
