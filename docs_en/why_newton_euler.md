# Why we do not use the Lagrangian equations for vehicle models

The starting point of this project is the following naive question.

> Robotics textbooks invariably set up the equations of motion with the Lagrangian equations.
> And yet the motion models of automobiles, aircraft, and ships are almost always
> written in the Newton–Euler formulation. Why the difference?

To state the conclusion first: **it is not that the Lagrangian formulation "cannot be used";
rather, in this problem domain other toolsets (Newton–Euler, Kane's method, state-space
representation) are more natural, and the advantages of the Lagrangian formulation do not
come into play.** The four simulations in this repository all numerically integrate equations
of motion written in the Newton–Euler formulation, and serve as worked examples of this.

Below, the reasons are explained in four points.

---

## 1. Nonholonomic constraints

An automobile's tire has the constraint "no side-slip (or the lateral force is determined as a
function of the slip angle)." This is a **constraint on velocity**, and it cannot be integrated
into a relation among coordinates. Such a constraint is called a **nonholonomic constraint**.

To handle this in the Lagrangian formulation requires the **Lagrange–d'Alembert equations**
(or Routh's procedure) with undetermined multipliers introduced, and the formulation becomes
cumbersome all at once. In the Newton–Euler formulation, by contrast, the constraint can be
written down naturally as an external-force term — "the force the tire generates."
This is why a vehicle's bicycle model can be written in a few lines.

The joint constraints of a robot arm are relations among coordinates (holonomic constraints),
so the constraint vanishes the moment generalized coordinates are chosen, and the Lagrangian
formulation becomes extremely natural. This is the decisive difference between the two.

## 2. Non-conservative forces govern the motion

Tire friction, air resistance, fluid drag, thrust — almost everything that determines the motion
of a mode of transportation is a **non-conservative force**.

The appeal of the Lagrangian formulation is that the equations of motion are derived — together
with the symmetries and conservation laws of the system — from a single scalar function, the
Lagrangian $L = T - V$ (kinetic energy − potential energy).
But non-conservative forces do not fit within the `L` framework and end up being pushed into
the right-hand side of the equation as generalized forces. Once this happens, the beauty
peculiar to the Lagrangian formulation hardly comes into play, and the effort is no different
from simply summing up forces in Newton–Euler.

## 3. Compatibility with a control-oriented approach

Modern control theory requires a state-space representation $\dot{x} = f(x, u)$.

The Newton–Euler formulation drops naturally into a first-order state equation
(this is also the form handled by this repository's `ode_solver.hpp`).
The Lagrangian formulation, on the other hand, comes out in the second-order form containing
the mass matrix, $M(q)\ddot{q} + C(q, \dot{q})\dot{q} + g(q) = \tau$, so making it into a state
equation involves the extra step of solving for `q̈` (the inverse of `M`).

If the final objective is the design and implementation of a control law, Newton–Euler — which
connects directly to a state equation from the start — is easier to handle.

## 4. For multibody systems, Kane's method is superior

For a full vehicle including the suspension (typically 14 degrees of freedom), the partial-derivative
terms of the energy explode combinatorially.

At this scale, **Kane's method** (a formulation of d'Alembert's principle based on partial velocities)
is more efficient than the Lagrangian formulation, and it is adopted internally by commercial
multibody software such as CarSim.

In other words, there is a "spectrum" of mechanical formulations:

| Formulation | Targets it excels at |
|---|---|
| Newton–Euler | Single rigid bodies, bicycle models, etc. Connects directly to the state equation. |
| Lagrangian | Holonomic constraints, conservative systems. The standard for robot arms. Nonholonomic constraints can be handled with undetermined multipliers but are cumbersome. |
| Kane's method | Multibody systems (14-DOF full vehicles, etc.). Avoids the explosion of terms. |

For automobiles, aircraft, and ships alike, Newton–Euler is central; multibody systems use Kane;
and the Lagrangian formulation is positioned as "in between, with few occasions to appear."

---

## The structure common to the three modes of transportation

The media and the constraints differ, but the four models in this repository all share the common
structure of **"writing the rigid-body dynamics in the Newton–Euler formulation and adding an
external-force model according to the medium and the constraints."**

- **Automobile** — nonholonomic constraint + road friction dominate. The response time is 0.1–1 second,
  and eigenmodes rarely become an issue.
- **Aircraft** — free space + aerodynamic force (low-density medium). The longitudinal and lateral motion
  separate, dividing into the short-period mode (a few seconds) and the phugoid mode (tens of seconds to minutes).
  Shaping the eigenmodes becomes the center of control design.
- **Ship** — free space + fluid force (high-density medium). The added mass cannot be neglected,
  and restoring forces also act. The response time is tens of seconds to minutes, with strong nonlinearity.

The Lagrangian formulation does not come to the center of the implementation in any of these fields.
The purpose of this repository is to let you confirm this fact with "code that actually runs."

---

## References

- R. Rajamani, *Vehicle Dynamics and Control*, Springer.
- B. Etkin and L. D. Reid, *Dynamics of Flight: Stability and Control*, Wiley.
- T. I. Fossen, *Handbook of Marine Craft Hydrodynamics and Motion Control*, Wiley.
- T. R. Kane and D. A. Levinson, *Dynamics: Theory and Applications*, McGraw-Hill.
