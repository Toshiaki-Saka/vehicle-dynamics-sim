# How to choose a mechanical formulation — using the right equations of motion for the job

> **The purpose of this document**  
> Rather than answering the question "which mechanical formulation is correct,"
> it shows **"which formulation is natural given the structure of the problem"**
> through simulations that actually run.

---

## Table of contents

1. [The central question](#1-the-central-question)
2. [An overview of the three mechanical formulations](#2-an-overview-of-the-three-mechanical-formulations)
3. [Where the Lagrangian method shines — the robot arm](#3-where-the-lagrangian-method-shines--the-robot-arm)
4. [Where the Newton–Euler method is natural — modes of transportation](#4-where-the-newtoneuler-method-is-natural--modes-of-transportation)
5. [Automobile: the archetype of a nonholonomic constraint](#5-automobile-the-archetype-of-a-nonholonomic-constraint)
6. [Aircraft: free space and eigenmodes](#6-aircraft-free-space-and-eigenmodes)
7. [Ship: high-density medium and added mass](#7-ship-high-density-medium-and-added-mass)
8. [The differences seen in simulation results](#8-the-differences-seen-in-simulation-results)
9. [A decision flow for choosing a formulation](#9-a-decision-flow-for-choosing-a-formulation)
10. [References](#10-references)

---

## 1. The central question

When you open a robotics textbook, the equations of motion are invariably derived by the Lagrangian method.

$$\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}_i}\right) - \frac{\partial L}{\partial q_i} = \tau_i \quad \text{(Lagrange's equations)}$$

Yet when you open automotive, aircraft, and ship textbooks, almost without exception
the Newton–Euler formulation is used.

$$m\mathbf{a} = \mathbf{F} \quad \text{(Newton's second law)}$$

$$I\boldsymbol{\alpha} = \mathbf{M} \quad \text{(Euler's rotational equation)}$$

**Why, while dealing with the same "rigid-body motion," are different equations used?**

The answer is not that "one of them is correct," but that depending on the **structure of the problem** — the
type of constraint, the nature of the governing forces, and the final application — which one is natural changes.

This document demonstrates that difference with simulation code that actually runs.

---

## 2. An overview of the three mechanical formulations

### A comparison of the formulations

| Formulation | Underlying quantity | Target it is best at | Situations it struggles with |
|---|---|---|---|
| **Newton–Euler** | Forces and moments (vector quantities) | Nonholonomic constraints, non-conservative forces, control-oriented design | The equations tend to become cumbersome for multibody systems |
| **Lagrangian** | Energy (scalar quantity) $L = T - V$ | Holonomic constraints, conservative systems, robot arms | Nonholonomic constraints require undetermined multipliers |
| **Kane's method** | d'Alembert's principle based on partial velocities | Large-scale multibody systems such as 14-DOF full vehicles | Intuition is hard to grasp by hand calculation |

### Which one is "correct"

Both are **correct equations that describe the same physical phenomenon**.
Setting up the equations of motion for an automobile using the Lagrangian method is, in principle, possible too.
However, as the next section shows, the "naturalness" differs greatly depending on the structure of the problem.

---

## 3. Where the Lagrangian method shines — the robot arm

### Why the Lagrangian method suits the robot arm

Let the generalized coordinates of a two-link robot arm be $q = (\theta_1, \theta_2)$.

```
       θ₁
  O ---+--- link1 ---+--- link2
       ↑              θ₂
    joint1          joint2
```

There are three reasons the Lagrangian method works naturally for this system.

**Reason 1: Holonomic constraint**

The joint constraint is a "relation among coordinates." Choosing the generalized coordinates $(\theta_1, \theta_2)$
makes the constraint vanish automatically. Lagrange multipliers (undetermined multipliers) are unnecessary.

Degrees of freedom after the constraint vanishes: exactly 2, $(\theta_1, \theta_2)$.
The Lagrangian becomes a simple scalar function:

$$L(q, \dot{q}) = T(q, \dot{q}) - V(q)$$

**Reason 2: Conservative forces play the leading role**

Gravity is a conservative force with a potential $V(q)$. It fits directly into the framework of the Lagrangian $L = T - V$.
The exchange between kinetic and potential energy is naturally represented.

**Reason 3: The equations come out of a scalar function**

Simply taking the partial derivatives of the scalar function $L$ with respect to each generalized coordinate
mechanically produces the equations of motion. There is no need to worry about the direction or sign of vectors.

```python
# excerpt from python/lagrangian_arm.py
def manipulator_matrices(q):
    """
    Computing d/dt(∂L/∂q̇) - ∂L/∂q = τ from L = T - V
    yields the mass matrix M(q) of a planar two-link arm in closed form
    """
    q1, q2 = q
    c2 = np.cos(q2)
    m11 = I1 + I2 + M1*LC1**2 + M2*(L1**2 + LC2**2 + 2*L1*LC2*c2)
    m12 = I2 + M2*(LC2**2 + L1*LC2*c2)
    m22 = I2 + M2*LC2**2
    return np.array([[m11, m12], [m12, m22]])
```

### The standard form of the equations of motion

Applying the Lagrangian method, the equations of motion of a robot arm always take this form:

$$M(q)\ddot{q} + C(q, \dot{q})\dot{q} + g(q) = \tau$$

- $M(q)$ : the mass (inertia) matrix — its construction is systematic and elegant
- $C(q, \dot{q})\dot{q}$ : the Coriolis and centrifugal terms
- $g(q)$ : the gravity term
- $\tau$ : the drive torque

This form is the **worldwide standard of robotics**, and textbooks, papers, and software
(such as ROS urdf) all assume it.

### Simulation demonstration: energy conservation

Running the demo `python/lagrangian_arm.py`, you can confirm that when the drive torque is $\tau = 0$
the total energy $T + V$ is conserved as theory predicts:

```
Simulation time      : 10.0 s (5000 steps)
Total energy (mean)  : 24.5250 J
Energy variation     : 6.453e-06 J (0.00003% of the mean)
```

A variation of **0.00003%** of the mean is within the range of RK4 numerical error.
It is numerical evidence that the Lagrangian formulation handles conservative systems naturally.

![Simulation result of the Lagrangian arm](../lagrangian_arm_result.png)

---

## 4. Where the Newton–Euler method is natural — modes of transportation

For modes of transportation (automobiles, aircraft, ships), the Newton–Euler formulation is the standard
for the following four reasons.

### Reason 1: Nonholonomic constraints

An automobile's tire has the constraint "no side-slip."
This is a **constraint on velocity**, and it cannot be integrated into a relation among coordinates.

The tire's nonholonomic constraint is $v_{y,\text{wheel}} = 0$ (the lateral velocity of the tire is zero),
which takes the form $A(q)\dot{q} = 0$ rather than $f(q) = 0$. It cannot be eliminated by a coordinate transformation alone.

To handle this in the Lagrangian formulation requires the Lagrange–d'Alembert equations:

$$\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}}\right) - \frac{\partial L}{\partial q} = \tau + A^T(q)\lambda \quad (\lambda: \text{undetermined multiplier})$$

The undetermined multiplier $\lambda$ adds to the count, and the system of equations grows larger.

**In Newton–Euler you simply write this constraint down as an external force — "the force the tire generates"**:

```cpp
// excerpt from src/car_dynamic.cpp
// Handle the tire lateral force directly as an external force
double a11 = -(Cf + Cr) / (m * vx);   // convert lateral force to lateral acceleration
double a21 = -(lf*Cf - lr*Cr) / (Iz * vx);  // yaw moment due to lateral force
```

### Reason 2: Non-conservative forces govern the motion

| Type of force | Example in transportation | Incorporation into the Lagrangian |
|---|---|---|
| Tire friction | Cornering force | Not possible (non-conservative force) |
| Air resistance | Drag $D = \frac{1}{2}\rho V^2 C_D S$ | Not possible (non-conservative force) |
| Fluid drag | Viscous resistance of a ship | Not possible (non-conservative force) |
| Thrust | Engine, thruster | Not possible (non-conservative force) |
| Gravity | Restoring force of aircraft/ships | Possible (conservative force) |

The Lagrangian $L = T - V$ can directly handle only conservative forces.
Non-conservative forces can only be added to the right-hand side as "generalized forces,"
and the advantage peculiar to the Lagrangian formulation (the beauty of the equations coming out
of a scalar function) no longer comes into play.

### Reason 3: Direct connection to control design

Modern control theory requires a **state equation**:

$$\dot{x} = f(x, u) \quad \text{or} \quad \dot{x} = Ax + Bu \quad \text{(in the linear case)}$$

**Newton–Euler → state equation: natural**

```cpp
// src/car_dynamic.cpp — building the linear state equation directly
Eigen::Matrix2d A;
A << a11, a12,
     a21, a22;
Eigen::Vector2d B(Cf/m, lf*Cf/Iz);
// can be integrated directly with RK4 as ẋ = Ax + B·δ
```

**Lagrangian → state equation: the inverse of M is required**

```python
# python/lagrangian_arm.py
def dynamics(state, tau):
    q, dq = state[:2], state[2:]
    M = manipulator_matrices(q)
    coriolis, gravity = manipulator_rhs(q, dq)
    # the extra step of taking the inverse of M is always required here
    ddq = np.linalg.solve(M, tau - coriolis - gravity)
    return np.concatenate([dq, ddq])
```

Since modes of transportation aim at the form of a state equation from the start,
Newton–Euler makes for a more natural design flow.

### Reason 4: For multibody systems, Kane's method is superior

For a full-vehicle model including the suspension and steering linkage
(typically 14 degrees of freedom), the partial-derivative terms of the energy
explode combinatorially.

At this scale, **Kane's method** (d'Alembert's principle based on partial velocities)
is the most efficient, and it is adopted internally by commercial multibody software
such as CarSim.

---

## 5. Automobile: the archetype of a nonholonomic constraint

### Kinematic bicycle model (low-speed regime)

Under the assumption of no side-slip at low speed, an automobile can be described by three state equations:

```math
\begin{align}
\dot{x} &= v\cos\psi \\
\dot{y} &= v\sin\psi \\
\dot{\psi} &= \frac{v}{L}\tan\delta \quad \text{← Ackermann geometry}
\end{align}
```

$L$ is the wheelbase and $\delta$ is the front-wheel steering angle.
The third equation is derived directly from the Ackermann condition, in which the left and right front wheels
share a common turning center. This simple form becomes the foundation of all of **Pure Pursuit, Stanley, LQR, and MPC**.

### Dynamic bicycle model (high-speed regime)

In the high-speed regime the tire takes a slip angle $\alpha$ and generates a lateral force $F_y = -C_\alpha\alpha$.
Linearizing Newton–Euler:

```math
\begin{align}
m(\dot{v}_y + v_x\dot{\psi}) &= F_{yf}\cos\delta + F_{yr} & \text{(lateral force balance)} \\
I_z\ddot{\psi} &= \ell_f F_{yf}\cos\delta - \ell_r F_{yr} & \text{(yaw moment balance)}
\end{align}
```

With state $\mathbf{x} = (v_y, \dot{\psi})^T$ and input $u = \delta$, it becomes a **linear time-invariant system $\dot{x} = Ax + Bu$**.
This form is the standard form of Rajamani's textbook Ch.2 and is the basis of ESC and LKAS design.

### Simulation results

```
Control law     RMS deviation [m]   Max steering [°]   Characteristic
Pure Pursuit       0.195         16.69   look-ahead arc tracking
Stanley            0.204         25.39   fast recovery using cross-track error explicitly
LQR                0.191         35.00   Riccati analytic solution + FF correction
MPC                0.225         24.00   constrained receding-horizon optimization

Yaw-rate settling time: 0.1–0.3 s (the automobile's characteristically fast response)
Stability factor Kv = 0.00104 (understeer)
```

A response time as short as 0.1–0.3 seconds is characteristic of the automobile.
ESC watches the difference between this "intended yaw rate" and the actual yaw rate
and intervenes within 0.1 second.

---

## 6. Aircraft: free space and eigenmodes

### Six-degree-of-freedom rigid-body motion (full equations)

Since an aircraft undergoes free-space motion with no ground constraint,
no nonholonomic constraint exists. Instead, the complete Newton–Euler equations
in the body coordinate frame are required:

```math
\begin{align}
m(\dot{u} + qw - rv) &= F_x - mg\sin\theta & \text{(including the Coriolis term about the pitch axis)} \\
m(\dot{v} + ru - pw) &= F_y + mg\cos\theta\sin\phi \\
m(\dot{w} + pv - qu) &= F_z + mg\cos\theta\cos\phi
\end{align}
```

```math
\begin{align}
I_x\dot{p} - (I_y - I_z)qr &= L & \text{(roll: controlled by aileron }\delta_a\text{)} \\
I_y\dot{q} - (I_z - I_x)rp &= M & \text{(pitch: controlled by elevator }\delta_e\text{)} \\
I_z\dot{r} - (I_x - I_y)pq &= N & \text{(yaw: controlled by rudder }\delta_r\text{)}
\end{align}
```

The external forces and moments are determined by aerodynamic forces:

```math
F = \frac{1}{2}\rho V^2 S \cdot C_*(\alpha,\, \beta,\, q,\, \delta_a,\, \delta_e,\, \delta_r,\, \ldots)
```

### Separation of longitudinal and lateral motion

Through linearization of a symmetric airframe shape, the motion **separates naturally** into two.
This too is not obvious in the Lagrangian formulation, but derived from the physical force balance
of Newton–Euler it can be understood with good clarity.

```
Longitudinal : (Δu, w, q, θ)  ← actuated by elevator δ_e
Lateral      : (β, p, r, φ)  ← actuated by aileron δ_a + rudder δ_r
```

### Eigenvalues of Boeing 747 longitudinal motion (simulation results)

```cpp
// src/aircraft.cpp — Boeing 747 cruise condition (altitude 12,200 m, V=235 m/s)
Eigen::Matrix4d A;
A << -0.00643,   0.0263,    0.0,  -9.81,
     -0.0941,   -0.624,  235.0,   0.0,
     -0.000222, -0.00153, -0.668,  0.0,
      0.0,       0.0,     1.0,    0.0;
```

Results of the eigenvalue analysis:

```
Short-period mode:
  Eigenvalue: -0.6455 ± 0.6007j
  Period    : 10.5 s
  Character : fast, highly damped. Directly tied to the pilot's handling feel.

Phugoid mode:
  Eigenvalue: -0.0037 ± 0.0074j
  Period    : 847 s (about 14 minutes)
  Character : slow, lightly damped. A long-period oscillation slowly exchanging speed and altitude.

Period ratio: 847 / 10.5 ≈ 81 times
```

These **two eigenmodes** are the central subject of aircraft control design.
The stability augmentation system (SAS) shapes the damping of these modes.

Eigenmodes almost never become a problem in automobiles, but in aircraft
they become an essential design challenge.
This too is a characteristic that comes from the **low density of the medium (air) and free-space motion**.

---

## 7. Ship: high-density medium and added mass

### The decisive difference from automobiles and aircraft

The point where a ship greatly differs from other modes of transportation is the "high density of the medium."

| Phenomenon | Automobile | Aircraft | Ship |
|---|---|---|---|
| Added mass | Negligible | Almost negligible | **Same order as the hull** |
| Fluid drag | Air resistance (small) | Air resistance (medium) | **Water resistance (large)** |
| Restoring force | Tire/road | Gravity (static stability) | **Buoyancy, metacenter** |
| Response time | 0.1–1 s | Seconds to minutes | **Tens of seconds to minutes** |

### Fossen's standard model

The industry-standard formulation by Thor I. Fossen (NTNU):

$$M\dot{\nu} + C(\nu)\nu + D(\nu)\nu + g(\eta) = \tau + \tau_{\text{wind}} + \tau_{\text{wave}}$$

where each variable is:

- $\eta$ : position and attitude in the inertial frame $(x, y, z, \phi, \theta, \psi)^T$
- $\nu$ : velocity and angular velocity in the body frame $(u, v, w, p, q, r)^T$
- $M = M_{RB} + M_A$ : rigid-body inertia + added mass
- $D(\nu)$ : fluid drag (linear + quadratic terms)
- $g(\eta)$ : restoring forces and moments

This form is a direct extension of the Newton–Euler formulation.
The added mass $M_A$ is the effect that "when accelerating, the surrounding water accelerates along with it,"
and although it can be represented in the Lagrangian formulation too, the physical intuition is easier to grasp with Newton–Euler.

### Nomoto first-order model (a simplified form for control)

For course control, Nomoto's simplified model is widely used:

$$T\ddot{\psi} + \dot{\psi} = K\delta$$

where $T = 50$ s (the mid-size container ship of this demo) and $K = 0.18$ (the rudder-effectiveness gain).

### Simulation results

```
Target heading : course change to 30 degrees
Settling time  : 124.8 s (within ±1.5 degrees)
Max rudder     : 35.0 degrees (rudder saturation)

Rudder saturates at the initial motion, then settles while decaying.
Against the automobile's yaw response (<1 s), the ship takes over 2 minutes.
→ the effect of added mass, fluid damping, and large hull inertia.
```

---

## 8. The differences seen in simulation results

### A three-order-of-magnitude difference in time scales

While dealing with the same "rigid-body rotational motion," the response time varies over
three orders of magnitude due to the differences in medium and inertia:

```
Automobile yaw response : 0.1–1 s        ← tire friction and road reaction respond instantly
Aircraft short period   : 1–10 s         ← the instantaneous responsiveness of aerodynamic forces
Aircraft phugoid        : 30–1000 s      ← the gentle exchange of speed ↔ altitude
Ship course change      : tens of seconds to minutes   ← high-density fluid and large inertia
Robot arm               : 0.1–a few s    ← holonomic constraint, electric actuators
```

### The difference in "naturalness" produced by the difference in formulation

| System | Formulation to use | Using the opposite formulation |
|---|---|---|
| Robot arm | Lagrangian ($M(q)\ddot{q} + C\dot{q} + g = \tau$) | With Newton–Euler, the trouble of tracking force directions through coordinate transformations |
| Automobile | Newton–Euler (direct connection to the state equation) | With Lagrangian, undetermined multipliers are required |
| Aircraft | Newton–Euler (modal analysis) | With Lagrangian, organizing the rotational terms takes effort |
| Ship | Newton–Euler + Fossen formulation | With Lagrangian, the treatment of added mass is complex |
| Full vehicle (14 DOF) | Kane's method | With both Lagrangian and Newton–Euler, the equations explode |

### How to read the generated graphs

**`vehicle_dynamics_results.png`** (transportation demo)

- **Graphs 1 & 2 (path tracking)**: all four control laws track the ellipse. Differences in steering amount and accuracy reflect the differences in the design philosophy of the control laws
- **Graph 3 (dynamic response)**: for 1 degree of steering, the yaw rate settles in 0.1–0.3 s. The fast response of a nonholonomic constraint system
- **Graph 4 (aircraft short period)**: pitch oscillation that damps in about 10 seconds. Without an SAS, a cause of motion sickness
- **Graph 5 (phugoid)**: speed and altitude oscillate slowly over a long period of 847 seconds. Without an autopilot, it decays naturally over 14 minutes
- **Graph 6 (ship course change)**: settles over 2 minutes. The initial rudder saturation and the effect of added mass are visible

**`lagrangian_arm_result.png`** (Lagrangian demo)

- **Left (joint angles)**: free motion like a double pendulum in which the two links rotate intricately
- **Right (energy conservation)**: the variation of $T + V$ is $6\times10^{-6}$ J against 24.5 J (0.00003%). Evidence that the Lagrangian formulation handles conservative systems naturally

---

## 9. A decision flow for choosing a formulation

```
You need to set up the equations of motion for a problem
│
├─ Q1: Is it a multibody system (connected rigid bodies with 6+ DOF)?
│  │
│  └─ Yes → [Kane's method / multibody analysis software]
│             CarSim, Adams, Modelica, etc.
│
└─ Single or a small number of rigid bodies
   │
   ├─ Q2: Is the constraint holonomic or nonholonomic?
   │  │
   │  ├─ Holonomic (expressible as a coordinate relation, e.g., a robot joint)
   │  │  │
   │  │  └─ Q3: Is it a conservative system (gravity/elastic force dominate)?
   │  │     │
   │  │     ├─ Yes → [Lagrangian method] ★ the beauty of energy conservation comes into play
   │  │     │          everything comes out of the scalar L = T - V
   │  │     │
   │  │     └─ No (non-conservative forces dominate)
   │  │        → [Lagrangian or Newton-Euler] either is about the same
   │  │          if control design is the objective, Newton-Euler is easier
   │  │
   │  └─ Nonholonomic (velocity constraint, e.g., no tire side-slip)
   │     │
   │     └─ [Newton-Euler] ★ the constraint can be represented naturally as an external force
   │        it is easier to avoid undetermined multipliers (Lagrange-d'Alembert)
   │
   └─ Q4: Is the final objective control design?
      │
      ├─ Yes → [Newton-Euler] connects directly to the state equation ẋ = f(x,u)
      │
      └─ No (theoretical analysis, energy methods)
         → judge according to the problem
```

### Summary

| Formulation to use | Structure of the problem | Typical examples |
|---|---|---|
| **Lagrangian** | Holonomic constraint + conservative system | Robot arm, planetary motion, vibration of a string |
| **Newton–Euler** | Nonholonomic constraint or control-oriented | Automobile, aircraft, ship, spacecraft attitude control |
| **Kane's method** | Large-scale multibody system | Full vehicle, human-body model, satellite deployment mechanism |

---

## 10. References

### Transportation (Newton–Euler)

- R. Rajamani, *Vehicle Dynamics and Control*, Springer, 2012.  
  The standard text for the automobile's two-degree-of-freedom linear model.

- B. Etkin and L. D. Reid, *Dynamics of Flight: Stability and Control*, Wiley, 1996.  
  The aircraft's six-degree-of-freedom model and eigenmode analysis.

- T. I. Fossen, *Handbook of Marine Craft Hydrodynamics and Motion Control*, Wiley, 2011.  
  The standard formulation for ships. The original source of the added-mass and Nomoto models.

### Robotics (Lagrangian)

- M. W. Spong, S. Hutchinson, M. Vidyasagar, *Robot Modeling and Control*, Wiley, 2005.  
  The standard derivation of $M(q)\ddot{q} + C(q,\dot{q})\dot{q} + g(q) = \tau$.

- B. Siciliano et al., *Robotics: Modelling, Planning and Control*, Springer, 2009.  
  A comprehensive text on manipulator dynamics by the Lagrangian method.

### Multibody systems (Kane's method)

- T. R. Kane and D. A. Levinson, *Dynamics: Theory and Applications*, McGraw-Hill, 1985.  
  The original source of Kane's method. A formulation of d'Alembert's principle based on partial velocities.

### Analytical mechanics (theoretical background)

- H. Goldstein, C. Poole, J. Safko, *Classical Mechanics*, Addison-Wesley, 2002.  
  The rigorous definition of holonomic and nonholonomic constraints and the Lagrange–d'Alembert equations.

---

*To run the simulations referenced in this document:*

```bash
# Transportation demo (Newton–Euler formulation)
python vehicle_dynamics_simulation.py

# Robot arm demo (Lagrangian method)
python python/lagrangian_arm.py
```
