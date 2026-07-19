"""
LQR gain computation (the part kept in Python)
==============================================
Solve the continuous-time algebraic Riccati equation (CARE) with scipy and
output the gain K.

  A^T P + P A - P B R^{-1} B^T P + Q = 0
  K = R^{-1} B^T P

The part corresponding to the "analytical derivation" of this equation is kept
in Python; C++ merely receives the numeric values of K and uses them for control.
"""

import sys
import json
import numpy as np
from scipy.linalg import solve_continuous_are


def compute_lqr_gain(
    m=1500.0, Iz=2500.0, lf=1.2, lr=1.5,
    Cf=80000.0, Cr=80000.0, vx=8.0,
    Q_diag=(10.0, 0.5, 10.0, 0.5),
    R_val=1.0,
):
    """
    Return the LQR gain for the 4th-order path-deviation system
    x = (e_y, ė_y, e_ψ, ė_ψ).
    Reference: Rajamani "Vehicle Dynamics and Control" Eq.(2.45)-(2.49)
    """
    A = np.array([
        [0, 1, 0, 0],
        [0, -(2*Cf + 2*Cr)/(m*vx),  (2*Cf + 2*Cr)/m,
             (-2*Cf*lf + 2*Cr*lr)/(m*vx)],
        [0, 0, 0, 1],
        [0, -(2*Cf*lf - 2*Cr*lr)/(Iz*vx), (2*Cf*lf - 2*Cr*lr)/Iz,
             -(2*Cf*lf**2 + 2*Cr*lr**2)/(Iz*vx)],
    ])
    B = np.array([[0], [2*Cf/m], [0], [2*Cf*lf/Iz]])
    Q = np.diag(Q_diag)
    R = np.array([[R_val]])

    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)   # shape (1, 4)
    return K.flatten().tolist()


if __name__ == "__main__":
    output_path = sys.argv[1] if len(sys.argv) > 1 else "lqr_k.json"
    K = compute_lqr_gain()
    with open(output_path, "w") as f:
        json.dump({"K": K}, f)
    print(f"LQR K = {[f'{v:.4f}' for v in K]}")
    print(f"Written: {output_path}")
