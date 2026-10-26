"""Two-degree-of-freedom quarter-car model and fixed-step RK4 integrator.

Sign convention (z axes point UP, all displacements relative to static equilibrium):

    d  = zs - zu                       suspension deflection  (>0: spring extended)
    dv = zs_dot - zu_dot               suspension velocity
    Fs = ks*d + cs*dv                  passive spring+damper force; acts on the
                                       sprung mass as -Fs and on the unsprung mass as +Fs
    Ft = kt*(zu - zr) + ct*(zu_dot - zr_dot)
                                       dynamic tyre force; acts on the unsprung
                                       mass as -Ft (Ft > 0: tyre compressed beyond static)
    u                                  actuator force; +u on the sprung mass (upward),
                                       -u on the unsprung mass (downward).

Equations of motion (tyre damping ct = 0 by default):

    ms * zs_ddot = -ks*(zs-zu) - cs*(zs_dot-zu_dot) + u
    mu * zu_ddot =  ks*(zs-zu) + cs*(zs_dot-zu_dot) - kt*(zu-zr) - ct*(zu_dot-zr_dot) - u

Because coordinates are measured from static equilibrium, the gravity forces
ms*g and mu*g are exactly balanced by the static spring/tyre deflections and
vanish from the equations.  The static tyre load (ms+mu)*g is still needed to
interpret the dynamic tyre force (contact-loss check) and is available from
VehicleParams.static_tyre_load.

State vector: x = [zs, zs_dot, zu, zu_dot].
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from .config import VehicleParams

State = Tuple[float, float, float, float]


class QuarterCar:
    """Stateless quarter-car dynamics + RK4 step (pure-Python floats for speed)."""

    def __init__(self, params: VehicleParams):
        self.p = params
        # Local copies avoid attribute lookups in the hot loop.
        self._ms = params.ms
        self._mu = params.mu
        self._ks = params.ks
        self._cs = params.cs
        self._kt = params.kt
        self._ct = params.ct

    # ------------------------------------------------------------------ dynamics
    def derivatives(self, x: Sequence[float], zr: float, zrd: float, u: float) -> State:
        """Return (zs_dot, zs_ddot, zu_dot, zu_ddot) for state x, road (zr, zr_dot), force u."""
        zs, vs, zu, vu = x
        fs = self._ks * (zs - zu) + self._cs * (vs - vu)
        ft = self._kt * (zu - zr) + self._ct * (vu - zrd)
        return (vs, (-fs + u) / self._ms, vu, (fs - ft - u) / self._mu)

    def sprung_acceleration(self, x: Sequence[float], zr: float, zrd: float, u: float) -> float:
        """Sprung-mass vertical acceleration zs_ddot at state x with force u [m/s^2]."""
        return self.derivatives(x, zr, zrd, u)[1]

    def tyre_force_dynamic(self, x: Sequence[float], zr: float, zrd: float) -> float:
        """Dynamic tyre force Ft = kt*(zu-zr) + ct*(zu_dot-zr_dot) [N]; >0 = compressed."""
        return self._kt * (x[2] - zr) + self._ct * (x[3] - zrd)

    # --------------------------------------------------------------- integration
    def rk4_step(
        self,
        x: Sequence[float],
        u: float,
        h: float,
        road0: Tuple[float, float],
        road_mid: Tuple[float, float],
        road1: Tuple[float, float],
    ) -> State:
        """One classical RK4 step of size h with actuator force u held constant (ZOH).

        road0/road_mid/road1 are (zr, zr_dot) at t, t+h/2, t+h.  The road is
        sampled, not interpolated, so the road disturbance is also handled at
        4th order for smooth profiles.
        """
        f = self.derivatives
        k1 = f(x, road0[0], road0[1], u)
        x2 = [x[i] + 0.5 * h * k1[i] for i in range(4)]
        k2 = f(x2, road_mid[0], road_mid[1], u)
        x3 = [x[i] + 0.5 * h * k2[i] for i in range(4)]
        k3 = f(x3, road_mid[0], road_mid[1], u)
        x4 = [x[i] + h * k3[i] for i in range(4)]
        k4 = f(x4, road1[0], road1[1], u)
        h6 = h / 6.0
        return (
            x[0] + h6 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
            x[1] + h6 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
            x[2] + h6 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]),
            x[3] + h6 * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]),
        )

    # ------------------------------------------------------------ linear analysis
    def matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Linear state-space form  x_dot = A x + B u + E [zr, zr_dot]^T.

        Returns (A, B, E) with shapes (4,4), (4,), (4,2).
        """
        p = self.p
        A = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [-p.ks / p.ms, -p.cs / p.ms, p.ks / p.ms, p.cs / p.ms],
                [0.0, 0.0, 0.0, 1.0],
                [p.ks / p.mu, p.cs / p.mu, -(p.ks + p.kt) / p.mu, -(p.cs + p.ct) / p.mu],
            ]
        )
        B = np.array([0.0, 1.0 / p.ms, 0.0, -1.0 / p.mu])
        E = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [p.kt / p.mu, p.ct / p.mu]])
        return A, B, E

    def total_energy(self, x: Sequence[float], zr: float = 0.0) -> float:
        """Mechanical energy 0.5 ms vs^2 + 0.5 mu vu^2 + 0.5 ks d^2 + 0.5 kt (zu-zr)^2 [J].

        Conserved for cs = ct = 0, u = 0 and constant road (used in tests).
        """
        p = self.p
        d = x[0] - x[2]
        return (
            0.5 * p.ms * x[1] ** 2
            + 0.5 * p.mu * x[3] ** 2
            + 0.5 * p.ks * d**2
            + 0.5 * p.kt * (x[2] - zr) ** 2
        )
