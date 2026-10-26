"""Configuration dataclasses for the quarter-car active-suspension environment.

Every physical constant, scale and weight used anywhere in the project lives
here (or in road_profiles.py for road presets).  Nothing numerical is
hard-coded in the dynamics, reward or environment code.

Units are SI throughout: metres, seconds, kilograms, newtons.
Coordinate convention: z axes point UP; displacements are measured relative
to the STATIC EQUILIBRIUM position (so gravity is already balanced and does
not appear in the equations of motion).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Tuple, Union

G = 9.81  # m/s^2, used only to compute the static tyre load (ms + mu) * g

SpeedSpec = Union[float, Tuple[float, float]]


@dataclass(frozen=True)
class VehicleParams:
    """Quarter-car parameters.

    Default values are typical, round-number values of the magnitude used for a
    mid-size passenger-car corner in the quarter-car literature.  They are NOT
    copied from one specific paper or vehicle datasheet and are a modelling
    choice; every one can be overridden.

    ms : sprung mass (quarter body)                      [kg]
    mu : unsprung mass (wheel, tyre, brake, hub)         [kg]
    ks : suspension spring stiffness                     [N/m]
    cs : passive suspension damping coefficient          [N s/m]
    kt : tyre vertical stiffness                         [N/m]
    ct : tyre vertical damping (0 = no tyre damping)     [N s/m]
    """

    ms: float = 290.0
    mu: float = 59.0
    ks: float = 16000.0
    cs: float = 1000.0
    kt: float = 190000.0
    ct: float = 0.0

    def __post_init__(self) -> None:
        for name in ("ms", "mu", "ks", "kt"):
            if not getattr(self, name) > 0:
                raise ValueError(f"VehicleParams.{name} must be > 0")
        for name in ("cs", "ct"):
            if getattr(self, name) < 0:
                raise ValueError(f"VehicleParams.{name} must be >= 0")

    @property
    def static_tyre_load(self) -> float:
        """Static vertical tyre load (ms + mu) * g  [N]."""
        return (self.ms + self.mu) * G

    @property
    def body_freq_hz(self) -> float:
        """Approximate body (sprung) bounce frequency, tyre treated as rigid [Hz]."""
        return math.sqrt(self.ks / self.ms) / (2 * math.pi)

    @property
    def wheel_hop_freq_hz(self) -> float:
        """Approximate wheel-hop frequency, body treated as fixed [Hz]."""
        return math.sqrt((self.ks + self.kt) / self.mu) / (2 * math.pi)

    @property
    def body_damping_ratio(self) -> float:
        """Approximate body damping ratio cs / (2 sqrt(ks ms))."""
        return self.cs / (2.0 * math.sqrt(self.ks * self.ms))


@dataclass(frozen=True)
class ActuatorParams:
    """Ideal force actuator between sprung and unsprung mass.

    u_max : force corresponding to a normalized command of +/-1   [N]
    The actuator has no dynamics (no lag, bandwidth limit or power limit);
    the only non-ideality is saturation at +/-u_max.  1500 N is roughly
    0.5 g acting on the quarter body mass: a design choice, not a datasheet value.
    """

    u_max: float = 1500.0

    def __post_init__(self) -> None:
        if not self.u_max > 0:
            raise ValueError("ActuatorParams.u_max must be > 0")


@dataclass(frozen=True)
class TimingParams:
    """Time discretisation.

    dt_sim          : fixed RK4 integration step                [s]
    dt_ctrl         : RL control step (actuator command is held constant, ZOH) [s]
    episode_duration: episode length                            [s]
    dt_ctrl must be an exact integer multiple of dt_sim (checked).
    """

    dt_sim: float = 0.001
    dt_ctrl: float = 0.01
    episode_duration: float = 10.0

    def __post_init__(self) -> None:
        if not (self.dt_sim > 0 and self.dt_ctrl > 0 and self.episode_duration > 0):
            raise ValueError("All time parameters must be > 0")
        ratio = self.dt_ctrl / self.dt_sim
        if abs(ratio - round(ratio)) > 1e-9 or round(ratio) < 1:
            raise ValueError("dt_ctrl must be an integer multiple of dt_sim")
        steps = self.episode_duration / self.dt_ctrl
        if abs(steps - round(steps)) > 1e-9:
            raise ValueError("episode_duration must be an integer multiple of dt_ctrl")

    @property
    def substeps(self) -> int:
        return int(round(self.dt_ctrl / self.dt_sim))

    @property
    def max_steps(self) -> int:
        return int(round(self.episode_duration / self.dt_ctrl))


@dataclass(frozen=True)
class RewardParams:
    """Reward = -(sum_i w_i * term_i), every term dimensionless.

    Reference scales (units in brackets):
      a_ref     : acceleration scale for ride comfort          [m/s^2]
      d_ref     : suspension-deflection scale                  [m]
      f_ref_frac: dynamic tyre force scale as a fraction of static tyre load [-]
      u_ref is the actuator max force (ActuatorParams.u_max).
    Weights w_* are dimensionless.

    Rationale for the reference scales: they are set to the order of magnitude of
    the PASSIVE suspension's RMS response on an ISO class-C road at 15 m/s in this
    simulator (measured with scripts/passive_damping_sweep.py: ~1.1 m/s^2,
    ~0.012 m, ~0.25 of static tyre load), rounded.  Each normalised term is then
    O(1) for a passive car on a typical rough road, so no term dominates only
    because of its units.  The importance of each term is expressed by its weight.
    """

    a_ref: float = 1.0
    d_ref: float = 0.015
    f_ref_frac: float = 0.25
    w_acc: float = 1.0
    w_defl: float = 0.3
    w_tyre: float = 0.3
    w_effort: float = 0.05
    w_rate: float = 0.05
    # Fail-safe: added per REMAINING step if the episode terminates early,
    # so that terminating can never be a way of avoiding negative reward.
    termination_cost_per_step: float = 10.0


@dataclass(frozen=True)
class SafetyParams:
    """Numerical / physical-validity guards (documented, tested).

    hard_suspension_limit : |zs-zu| above this => terminated ("suspension_hard_limit")
    hard_tyre_limit       : |zu-zr| above this => terminated ("tyre_hard_limit")
    Non-finite state => terminated ("non_finite_state").
    These are fail-safes far outside normal operation (the plant is passively
    stable and the actuator force is bounded), not part of the dynamics.
    travel_limit_terminates / contact_loss_terminates are OFF by default; the
    events (|zs-zu| > travel_limit, tyre load ratio <= 0) are only counted and
    reported in `info`.
    """

    hard_suspension_limit: float = 0.30
    hard_tyre_limit: float = 0.15
    travel_limit: float = 0.10   # |zs-zu| [m] counted as a travel (rattle-space) violation
    travel_limit_terminates: bool = False
    contact_loss_terminates: bool = False


@dataclass(frozen=True)
class ObservationScales:
    """Normalisation scales: obs_i = physical_i / scale_i (no clipping applied).

    Chosen as typical magnitudes of each quantity, so that normalised values
    are O(1).  Units in brackets.
    """

    acc: float = 2.0           # [m/s^2]
    defl: float = 0.05         # [m]     suspension deflection  zs-zu
    defl_rate: float = 0.5     # [m/s]   suspension velocity
    tyre_defl: float = 0.01    # [m]     tyre deflection zu-zr
    unsprung_vel: float = 0.5  # [m/s]   unsprung velocity


@dataclass(frozen=True)
class EnvConfig:
    """Complete environment configuration."""

    vehicle: VehicleParams = field(default_factory=VehicleParams)
    actuator: ActuatorParams = field(default_factory=ActuatorParams)
    timing: TimingParams = field(default_factory=TimingParams)
    reward: RewardParams = field(default_factory=RewardParams)
    safety: SafetyParams = field(default_factory=SafetyParams)
    obs_scales: ObservationScales = field(default_factory=ObservationScales)
    # Road specification (see road_profiles.sample_road).  "train" = training mixture.
    road: Any = "train"
    # Default speed [m/s] for road specs that do not carry their own "speed".
    speed: SpeedSpec = 15.0
    # Std-dev of random perturbation of the initial vehicle state (0 = none).
    initial_state_noise: float = 0.0

    def with_(self, **kwargs: Any) -> "EnvConfig":
        """Return a copy with some fields replaced."""
        return replace(self, **kwargs)
