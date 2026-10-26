"""Reproducible road profiles and spatial -> time conversion.

A road profile is a smooth function of travelled distance x [m]: height z(x) [m]
and slope dz/dx [-] (both analytic, vectorised).  For a constant vehicle speed v
[m/s] the time-domain disturbance seen by the wheel is

    zr(t)     = z(v t)
    zr_dot(t) = v * dz/dx (v t)

All profiles are at least C^1 (continuous height and slope), so no unintended
impulses are fed to the tyre.  The road height at x = 0 is defined as 0.

Profiles
--------
FlatRoad      : z = 0.
BumpRoad      : raised-cosine speed hump  z = h/2 (1 - cos(2 pi (x-x0)/L))  on [x0, x0+L].
RandomRoad    : stochastic ISO-8608-style profile synthesised as a sum of cosines with
                random phases (spectral representation), PSD  Gd(n) = Gd(n0) (n/n0)^-2
                with n0 = 0.1 cycle/m.  Reproducible from an integer seed.

Road specification mini-language (used by the environment, see `sample_road`)
----------------------------------------------------------------------------
  "flat", "bump", "rough_A" ... "rough_E" (ISO class), "rough" (= rough_C),
  "train" (training mixture), "test" (held-out mixture)
  dict  {"kind": "bump", "height": (0.02, 0.05), "speed": (8, 16), ...}
        a 2-tuple of numbers is sampled uniformly (with the environment RNG)
  RoadProfile instance (used as-is; speed taken from the environment)
  list/tuple of any of the above: one is chosen uniformly at each reset
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

# ISO 8608 road-class roughness coefficient Gd(n0) at n0 = 0.1 cycle/m, in m^3
# (class centre values; A = very smooth ... E = very rough).
ISO_GD_N0 = {
    "A": 16e-6,
    "B": 64e-6,
    "C": 256e-6,
    "D": 1024e-6,
    "E": 4096e-6,
}
ISO_N0 = 0.1  # cycle/m


class RoadProfile:
    """Base class: a road height profile z(x)."""

    name: str = "road"

    def height(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def slope(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def time_series(self, speed: float, dt: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Road height and rate at times t_i = i*dt, i = 0..n-1, for constant speed.

        Returns (zr, zr_dot) as float64 arrays.
        """
        if not speed > 0:
            raise ValueError("speed must be > 0")
        x = speed * dt * np.arange(n, dtype=np.float64)
        return self.height(x), speed * self.slope(x)

    def describe(self) -> str:
        return self.name


class FlatRoad(RoadProfile):
    name = "flat"

    def height(self, x):
        return np.zeros_like(np.asarray(x, dtype=np.float64))

    def slope(self, x):
        return np.zeros_like(np.asarray(x, dtype=np.float64))


class BumpRoad(RoadProfile):
    """One or several raised-cosine humps.

    height  : hump height h [m]
    length  : hump length L along the road [m]
    start   : position x0 of the first hump's leading edge [m]
    count   : number of humps (>=1)
    spacing : distance between leading edges of consecutive humps [m] (>= length)
    The profile and its slope are continuous (slope is 0 at both ends).
    """

    def __init__(self, height: float = 0.04, length: float = 2.5, start: float = 10.0,
                 count: int = 1, spacing: float = 30.0):
        if not (height >= 0 and length > 0 and start >= 0 and count >= 1):
            raise ValueError("invalid bump parameters")
        if count > 1 and spacing < length:
            raise ValueError("spacing must be >= length so humps do not overlap")
        self.h, self.L, self.x0, self.count, self.spacing = height, length, start, int(count), spacing
        self.name = f"bump(h={height:.3f}m,L={length:.2f}m,n={self.count})"

    def height(self, x):
        x = np.asarray(x, dtype=np.float64)
        z = np.zeros_like(x)
        for i in range(self.count):
            s = (x - (self.x0 + i * self.spacing)) / self.L
            inside = (s >= 0.0) & (s <= 1.0)
            z = z + np.where(inside, 0.5 * self.h * (1.0 - np.cos(2.0 * np.pi * s)), 0.0)
        return z

    def slope(self, x):
        x = np.asarray(x, dtype=np.float64)
        dz = np.zeros_like(x)
        for i in range(self.count):
            s = (x - (self.x0 + i * self.spacing)) / self.L
            inside = (s >= 0.0) & (s <= 1.0)
            dz = dz + np.where(inside, 0.5 * self.h * 2.0 * np.pi / self.L * np.sin(2.0 * np.pi * s), 0.0)
        return dz


class RandomRoad(RoadProfile):
    """Stochastic road (ISO 8608-style PSD) from a seeded sum of cosines.

    z(x) = sum_i A_i cos(2 pi n_i x + phi_i) - (value at x = 0)

    Spatial-frequency range [n_min, n_max] (cycle/m) is split into `n_components`
    logarithmically spaced bins; in bin i the frequency n_i is drawn uniformly
    inside the bin and the amplitude is A_i = sqrt(2 Gd(n_i) dn_i) so the
    discrete spectrum approximates the PSD Gd(n) = Gd(n0) (n/n0)^-2.
    Phases are uniform on [0, 2 pi).  The same seed always gives the same road.
    """

    def __init__(self, iso_class: str = "C", seed: int = 0, roughness: Optional[float] = None,
                 n_components: int = 100, n_min: float = 0.02, n_max: float = 4.0):
        if roughness is None:
            if iso_class not in ISO_GD_N0:
                raise ValueError(f"iso_class must be one of {sorted(ISO_GD_N0)}")
            gd0 = ISO_GD_N0[iso_class]
        else:
            if not roughness > 0:
                raise ValueError("roughness must be > 0")
            gd0 = float(roughness)
        if not (0 < n_min < n_max and n_components >= 1):
            raise ValueError("invalid spatial-frequency range")
        self.iso_class, self.seed, self.gd0 = iso_class, int(seed), gd0
        rng = np.random.default_rng(self.seed)
        edges = np.geomspace(n_min, n_max, n_components + 1)
        n_i = rng.uniform(edges[:-1], edges[1:])
        dn = np.diff(edges)
        gd = gd0 * (n_i / ISO_N0) ** -2.0
        self._amp = np.sqrt(2.0 * gd * dn)
        self._w = 2.0 * np.pi * n_i
        self._phi = rng.uniform(0.0, 2.0 * np.pi, size=n_components)
        self._z0 = float(np.sum(self._amp * np.cos(self._phi)))
        self.name = f"random(ISO {iso_class if roughness is None else 'custom'}, seed={self.seed})"

    def height(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        out = np.empty_like(x)
        # chunked to bound memory: (chunk x n_components)
        for a in range(0, x.size, 4096):
            xs = x[a:a + 4096, None]
            out[a:a + 4096] = np.cos(self._w[None, :] * xs + self._phi[None, :]) @ self._amp
        return out - self._z0

    def slope(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        out = np.empty_like(x)
        aw = self._amp * self._w
        for a in range(0, x.size, 4096):
            xs = x[a:a + 4096, None]
            out[a:a + 4096] = -(np.sin(self._w[None, :] * xs + self._phi[None, :]) @ aw)
        return out


# ---------------------------------------------------------------- specification
PRESETS: Dict[str, Any] = {
    "flat": {"kind": "flat"},
    "bump": {"kind": "bump"},
    "rough": {"kind": "random", "iso_class": "C"},
    **{f"rough_{c}": {"kind": "random", "iso_class": c} for c in ISO_GD_N0},
}

# Training mixture (ranges are sampled per episode with the env RNG).
TRAIN_ROADS = [
    {"kind": "flat", "speed": (10.0, 20.0)},
    {"kind": "bump", "height": (0.02, 0.04), "length": (1.5, 3.0), "start": (5.0, 15.0), "speed": (8.0, 14.0)},
    {"kind": "random", "iso_class": "B", "speed": (10.0, 20.0)},
    {"kind": "random", "iso_class": "C", "speed": (10.0, 20.0)},
]

# Held-out mixture: ISO classes, bump heights and speeds that are NOT covered by
# TRAIN_ROADS (class A and D roads, taller bumps, faster driving).
TEST_ROADS = [
    {"kind": "bump", "height": (0.05, 0.06), "length": (3.0, 4.0), "start": (5.0, 15.0), "speed": (14.0, 18.0)},
    {"kind": "random", "iso_class": "A", "speed": (22.0, 28.0)},
    {"kind": "random", "iso_class": "D", "speed": (8.0, 12.0)},
]

PRESETS["train"] = TRAIN_ROADS
PRESETS["test"] = TEST_ROADS

_KINDS = {"flat": FlatRoad, "bump": BumpRoad, "random": RandomRoad}


@dataclass
class RoadSample:
    """A concrete road + speed for one episode."""

    profile: RoadProfile
    speed: float
    seed: Optional[int]
    label: str


def _is_range(v: Any) -> bool:
    return (isinstance(v, (tuple, list)) and len(v) == 2
            and all(isinstance(a, (int, float, np.floating, np.integer)) for a in v))


def _draw(v: Any, rng: np.random.Generator) -> Any:
    if _is_range(v):
        lo, hi = float(v[0]), float(v[1])
        if hi < lo:
            raise ValueError(f"invalid range {v}")
        return float(rng.uniform(lo, hi))
    return v


def sample_road(spec: Any, rng: np.random.Generator, default_speed: Any = 15.0,
                road_seed: Optional[int] = None) -> RoadSample:
    """Resolve a road specification into a concrete (profile, speed).

    `rng` is consumed ONLY here (once per reset), so the result depends only on
    the RNG state, never on the agent's actions.  If `road_seed` is given it
    overrides the seed of stochastic roads (but not other random draws).
    """
    if isinstance(spec, RoadProfile):
        speed = float(_draw(default_speed, rng))
        return RoadSample(spec, speed, getattr(spec, "seed", None), spec.describe())
    if isinstance(spec, str):
        if spec not in PRESETS:
            raise ValueError(f"unknown road preset '{spec}'; options: {sorted(PRESETS)}")
        return sample_road(PRESETS[spec], rng, default_speed, road_seed)
    if isinstance(spec, (list, tuple)) and not _is_range(spec):
        if len(spec) == 0:
            raise ValueError("empty road specification list")
        idx = int(rng.integers(len(spec)))
        return sample_road(spec[idx], rng, default_speed, road_seed)
    if isinstance(spec, dict):
        params = copy.deepcopy(spec)
        kind = params.pop("kind", None)
        if kind not in _KINDS:
            raise ValueError(f"road 'kind' must be one of {sorted(_KINDS)}, got {kind!r}")
        speed = float(_draw(params.pop("speed", default_speed), rng))
        seed_in = params.pop("seed", None)
        kwargs = {k: _draw(v, rng) for k, v in params.items()}
        seed = None
        if kind == "random":
            if road_seed is not None:
                seed = int(road_seed)
            elif seed_in is not None:
                seed = int(seed_in)
            else:
                seed = int(rng.integers(2**31 - 1))
            kwargs["seed"] = seed
        if kind == "bump" and "count" in kwargs:
            kwargs["count"] = int(kwargs["count"])
        profile = _KINDS[kind](**kwargs)
        return RoadSample(profile, speed, seed, f"{profile.describe()} @ {speed:.1f} m/s")
    raise TypeError(f"unsupported road specification of type {type(spec).__name__}")
