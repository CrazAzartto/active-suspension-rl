"""Physics / numerics tests for the quarter-car model and road profiles."""
import numpy as np
import pytest

from environment import (BumpRoad, FlatRoad, QuarterCar, RandomRoad, TimingParams,
                         VehicleParams)
from environment.config import EnvConfig

P = VehicleParams()
CAR = QuarterCar(P)


def simulate(car, x0, u_fn, road_fn, t_end, h=0.001):
    """Fixed-step RK4 helper. road_fn(t) -> (zr, zr_dot)."""
    x = tuple(x0)
    n = int(round(t_end / h))
    traj = [x]
    for k in range(n):
        t = k * h
        x = car.rk4_step(x, u_fn(t), h, road_fn(t), road_fn(t + h / 2), road_fn(t + h))
        traj.append(x)
    return np.array(traj)


def flat(t):
    return (0.0, 0.0)


# 1. force signs / equal-and-opposite --------------------------------------------------
def test_actuator_force_equal_and_opposite():
    x = (0.0, 0.0, 0.0, 0.0)
    for u in (500.0, -800.0):
        _, a_s, _, a_u = CAR.derivatives(x, 0.0, 0.0, u)
        assert a_s == pytest.approx(u / P.ms)           # + on sprung mass
        assert a_u == pytest.approx(-u / P.mu)          # - on unsprung mass
        assert P.ms * a_s + P.mu * a_u == pytest.approx(0.0, abs=1e-9)  # equal & opposite forces


def test_spring_and_tyre_force_signs():
    # sprung mass displaced up: spring pulls it down, pushes wheel up
    _, a_s, _, a_u = CAR.derivatives((0.01, 0, 0, 0), 0.0, 0.0, 0.0)
    assert a_s < 0 and a_u > 0
    # wheel displaced up with flat road: tyre pushes it down
    _, a_s, _, a_u = CAR.derivatives((0, 0, 0.01, 0), 0.0, 0.0, 0.0)
    assert a_u < 0
    # road raised above wheel: tyre pushes wheel up
    _, _, _, a_u = CAR.derivatives((0, 0, 0, 0), 0.01, 0.0, 0.0)
    assert a_u == pytest.approx(P.kt * 0.01 / P.mu)


def test_damper_force_sign_and_tyre_damping():
    _, a_s, _, a_u = CAR.derivatives((0, 1.0, 0, 0), 0, 0, 0)  # sprung moving up
    assert a_s == pytest.approx(-P.cs / P.ms) and a_u == pytest.approx(P.cs / P.mu)
    car_ct = QuarterCar(VehicleParams(cs=0.0, ct=500.0))  # isolate the tyre damper
    _, _, _, a_u = car_ct.derivatives((0, 0, 0, 1.0), 0, 0, 0)  # wheel moving up, road still
    assert a_u == pytest.approx(-500.0 / P.mu)
    _, _, _, a_u = car_ct.derivatives((0, 0, 0, 0), 0, 1.0, 0)  # road moving up
    assert a_u == pytest.approx(500.0 / P.mu)


# 2. equilibrium and analytic cases -----------------------------------------------------
def test_equilibrium_stays_at_rest():
    tr = simulate(CAR, (0, 0, 0, 0), lambda t: 0.0, flat, 5.0)
    assert np.max(np.abs(tr)) < 1e-15


def test_constant_force_steady_state_analytic():
    # Steady state under constant u on a flat road: ks*d = u, zu = 0  => zs = u/ks
    u = 600.0
    tr = simulate(CAR, (0, 0, 0, 0), lambda t: u, flat, 15.0)
    assert tr[-1, 0] == pytest.approx(u / P.ks, rel=1e-4)
    assert tr[-1, 2] == pytest.approx(0.0, abs=1e-6)
    assert tr[-1, 1] == pytest.approx(0.0, abs=1e-6) and tr[-1, 3] == pytest.approx(0.0, abs=1e-6)


def test_constant_road_offset_analytic():
    # A constant road offset c is followed exactly: zs = zu = c at equilibrium
    c = 0.02
    tr = simulate(CAR, (c, 0, c, 0), lambda t: 0.0, lambda t: (c, 0.0), 3.0)
    assert np.max(np.abs(tr - np.array([c, 0, c, 0]))) < 1e-12


def test_energy_conserved_undamped():
    car = QuarterCar(VehicleParams(cs=0.0, ct=0.0))
    x0 = (0.03, 0.0, -0.002, 0.1)
    tr = simulate(car, x0, lambda t: 0.0, flat, 5.0)
    e = np.array([car.total_energy(x) for x in tr])
    assert (e.max() - e.min()) / e[0] < 1e-6


def test_rk4_matches_matrix_exponential():
    A, _, _ = CAR.matrices()
    x0 = np.array([0.02, 0.0, 0.005, 0.0])
    t_end = 1.0
    w, V = np.linalg.eig(A)
    exact = (V @ np.diag(np.exp(w * t_end)) @ np.linalg.solve(V, x0)).real
    num = simulate(CAR, x0, lambda t: 0.0, flat, t_end)[-1]
    assert np.max(np.abs(num - exact)) < 1e-9


def test_rk4_fourth_order_convergence():
    x0 = (0.02, 0.0, 0.005, 0.0)
    ref = simulate(CAR, x0, lambda t: 0.0, flat, 0.5, h=1e-4)[-1]
    e1 = np.max(np.abs(simulate(CAR, x0, lambda t: 0.0, flat, 0.5, h=0.004)[-1] - ref))
    e2 = np.max(np.abs(simulate(CAR, x0, lambda t: 0.0, flat, 0.5, h=0.002)[-1] - ref))
    assert e1 / e2 > 10.0  # ~16 expected for 4th order


def test_passive_system_is_stable():
    A, _, _ = CAR.matrices()
    assert np.all(np.linalg.eigvals(A).real < 0)


def test_default_parameters_physical_ranges():
    assert 0.8 < P.body_freq_hz < 2.0
    assert 8.0 < P.wheel_hop_freq_hz < 15.0
    assert 0.1 < P.body_damping_ratio < 0.5


# 3. bump response ----------------------------------------------------------------------
def test_bump_response_plausible():
    speed = 10.0
    road = BumpRoad(height=0.04, length=2.5, start=2.0)
    h = 0.001
    n = int(6.0 / h)
    zr, zrd = road.time_series(speed, h / 2, 2 * n + 3)
    x = (0.0, 0.0, 0.0, 0.0)
    zs_hist, acc_hist, zu_hist = [], [], []
    for k in range(n):
        r = lambda i: (zr[i], zrd[i])  # noqa: E731
        acc_hist.append(CAR.sprung_acceleration(x, zr[2 * k], zrd[2 * k], 0.0))
        x = CAR.rk4_step(x, 0.0, h, r(2 * k), r(2 * k + 1), r(2 * k + 2))
        zs_hist.append(x[0]); zu_hist.append(x[2])
    zs_hist, zu_hist, acc_hist = map(np.array, (zs_hist, zu_hist, acc_hist))
    assert zs_hist.max() > 0.005                      # body is lifted
    assert zs_hist.max() < 0.04 * 1.5                 # but not amplified absurdly
    assert np.max(np.abs(acc_hist)) < 20.0            # bounded acceleration
    assert abs(zs_hist[-1]) < 1e-3 and abs(zu_hist[-1]) < 1e-4   # returns to equilibrium


# road profiles -------------------------------------------------------------------------
def test_bump_profile_geometry_and_smoothness():
    r = BumpRoad(height=0.05, length=3.0, start=4.0)
    assert r.height(np.array([0.0, 3.99, 7.01, 100.0])).max() == 0.0
    assert r.height(np.array([5.5]))[0] == pytest.approx(0.05)       # peak at mid-length
    x = np.linspace(3.9, 7.1, 20001)
    z = r.height(x)
    assert np.max(np.abs(np.diff(z))) < 1e-3                         # no steps
    assert np.allclose(np.gradient(z, x), r.slope(x), atol=2e-4)     # analytic slope consistent
    assert r.slope(np.array([4.0]))[0] == pytest.approx(0.0, abs=1e-12)


def test_flat_road_is_flat():
    z, zd = FlatRoad().time_series(10.0, 0.001, 100)
    assert np.all(z == 0) and np.all(zd == 0)


def test_random_road_reproducible_and_seed_dependent():
    a = RandomRoad("C", seed=5).height(np.linspace(0, 100, 500))
    b = RandomRoad("C", seed=5).height(np.linspace(0, 100, 500))
    c = RandomRoad("C", seed=6).height(np.linspace(0, 100, 500))
    assert np.array_equal(a, b) and not np.allclose(a, c)
    assert RandomRoad("C", seed=5).height(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-12)


def test_random_road_slope_consistent_and_roughness_ordering():
    r = RandomRoad("C", seed=1)
    x = np.linspace(0, 50, 200001)
    assert np.allclose(np.gradient(r.height(x), x), r.slope(x), atol=1e-4)
    xs = np.linspace(0, 2000, 40001)
    std = {c: np.std(RandomRoad(c, seed=3).height(xs)) for c in "ABCDE"}
    assert std["A"] < std["B"] < std["C"] < std["D"] < std["E"]


def test_time_conversion_uses_speed():
    r = BumpRoad(height=0.05, length=2.0, start=0.0)
    # at 10 m/s the bump peak (x=1 m) occurs at t = 0.1 s; at 20 m/s at t = 0.05 s
    for v, tpk in ((10.0, 0.1), (20.0, 0.05)):
        z, _ = r.time_series(v, 0.001, 300)
        assert np.argmax(z) * 0.001 == pytest.approx(tpk, abs=0.002)


def test_timing_params_validation():
    assert TimingParams().substeps == 10 and TimingParams().max_steps == 1000
    with pytest.raises(ValueError):
        TimingParams(dt_sim=0.003, dt_ctrl=0.01)       # does not divide exactly
    with pytest.raises(ValueError):
        VehicleParams(ms=-1.0)


# 9. numerical stability ----------------------------------------------------------------
def test_long_rough_road_simulation_finite_and_bounded():
    road = RandomRoad("E", seed=11)
    h = 0.001
    n = 60000                                   # 60 s
    zr, zrd = road.time_series(25.0, h / 2, 2 * n + 3)
    x = (0.0, 0.0, 0.0, 0.0)
    worst_disp = worst_vel = 0.0
    for k in range(n):
        u = 1500.0 if (k // 50) % 2 else -1500.0   # worst-case square-wave force
        x = CAR.rk4_step(x, u, h, (zr[2 * k], zrd[2 * k]), (zr[2 * k + 1], zrd[2 * k + 1]),
                         (zr[2 * k + 2], zrd[2 * k + 2]))
        worst_disp = max(worst_disp, abs(x[0]), abs(x[2]))
        worst_vel = max(worst_vel, abs(x[1]), abs(x[3]))
    # class-E road at 25 m/s with worst-case bang-bang force: bounded, physically sane magnitudes
    assert np.all(np.isfinite(x)) and worst_disp < 0.5 and worst_vel < 20.0


def test_default_config_is_valid():
    cfg = EnvConfig()
    assert cfg.timing.dt_ctrl / cfg.timing.dt_sim == pytest.approx(10.0)
