"""Gymnasium-API, reward, reproducibility and passive-baseline tests."""
import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import environment  # noqa: F401  (registers the id)
from environment import (ActiveSuspensionEnv, ActuatorParams, BumpRoad, EnvConfig, OBS_NAMES,
                         RewardParams, SafetyParams, TimingParams, VehicleParams)
from environment.evaluation import (PassiveController, SkyhookController, episode_metrics,
                                    evaluate, rollout)


def run_episode(env, policy_fn, seed=0, options=None):
    obs, info = env.reset(seed=seed, options=options)
    obs_l, rew_l, infos = [obs], [], []
    done = False
    while not done:
        obs, r, te, tr, info = env.step(policy_fn(obs))
        obs_l.append(obs); rew_l.append(r); infos.append(info)
        done = te or tr
    return np.array(obs_l), np.array(rew_l), infos, te, tr


zero = lambda obs: np.zeros(1, dtype=np.float32)  # noqa: E731


# 4. spaces / shapes ------------------------------------------------------------------------
def test_spaces_and_shapes():
    env = ActiveSuspensionEnv()
    assert env.action_space.shape == (1,) and env.action_space.dtype == np.float32
    assert env.action_space.low[0] == -1.0 and env.action_space.high[0] == 1.0
    assert env.observation_space.shape == (6,) and len(OBS_NAMES) == 6
    obs, info = env.reset(seed=0)
    assert obs.shape == (6,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    obs, r, te, tr, info = env.step(env.action_space.sample())
    assert obs.shape == (6,) and isinstance(r, float) and isinstance(te, bool) and isinstance(tr, bool)


# 7. API compatibility ----------------------------------------------------------------------
@pytest.mark.parametrize("road", ["flat", "bump", "rough_C", "train", "test"])
def test_gymnasium_env_checker(road):
    env = ActiveSuspensionEnv(road=road)
    check_env(env, skip_render_check=True)


def test_gym_make_registered():
    env = gym.make("ActiveSuspension-v0")
    obs, _ = env.reset(seed=1)
    assert obs.shape == (6,)
    assert env.unwrapped.max_steps == 1000


def test_step_before_reset_and_after_done_raises():
    env = ActiveSuspensionEnv()
    with pytest.raises(RuntimeError):
        env.step(np.zeros(1, dtype=np.float32))
    cfg = EnvConfig(timing=TimingParams(episode_duration=0.05))
    env = ActiveSuspensionEnv(cfg)
    env.reset(seed=0)
    for _ in range(5):
        env.step(np.zeros(1, dtype=np.float32))
    with pytest.raises(RuntimeError):
        env.step(np.zeros(1, dtype=np.float32))


def test_truncation_semantics_and_episode_length():
    env = ActiveSuspensionEnv(road="rough_C")
    _, _, infos, te, tr = run_episode(env, zero)
    assert len(infos) == 1000 and tr is True and te is False
    assert infos[-1]["termination_reason"] is None
    assert infos[-1]["time"] == pytest.approx(10.0)


# 5. action bounds / saturation / mapping ----------------------------------------------------
def test_action_mapping_to_newtons_and_saturation():
    cfg = EnvConfig(actuator=ActuatorParams(u_max=1200.0))
    env = ActiveSuspensionEnv(cfg, road="flat")
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([0.5], dtype=np.float32))
    assert info["actuator_force"] == pytest.approx(600.0) and not info["action_saturated"]
    _, _, _, _, info = env.step(np.array([-0.25], dtype=np.float32))
    assert info["actuator_force"] == pytest.approx(-300.0)
    _, _, _, _, info = env.step(np.array([7.0], dtype=np.float32))
    assert info["actuator_force"] == pytest.approx(1200.0) and info["action_saturated"]
    _, _, _, _, info = env.step(np.array([-3.0], dtype=np.float32))
    assert info["actuator_force"] == pytest.approx(-1200.0) and info["action_saturated"]


def test_invalid_actions_rejected():
    env = ActiveSuspensionEnv(road="flat")
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(np.array([np.nan], dtype=np.float32))
    with pytest.raises(ValueError):
        env.step(np.zeros(2, dtype=np.float32))


def test_scalar_and_list_actions_accepted():
    env = ActiveSuspensionEnv(road="flat")
    env.reset(seed=0)
    env.step(0.1)
    env.step([0.1])
    env.step(np.array([0.1]))


def test_actuator_acts_on_dynamics_with_correct_sign():
    env = ActiveSuspensionEnv(road="flat")
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([1.0], dtype=np.float32))
    zs, vs, zu, vu = info["state"]
    assert vs > 0 and vu < 0                      # +u pushes body up and wheel down
    # momentum of the pair is only approximately balanced (tyre force is external): within 10 %
    assert abs(290.0 * vs + 59.0 * vu) < 0.1 * 290.0 * abs(vs)


# 6. reset / reproducibility -----------------------------------------------------------------
def test_seeded_reproducibility_same_seed_same_trajectory():
    pol = lambda obs: np.array([np.tanh(obs[2])], dtype=np.float32)  # noqa: E731
    e1, e2 = ActiveSuspensionEnv(road="train"), ActiveSuspensionEnv(road="train")
    o1, r1, _, _, _ = run_episode(e1, pol, seed=42)
    o2, r2, _, _, _ = run_episode(e2, pol, seed=42)
    assert np.array_equal(o1, o2) and np.array_equal(r1, r2)
    # re-reset of the same env instance with the same seed also reproduces
    o3, r3, _, _, _ = run_episode(e1, pol, seed=42)
    assert np.array_equal(o1, o3) and np.array_equal(r1, r3)


def test_different_seeds_give_different_random_roads():
    env = ActiveSuspensionEnv(road="rough_C")
    _, i1 = env.reset(seed=1)
    _, i2 = env.reset(seed=2)
    assert i1["road_seed"] != i2["road_seed"]


def test_explicit_road_seed_and_speed_options():
    env = ActiveSuspensionEnv(road="rough_C")
    _, info = env.reset(seed=0, options={"road_seed": 777, "speed": 22.0})
    assert info["road_seed"] == 777 and info["speed"] == 22.0
    env.reset(seed=1, options={"road": "bump"})
    assert env.road_sample.label.startswith("bump")
    env.reset(seed=1, options={"road": BumpRoad(height=0.05, length=3.0)})
    assert env.road_sample.profile.h == 0.05


def test_road_independent_of_policy_actions():
    """Identical seed => identical road/speed regardless of what the policy does."""
    env = ActiveSuspensionEnv(road="train")
    env.reset(seed=9)
    z_a = list(env._zr[:200]); label_a = env.road_sample.label
    env.reset(seed=9)
    for _ in range(50):
        env.step(np.array([1.0], dtype=np.float32))
    env.reset(seed=9)
    assert env.road_sample.label == label_a and z_a == env._zr[:200]


def test_reset_state_is_road_following_equilibrium():
    env = ActiveSuspensionEnv(road="rough_C")
    obs, info = env.reset(seed=3)
    assert np.allclose(obs[[0, 1, 2, 3, 5]], 0.0, atol=1e-9)    # at rest relative to the road
    zs, vs, zu, vu = info["state"] if "state" in info else env._x
    assert zs == pytest.approx(zu) and vs == pytest.approx(vu)


def test_initial_state_option():
    env = ActiveSuspensionEnv(road="flat")
    obs, info = env.reset(seed=0, options={"initial_state": [0.01, 0, 0, 0]})
    assert obs[1] == pytest.approx(0.01 / env.config.obs_scales.defl)


# 8. finiteness -------------------------------------------------------------------------------
@pytest.mark.parametrize("road", ["flat", "bump", "rough_E", "train", "test"])
def test_finite_observations_rewards_states(road):
    env = ActiveSuspensionEnv(road=road)
    rng = np.random.default_rng(0)
    obs, rew, infos, te, tr = run_episode(env, lambda o: rng.uniform(-1, 1, 1).astype(np.float32), seed=5)
    assert np.all(np.isfinite(obs)) and np.all(np.isfinite(rew))
    assert all(np.all(np.isfinite(i["state"])) for i in infos)
    assert not te


def test_worst_case_bangbang_force_is_stable():
    env = ActiveSuspensionEnv(road="rough_E")
    sq = {"k": 0}

    def pol(o):
        sq["k"] += 1
        return np.array([1.0 if (sq["k"] // 5) % 2 else -1.0], dtype=np.float32)

    obs, rew, infos, te, tr = run_episode(env, pol, seed=2)
    assert tr and not te and np.all(np.isfinite(rew))


# termination guards ----------------------------------------------------------------------------
def test_hard_limit_termination_and_penalty():
    cfg = EnvConfig(safety=SafetyParams(hard_suspension_limit=0.01))
    env = ActiveSuspensionEnv(cfg, road="bump", speed=10.0)
    obs, rew, infos, te, tr = run_episode(env, zero, seed=0)
    assert te and not tr
    assert infos[-1]["termination_reason"] == "suspension_hard_limit"
    n = len(infos)
    assert rew[-1] < -cfg.reward.termination_cost_per_step * (env.max_steps - n) + 1e-9


def test_default_guards_not_triggered_in_normal_operation():
    for road in ("flat", "bump", "rough_D", "train", "test"):
        env = ActiveSuspensionEnv(road=road)
        _, _, infos, te, _ = run_episode(env, zero, seed=100001)
        assert not te


def test_contact_loss_flag_and_optional_termination():
    # A tall, short bump at speed throws the wheel off the road => load ratio <= 0
    road = BumpRoad(height=0.08, length=1.0, start=3.0)
    env = ActiveSuspensionEnv(road=road, speed=20.0)
    _, _, infos, te, tr = run_episode(env, zero, seed=0)
    assert any(i["contact_loss_in_interval"] for i in infos) and not te
    cfg = EnvConfig(safety=SafetyParams(contact_loss_terminates=True))
    env = ActiveSuspensionEnv(cfg, road=road, speed=20.0)
    _, _, infos, te, tr = run_episode(env, zero, seed=0)
    assert te and infos[-1]["termination_reason"] == "contact_loss"


# reward --------------------------------------------------------------------------------------------
def test_reward_is_zero_at_rest_and_negative_otherwise():
    env = ActiveSuspensionEnv(road="flat")
    env.reset(seed=0)
    _, r, *_ = env.step(np.zeros(1, dtype=np.float32))
    assert r == 0.0
    _, r, *_ = env.step(np.array([1.0], dtype=np.float32))
    assert r < 0.0


def test_reward_terms_sum_to_reward_and_weights_configurable():
    env = ActiveSuspensionEnv(road="rough_C")
    env.reset(seed=0)
    for _ in range(20):
        _, r, _, _, info = env.step(np.array([0.3], dtype=np.float32))
        assert r == pytest.approx(-sum(info["reward_terms"].values()))
    cfg = EnvConfig(reward=RewardParams(w_acc=0.0, w_defl=0.0, w_tyre=0.0, w_effort=1.0, w_rate=0.0))
    env = ActiveSuspensionEnv(cfg, road="flat")
    env.reset(seed=0)
    _, r, *_ = env.step(np.array([0.5], dtype=np.float32))
    assert r == pytest.approx(-0.25)


def test_reward_independent_of_control_step_scale():
    """Mean-over-interval reward: per-step magnitude should not scale with dt_ctrl."""
    def mean_r(dt_ctrl):
        cfg = EnvConfig(timing=TimingParams(dt_sim=0.001, dt_ctrl=dt_ctrl, episode_duration=10.0))
        env = ActiveSuspensionEnv(cfg, road="rough_C", speed=15.0)
        _, r, *_ = run_episode(env, zero, seed=4)
        return np.mean(r)
    assert mean_r(0.01) == pytest.approx(mean_r(0.02), rel=0.1)


# 10. passive controller / identical conditions --------------------------------------------------------
def test_passive_applies_zero_force_everywhere():
    env = ActiveSuspensionEnv(road="rough_C")
    traj = rollout(env, PassiveController(), seed=3)
    assert np.all(traj["actuator_force"] == 0.0) and np.all(traj["actuator_power"] == 0.0)


def test_passive_equals_zero_action_policy():
    env = ActiveSuspensionEnv(road="rough_C")
    a = rollout(env, PassiveController(), seed=7)
    b = rollout(env, lambda o: np.zeros(1, dtype=np.float32), seed=7)
    assert np.array_equal(a["sprung_acceleration"], b["sprung_acceleration"])


def test_identical_conditions_for_different_controllers():
    env = ActiveSuspensionEnv(road="rough_C")
    p = rollout(env, PassiveController(), seed=11)
    label_p = env.road_sample.label
    s = rollout(env, SkyhookController(env.config), seed=11)
    assert env.road_sample.label == label_p
    assert np.array_equal(p["road_height"], s["road_height"])   # same road under both controllers
    assert not np.array_equal(p["sprung_acceleration"], s["sprung_acceleration"])


def test_evaluate_returns_physical_metrics():
    res = evaluate(PassiveController(), road="rough_C", seeds=range(100000, 100003))
    for k in ("rms_acc", "peak_acc", "rms_defl", "max_defl", "rms_dtl", "contact_loss_frac",
              "travel_viol_frac", "rms_force", "mean_power", "return"):
        assert k in res["mean"] and np.isfinite(res["mean"][k])
    assert res["mean"]["rms_force"] == 0.0


def test_skyhook_reference_improves_ride_over_passive():
    """Sanity: the action channel can improve ride comfort and the reward (reference controller only)."""
    cfg = EnvConfig()
    seeds = range(100000, 100008)
    p = evaluate(PassiveController(), cfg, road="rough_C", seeds=seeds)["mean"]
    s = evaluate(SkyhookController(cfg), cfg, road="rough_C", seeds=seeds)["mean"]
    assert s["rms_acc"] < p["rms_acc"] and s["return"] > p["return"]


# info contents -------------------------------------------------------------------------------------------------
def test_info_contains_documented_keys():
    env = ActiveSuspensionEnv(road="rough_C")
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([0.2], dtype=np.float32))
    for k in ("sprung_acceleration", "suspension_deflection", "suspension_velocity", "tyre_deflection",
              "unsprung_velocity", "tyre_force_dynamic", "tyre_load_ratio", "actuator_force",
              "actuator_power", "contact_loss", "contact_loss_in_interval", "reward_terms",
              "termination_reason", "road_height", "time", "state", "action_applied"):
        assert k in info, k
    # observation channels are the normalised info values
    s = env.config.obs_scales
    obs = env._obs({k: info[k] for k in ("sprung_acceleration", "suspension_deflection", "suspension_velocity",
                                          "tyre_deflection", "unsprung_velocity")}, info["action_applied"])
    assert obs[0] == pytest.approx(info["sprung_acceleration"] / s.acc, rel=1e-5)
    assert obs[5] == pytest.approx(0.2, rel=1e-5)


def test_observation_matches_info_after_step():
    env = ActiveSuspensionEnv(road="rough_C")
    env.reset(seed=0)
    obs, _, _, _, info = env.step(np.array([-0.4], dtype=np.float32))
    s = env.config.obs_scales
    expect = np.array([info["sprung_acceleration"] / s.acc, info["suspension_deflection"] / s.defl,
                       info["suspension_velocity"] / s.defl_rate, info["tyre_deflection"] / s.tyre_defl,
                       info["unsprung_velocity"] / s.unsprung_vel, -0.4], dtype=np.float32)
    assert np.allclose(obs, expect, rtol=1e-5, atol=1e-7)


def test_vehicle_params_override_changes_dynamics():
    a = rollout(ActiveSuspensionEnv(EnvConfig(vehicle=VehicleParams(cs=500.0)), road="bump", speed=10.0),
                PassiveController(), seed=0)
    b = rollout(ActiveSuspensionEnv(EnvConfig(vehicle=VehicleParams(cs=3000.0)), road="bump", speed=10.0),
                PassiveController(), seed=0)
    assert not np.allclose(a["sprung_acceleration"], b["sprung_acceleration"])
