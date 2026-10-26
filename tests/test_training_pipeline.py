"""Stage 2 pipeline tests: integration, PPO build/train/save/load, metrics, matched evaluation."""
import numpy as np
import pytest

sb3 = pytest.importorskip("stable_baselines3")
from stable_baselines3 import PPO  # noqa: E402

import environment  # noqa: E402,F401
from environment import ActiveSuspensionEnv, EnvConfig, TimingParams, VehicleParams  # noqa: E402
from environment.evaluation import PassiveController, rollout  # noqa: E402
from training.common import build_env_config, make_vec_env  # noqa: E402
from training.metrics import compute_metrics, improvement_pct  # noqa: E402
import evaluate as ev  # noqa: E402

SHORT = EnvConfig(timing=TimingParams(episode_duration=2.0))


def test_env_import_and_reset_step():
    env = ActiveSuspensionEnv(SHORT)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (6,) and env.action_space.shape == (1,)
    done, n = False, 0
    while not done:
        obs, r, te, tr, info = env.step(env.action_space.sample())
        done, n = te or tr, n + 1
        assert np.isfinite(r) and np.all(np.isfinite(obs))
    assert n == 200 and tr and not te


def test_build_env_config_defaults_and_overrides():
    assert build_env_config(None) == EnvConfig()
    c = build_env_config({"vehicle": {"cs": 1500.0}, "reward": {"w_acc": 2.0}})
    assert c.vehicle.cs == 1500.0 and c.reward.w_acc == 2.0 and c.vehicle.ms == 290.0
    with pytest.raises(ValueError):
        build_env_config({"bogus": 1})


def test_vec_env_seeding_is_reproducible():
    def first_obs():
        v = make_vec_env(SHORT.with_(road="rough_C"), n_envs=2, seed=3)
        o = v.reset(); return o
    assert np.array_equal(first_obs(), first_obs())


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    venv = make_vec_env(SHORT, n_envs=2, seed=0)
    m = PPO("MlpPolicy", venv, n_steps=64, batch_size=64, n_epochs=2, device="cpu", seed=0, verbose=0)
    m.learn(total_timesteps=512)
    p = tmp_path_factory.mktemp("m") / "tiny"
    m.save(str(p))
    return m, str(p) + ".zip"


def test_ppo_save_load_predict_valid(tiny_model):
    m, path = tiny_model
    m2 = PPO.load(path, device="cpu")
    obs = np.zeros(6, dtype=np.float32)
    a1, _ = m.predict(obs, deterministic=True); a2, _ = m2.predict(obs, deterministic=True)
    assert a1.shape == (1,) and np.all(np.abs(a2) <= 1.0 + 1e-6) and np.array_equal(a1, a2)


def test_metrics_known_example():
    n = 4
    traj = {"sprung_acceleration": np.array([2.0, -2.0, 2.0, -2.0]), "suspension_deflection": np.array([0.01, -0.03, 0.01, 0.01]),
            "tyre_force_dynamic": np.array([100.0, -300.0, 100.0, 100.0]), "actuator_force": np.array([0.0, 1500.0, -1500.0, 750.0]),
            "tyre_deflection": np.array([0.001, -0.002, 0.0, 0.0]), "tyre_load_ratio": np.array([1.0, 0.9, 1.0, 0.95]),
            "contact_loss_in_interval": np.array([0, 0, 0, 1], bool), "travel_limit_exceeded_in_interval": np.zeros(n, bool),
            "actuator_power": np.array([0.0, 10.0, 10.0, 0.0]), "reward": np.array([-1.0, -2.0, -3.0, -4.0]),
            "terminated": np.asarray(False)}
    m = compute_metrics(traj, static_tyre_load=1000.0, u_max=1500.0)
    assert m["rms_acc"] == pytest.approx(2.0) and m["peak_acc"] == pytest.approx(2.0)
    assert m["max_defl"] == pytest.approx(0.03) and m["rms_defl"] == pytest.approx(np.sqrt((1e-4 * 3 + 9e-4) / 4))
    assert m["rms_dtl"] == pytest.approx(np.sqrt((3 * 1e4 + 9e4) / 4) / 1000.0)
    assert m["max_tyre_force"] == pytest.approx(300.0) and m["peak_force"] == pytest.approx(1500.0)
    assert m["sat_frac"] == pytest.approx(0.5) and m["contact_loss_frac"] == pytest.approx(0.25)
    assert m["min_tyre_load_ratio"] == pytest.approx(0.9) and m["return"] == pytest.approx(-10.0)
    assert m["rms_force"] == pytest.approx(np.sqrt((0 + 1500**2 * 2 + 750**2) / 4))


def test_improvement_pct_sign_and_nan():
    assert improvement_pct("rms_acc", 2.0, 1.5) == pytest.approx(25.0)      # lower is better
    assert improvement_pct("return", -100.0, -80.0) == pytest.approx(20.0)  # higher is better, negative base
    assert improvement_pct("rms_acc", 2.0, 3.0) == pytest.approx(-50.0)     # degradation is negative
    assert np.isnan(improvement_pct("rms_acc", 0.0, 0.1))                    # passive ~ 0
    assert np.isnan(improvement_pct("rms_force", 0.0, 100.0))                # effort metric


def test_evaluation_output_and_matched_conditions(tiny_model):
    m, _ = tiny_model
    sc = [{"name": "t_rough", "road": "rough_C", "speed": 15.0, "n_seeds": 2},
          {"name": "t_pert", "road": "flat", "n_seeds": 1, "options": {"initial_state": [0.02, 0, 0, 0]}, "deterministic": True}]
    ctl = {"passive": PassiveController(), "ppo": m}
    ep, ts = ev.evaluate_scenarios(ctl, SHORT, sc, 100000, "tiny", ts_seeds=1)
    assert len(ep) == (2 + 1) * 2 and set(ep.controller) == {"passive", "ppo"}
    assert np.all(np.isfinite(ep["rms_acc"])) and (ep[ep.controller == "passive"].rms_force == 0).all()
    assert {"veh_ms", "veh_cs", "model_id", "seed", "return"} <= set(ep.columns)
    cmp_df = ev.compare(ep)
    assert set(cmp_df.controller) == {"ppo"} and len(ts) > 0
    # reproducible: repeating the evaluation gives identical numbers
    ep2, _ = ev.evaluate_scenarios(ctl, SHORT, sc, 100000, "tiny", ts_seeds=1)
    assert np.array_equal(ep["rms_acc"].to_numpy(), ep2["rms_acc"].to_numpy())


def test_same_conditions_for_passive_and_policy():
    env = ActiveSuspensionEnv(SHORT, road="rough_C")
    a = rollout(env, PassiveController(), seed=100003); la = env.road_sample.label
    b = rollout(env, lambda o: np.array([0.3], np.float32), seed=100003)
    assert env.road_sample.label == la and np.array_equal(a["road_height"], b["road_height"])
