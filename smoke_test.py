"""Quick integration smoke test of the Stage 1 environment as Stage 2 uses it (no training).

  python smoke_test.py
"""
import numpy as np
import gymnasium as gym

import environment  # noqa: F401  registers ActiveSuspension-v0
from environment import ActiveSuspensionEnv, EnvConfig, OBS_NAMES

env = ActiveSuspensionEnv(EnvConfig())
print("class:", type(env).__module__ + "." + type(env).__name__)
print("obs space:", env.observation_space, "| names:", OBS_NAMES)
print("act space:", env.action_space, "| u_max [N]:", env.config.actuator.u_max)
obs, info = env.reset(seed=0, options={"road": "rough_C", "speed": 15.0, "road_seed": 123})
assert obs.shape == (6,) and obs.dtype == np.float32
ret, n, done = 0.0, 0, False
while not done:
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    ret += r; n += 1; done = term or trunc
    assert np.all(np.isfinite(obs)) and np.isfinite(r)
print(f"random-policy episode: {n} steps, return {ret:.1f}, terminated={term}, truncated={trunc}, reason={info['termination_reason']}")
e2 = gym.make("ActiveSuspension-v0"); e2.reset(seed=1)
print("gym.make OK; reward weights:", env.config.reward)
print("SMOKE TEST PASSED")
