"""Quarter-car active-suspension RL environment (Stage 1)."""
import gymnasium as _gym

from .config import (ActuatorParams, EnvConfig, ObservationScales, RewardParams,
                     SafetyParams, TimingParams, VehicleParams)
from .quarter_car import QuarterCar
from .road_profiles import (BumpRoad, FlatRoad, RandomRoad, RoadProfile, TEST_ROADS,
                            TRAIN_ROADS, sample_road)
from .suspension_env import OBS_NAMES, ActiveSuspensionEnv

ENV_ID = "ActiveSuspension-v0"
if ENV_ID not in _gym.registry:
    # No max_episode_steps here: the environment truncates itself (TimeLimit would duplicate it).
    _gym.register(id=ENV_ID, entry_point="environment.suspension_env:ActiveSuspensionEnv")

__all__ = [
    "ActiveSuspensionEnv", "EnvConfig", "VehicleParams", "ActuatorParams", "TimingParams",
    "RewardParams", "SafetyParams", "ObservationScales", "QuarterCar", "RoadProfile",
    "FlatRoad", "BumpRoad", "RandomRoad", "sample_road", "TRAIN_ROADS", "TEST_ROADS",
    "OBS_NAMES", "ENV_ID",
]
