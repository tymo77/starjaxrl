"""Thin Gymnasium wrapper around StarshipEnv for evaluation and rendering."""

import numpy as np
import jax
import jax.numpy as jnp
import gymnasium as gym
from gymnasium import spaces

from starjaxrl.env.starship_env import (
    EnvParams,
    DEFAULT_ENV_PARAMS,
    reset,
    step,
    get_obs,
    StarshipEnv,
)


class StarshipGymEnv(gym.Env):
    """Gymnasium-compatible wrapper for use in evaluation and rendering.

    Not used during JAX training (which uses the raw functional API).
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, params: EnvParams = DEFAULT_ENV_PARAMS, seed: int = 0):
        super().__init__()
        self.params = params
        self._key = jax.random.PRNGKey(seed)
        self._state = None

        obs_high = np.full(StarshipEnv.OBS_DIM, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        act_high = np.array([1.0, params.delta_max], dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([0.0, -params.delta_max], dtype=np.float32),
            high=act_high,
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._key = jax.random.PRNGKey(seed)
        self._key, subkey = jax.random.split(self._key)
        self._state = reset(subkey, self.params)
        obs = np.array(get_obs(self._state), dtype=np.float32)
        return obs, {}

    def step(self, action):
        action = jnp.array(action, dtype=jnp.float32)
        self._state, obs, reward, done, info = step(self._state, action, self.params)
        obs    = np.array(obs, dtype=np.float32)
        reward = float(reward)
        done   = bool(done)
        return obs, reward, done, False, {"success": bool(info.success)}
