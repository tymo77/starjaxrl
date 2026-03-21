"""Tests for logging utilities and greedy eval rollout."""

import jax
import pytest

from starjaxrl.env.starship_env import StarshipEnv, env_params_from_cfg, get_obs, reset
from starjaxrl.training.logging import init_logging, log_metrics, run_eval_episode
from starjaxrl.training.runner import init_runner
from starjaxrl.agents import TrainMetrics

KEY = jax.random.PRNGKey(55)


@pytest.fixture(scope="module")
def runner_and_graphdef(train_cfg):
    env_params = env_params_from_cfg(train_cfg.env)
    runner_state, graphdef, _ = init_runner(
        train_cfg, KEY, env_params, reset, get_obs,
        obs_dim=StarshipEnv.OBS_DIM, action_dim=StarshipEnv.ACTION_DIM,
    )
    return runner_state, graphdef


# ---------------------------------------------------------------------------
# init_logging
# ---------------------------------------------------------------------------

def test_init_logging_disabled(train_cfg):
    """Disabled mode returns False without error."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.merge(train_cfg, {"wandb": {"mode": "disabled"}})
    active = init_logging(cfg)
    assert active is False


# ---------------------------------------------------------------------------
# log_metrics
# ---------------------------------------------------------------------------

def test_log_metrics_noop_when_inactive():
    """log_metrics must not raise when wandb_active=False."""
    import jax.numpy as jnp
    metrics = TrainMetrics(
        total_loss=jnp.array(1.0),
        pg_loss=jnp.array(0.5),
        vf_loss=jnp.array(2.0),
        entropy=jnp.array(1.0),
        mean_reward=jnp.array(-1.0),
    )
    log_metrics(metrics, step=1, wandb_active=False)  # should not raise


# ---------------------------------------------------------------------------
# run_eval_episode
# ---------------------------------------------------------------------------

def test_eval_episode_returns_trajectory(runner_and_graphdef, train_cfg):
    runner_state, graphdef = runner_and_graphdef
    env_params = env_params_from_cfg(train_cfg.env)

    states, actions, success, total_reward = run_eval_episode(
        runner_state.agent_state, graphdef, env_params, KEY
    )

    assert len(states) >= 2        # at least one step taken
    assert len(actions) >= 1
    assert isinstance(success, bool)
    assert isinstance(total_reward, float)


def test_eval_episode_states_finite(runner_and_graphdef, train_cfg):
    runner_state, graphdef = runner_and_graphdef
    env_params = env_params_from_cfg(train_cfg.env)

    states, _actions, _success, _reward = run_eval_episode(
        runner_state.agent_state, graphdef, env_params, KEY
    )

    import jax.numpy as jnp
    for state in states:
        assert jnp.isfinite(state.x)
        assert jnp.isfinite(state.y)
        assert jnp.isfinite(state.vy)


def test_eval_episode_terminates(runner_and_graphdef, train_cfg):
    """Episode must end before max_steps with a random initial policy."""
    runner_state, graphdef = runner_and_graphdef
    env_params = env_params_from_cfg(train_cfg.env)

    states, _actions, _success, _reward = run_eval_episode(
        runner_state.agent_state, graphdef, env_params, KEY,
        max_steps=5000,
    )

    # Starship starts at 3000 m and falls — should terminate within budget
    assert len(states) < 5000
