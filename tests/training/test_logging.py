"""Tests for logging utilities and greedy eval rollout."""

import jax
import pytest

from starjaxrl.env.starship_env import StarshipEnv, env_params_from_cfg, get_obs, reset
from starjaxrl.training.logging import init_logging, log_metrics, log_trajectory_artifact, run_eval_episode
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


# ---------------------------------------------------------------------------
# log_trajectory_artifact
# ---------------------------------------------------------------------------

def test_log_trajectory_artifact_noop_when_inactive(runner_and_graphdef, train_cfg, tmp_path):
    """log_trajectory_artifact must not raise when wandb_active=False."""
    runner_state, graphdef = runner_and_graphdef
    env_params = env_params_from_cfg(train_cfg.env)

    states, actions, _, _ = run_eval_episode(
        runner_state.agent_state, graphdef, env_params, KEY
    )
    # Should be a no-op — no GIF written, no error raised
    log_trajectory_artifact(
        states, actions, env_params, step=1,
        wandb_active=False, render_dir=str(tmp_path),
    )
    assert list(tmp_path.iterdir()) == []


def test_log_trajectory_artifact_writes_gif(runner_and_graphdef, train_cfg, tmp_path):
    """When wandb is inactive the function is a no-op, but we can verify the
    render path by monkey-patching wandb so the GIF-writing code is exercised."""
    import types
    import sys

    runner_state, graphdef = runner_and_graphdef
    env_params = env_params_from_cfg(train_cfg.env)

    states, actions, _, _ = run_eval_episode(
        runner_state.agent_state, graphdef, env_params, KEY
    )

    # Build a minimal wandb stub so we can exercise the render path without a
    # real wandb run.
    logged_artifacts: list = []
    logged_media: list = []

    class _FakeArtifact:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.files: list = []
        def add_file(self, path):
            self.files.append(path)

    class _FakeVideo:
        def __init__(self, path, **kwargs):
            self.path = path

    fake_wandb = types.SimpleNamespace(
        Artifact=lambda **kw: _FakeArtifact(**kw),
        Video=_FakeVideo,
        log_artifact=lambda art: logged_artifacts.append(art),
        log=lambda payload, step=None: logged_media.append(payload),
    )

    original = sys.modules.get("wandb")
    sys.modules["wandb"] = fake_wandb
    try:
        log_trajectory_artifact(
            states, actions, env_params, step=42,
            wandb_active=True, render_dir=str(tmp_path),
        )
    finally:
        if original is None:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = original

    # A GIF file should have been written to the render directory
    gif_files = list(tmp_path.glob("*.gif"))
    assert len(gif_files) == 1, f"Expected one GIF, found: {gif_files}"
    assert gif_files[0].name == "trajectory_0042.gif"

    # Artifact and media should have been logged via the stub
    assert len(logged_artifacts) == 1
    assert len(logged_media) == 1
