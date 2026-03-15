"""End-to-end integration test: training loop runs and reward improves.

Uses a tiny configuration (few envs, short rollouts, few updates) so the
test completes quickly while still exercising the full training pipeline.
"""

import jax
import jax.numpy as jnp
import pytest
from omegaconf import OmegaConf

from starjaxrl.training.runner import init_runner, make_train_step


# ---------------------------------------------------------------------------
# Mini config — fast enough for a test but exercises the full pipeline
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mini_cfg():
    """Full merged config with reduced scale for fast integration tests."""
    env     = OmegaConf.load("configs/env.yaml")
    ppo     = OmegaConf.load("configs/ppo.yaml")
    network = OmegaConf.load("configs/network.yaml")
    base    = OmegaConf.load("configs/train.yaml")
    cfg = OmegaConf.merge(
        OmegaConf.create({"env": env, "ppo": ppo, "network": network}),
        {k: v for k, v in base.items() if k not in ("defaults",)},
    )
    return OmegaConf.merge(cfg, {
        "ppo": {
            "n_envs":         4,
            "rollout_len":    32,
            "n_epochs":       2,
            "minibatch_size": 16,
        },
        "n_updates": 30,
        "wandb": {"mode": "disabled"},
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_n_updates(cfg, n: int):
    """Run n training updates and return list of TrainMetrics."""
    key          = jax.random.PRNGKey(0)
    runner_state, graphdef, optimizer = init_runner(cfg, key)
    train_step   = jax.jit(make_train_step(graphdef, optimizer,
                                           __import__("starjaxrl.env.starship_env",
                                                      fromlist=["env_params_from_cfg"])
                                           .env_params_from_cfg(cfg.env),
                                           cfg))
    metrics_log = []
    for _ in range(n):
        runner_state, metrics = train_step(runner_state, None)
        metrics_log.append(metrics)
    return metrics_log


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_training_loop_completes(mini_cfg):
    """Full training loop runs 30 updates without crashing."""
    from starjaxrl.env.starship_env import env_params_from_cfg
    key = jax.random.PRNGKey(1)
    runner_state, graphdef, optimizer = init_runner(mini_cfg, key)
    env_params = env_params_from_cfg(mini_cfg.env)
    train_step = jax.jit(make_train_step(graphdef, optimizer, env_params, mini_cfg))

    for _ in range(30):
        runner_state, metrics = train_step(runner_state, None)

    # final step completed — runner_state is valid
    assert runner_state.step is not None


def test_metrics_finite(mini_cfg):
    """All losses and rewards remain finite throughout training."""
    from starjaxrl.env.starship_env import env_params_from_cfg
    key = jax.random.PRNGKey(2)
    runner_state, graphdef, optimizer = init_runner(mini_cfg, key)
    env_params = env_params_from_cfg(mini_cfg.env)
    train_step = jax.jit(make_train_step(graphdef, optimizer, env_params, mini_cfg))

    for _ in range(30):
        runner_state, metrics = train_step(runner_state, None)
        assert jnp.isfinite(metrics.total_loss),  f"total_loss not finite: {metrics.total_loss}"
        assert jnp.isfinite(metrics.pg_loss),     f"pg_loss not finite: {metrics.pg_loss}"
        assert jnp.isfinite(metrics.vf_loss),     f"vf_loss not finite: {metrics.vf_loss}"
        assert jnp.isfinite(metrics.entropy),     f"entropy not finite: {metrics.entropy}"
        assert jnp.isfinite(metrics.mean_reward), f"mean_reward not finite: {metrics.mean_reward}"


def test_reward_does_not_collapse(mini_cfg):
    """Mean reward in the last 10 updates is no worse than in the first 10."""
    from starjaxrl.env.starship_env import env_params_from_cfg
    key = jax.random.PRNGKey(3)
    runner_state, graphdef, optimizer = init_runner(mini_cfg, key)
    env_params = env_params_from_cfg(mini_cfg.env)
    train_step = jax.jit(make_train_step(graphdef, optimizer, env_params, mini_cfg))

    rewards = []
    for _ in range(30):
        runner_state, metrics = train_step(runner_state, None)
        rewards.append(float(metrics.mean_reward))

    first_half_mean = sum(rewards[:10]) / 10
    last_half_mean  = sum(rewards[20:]) / 10
    # Reward should not significantly collapse; allow 20 % slack
    assert last_half_mean >= first_half_mean - abs(first_half_mean) * 0.20, (
        f"Reward collapsed: first={first_half_mean:.4f} last={last_half_mean:.4f}"
    )


def test_entropy_decreases(mini_cfg):
    """Policy entropy should trend downward as the agent commits to actions."""
    from starjaxrl.env.starship_env import env_params_from_cfg
    key = jax.random.PRNGKey(4)
    runner_state, graphdef, optimizer = init_runner(mini_cfg, key)
    env_params = env_params_from_cfg(mini_cfg.env)
    train_step = jax.jit(make_train_step(graphdef, optimizer, env_params, mini_cfg))

    entropies = []
    for _ in range(30):
        runner_state, metrics = train_step(runner_state, None)
        entropies.append(float(metrics.entropy))

    first_mean = sum(entropies[:10]) / 10
    last_mean  = sum(entropies[20:]) / 10
    # Entropy should not increase substantially (allow 10 % slack)
    assert last_mean <= first_mean * 1.10, (
        f"Entropy increased: first={first_mean:.4f} last={last_mean:.4f}"
    )


def test_grad_step_updates_params(mini_cfg):
    """Verifies that a gradient step actually changes the network weights."""
    from starjaxrl.env.starship_env import env_params_from_cfg
    import jax
    key = jax.random.PRNGKey(5)
    runner_state, graphdef, optimizer = init_runner(mini_cfg, key)
    env_params = env_params_from_cfg(mini_cfg.env)
    train_step = jax.jit(make_train_step(graphdef, optimizer, env_params, mini_cfg))

    params_before = jax.tree.map(lambda x: x.copy(), runner_state.agent_state)
    runner_state, _ = train_step(runner_state, None)
    params_after = runner_state.agent_state

    leaves_before = jax.tree.leaves(params_before)
    leaves_after  = jax.tree.leaves(params_after)
    changed = any(
        not jnp.array_equal(b, a)
        for b, a in zip(leaves_before, leaves_after)
    )
    assert changed, "Network weights did not change after a gradient step"
