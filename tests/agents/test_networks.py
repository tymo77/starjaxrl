"""Tests for Actor, Critic, PPOAgent, and probability utilities."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from starjaxrl.agents import (
    Actor,
    Critic,
    PPOAgent,
    agent_from_cfg,
    gaussian_entropy,
    gaussian_log_prob,
)
from starjaxrl.env.starship_env import StarshipEnv

OBS_DIM    = StarshipEnv.OBS_DIM     # 7
ACTION_DIM = StarshipEnv.ACTION_DIM  # 2
HIDDEN_DIM = 64
N_HIDDEN   = 2
KEY        = jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rngs():
    return nnx.Rngs(params=KEY)


@pytest.fixture(scope="module")
def actor(rngs):
    return Actor(OBS_DIM, ACTION_DIM, HIDDEN_DIM, N_HIDDEN, rngs)


@pytest.fixture(scope="module")
def critic(rngs):
    return Critic(OBS_DIM, HIDDEN_DIM, N_HIDDEN, rngs)


@pytest.fixture(scope="module")
def agent():
    rngs = nnx.Rngs(params=KEY)
    return PPOAgent(OBS_DIM, ACTION_DIM, HIDDEN_DIM, N_HIDDEN, rngs)


@pytest.fixture
def obs():
    return jnp.ones(OBS_DIM)


@pytest.fixture
def batch_obs():
    return jnp.ones((16, OBS_DIM))


# ---------------------------------------------------------------------------
# gaussian_log_prob
# ---------------------------------------------------------------------------

def test_log_prob_shape_scalar():
    mu      = jnp.zeros(ACTION_DIM)
    log_std = jnp.zeros(ACTION_DIM)
    action  = jnp.zeros(ACTION_DIM)
    lp = gaussian_log_prob(action, mu, log_std)
    assert lp.shape == ()


def test_log_prob_shape_batched():
    N       = 16
    mu      = jnp.zeros((N, ACTION_DIM))
    log_std = jnp.zeros((N, ACTION_DIM))
    action  = jnp.zeros((N, ACTION_DIM))
    lp = gaussian_log_prob(action, mu, log_std)
    assert lp.shape == (N,)


def test_log_prob_finite():
    mu      = jnp.array([0.5, -0.3])
    log_std = jnp.zeros(ACTION_DIM)
    action  = jnp.array([0.4, 0.1])
    assert jnp.isfinite(gaussian_log_prob(action, mu, log_std))


def test_log_prob_maximised_at_mean():
    """log_prob is highest when action == mean."""
    mu      = jnp.array([0.5, -0.3])
    log_std = jnp.zeros(ACTION_DIM)
    lp_at_mean  = gaussian_log_prob(mu, mu, log_std)
    lp_off_mean = gaussian_log_prob(mu + 1.0, mu, log_std)
    assert lp_at_mean > lp_off_mean


def test_log_prob_negative():
    """Log probability of a continuous distribution can be negative but is finite."""
    mu      = jnp.zeros(ACTION_DIM)
    log_std = jnp.zeros(ACTION_DIM)
    lp = gaussian_log_prob(jnp.zeros(ACTION_DIM), mu, log_std)
    assert jnp.isfinite(lp)


# ---------------------------------------------------------------------------
# gaussian_entropy
# ---------------------------------------------------------------------------

def test_entropy_shape_scalar():
    log_std = jnp.zeros(ACTION_DIM)
    assert gaussian_entropy(log_std).shape == ()


def test_entropy_positive():
    """Entropy of a Gaussian with std=1 is positive."""
    log_std = jnp.zeros(ACTION_DIM)  # std = 1
    assert float(gaussian_entropy(log_std)) > 0.0


def test_entropy_increases_with_std():
    """Wider distribution → higher entropy."""
    h_narrow = float(gaussian_entropy(jnp.full(ACTION_DIM, -1.0)))
    h_wide   = float(gaussian_entropy(jnp.full(ACTION_DIM,  1.0)))
    assert h_wide > h_narrow


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------

def test_actor_output_shapes(actor, obs):
    mu, log_std = actor(obs)
    assert mu.shape      == (ACTION_DIM,)
    assert log_std.shape == (ACTION_DIM,)


def test_actor_outputs_finite(actor, obs):
    mu, log_std = actor(obs)
    assert jnp.all(jnp.isfinite(mu))
    assert jnp.all(jnp.isfinite(log_std))


def test_actor_log_std_clipped(actor, obs):
    """log_std must stay within [-1, 2] regardless of learned values."""
    _, log_std = actor(obs)
    assert jnp.all(log_std >= -1.0)
    assert jnp.all(log_std <= 2.0)


def test_actor_batched(actor, batch_obs):
    """vmap actor over a batch of observations."""
    batched = jax.vmap(actor)
    mu, log_std = batched(batch_obs)
    assert mu.shape      == (16, ACTION_DIM)
    assert log_std.shape == (16, ACTION_DIM)  # vmap replicates all outputs


def test_actor_jittable(actor, obs):
    mu, log_std = jax.jit(actor)(obs)
    assert jnp.all(jnp.isfinite(mu))


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

def test_critic_output_shape(critic, obs):
    v = critic(obs)
    assert v.shape == ()


def test_critic_output_finite(critic, obs):
    assert jnp.isfinite(critic(obs))


def test_critic_batched(critic, batch_obs):
    batched = jax.vmap(critic)
    v = batched(batch_obs)
    assert v.shape == (16,)
    assert jnp.all(jnp.isfinite(v))


def test_critic_jittable(critic, obs):
    v = jax.jit(critic)(obs)
    assert jnp.isfinite(v)


# ---------------------------------------------------------------------------
# PPOAgent — get_action_and_value
# ---------------------------------------------------------------------------

def test_get_action_and_value_shapes(agent, obs):
    action, log_prob, value, entropy = agent.get_action_and_value(obs, KEY)
    assert action.shape   == (ACTION_DIM,)
    assert log_prob.shape == ()
    assert value.shape    == ()
    assert entropy.shape  == ()


def test_get_action_and_value_finite(agent, obs):
    action, log_prob, value, entropy = agent.get_action_and_value(obs, KEY)
    assert jnp.all(jnp.isfinite(action))
    assert jnp.isfinite(log_prob)
    assert jnp.isfinite(value)
    assert jnp.isfinite(entropy)


def test_get_action_entropy_positive(agent, obs):
    _, _, _, entropy = agent.get_action_and_value(obs, KEY)
    assert float(entropy) > 0.0


def test_get_action_different_keys_differ(agent, obs):
    """Different PRNG keys should (almost certainly) produce different actions."""
    key1, key2 = jax.random.split(KEY)
    a1, _, _, _ = agent.get_action_and_value(obs, key1)
    a2, _, _, _ = agent.get_action_and_value(obs, key2)
    assert not jnp.allclose(a1, a2)


# ---------------------------------------------------------------------------
# PPOAgent — evaluate_actions
# ---------------------------------------------------------------------------

def test_evaluate_actions_shapes(agent, obs):
    action = jnp.zeros(ACTION_DIM)
    log_prob, entropy, value = agent.evaluate_actions(obs, action)
    assert log_prob.shape == ()
    assert entropy.shape  == ()
    assert value.shape    == ()


def test_evaluate_actions_finite(agent, obs):
    action = jnp.zeros(ACTION_DIM)
    log_prob, entropy, value = agent.evaluate_actions(obs, action)
    assert jnp.isfinite(log_prob)
    assert jnp.isfinite(entropy)
    assert jnp.isfinite(value)


def test_evaluate_consistent_with_sample(agent, obs):
    """log_prob from evaluate_actions matches log_prob from get_action_and_value."""
    action, lp_sample, _, _ = agent.get_action_and_value(obs, KEY)
    lp_eval, _, _           = agent.evaluate_actions(obs, action)
    assert lp_sample == pytest.approx(float(lp_eval), rel=1e-5)


def test_evaluate_batched(agent, batch_obs):
    """vmap evaluate_actions over a batch."""
    actions = jnp.zeros((16, ACTION_DIM))
    batched = jax.vmap(lambda o, a: agent.evaluate_actions(o, a))
    log_probs, entropies, values = batched(batch_obs, actions)
    assert log_probs.shape == (16,)
    assert values.shape    == (16,)
    assert jnp.all(jnp.isfinite(log_probs))


# ---------------------------------------------------------------------------
# agent_from_cfg
# ---------------------------------------------------------------------------

def test_agent_from_cfg(train_cfg):
    agent = agent_from_cfg(train_cfg, KEY, obs_dim=OBS_DIM, action_dim=ACTION_DIM)
    obs = jnp.ones(OBS_DIM)
    action, log_prob, value, entropy = agent.get_action_and_value(obs, KEY)
    assert action.shape == (ACTION_DIM,)
    assert jnp.isfinite(log_prob)
