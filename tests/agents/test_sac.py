"""Tests for SAC: replay buffer, networks, and training loop (Starship + CartPole)."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from omegaconf import OmegaConf

from starjaxrl.agents.sac import (
    SACActor,
    SACQNetwork,
    SACTwinnedQ,
    SACTrainMetrics,
    ReplayBuffer,
    make_replay_buffer,
    buffer_add_batch,
    buffer_sample,
)
from starjaxrl.training.sac_runner import (
    SACRunnerState,
    init_sac_runner,
    make_sac_collect_step,
    make_sac_train_step,
)
from starjaxrl.env.starship_env import (
    StarshipEnv,
    env_params_from_cfg,
    get_obs as starship_get_obs,
    reset as starship_reset,
    step as starship_step,
)
from starjaxrl.env.cartpole_env import (
    CartPoleEnv,
    env_params_from_cfg as cartpole_params_from_cfg,
    get_obs as cartpole_get_obs,
    reset as cartpole_reset,
    step as cartpole_step,
)

KEY = jax.random.PRNGKey(42)

# ---------------------------------------------------------------------------
# Shared config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sac_cfg():
    """Merged config with SAC hyperparams (small buffer/batch for fast tests)."""
    env     = OmegaConf.load("configs/env/env.yaml")
    sac     = OmegaConf.load("configs/sac/sac.yaml")
    network = OmegaConf.load("configs/network/network.yaml")
    base    = OmegaConf.load("configs/train.yaml")
    cfg = OmegaConf.merge(
        OmegaConf.create({"env": env, "sac": sac, "network": network}),
        {k: v for k, v in base.items() if k not in ("defaults",)},
    )
    # Shrink sizes for fast test execution
    return OmegaConf.merge(cfg, OmegaConf.create({
        "sac": {
            "n_envs": 2,
            "buffer_size": 512,
            "batch_size": 32,
            "learning_starts": 10,
            "collect_steps": 1,
        }
    }))


@pytest.fixture(scope="module")
def cartpole_sac_cfg():
    """Same as sac_cfg but for the CartPole environment."""
    env     = OmegaConf.load("configs/env/cartpole.yaml")
    sac     = OmegaConf.load("configs/sac/sac.yaml")
    network = OmegaConf.load("configs/network/network.yaml")
    base    = OmegaConf.load("configs/cartpole.yaml")
    cfg = OmegaConf.merge(
        OmegaConf.create({"env": env, "sac": sac, "network": network}),
        {k: v for k, v in base.items() if k not in ("defaults",)},
    )
    return OmegaConf.merge(cfg, OmegaConf.create({
        "sac": {
            "n_envs": 2,
            "buffer_size": 512,
            "batch_size": 32,
            "learning_starts": 10,
            "collect_steps": 1,
        }
    }))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_starship(cfg, key=KEY):
    env_params = env_params_from_cfg(cfg.env)
    return init_sac_runner(
        cfg, key, env_params, starship_reset, starship_get_obs,
        obs_dim=StarshipEnv.OBS_DIM, action_dim=StarshipEnv.ACTION_DIM,
    )


def _init_cartpole(cfg, key=KEY):
    env_params = cartpole_params_from_cfg(cfg.env)
    return init_sac_runner(
        cfg, key, env_params, cartpole_reset, cartpole_get_obs,
        obs_dim=CartPoleEnv.OBS_DIM, action_dim=CartPoleEnv.ACTION_DIM,
    )


def _make_starship_train_step(actor_gd, critic_gd, opt, env_params, cfg):
    return make_sac_train_step(
        actor_gd, critic_gd, opt, env_params, cfg,
        starship_reset, starship_get_obs, starship_step,
        action_dim=StarshipEnv.ACTION_DIM,
    )


def _make_cartpole_train_step(actor_gd, critic_gd, opt, env_params, cfg):
    return make_sac_train_step(
        actor_gd, critic_gd, opt, env_params, cfg,
        cartpole_reset, cartpole_get_obs, cartpole_step,
        action_dim=CartPoleEnv.ACTION_DIM,
    )


# ===========================================================================
# Replay buffer
# ===========================================================================

class TestReplayBuffer:
    OBS_DIM    = 7
    ACTION_DIM = 2
    CAPACITY   = 64

    @pytest.fixture(autouse=True)
    def buf(self):
        self.buf = make_replay_buffer(self.CAPACITY, self.OBS_DIM, self.ACTION_DIM)

    def test_initial_shapes(self):
        assert self.buf.obs.shape      == (self.CAPACITY, self.OBS_DIM)
        assert self.buf.action.shape   == (self.CAPACITY, self.ACTION_DIM)
        assert self.buf.reward.shape   == (self.CAPACITY,)
        assert self.buf.next_obs.shape == (self.CAPACITY, self.OBS_DIM)
        assert self.buf.done.shape     == (self.CAPACITY,)

    def test_initial_ptr_and_size(self):
        assert int(self.buf.ptr)  == 0
        assert int(self.buf.size) == 0

    def test_add_batch_updates_ptr_and_size(self):
        n        = 8
        obs      = jnp.ones((n, self.OBS_DIM))
        action   = jnp.zeros((n, self.ACTION_DIM))
        reward   = jnp.ones(n)
        next_obs = jnp.ones((n, self.OBS_DIM)) * 2.0
        done     = jnp.zeros(n, dtype=bool)
        buf2 = buffer_add_batch(self.buf, obs, action, reward, next_obs, done)
        assert int(buf2.ptr)  == n
        assert int(buf2.size) == n

    def test_add_batch_wraps_around(self):
        n      = self.CAPACITY - 2
        obs    = jnp.ones((n, self.OBS_DIM))
        action = jnp.zeros((n, self.ACTION_DIM))
        reward = jnp.ones(n)
        nobs   = jnp.ones((n, self.OBS_DIM))
        done   = jnp.zeros(n, dtype=bool)
        buf2 = buffer_add_batch(self.buf, obs, action, reward, nobs, done)

        # Now add 4 more — wraps around
        n2     = 4
        obs2   = jnp.ones((n2, self.OBS_DIM)) * 9.0
        act2   = jnp.zeros((n2, self.ACTION_DIM))
        rew2   = jnp.ones(n2) * 9.0
        nobs2  = jnp.ones((n2, self.OBS_DIM)) * 9.0
        done2  = jnp.zeros(n2, dtype=bool)
        buf3 = buffer_add_batch(buf2, obs2, act2, rew2, nobs2, done2)
        assert int(buf3.size) == self.CAPACITY   # capped at capacity
        assert int(buf3.ptr)  == (n + n2) % self.CAPACITY

    def test_add_batch_data_stored_correctly(self):
        n      = 4
        obs    = jax.random.normal(KEY, (n, self.OBS_DIM))
        action = jax.random.normal(KEY, (n, self.ACTION_DIM))
        reward = jax.random.normal(KEY, (n,))
        nobs   = jax.random.normal(KEY, (n, self.OBS_DIM))
        done   = jnp.array([False, True, False, True])
        buf2 = buffer_add_batch(self.buf, obs, action, reward, nobs, done)
        assert jnp.allclose(buf2.obs[:n],      obs)
        assert jnp.allclose(buf2.action[:n],   action)
        assert jnp.allclose(buf2.reward[:n],   reward)
        assert jnp.allclose(buf2.next_obs[:n], nobs)
        assert jnp.all(buf2.done[:n] == done)

    def test_sample_shapes(self):
        # Fill buffer first
        n   = 32
        obs = jax.random.normal(KEY, (n, self.OBS_DIM))
        act = jax.random.normal(KEY, (n, self.ACTION_DIM))
        rew = jax.random.normal(KEY, (n,))
        nob = jax.random.normal(KEY, (n, self.OBS_DIM))
        don = jnp.zeros(n, dtype=bool)
        buf = buffer_add_batch(self.buf, obs, act, rew, nob, don)

        batch = 16
        s_obs, s_act, s_rew, s_nobs, s_done = buffer_sample(buf, batch, KEY)
        assert s_obs.shape  == (batch, self.OBS_DIM)
        assert s_act.shape  == (batch, self.ACTION_DIM)
        assert s_rew.shape  == (batch,)
        assert s_nobs.shape == (batch, self.OBS_DIM)
        assert s_done.shape == (batch,)

    def test_sample_empty_buffer_does_not_crash(self):
        """Sampling from an empty buffer should not crash (returns zeros)."""
        s_obs, _, _, _, _ = buffer_sample(self.buf, 8, KEY)
        assert s_obs.shape == (8, self.OBS_DIM)


# ===========================================================================
# Networks
# ===========================================================================

class TestSACActor:
    OBS_DIM    = 7
    ACTION_DIM = 2

    @pytest.fixture(autouse=True)
    def actor(self):
        self.actor = SACActor(
            obs_dim=self.OBS_DIM, action_dim=self.ACTION_DIM,
            hidden_dim=32, n_hidden=2, rngs=nnx.Rngs(params=KEY),
        )

    def test_call_output_shapes(self):
        obs = jnp.zeros(self.OBS_DIM)
        mu, log_std = self.actor(obs)
        assert mu.shape      == (self.ACTION_DIM,)
        assert log_std.shape == (self.ACTION_DIM,)

    def test_call_output_finite(self):
        obs = jnp.zeros(self.OBS_DIM)
        mu, log_std = self.actor(obs)
        assert jnp.all(jnp.isfinite(mu))
        assert jnp.all(jnp.isfinite(log_std))

    def test_log_std_within_bounds(self):
        obs = jax.random.normal(KEY, (self.OBS_DIM,)) * 10  # extreme input
        _, log_std = self.actor(obs)
        assert jnp.all(log_std >= -5.0)
        assert jnp.all(log_std <=  2.0)

    def test_sample_output_shapes(self):
        obs = jnp.zeros(self.OBS_DIM)
        key = jax.random.PRNGKey(0)
        action, log_prob = self.actor.sample(obs, key)
        assert action.shape   == (self.ACTION_DIM,)
        assert log_prob.shape == ()

    def test_sample_action_in_minus1_plus1(self):
        obs = jax.random.normal(KEY, (self.OBS_DIM,))
        action, _ = self.actor.sample(obs, KEY)
        assert jnp.all(action > -1.0)
        assert jnp.all(action <  1.0)

    def test_sample_log_prob_finite(self):
        obs = jax.random.normal(KEY, (self.OBS_DIM,))
        _, log_prob = self.actor.sample(obs, KEY)
        assert jnp.isfinite(log_prob)

    def test_mean_action_shape_and_bounds(self):
        obs = jnp.zeros(self.OBS_DIM)
        a   = self.actor.mean_action(obs)
        assert a.shape == (self.ACTION_DIM,)
        assert jnp.all(jnp.abs(a) < 1.0)

    def test_vmap_sample(self):
        """Confirm that jax.vmap over (obs, key) works correctly."""
        batch   = 8
        obs_b   = jax.random.normal(KEY, (batch, self.OBS_DIM))
        keys_b  = jax.random.split(KEY, batch)
        actions, log_probs = jax.vmap(lambda o, k: self.actor.sample(o, k))(obs_b, keys_b)
        assert actions.shape    == (batch, self.ACTION_DIM)
        assert log_probs.shape  == (batch,)
        assert jnp.all(jnp.isfinite(actions))
        assert jnp.all(jnp.isfinite(log_probs))


class TestSACQNetwork:
    OBS_DIM    = 7
    ACTION_DIM = 2

    @pytest.fixture(autouse=True)
    def net(self):
        self.q = SACQNetwork(
            obs_dim=self.OBS_DIM, action_dim=self.ACTION_DIM,
            hidden_dim=32, n_hidden=2, rngs=nnx.Rngs(params=KEY),
        )

    def test_output_shape_single(self):
        obs    = jnp.zeros(self.OBS_DIM)
        action = jnp.zeros(self.ACTION_DIM)
        q_val  = self.q(obs, action)
        assert q_val.shape == ()

    def test_output_finite(self):
        obs    = jax.random.normal(KEY, (self.OBS_DIM,))
        action = jax.random.normal(KEY, (self.ACTION_DIM,))
        assert jnp.isfinite(self.q(obs, action))

    def test_vmap_batch(self):
        batch   = 16
        obs_b   = jax.random.normal(KEY, (batch, self.OBS_DIM))
        act_b   = jax.random.normal(KEY, (batch, self.ACTION_DIM))
        q_vals  = jax.vmap(self.q)(obs_b, act_b)
        assert q_vals.shape == (batch,)
        assert jnp.all(jnp.isfinite(q_vals))


class TestSACTwinnedQ:
    OBS_DIM    = 7
    ACTION_DIM = 2

    @pytest.fixture(autouse=True)
    def net(self):
        self.tq = SACTwinnedQ(
            obs_dim=self.OBS_DIM, action_dim=self.ACTION_DIM,
            hidden_dim=32, n_hidden=2, rngs=nnx.Rngs(params=KEY),
        )

    def test_output_shapes(self):
        obs    = jnp.zeros(self.OBS_DIM)
        action = jnp.zeros(self.ACTION_DIM)
        q1, q2 = self.tq(obs, action)
        assert q1.shape == ()
        assert q2.shape == ()

    def test_q1_q2_differ(self):
        """Two networks with different initialisations should give different values."""
        obs    = jax.random.normal(KEY, (self.OBS_DIM,))
        action = jax.random.normal(KEY, (self.ACTION_DIM,))
        q1, q2 = self.tq(obs, action)
        assert not jnp.allclose(q1, q2), "Q1 and Q2 should differ (different init)"

    def test_vmap_batch(self):
        batch  = 8
        obs_b  = jax.random.normal(KEY, (batch, self.OBS_DIM))
        act_b  = jax.random.normal(KEY, (batch, self.ACTION_DIM))
        q1, q2 = jax.vmap(self.tq)(obs_b, act_b)
        assert q1.shape == (batch,)
        assert q2.shape == (batch,)


# ===========================================================================
# Starship runner
# ===========================================================================

class TestInitSACRunnerStarship:
    def test_obs_shape(self, sac_cfg):
        runner_state, _, _, _ = _init_starship(sac_cfg)
        n_envs  = int(sac_cfg.sac.n_envs)
        obs_dim = StarshipEnv.OBS_DIM
        assert runner_state.obs.shape == (n_envs, obs_dim)

    def test_obs_finite(self, sac_cfg):
        runner_state, _, _, _ = _init_starship(sac_cfg)
        assert jnp.all(jnp.isfinite(runner_state.obs))

    def test_actor_state_finite(self, sac_cfg):
        runner_state, _, _, _ = _init_starship(sac_cfg)
        for leaf in jax.tree.leaves(runner_state.actor_state):
            assert jnp.all(jnp.isfinite(leaf)), "Non-finite actor param at init"

    def test_critic_state_finite(self, sac_cfg):
        runner_state, _, _, _ = _init_starship(sac_cfg)
        for leaf in jax.tree.leaves(runner_state.critic_state):
            assert jnp.all(jnp.isfinite(leaf)), "Non-finite critic param at init"

    def test_target_equals_critic_at_init(self, sac_cfg):
        runner_state, _, _, _ = _init_starship(sac_cfg)
        for t, c in zip(
            jax.tree.leaves(runner_state.target_state),
            jax.tree.leaves(runner_state.critic_state),
        ):
            assert jnp.allclose(t, c), "Target and critic should match at init"

    def test_buffer_empty_at_init(self, sac_cfg):
        runner_state, _, _, _ = _init_starship(sac_cfg)
        assert int(runner_state.buffer.size) == 0

    def test_step_zero_at_init(self, sac_cfg):
        runner_state, _, _, _ = _init_starship(sac_cfg)
        assert int(runner_state.step) == 0


# ---------------------------------------------------------------------------
# Starship collect_step
# ---------------------------------------------------------------------------

class TestSACCollectStepStarship:
    @pytest.fixture(scope="class")
    def after_collect(self, sac_cfg):
        env_params = env_params_from_cfg(sac_cfg.env)
        runner_state, actor_gd, _, _ = _init_starship(sac_cfg)
        collect_fn = jax.jit(make_sac_collect_step(
            actor_gd, env_params, sac_cfg,
            starship_reset, starship_get_obs, starship_step,
        ))
        current_g = jnp.array(float(sac_cfg.env.g), dtype=jnp.float32)
        new_runner = collect_fn(runner_state, current_g)
        return runner_state, new_runner

    def test_buffer_grows(self, after_collect, sac_cfg):
        old, new = after_collect
        n_envs = int(sac_cfg.sac.n_envs)
        assert int(new.buffer.size) == int(old.buffer.size) + n_envs

    def test_step_increments(self, after_collect):
        old, new = after_collect
        assert int(new.step) == int(old.step) + 1

    def test_obs_finite(self, after_collect):
        _, new = after_collect
        assert jnp.all(jnp.isfinite(new.obs))

    def test_obs_shape_unchanged(self, after_collect):
        old, new = after_collect
        assert new.obs.shape == old.obs.shape


# ---------------------------------------------------------------------------
# Starship train_step
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def starship_runner_and_step(sac_cfg):
    """Initialise, warm up the buffer, and run one train_step."""
    env_params = env_params_from_cfg(sac_cfg.env)
    runner_state, actor_gd, critic_gd, opt = _init_starship(sac_cfg)
    current_g = jnp.array(float(sac_cfg.env.g), dtype=jnp.float32)

    # Fill buffer past batch_size so gradients are meaningful
    collect_fn = jax.jit(make_sac_collect_step(
        actor_gd, env_params, sac_cfg,
        starship_reset, starship_get_obs, starship_step,
    ))
    for _ in range(int(sac_cfg.sac.learning_starts)):
        runner_state = collect_fn(runner_state, current_g)

    train_fn = jax.jit(_make_starship_train_step(actor_gd, critic_gd, opt, env_params, sac_cfg))
    new_runner, metrics = train_fn(runner_state, current_g)
    return runner_state, new_runner, metrics


def test_starship_train_step_returns_metrics(starship_runner_and_step):
    _, _, metrics = starship_runner_and_step
    assert isinstance(metrics, SACTrainMetrics)


def test_starship_train_step_metrics_finite(starship_runner_and_step):
    _, _, metrics = starship_runner_and_step
    for field in metrics:
        assert jnp.isfinite(field), f"Non-finite metric: {field}"


def test_starship_train_step_alpha_positive(starship_runner_and_step):
    _, _, metrics = starship_runner_and_step
    assert float(metrics.alpha) > 0.0


def test_starship_train_step_actor_params_change(starship_runner_and_step):
    old, new, _ = starship_runner_and_step
    old_leaves = jax.tree.leaves(old.actor_state)
    new_leaves = jax.tree.leaves(new.actor_state)
    changed = any(not jnp.allclose(o, n) for o, n in zip(old_leaves, new_leaves))
    assert changed, "Actor parameters should change after a gradient update"


def test_starship_train_step_critic_params_change(starship_runner_and_step):
    old, new, _ = starship_runner_and_step
    old_leaves = jax.tree.leaves(old.critic_state)
    new_leaves = jax.tree.leaves(new.critic_state)
    changed = any(not jnp.allclose(o, n) for o, n in zip(old_leaves, new_leaves))
    assert changed, "Critic parameters should change after a gradient update"


def test_starship_train_step_target_soft_updated(starship_runner_and_step):
    """Target should be between old target and new critic (EMA update applied)."""
    old, new, _ = starship_runner_and_step
    # Target must differ from old target (was updated)
    old_t = jax.tree.leaves(old.target_state)
    new_t = jax.tree.leaves(new.target_state)
    changed = any(not jnp.allclose(ot, nt) for ot, nt in zip(old_t, new_t))
    assert changed, "Target networks should change via soft update"


def test_starship_train_step_all_params_finite(starship_runner_and_step):
    _, new, _ = starship_runner_and_step
    for leaf in jax.tree.leaves(new.actor_state):
        assert jnp.all(jnp.isfinite(leaf)), "NaN/Inf in actor state after update"
    for leaf in jax.tree.leaves(new.critic_state):
        assert jnp.all(jnp.isfinite(leaf)), "NaN/Inf in critic state after update"


def test_starship_train_step_increments_step(starship_runner_and_step):
    old, new, _ = starship_runner_and_step
    assert int(new.step) == int(old.step) + 1


def test_starship_multiple_train_steps_stable(sac_cfg):
    """Five consecutive train_steps must keep all metrics finite."""
    env_params = env_params_from_cfg(sac_cfg.env)
    runner_state, actor_gd, critic_gd, opt = _init_starship(sac_cfg)
    current_g = jnp.array(float(sac_cfg.env.g), dtype=jnp.float32)

    collect_fn = jax.jit(make_sac_collect_step(
        actor_gd, env_params, sac_cfg,
        starship_reset, starship_get_obs, starship_step,
    ))
    train_fn = jax.jit(_make_starship_train_step(actor_gd, critic_gd, opt, env_params, sac_cfg))

    for _ in range(int(sac_cfg.sac.learning_starts)):
        runner_state = collect_fn(runner_state, current_g)

    for _ in range(5):
        runner_state, metrics = train_fn(runner_state, current_g)
        for field in metrics:
            assert jnp.isfinite(field), "Divergence detected in multi-step SAC test"


# ===========================================================================
# CartPole runner — same structural tests for the second supported problem
# ===========================================================================

@pytest.fixture(scope="module")
def cartpole_runner_and_step(cartpole_sac_cfg):
    """Initialise, warm up the buffer, and run one train_step on CartPole."""
    env_params = cartpole_params_from_cfg(cartpole_sac_cfg.env)
    runner_state, actor_gd, critic_gd, opt = _init_cartpole(cartpole_sac_cfg)
    current_g = jnp.array(float(cartpole_sac_cfg.env.g), dtype=jnp.float32)

    collect_fn = jax.jit(make_sac_collect_step(
        actor_gd, env_params, cartpole_sac_cfg,
        cartpole_reset, cartpole_get_obs, cartpole_step,
    ))
    for _ in range(int(cartpole_sac_cfg.sac.learning_starts)):
        runner_state = collect_fn(runner_state, current_g)

    train_fn = jax.jit(_make_cartpole_train_step(
        actor_gd, critic_gd, opt, env_params, cartpole_sac_cfg
    ))
    new_runner, metrics = train_fn(runner_state, current_g)
    return runner_state, new_runner, metrics


def test_cartpole_init_obs_shape(cartpole_sac_cfg):
    runner_state, _, _, _ = _init_cartpole(cartpole_sac_cfg)
    n_envs  = int(cartpole_sac_cfg.sac.n_envs)
    obs_dim = CartPoleEnv.OBS_DIM
    assert runner_state.obs.shape == (n_envs, obs_dim)


def test_cartpole_train_step_returns_metrics(cartpole_runner_and_step):
    _, _, metrics = cartpole_runner_and_step
    assert isinstance(metrics, SACTrainMetrics)


def test_cartpole_train_step_metrics_finite(cartpole_runner_and_step):
    _, _, metrics = cartpole_runner_and_step
    for field in metrics:
        assert jnp.isfinite(field), f"Non-finite CartPole metric: {field}"


def test_cartpole_train_step_actor_params_change(cartpole_runner_and_step):
    old, new, _ = cartpole_runner_and_step
    old_leaves = jax.tree.leaves(old.actor_state)
    new_leaves = jax.tree.leaves(new.actor_state)
    changed = any(not jnp.allclose(o, n) for o, n in zip(old_leaves, new_leaves))
    assert changed, "CartPole actor parameters should change after gradient update"


def test_cartpole_train_step_all_params_finite(cartpole_runner_and_step):
    _, new, _ = cartpole_runner_and_step
    for leaf in jax.tree.leaves(new.actor_state):
        assert jnp.all(jnp.isfinite(leaf)), "NaN/Inf in CartPole actor state"
    for leaf in jax.tree.leaves(new.critic_state):
        assert jnp.all(jnp.isfinite(leaf)), "NaN/Inf in CartPole critic state"


def test_cartpole_multiple_train_steps_stable(cartpole_sac_cfg):
    """Five consecutive CartPole train_steps must keep all metrics finite."""
    env_params = cartpole_params_from_cfg(cartpole_sac_cfg.env)
    runner_state, actor_gd, critic_gd, opt = _init_cartpole(cartpole_sac_cfg)
    current_g = jnp.array(float(cartpole_sac_cfg.env.g), dtype=jnp.float32)

    collect_fn = jax.jit(make_sac_collect_step(
        actor_gd, env_params, cartpole_sac_cfg,
        cartpole_reset, cartpole_get_obs, cartpole_step,
    ))
    train_fn = jax.jit(_make_cartpole_train_step(
        actor_gd, critic_gd, opt, env_params, cartpole_sac_cfg
    ))

    for _ in range(int(cartpole_sac_cfg.sac.learning_starts)):
        runner_state = collect_fn(runner_state, current_g)

    for _ in range(5):
        runner_state, metrics = train_fn(runner_state, current_g)
        for field in metrics:
            assert jnp.isfinite(field), "Divergence in CartPole multi-step SAC test"
