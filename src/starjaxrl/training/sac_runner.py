"""SAC training runner: SACRunnerState, train_step factory, and top-level sac_train()."""

from pathlib import Path
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from omegaconf import DictConfig

from starjaxrl.agents.sac import (
    SACActor,
    SACTwinnedQ,
    SACTrainMetrics,
    ReplayBuffer,
    make_replay_buffer,
    buffer_add_batch,
    buffer_sample,
)


# ---------------------------------------------------------------------------
# Runner state
# ---------------------------------------------------------------------------

class SACRunnerState(NamedTuple):
    """Carry for the SAC training loop. All fields are pytree-compatible."""
    env_states:   Any           # batched env states (fields shape (N,))
    obs:          jax.Array     # (N, obs_dim) — current observations
    buffer:       ReplayBuffer  # circular replay buffer
    actor_state:  Any           # nnx.State of SACActor
    critic_state: Any           # nnx.State of SACTwinnedQ
    target_state: Any           # nnx.State of SACTwinnedQ (EMA of critics)
    log_alpha:    jax.Array     # scalar — learned log-temperature
    actor_opt:    Any           # optax state for actor
    critic_opt:   Any           # optax state for critics
    alpha_opt:    Any           # optax state for log_alpha
    key:          jax.Array     # PRNG key
    step:         jax.Array     # int scalar — number of updates so far


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_optimizer(cfg: DictConfig) -> optax.GradientTransformation:
    """Actor and critic optimizer: gradient clipping + Adam."""
    return optax.chain(
        optax.clip_by_global_norm(float(cfg.sac.grad_clip)),
        optax.adam(float(cfg.sac.lr)),
    )


# ---------------------------------------------------------------------------
# collect_step factory (no gradient updates, just env interaction)
# ---------------------------------------------------------------------------

def make_sac_collect_step(
    actor_graphdef:  Any,
    base_env_params: Any,
    cfg:             DictConfig,
    env_reset:       Callable,
    env_get_obs:     Callable,
    env_step:        Callable,
) -> Callable[[SACRunnerState, jax.Array], SACRunnerState]:
    """Return a jit-able step that collects one env transition per parallel env.

    Used during the ``learning_starts`` warm-up phase where the buffer is
    filled before any gradient updates are applied.
    """
    n_envs = int(cfg.sac.n_envs)

    def collect_step(
        runner_state: SACRunnerState, current_g: jax.Array
    ) -> SACRunnerState:
        env_params = base_env_params._replace(g=current_g)

        actor = nnx.merge(actor_graphdef, runner_state.actor_state)

        key, act_key, rst_key = jax.random.split(runner_state.key, 3)
        act_keys = jax.random.split(act_key, n_envs)

        actions, _ = jax.vmap(lambda o, k: actor.sample(o, k))(runner_state.obs, act_keys)

        next_states, next_obs, rewards, dones, _ = jax.vmap(
            lambda s, a: env_step(s, a, env_params)
        )(runner_state.env_states, actions)

        rst_keys     = jax.random.split(rst_key, n_envs)
        fresh_states = jax.vmap(lambda k: env_reset(k, env_params))(rst_keys)
        fresh_obs    = jax.vmap(env_get_obs)(fresh_states)

        env_states_next = jax.tree.map(
            lambda f, n: jnp.where(dones, f, n), fresh_states, next_states
        )
        obs_next = jnp.where(dones[:, None], fresh_obs, next_obs)

        new_buffer = buffer_add_batch(
            runner_state.buffer, runner_state.obs, actions, rewards, next_obs, dones
        )

        return SACRunnerState(
            env_states   = env_states_next,
            obs          = obs_next,
            buffer       = new_buffer,
            actor_state  = runner_state.actor_state,
            critic_state = runner_state.critic_state,
            target_state = runner_state.target_state,
            log_alpha    = runner_state.log_alpha,
            actor_opt    = runner_state.actor_opt,
            critic_opt   = runner_state.critic_opt,
            alpha_opt    = runner_state.alpha_opt,
            key          = key,
            step         = runner_state.step + 1,
        )

    return collect_step


# ---------------------------------------------------------------------------
# train_step factory
# ---------------------------------------------------------------------------

def make_sac_train_step(
    actor_graphdef:  Any,
    critic_graphdef: Any,
    optimizer:       optax.GradientTransformation,
    base_env_params: Any,
    cfg:             DictConfig,
    env_reset:       Callable,
    env_get_obs:     Callable,
    env_step:        Callable,
    action_dim:      int,
) -> Callable[[SACRunnerState, jax.Array], tuple[SACRunnerState, SACTrainMetrics]]:
    """Return a jit-able SAC train_step closed over static config.

    Each call:
      1. Collects ``collect_steps`` env transitions per parallel env.
      2. Adds them to the replay buffer.
      3. Samples a minibatch and applies one gradient update to critics,
         actor, and (optionally) the entropy temperature α.
      4. Updates target networks via exponential moving average.
    """
    n_envs         = int(cfg.sac.n_envs)
    collect_steps  = int(cfg.sac.collect_steps)
    gamma          = float(cfg.sac.gamma)
    tau            = float(cfg.sac.tau)
    batch_size     = int(cfg.sac.batch_size)
    auto_alpha     = bool(cfg.sac.auto_alpha)
    target_entropy = float(cfg.sac.target_entropy_scale) * action_dim

    # Separate Adam for the scalar log_alpha (no gradient clipping needed)
    alpha_optimizer = optax.adam(float(cfg.sac.lr))

    # ------------------------------------------------------------------ #
    # Inner scan body: one env step across all parallel envs              #
    # ------------------------------------------------------------------ #

    def _collect_one(
        carry: tuple, _: Any
    ) -> tuple[tuple, jax.Array]:
        env_states, obs, buffer, actor_state, key, env_params = carry

        actor    = nnx.merge(actor_graphdef, actor_state)
        key, act_key, rst_key = jax.random.split(key, 3)
        act_keys = jax.random.split(act_key, n_envs)

        actions, _ = jax.vmap(lambda o, k: actor.sample(o, k))(obs, act_keys)

        next_states, next_obs, rewards, dones, _ = jax.vmap(
            lambda s, a: env_step(s, a, env_params)
        )(env_states, actions)

        rst_keys     = jax.random.split(rst_key, n_envs)
        fresh_states = jax.vmap(lambda k: env_reset(k, env_params))(rst_keys)
        fresh_obs    = jax.vmap(env_get_obs)(fresh_states)

        env_states_next = jax.tree.map(
            lambda f, n: jnp.where(dones, f, n), fresh_states, next_states
        )
        obs_next = jnp.where(dones[:, None], fresh_obs, next_obs)

        new_buffer = buffer_add_batch(buffer, obs, actions, rewards, next_obs, dones)

        return (env_states_next, obs_next, new_buffer, actor_state, key, env_params), rewards

    # ------------------------------------------------------------------ #
    # Main train_step                                                     #
    # ------------------------------------------------------------------ #

    def train_step(
        runner_state: SACRunnerState, current_g: jax.Array
    ) -> tuple[SACRunnerState, SACTrainMetrics]:
        env_params = base_env_params._replace(g=current_g)

        # --- 1. Collect environment transitions ---
        key, collect_key = jax.random.split(runner_state.key)
        collect_carry = (
            runner_state.env_states, runner_state.obs, runner_state.buffer,
            runner_state.actor_state, collect_key, env_params,
        )
        (env_states, obs, buffer, _, _, _), rewards = jax.lax.scan(
            _collect_one, collect_carry, None, collect_steps
        )
        # rewards shape: (collect_steps, n_envs)

        # --- 2. Sample minibatch from replay buffer ---
        key, sample_key, next_act_key, actor_act_key = jax.random.split(key, 4)
        b_obs, b_act, b_rew, b_next_obs, b_done = buffer_sample(
            buffer, batch_size, sample_key
        )

        # --- 3. Compute Bellman targets using frozen target networks ---
        actor_now     = nnx.merge(actor_graphdef, runner_state.actor_state)
        next_act_keys = jax.random.split(next_act_key, batch_size)
        next_actions, next_log_probs = jax.vmap(
            lambda o, k: actor_now.sample(o, k)
        )(b_next_obs, next_act_keys)

        target_critics = nnx.merge(critic_graphdef, runner_state.target_state)
        q1_next, q2_next = jax.vmap(target_critics)(b_next_obs, next_actions)

        alpha      = jnp.exp(runner_state.log_alpha)
        min_q_next = jnp.minimum(q1_next, q2_next) - alpha * next_log_probs
        targets    = (
            b_rew
            + gamma * (1.0 - b_done.astype(jnp.float32)) * min_q_next
        )
        targets = jax.lax.stop_gradient(targets)

        # --- 4. Critic update ---
        def critic_loss_fn(critic_state: Any) -> jax.Array:
            critics = nnx.merge(critic_graphdef, critic_state)
            q1, q2  = jax.vmap(critics)(b_obs, b_act)
            return jnp.mean((q1 - targets) ** 2 + (q2 - targets) ** 2)

        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(
            runner_state.critic_state
        )
        critic_updates, critic_opt = optimizer.update(
            critic_grads, runner_state.critic_opt, runner_state.critic_state
        )
        critic_state = optax.apply_updates(runner_state.critic_state, critic_updates)

        # --- 5. Actor update (uses freshly updated critics) ---
        actor_act_keys = jax.random.split(actor_act_key, batch_size)

        def actor_loss_fn(actor_state: Any) -> tuple[jax.Array, jax.Array]:
            actor   = nnx.merge(actor_graphdef, actor_state)
            acts_pi, log_probs_pi = jax.vmap(
                lambda o, k: actor.sample(o, k)
            )(b_obs, actor_act_keys)
            # Use updated critic_state for the Q-value signal
            critics = nnx.merge(critic_graphdef, critic_state)
            q1, q2  = jax.vmap(critics)(b_obs, acts_pi)
            min_q   = jnp.minimum(q1, q2)
            a       = jnp.exp(runner_state.log_alpha)
            loss    = jnp.mean(a * log_probs_pi - min_q)
            return loss, log_probs_pi.mean()

        (actor_loss, mean_log_prob), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(runner_state.actor_state)
        actor_updates, actor_opt = optimizer.update(
            actor_grads, runner_state.actor_opt, runner_state.actor_state
        )
        actor_state = optax.apply_updates(runner_state.actor_state, actor_updates)

        # --- 6. Temperature (alpha) update ---
        def alpha_loss_fn(log_alpha: jax.Array) -> jax.Array:
            return -jnp.exp(log_alpha) * jax.lax.stop_gradient(
                mean_log_prob + target_entropy
            )

        alpha_loss, alpha_grads = jax.value_and_grad(alpha_loss_fn)(runner_state.log_alpha)
        alpha_updates, alpha_opt = alpha_optimizer.update(
            alpha_grads, runner_state.alpha_opt
        )
        new_log_alpha = optax.apply_updates(runner_state.log_alpha, alpha_updates)
        # Revert to previous alpha when auto-tuning is disabled
        log_alpha = jnp.where(auto_alpha, new_log_alpha, runner_state.log_alpha)

        # --- 7. Soft target network update (EMA) ---
        target_state = jax.tree.map(
            lambda t, c: (1.0 - tau) * t + tau * c,
            runner_state.target_state, critic_state,
        )

        metrics = SACTrainMetrics(
            actor_loss  = actor_loss,
            critic_loss = critic_loss,
            alpha_loss  = alpha_loss,
            alpha       = jnp.exp(log_alpha),
            mean_reward = rewards.mean(),
        )

        new_runner = SACRunnerState(
            env_states   = env_states,
            obs          = obs,
            buffer       = buffer,
            actor_state  = actor_state,
            critic_state = critic_state,
            target_state = target_state,
            log_alpha    = log_alpha,
            actor_opt    = actor_opt,
            critic_opt   = critic_opt,
            alpha_opt    = alpha_opt,
            key          = key,
            step         = runner_state.step + 1,
        )
        return new_runner, metrics

    return train_step


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_sac_runner(
    cfg:         DictConfig,
    key:         jax.Array,
    env_params:  Any,
    env_reset:   Callable,
    env_get_obs: Callable,
    obs_dim:     int,
    action_dim:  int,
) -> tuple[SACRunnerState, Any, Any, optax.GradientTransformation]:
    """Initialise all SAC training state.

    Returns:
        runner_state:    Initial SACRunnerState.
        actor_graphdef:  Static NNX graph structure for the actor.
        critic_graphdef: Static NNX graph structure for the critics.
        optimizer:       Shared optax optimizer for actor and critics.
    """
    n_envs      = int(cfg.sac.n_envs)
    buffer_size = int(cfg.sac.buffer_size)
    hidden_dim  = int(cfg.network.hidden_dim)
    n_hidden    = int(cfg.network.n_hidden)

    key, actor_key, critic_key, env_key = jax.random.split(key, 4)

    # Build actor and critics
    actor   = SACActor(
        obs_dim, action_dim, hidden_dim, n_hidden, nnx.Rngs(params=actor_key)
    )
    critics = SACTwinnedQ(
        obs_dim, action_dim, hidden_dim, n_hidden, nnx.Rngs(params=critic_key)
    )

    actor_graphdef,  actor_state  = nnx.split(actor)
    critic_graphdef, critic_state = nnx.split(critics)
    # Target networks start as an exact copy of the critics
    target_state = jax.tree.map(lambda x: x, critic_state)

    # Build shared optimizer and per-network optimizer states
    optimizer  = _build_optimizer(cfg)
    actor_opt  = optimizer.init(actor_state)
    critic_opt = optimizer.init(critic_state)

    # Scalar log_alpha with its own Adam optimizer (no gradient clipping)
    log_alpha = jnp.log(jnp.array(float(cfg.sac.alpha_init)))
    alpha_opt = optax.adam(float(cfg.sac.lr)).init(log_alpha)

    # Initialise parallel environments
    env_keys   = jax.random.split(env_key, n_envs)
    env_states = jax.vmap(lambda k: env_reset(k, env_params))(env_keys)
    obs        = jax.vmap(env_get_obs)(env_states)

    # Allocate empty replay buffer
    buffer = make_replay_buffer(buffer_size, obs_dim, action_dim)

    runner_state = SACRunnerState(
        env_states   = env_states,
        obs          = obs,
        buffer       = buffer,
        actor_state  = actor_state,
        critic_state = critic_state,
        target_state = target_state,
        log_alpha    = log_alpha,
        actor_opt    = actor_opt,
        critic_opt   = critic_opt,
        alpha_opt    = alpha_opt,
        key          = key,
        step         = jnp.zeros((), dtype=jnp.int32),
    )

    return runner_state, actor_graphdef, critic_graphdef, optimizer


# ---------------------------------------------------------------------------
# Top-level training loops
# ---------------------------------------------------------------------------

def sac_train(cfg: DictConfig) -> tuple[SACRunnerState, list[SACTrainMetrics]]:
    """Run the full SAC training loop for the Starship environment."""
    from starjaxrl.env.starship_env import (
        StarshipEnv,
        env_params_from_cfg,
        get_obs,
        reset,
        step as env_step,
    )
    from starjaxrl.training.checkpoint import CheckpointManager
    from starjaxrl.training.logging import init_logging, log_metrics, finish_logging

    key              = jax.random.PRNGKey(int(cfg.seed))
    base_env_params  = env_params_from_cfg(cfg.env)
    n_updates        = int(cfg.n_updates)
    log_every        = int(cfg.log_every)
    checkpoint_every = int(cfg.checkpoint_every)
    learning_starts  = int(cfg.sac.learning_starts)

    # Gravity curriculum (same as PPO runner)
    g_start   = float(cfg.curriculum.g_start)
    g_end     = float(cfg.env.g)
    g_updates = int(cfg.curriculum.g_updates)

    runner_state, actor_graphdef, critic_graphdef, optimizer = init_sac_runner(
        cfg, key, base_env_params, reset, get_obs,
        obs_dim=StarshipEnv.OBS_DIM, action_dim=StarshipEnv.ACTION_DIM,
    )

    collect_step = jax.jit(make_sac_collect_step(
        actor_graphdef, base_env_params, cfg, reset, get_obs, env_step
    ))
    train_step = jax.jit(make_sac_train_step(
        actor_graphdef, critic_graphdef, optimizer, base_env_params,
        cfg, reset, get_obs, env_step, action_dim=StarshipEnv.ACTION_DIM,
    ))

    wandb_active = init_logging(cfg)
    ckpt_manager = CheckpointManager(Path("checkpoints"))

    current_g = jnp.array(g_start, dtype=jnp.float32)

    # Warm-up: fill the replay buffer before training
    print(f"Filling replay buffer ({learning_starts} transitions)…")
    for _ in range(learning_starts):
        runner_state = collect_step(runner_state, current_g)

    all_metrics: list[SACTrainMetrics] = []

    for update in range(n_updates):
        frac      = min(1.0, update / max(1, g_updates))
        current_g = jnp.array(g_start + frac * (g_end - g_start), dtype=jnp.float32)

        runner_state, metrics = train_step(runner_state, current_g)
        all_metrics.append(metrics)
        step = update + 1

        if step % log_every == 0:
            print(
                f"update {step:4d}/{n_updates} | "
                f"g={float(current_g):.2f} | "
                f"reward {float(metrics.mean_reward):+.3f} | "
                f"actor {float(metrics.actor_loss):.4f} | "
                f"critic {float(metrics.critic_loss):.4f} | "
                f"alpha {float(metrics.alpha):.4f}"
            )

        if wandb_active and step % log_every == 0:
            log_metrics(metrics, step, wandb_active=wandb_active)

        if step % checkpoint_every == 0:
            ckpt_manager.save_periodic(runner_state.actor_state, step)

    finish_logging(wandb_active)
    return runner_state, all_metrics
