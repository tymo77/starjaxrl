"""Training runner: RunnerState, train_step factory, and top-level train()."""

from pathlib import Path
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from omegaconf import DictConfig

from starjaxrl.agents.ppo import (
    PPOAgent,
    Transition,
    TrainMetrics,
    agent_from_cfg,
    compute_gae,
)


# ---------------------------------------------------------------------------
# Runner state
# ---------------------------------------------------------------------------

class RunnerState(NamedTuple):
    """Carry for the JAX training loop. All fields are pytree-compatible."""
    env_states:  Any           # batched env states (fields shape (N,))
    obs:         jax.Array     # (N, obs_dim) — current observations
    agent_state: Any           # nnx.State pytree of parameter arrays
    opt_state:   Any           # optax optimizer state
    key:         jax.Array     # PRNG key
    step:        jax.Array     # int scalar — number of updates so far


# ---------------------------------------------------------------------------
# train_step factory
# ---------------------------------------------------------------------------

def make_train_step(
    graphdef:        Any,
    optimizer:       optax.GradientTransformation,
    base_env_params: Any,
    cfg:             DictConfig,
    env_reset:       Callable,
    env_get_obs:     Callable,
    env_step:        Callable,
) -> Callable[[RunnerState, Any], tuple[RunnerState, TrainMetrics]]:
    """Return a jit-able train_step function closed over static config.

    The returned function accepts ``current_g`` as its second argument so
    that a gravity curriculum can be applied without recompilation.

    Args:
        graphdef:        Static NNX graph structure.
        optimizer:       Optax optimizer.
        base_env_params: Environment parameters (NamedTuple with a ``g`` field).
        cfg:             Hydra config.
        env_reset:       ``reset(key, params) -> state`` function for the env.
        env_get_obs:     ``get_obs(state) -> obs`` function for the env.
        env_step:        ``step(state, action, params) -> (state, obs, r, done, info)``
                         function for the env.
    """

    n_envs         = int(cfg.ppo.n_envs)
    rollout_len    = int(cfg.ppo.rollout_len)
    n_epochs       = int(cfg.ppo.n_epochs)
    minibatch_size = int(cfg.ppo.minibatch_size)
    batch_size     = n_envs * rollout_len
    n_minibatches  = batch_size // minibatch_size
    gamma          = float(cfg.ppo.gamma)
    gae_lambda     = float(cfg.ppo.gae_lambda)
    clip_eps       = float(cfg.ppo.clip_eps)
    vf_coef        = float(cfg.ppo.value_coeff)
    ent_coef       = float(cfg.ppo.entropy_coeff)
    normalize_adv  = bool(cfg.ppo.normalize_advantages)

    def train_step(
        runner_state: RunnerState, current_g: jax.Array
    ) -> tuple[RunnerState, TrainMetrics]:
        # Patch gravity for this update (enables curriculum without recompilation)
        env_params = base_env_params._replace(g=current_g)

        env_states, obs, agent_state, opt_state, key, step = runner_state

        # Reconstruct agent from pytree state for the rollout
        agent = nnx.merge(graphdef, agent_state)

        # ------------------------------------------------------------------ #
        # 1. Rollout collection                                               #
        # ------------------------------------------------------------------ #

        def collect_step(
            carry: tuple, _: Any
        ) -> tuple[tuple, Transition]:
            env_states, obs, key = carry

            key, act_key, rst_key = jax.random.split(key, 3)
            act_keys = jax.random.split(act_key, n_envs)

            # Action for each env (vmapped over envs)
            actions, log_probs, values, _ = jax.vmap(
                lambda o, k: agent.get_action_and_value(o, k)
            )(obs, act_keys)

            # Step all envs in parallel
            next_states, next_obs, rewards, dones, _ = jax.vmap(
                lambda s, a: env_step(s, a, env_params)
            )(env_states, actions)

            # Auto-reset episodes that just ended
            rst_keys   = jax.random.split(rst_key, n_envs)
            fresh_states = jax.vmap(lambda k: env_reset(k, env_params))(rst_keys)
            fresh_obs    = jax.vmap(env_get_obs)(fresh_states)

            # For done envs: carry fresh state/obs into next step
            env_states_next = jax.tree.map(
                lambda f, n: jnp.where(dones, f, n),
                fresh_states, next_states,
            )
            obs_next = jnp.where(dones[:, None], fresh_obs, next_obs)

            transition = Transition(obs, actions, log_probs, values, rewards, dones)
            return (env_states_next, obs_next, key), transition

        key, rollout_key = jax.random.split(key)
        (env_states, last_obs, _), traj = jax.lax.scan(
            collect_step, (env_states, obs, rollout_key), None, rollout_len
        )
        # traj fields shape: (T, N, ...)

        # ------------------------------------------------------------------ #
        # 2. Bootstrap last value and compute GAE                             #
        # ------------------------------------------------------------------ #

        last_values = jax.vmap(agent.critic)(last_obs)   # (N,)

        advantages, returns = compute_gae(
            traj.reward, traj.value, traj.done,
            last_values, gamma, gae_lambda,
        )                                                 # (T, N)

        # ------------------------------------------------------------------ #
        # 3. Flatten (T, N, ...) → (T*N, ...)                                #
        # ------------------------------------------------------------------ #

        def _flat(x: jax.Array) -> jax.Array:
            return x.reshape((-1,) + x.shape[2:])

        flat_obs       = _flat(traj.obs)        # (B, obs_dim)
        flat_actions   = _flat(traj.action)    # (B, action_dim)
        flat_log_probs = _flat(traj.log_prob)  # (B,)
        flat_adv       = advantages.reshape(-1) # (B,)
        flat_returns   = returns.reshape(-1)    # (B,)

        if normalize_adv:
            flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        # Rebuild state so gradient can flow through it
        _, agent_state = nnx.split(agent)

        # ------------------------------------------------------------------ #
        # 4. PPO update: K epochs × M minibatches                            #
        # ------------------------------------------------------------------ #

        def update_epoch(
            carry: tuple, _: Any
        ) -> tuple[tuple, TrainMetrics]:
            agent_state, opt_state, key = carry
            key, perm_key = jax.random.split(key)
            perm       = jax.random.permutation(perm_key, batch_size)
            mb_indices = perm.reshape(n_minibatches, minibatch_size)

            def update_minibatch(
                carry: tuple, mb_idx: jax.Array
            ) -> tuple[tuple, TrainMetrics]:
                agent_state, opt_state = carry

                mb_obs    = flat_obs[mb_idx]
                mb_act    = flat_actions[mb_idx]
                mb_lp_old = flat_log_probs[mb_idx]
                mb_adv    = flat_adv[mb_idx]
                mb_ret    = flat_returns[mb_idx]

                def loss_fn(
                    agent_state: Any,
                ) -> tuple[jax.Array, tuple]:
                    ag = nnx.merge(graphdef, agent_state)
                    log_probs, entropies, values = jax.vmap(
                        lambda o, a: ag.evaluate_actions(o, a)
                    )(mb_obs, mb_act)

                    # Clipped policy loss
                    ratio = jnp.exp(log_probs - mb_lp_old)
                    pg1   = -mb_adv * ratio
                    pg2   = -mb_adv * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                    pg_loss = jnp.mean(jnp.maximum(pg1, pg2))

                    # Value function loss (MSE)
                    vf_loss = jnp.mean((values - mb_ret) ** 2)

                    # Entropy bonus
                    entropy = jnp.mean(entropies)

                    total = pg_loss + vf_coef * vf_loss - ent_coef * entropy
                    return total, (pg_loss, vf_loss, entropy)

                (total, (pg, vf, ent)), grads = jax.value_and_grad(
                    loss_fn, has_aux=True
                )(agent_state)

                updates, opt_state = optimizer.update(grads, opt_state, agent_state)
                agent_state = optax.apply_updates(agent_state, updates)

                mb_metrics = TrainMetrics(total, pg, vf, ent, traj.reward.mean())
                return (agent_state, opt_state), mb_metrics

            (agent_state, opt_state), mb_metrics = jax.lax.scan(
                update_minibatch, (agent_state, opt_state), mb_indices
            )
            return (agent_state, opt_state, key), mb_metrics

        key, upd_key = jax.random.split(key)
        (agent_state, opt_state, _), epoch_metrics = jax.lax.scan(
            update_epoch, (agent_state, opt_state, upd_key), None, n_epochs
        )

        metrics = TrainMetrics(
            total_loss  = epoch_metrics.total_loss.mean(),
            pg_loss     = epoch_metrics.pg_loss.mean(),
            vf_loss     = epoch_metrics.vf_loss.mean(),
            entropy     = epoch_metrics.entropy.mean(),
            mean_reward = traj.reward.mean(),
        )

        new_runner = RunnerState(env_states, last_obs, agent_state, opt_state, key, step + 1)
        return new_runner, metrics

    return train_step


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------

def build_optimizer(cfg: DictConfig, n_updates: int) -> optax.GradientTransformation:
    """Construct the optax optimizer (with optional LR annealing)."""
    n_envs         = int(cfg.ppo.n_envs)
    rollout_len    = int(cfg.ppo.rollout_len)
    n_epochs       = int(cfg.ppo.n_epochs)
    minibatch_size = int(cfg.ppo.minibatch_size)
    batch_size     = n_envs * rollout_len
    n_minibatches  = batch_size // minibatch_size
    total_steps    = n_updates * n_epochs * n_minibatches

    if cfg.ppo.lr_anneal:
        lr = optax.linear_schedule(
            init_value=float(cfg.ppo.lr),
            end_value=0.0,
            transition_steps=total_steps,
        )
    else:
        lr = float(cfg.ppo.lr)

    return optax.chain(
        optax.clip_by_global_norm(float(cfg.ppo.grad_clip)),
        optax.adam(lr),
    )


def init_runner(
    cfg:         DictConfig,
    key:         jax.Array,
    env_params:  Any,
    env_reset:   Callable,
    env_get_obs: Callable,
    obs_dim:     int,
    action_dim:  int,
) -> tuple[RunnerState, Any, optax.GradientTransformation]:
    """Initialise all training state.

    Args:
        cfg:         Hydra config.
        key:         PRNG key.
        env_params:  Environment parameters (already built from cfg).
        env_reset:   ``reset(key, params) -> state`` for the env.
        env_get_obs: ``get_obs(state) -> obs`` for the env.
        obs_dim:     Observation dimensionality (env-specific).
        action_dim:  Action dimensionality (env-specific).

    Returns:
        runner_state: initial RunnerState
        graphdef:     static NNX graph structure (pass to make_train_step)
        optimizer:    optax optimizer (pass to make_train_step)
    """
    n_envs    = int(cfg.ppo.n_envs)
    n_updates = int(cfg.n_updates)

    key, agent_key, env_key = jax.random.split(key, 3)

    # Build agent and split into static graphdef + mutable state
    agent      = agent_from_cfg(cfg, agent_key, obs_dim, action_dim)
    graphdef, agent_state = nnx.split(agent)

    # Build optimizer and initialise its state
    optimizer = build_optimizer(cfg, n_updates)
    opt_state = optimizer.init(agent_state)

    # Initialise all N environments
    env_keys   = jax.random.split(env_key, n_envs)
    env_states = jax.vmap(lambda k: env_reset(k, env_params))(env_keys)
    obs        = jax.vmap(env_get_obs)(env_states)          # (N, obs_dim)

    runner_state = RunnerState(
        env_states  = env_states,
        obs         = obs,
        agent_state = agent_state,
        opt_state   = opt_state,
        key         = key,
        step        = jnp.zeros((), dtype=jnp.int32),
    )

    return runner_state, graphdef, optimizer


# ---------------------------------------------------------------------------
# Top-level training loop (Starship)
# ---------------------------------------------------------------------------

def train(cfg: DictConfig) -> tuple[RunnerState, list[TrainMetrics]]:
    """Run the full PPO training loop for the Starship environment.

    Uses a Python for-loop over jit-compiled train_steps so that
    logging/checkpointing can be interleaved.
    """
    from starjaxrl.env.starship_env import (
        env_params_from_cfg,
        get_obs,
        reset,
        step as env_step,
        StarshipEnv,
    )
    from starjaxrl.training.checkpoint import CheckpointManager
    from starjaxrl.training.logging import (
        finish_logging,
        init_logging,
        log_metrics,
        run_eval_episode,
    )

    key               = jax.random.PRNGKey(int(cfg.seed))
    base_env_params   = env_params_from_cfg(cfg.env)
    n_updates         = int(cfg.n_updates)
    log_every         = int(cfg.log_every)
    checkpoint_every  = int(cfg.checkpoint_every)
    eval_every        = int(cfg.eval_every)

    # Gravity curriculum
    g_start    = float(cfg.curriculum.g_start)
    g_end      = float(cfg.env.g)
    g_updates  = int(cfg.curriculum.g_updates)

    runner_state, graphdef, optimizer = init_runner(
        cfg, key, base_env_params, reset, get_obs,
        obs_dim=StarshipEnv.OBS_DIM, action_dim=StarshipEnv.ACTION_DIM,
    )
    train_step = jax.jit(make_train_step(
        graphdef, optimizer, base_env_params, cfg, reset, get_obs, env_step
    ))

    wandb_active = init_logging(cfg)
    ckpt_manager = CheckpointManager(Path("checkpoints"))

    all_metrics: list[TrainMetrics] = []

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
                f"pg {float(metrics.pg_loss):.4f} | "
                f"vf {float(metrics.vf_loss):.4f} | "
                f"ent {float(metrics.entropy):.4f}"
            )

        # --- Logging ---
        if wandb_active and step % log_every == 0:
            log_metrics(metrics, step, wandb_active=wandb_active)

        # --- Eval + best checkpoint ---
        if step % eval_every == 0:
            from starjaxrl.utils.visualization import plot_trajectory
            current_env_params = base_env_params._replace(g=float(current_g))
            key, eval_key = jax.random.split(runner_state.key)
            eval_states, eval_acts, success, ep_return = run_eval_episode(
                runner_state.agent_state, graphdef, current_env_params, eval_key
            )
            print(f"  eval | g={float(current_g):.2f} | return {ep_return:.2f} | "
                  f"{'SUCCESS' if success else 'crash'} | steps {len(eval_states)-1}")
            plot_trajectory(
                eval_states, eval_acts,
                path=Path("renders") / f"eval_{step:04d}.png",
                env_params=current_env_params,
                title=(f"eval @ update {step}/{n_updates} | g={float(current_g):.2f} m/s² | "
                       f"return {ep_return:.1f} | {'SUCCESS' if success else 'crash'}"),
            )

            if wandb_active:
                log_metrics(
                    metrics, step, wandb_active=wandb_active,
                    extra={"eval/return": ep_return, "eval/success": float(success),
                           "curriculum/g": float(current_g)},
                )

            ckpt_manager.maybe_save_best(runner_state.agent_state, ep_return, step)

        # --- Periodic checkpoint ---
        if step % checkpoint_every == 0:
            ckpt_manager.save_periodic(runner_state.agent_state, step)

    finish_logging(wandb_active)
    return runner_state, all_metrics
