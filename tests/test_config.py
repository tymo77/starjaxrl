"""Tests for Hydra configuration loading and structure."""

from omegaconf import OmegaConf


# --- env config ---

def test_env_cfg_loads(env_cfg):
    assert env_cfg is not None


def test_env_cfg_physics_keys(env_cfg):
    required = ["m_dry", "m_prop_max", "T_max", "Isp", "T_min", "delta_max", "L", "g"]
    for key in required:
        assert key in env_cfg, f"Missing env key: {key}"


def test_env_cfg_initial_conditions(env_cfg):
    required = ["x0", "y0", "vx0", "vy0", "theta0", "omega0", "mprop0"]
    for key in required:
        assert key in env_cfg, f"Missing initial condition key: {key}"


def test_env_cfg_termination_keys(env_cfg):
    required = ["y_catch", "x_max", "theta_max", "t_max"]
    for key in required:
        assert key in env_cfg, f"Missing termination key: {key}"


def test_env_cfg_reward_keys(env_cfg):
    required = ["w_x", "w_vy", "w_vx", "w_theta", "R_success"]
    for key in required:
        assert key in env_cfg, f"Missing reward key: {key}"


def test_env_cfg_values_positive(env_cfg):
    for key in ["m_dry", "m_prop_max", "T_max", "Isp", "g", "dt", "T_max", "R_success"]:
        assert env_cfg[key] > 0, f"{key} should be positive"


def test_env_cfg_throttle_bounds(env_cfg):
    assert 0.0 < env_cfg.T_min < 1.0


# --- ppo config ---

def test_ppo_cfg_loads(ppo_cfg):
    assert ppo_cfg is not None


def test_ppo_cfg_keys(ppo_cfg):
    required = [
        "gamma", "gae_lambda", "clip_eps", "lr", "lr_anneal",
        "grad_clip", "n_epochs", "minibatch_size", "rollout_len",
        "n_envs", "entropy_coeff", "value_coeff", "normalize_advantages",
    ]
    for key in required:
        assert key in ppo_cfg, f"Missing PPO key: {key}"


def test_ppo_cfg_gamma_in_range(ppo_cfg):
    assert 0.0 < ppo_cfg.gamma <= 1.0


def test_ppo_cfg_lr_positive(ppo_cfg):
    assert ppo_cfg.lr > 0


# --- network config ---

def test_network_cfg_loads(network_cfg):
    assert network_cfg is not None


def test_network_cfg_keys(network_cfg):
    required = ["hidden_dim", "n_hidden", "activation", "orthogonal_init"]
    for key in required:
        assert key in network_cfg, f"Missing network key: {key}"


def test_network_cfg_hidden_dim_positive(network_cfg):
    assert network_cfg.hidden_dim > 0


# --- merged train config ---

def test_train_cfg_loads(train_cfg):
    assert train_cfg is not None


def test_train_cfg_has_sub_configs(train_cfg):
    assert "env" in train_cfg
    assert "ppo" in train_cfg
    assert "network" in train_cfg


def test_train_cfg_top_level_keys(train_cfg):
    required = ["seed", "n_updates", "checkpoint_every", "eval_every", "log_every"]
    for key in required:
        assert key in train_cfg, f"Missing top-level train key: {key}"


def test_train_cfg_override(env_cfg):
    """Verify OmegaConf overrides work correctly."""
    overridden = OmegaConf.merge(env_cfg, {"g": 1.62})
    assert overridden.g == 1.62
    assert env_cfg.g == 9.81  # original unchanged
