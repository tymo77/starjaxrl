# Environments

## Shared types

::: starjaxrl.env.types

::: starjaxrl.env.reward_utils

## Starship environment

::: starjaxrl.env.starship_env
    options:
      members:
        - EnvParams
        - StarshipEnv
        - reset
        - get_obs
        - step
        - is_done
        - is_success
        - compute_reward
        - env_params_from_cfg

## CartPole environment

::: starjaxrl.env.cartpole_env
    options:
      members:
        - CartPoleEnvParams
        - CartPoleEnv
        - reset
        - get_obs
        - step
        - is_done
        - is_success
        - compute_reward
        - env_params_from_cfg
