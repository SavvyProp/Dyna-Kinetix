"""Train a PPO residual-torque policy on the standup level."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    repository_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository_root))

import hydra
import jax
import wandb
from flax.serialization import to_state_dict
from omegaconf import OmegaConf

from experiments.ppo import make_train
from kinetix.environment import LogWrapper
from kinetix.util import generate_params_from_config, init_wandb, normalise_config
from kinetix.util.saving import save_params

from dynak.standup.residual_torque_env import make_residual_torque_env


def make_residual_standup_env(
    config,
    static_env_params,
    env_params,
    reset_fn=None,
):
    """Environment factory matching ``kinetix.util.train_utils.make_env``."""
    return LogWrapper(
        make_residual_torque_env(
            observation_type=config["observation_type"],
            reset_fn=reset_fn,
            env_params=env_params,
            static_env_params=static_env_params,
            auto_reset=True,
            residual_torque_limit_nm=config["residual_torque_limit_nm"],
            total_torque_limit_nm=config["total_torque_limit_nm"],
            energy_penalty_coefficient=config["energy_penalty_coefficient"],
            underlying_controller=config.get("underlying_controller", "pd"),
            pd_gain_randomization_fraction=config.get(
                "pd_gain_randomization_fraction",
                0.2,
            ),
            bang_bang_torque_randomization_fraction=config.get(
                "bang_bang_torque_randomization_fraction",
                0.2,
            ),
            controller_torque_noise_std_nm=config.get(
                "controller_torque_noise_std_nm",
                0.2,
            ),
            goal_inside_reward_per_second=config.get(
                "goal_inside_reward_per_second",
                1.0,
            ),
            goal_hold_duration_seconds=config["goal_hold_duration_seconds"],
            goal_linear_velocity_threshold_mps=config[
                "goal_linear_velocity_threshold_mps"
            ],
            goal_angular_velocity_threshold_rad_s=config[
                "goal_angular_velocity_threshold_rad_s"
            ],
        )
    )


def run_residual_standup_training(
    hydra_config,
    experiment_name: str,
) -> None:
    """Compile and run PPO, then save a self-describing local checkpoint."""
    process_id = jax.process_index()
    config = normalise_config(
        OmegaConf.to_container(hydra_config, resolve=True),
        experiment_name,
    )
    env_params, static_env_params = generate_params_from_config(config)
    config["env_params"] = to_state_dict(env_params)
    config["static_env_params"] = to_state_dict(static_env_params)

    if config["use_wandb"]:
        if process_id == 0:
            init_wandb(
                config,
                experiment_name,
                settings=wandb.Settings(quiet=True),
            )
        else:
            os.environ["WANDB_MODE"] = "disabled"

    rng = jax.random.PRNGKey(config["seed"])
    rng, train_rng = jax.random.split(rng)

    print(
        "Compiling residual standup PPO "
        f"with {config['underlying_controller']} controller "
        f"({config['num_train_envs']} envs, "
        f"{config['total_timesteps']} total environment steps)..."
    )
    compile_start = time.time()
    train = (
        jax.jit(
            make_train(
                config,
                env_params,
                static_env_params,
                env_factory=make_residual_standup_env,
            )
        )
        .lower(train_rng)
        .compile()
    )
    print(f"Compilation finished in {time.time() - compile_start:.1f}s; training...")

    result = train(train_rng)
    final_train_state = jax.device_get(result["runner_state"].train_state)
    final_extra = jax.device_get(result["runner_state"].extra)

    if process_id == 0:
        checkpoint_path = Path(config["final_checkpoint_path"])
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        save_params(
            {
                "step": final_train_state.step,
                "params": final_train_state.params,
                "opt_state": final_train_state.opt_state,
                "extra": final_extra,
                "config": config,
            },
            checkpoint_path,
        )
        print(f"Final policy saved to {checkpoint_path.resolve()}")

    if config["use_wandb"] and process_id == 0:
        wandb.finish()


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="residual_standup_ppo",
)
def main(hydra_config) -> None:
    """Train the default residual standup policy with PD control."""
    run_residual_standup_training(hydra_config, "ResidualStandupPDPPO")


if __name__ == "__main__":
    main()
