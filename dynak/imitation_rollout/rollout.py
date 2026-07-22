"""Build batched residual-standup expert rollout functions.

This module is intentionally limited to stepping policies and environments.
Command-line parsing, success filtering, and dataset persistence live in
``dynak.scripts``.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from dynak.standup.residual_torque_env import make_residual_torque_env
from kinetix.environment.spaces import ActionType, ObservationType
from kinetix.models import GeneralActorCriticRNN, make_network_from_config
from kinetix.util import rms_normalise


def _config_value(config: dict[str, Any], name: str, default: Any) -> Any:
    value = config.get(name)
    return default if value is None else value


def make_batched_rollout_function(
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    initial_level,
    static_env_params,
    env_params,
    deterministic: bool,
) -> Callable[[jax.Array], dict[str, jax.Array]]:
    """Return a jitted function that rolls out multiple episodes in parallel.

    The returned function accepts PRNG keys with shape ``(num_envs, 2)`` and
    returns transition arrays with leading shape ``(num_envs, max_timesteps)``.
    Each environment stops logically at its first terminal transition; the
    remaining timesteps are zero padded and marked false in ``valid_mask``.
    """
    if config["action_type"] != ActionType.CONTINUOUS:
        raise ValueError("Residual standup checkpoints must use continuous actions")
    if config["observation_type"] not in (
        ObservationType.PIXELS,
        ObservationType.SYMBOLIC_FLAT,
    ):
        raise ValueError(
            "Imitation rollouts require pixels or symbolic_flat observations"
        )

    use_pixels = config["observation_type"] == ObservationType.PIXELS

    env = make_residual_torque_env(
        observation_type=config["observation_type"],
        reset_fn=lambda _rng: initial_level,
        env_params=env_params,
        static_env_params=static_env_params,
        auto_reset=False,
        residual_torque_limit_nm=config["residual_torque_limit_nm"],
        total_torque_limit_nm=config["total_torque_limit_nm"],
        energy_penalty_coefficient=config["energy_penalty_coefficient"],
        underlying_controller=config["underlying_controller"],
        pd_gain_randomization_fraction=_config_value(
            config,
            "pd_gain_randomization_fraction",
            0.2,
        ),
        bang_bang_torque_randomization_fraction=_config_value(
            config,
            "bang_bang_torque_randomization_fraction",
            0.2,
        ),
        controller_torque_noise_std_nm=_config_value(
            config,
            "controller_torque_noise_std_nm",
            0.2,
        ),
        goal_hold_duration_seconds=_config_value(
            config,
            "goal_hold_duration_seconds",
            1.0,
        ),
        goal_linear_velocity_threshold_mps=_config_value(
            config,
            "goal_linear_velocity_threshold_mps",
            0.1,
        ),
        goal_angular_velocity_threshold_rad_s=_config_value(
            config,
            "goal_angular_velocity_threshold_rad_s",
            0.1,
        ),
    )
    network = make_network_from_config(env, env_params, config)
    policy_params = checkpoint["params"]

    rms = None
    if config.get("rms_norm", False):
        rms = (checkpoint.get("extra") or {}).get("rms")
        if rms is None:
            raise ValueError(
                "Checkpoint enables RMS normalization but contains no RMS state"
            )

    zero_action = env.action_type.noop_action()
    zero_scalar = jnp.asarray(0.0, dtype=jnp.float32)
    max_timesteps = int(env_params.max_timesteps)

    def policy_step(hstate, observation, done, action_key):
        batched_observation = jax.tree.map(lambda x: x[None, ...], observation)
        if rms is not None:
            batched_observation = rms_normalise(
                rms,
                batched_observation,
                flatten="auto",
            )
        network_observation = jax.tree.map(
            lambda x: x[None, ...],
            batched_observation,
        )
        network_done = jnp.asarray(done)[None, None]
        hstate, distribution, _ = network.apply(
            policy_params,
            hstate,
            (network_observation, network_done),
        )
        action = (
            distribution.mode()
            if deterministic
            else distribution.sample(seed=action_key)
        )
        return hstate, action[0, 0]

    def rollout_episode(episode_key):
        reset_key, scan_key = jax.random.split(episode_key)
        observation, state = env.reset(reset_key, env_params)
        hstate = GeneralActorCriticRNN.initialize_carry(1)

        def scan_step(carry, unused):
            del unused
            rng, current_observation, current_state, current_hstate, finished = carry
            rng, policy_key, step_key = jax.random.split(rng, 3)

            def active_step(active_inputs):
                obs, env_state, policy_hstate = active_inputs
                next_hstate, policy_action = policy_step(
                    policy_hstate,
                    obs,
                    jnp.asarray(False),
                    policy_key,
                )
                next_obs, next_state, reward, done, info = env.step(
                    step_key,
                    env_state,
                    policy_action,
                    env_params,
                )
                observation_fields = (
                    {
                        "image": jnp.rint(obs.image * 255.0).astype(jnp.uint8),
                        "global_info": obs.global_info.astype(jnp.float32),
                    }
                    if use_pixels
                    else {"observation": obs}
                )
                transition = {
                    **observation_fields,
                    "policy_action": policy_action,
                    "residual_torque_nm": info["residual_torque_nm"],
                    "underlying_torque_nm": info["controller_torque_nm"],
                    "total_torque_nm": info["total_torque_nm"],
                    "reward": reward.astype(jnp.float32),
                    "done": done,
                    "success": info["GoalR"],
                    "valid_mask": jnp.asarray(True),
                    "goal_inside": info["goal_inside"],
                    "goal_steady": info["goal_steady"],
                    "goal_hold_time_seconds": info["goal_hold_time_seconds"].astype(
                        jnp.float32
                    ),
                    "goal_max_linear_speed_mps": info[
                        "goal_max_linear_speed_mps"
                    ].astype(jnp.float32),
                    "goal_max_angular_speed_rad_s": info[
                        "goal_max_angular_speed_rad_s"
                    ].astype(jnp.float32),
                }
                next_carry = (
                    rng,
                    next_obs,
                    next_state,
                    next_hstate,
                    done,
                )
                return next_carry, transition

            def padded_step(padded_inputs):
                obs, env_state, policy_hstate = padded_inputs
                observation_fields = (
                    {
                        "image": jnp.zeros_like(obs.image, dtype=jnp.uint8),
                        "global_info": jnp.zeros_like(
                            obs.global_info,
                            dtype=jnp.float32,
                        ),
                    }
                    if use_pixels
                    else {"observation": jnp.zeros_like(obs)}
                )
                transition = {
                    **observation_fields,
                    "policy_action": zero_action,
                    "residual_torque_nm": zero_action,
                    "underlying_torque_nm": zero_action,
                    "total_torque_nm": zero_action,
                    "reward": zero_scalar,
                    "done": jnp.asarray(False),
                    "success": jnp.asarray(False),
                    "valid_mask": jnp.asarray(False),
                    "goal_inside": jnp.asarray(False),
                    "goal_steady": jnp.asarray(False),
                    "goal_hold_time_seconds": zero_scalar,
                    "goal_max_linear_speed_mps": zero_scalar,
                    "goal_max_angular_speed_rad_s": zero_scalar,
                }
                return (
                    rng,
                    obs,
                    env_state,
                    policy_hstate,
                    jnp.asarray(True),
                ), transition

            return jax.lax.cond(
                finished,
                padded_step,
                active_step,
                (current_observation, current_state, current_hstate),
            )

        initial_carry = (
            scan_key,
            observation,
            state,
            hstate,
            jnp.asarray(False),
        )
        _, trajectory = jax.lax.scan(
            scan_step,
            initial_carry,
            xs=None,
            length=max_timesteps,
        )
        return trajectory

    return jax.jit(jax.vmap(rollout_episode))
