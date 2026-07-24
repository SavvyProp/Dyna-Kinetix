"""Reusable flow-policy inference and standup-environment rollout helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp

from dynak.flow_action_chunking import (
    FlowMatchingPolicy,
    FlowModelConfig,
    sample_action_chunks,
)
from dynak.standup.controllers import UnderlyingControllerType
from dynak.standup.stand_bb import DEFAULT_BB_TORQUE_RANDOMIZATION_FRACTION
from dynak.standup.stand_pd import (
    DEFAULT_CONTROLLER_TORQUE_NOISE_STD_NM,
    DEFAULT_PD_GAIN_RANDOMIZATION_FRACTION,
)
from dynak.standup.residual_torque_env import (
    DEFAULT_ENERGY_PENALTY_COEFFICIENT,
    DEFAULT_GOAL_HOLD_DURATION_SECONDS,
    DEFAULT_GOAL_INSIDE_REWARD_PER_SECOND,
    DEFAULT_GOAL_LINEAR_VELOCITY_THRESHOLD_MPS,
    DEFAULT_TOTAL_TORQUE_LIMIT_NM,
    default_residual_torque_limit_nm,
    make_residual_torque_env,
)
from kinetix.environment.spaces import ObservationType
from kinetix.util.saving import load_params

RESIDUAL_CONTROLLER_NAMES = tuple(
    controller.value for controller in UnderlyingControllerType
)


@dataclass(frozen=True)
class LoadedFlowPolicy:
    """Model objects and preprocessing metadata restored from a checkpoint."""

    model: FlowMatchingPolicy
    params: Any
    model_config: FlowModelConfig
    image_shape: tuple[int, int, int]
    residual_torque_limit_nm: float
    prediction_target: str
    pd_gain_randomization_fraction: float
    bang_bang_torque_randomization_fraction: float
    controller_torque_noise_std_nm: float


def flow_policy_from_checkpoint(checkpoint: dict[str, Any]) -> LoadedFlowPolicy:
    """Validate and reconstruct a flow policy checkpoint in memory."""
    if "params" not in checkpoint:
        raise ValueError("Flow checkpoint contains no model parameters")
    if "model_config" not in checkpoint:
        raise ValueError("Flow checkpoint contains no model_config")
    if "data_config" not in checkpoint:
        raise ValueError("Flow checkpoint contains no data_config")

    model_config = FlowModelConfig(**checkpoint["model_config"])
    data_config = checkpoint["data_config"]
    observation_inputs = data_config.get("observation_inputs")
    if observation_inputs != ["image"]:
        raise ValueError(
            "Evaluation requires an image-only flow checkpoint; got "
            f"observation_inputs={observation_inputs!r}"
        )
    if model_config.action_dim != 3:
        raise ValueError(
            "Residual standup requires three action dimensions; checkpoint has "
            f"{model_config.action_dim}"
        )

    image_shape = tuple(int(value) for value in data_config["image_shape"])
    if len(image_shape) != 3 or image_shape[-1] != 3:
        raise ValueError(f"Invalid checkpoint image shape: {image_shape}")
    torque_normalization_limit_nm = data_config.get("torque_normalization_limit_nm")
    if torque_normalization_limit_nm is None:
        torque_normalization_limit_nm = data_config["residual_torque_limit_nm"]
    residual_torque_limit_nm = float(torque_normalization_limit_nm)
    if residual_torque_limit_nm <= 0:
        raise ValueError("Checkpoint torque normalization limit must be positive")
    prediction_target = str(data_config.get("prediction_target", "residual_torque_nm"))
    if prediction_target not in ("residual_torque_nm", "total_torque_nm"):
        raise ValueError(
            f"Unsupported flow checkpoint prediction_target={prediction_target!r}"
        )

    return LoadedFlowPolicy(
        model=FlowMatchingPolicy(model_config),
        params=checkpoint["params"],
        model_config=model_config,
        image_shape=image_shape,
        residual_torque_limit_nm=residual_torque_limit_nm,
        prediction_target=prediction_target,
        pd_gain_randomization_fraction=float(
            data_config.get(
                "pd_gain_randomization_fraction",
                DEFAULT_PD_GAIN_RANDOMIZATION_FRACTION,
            )
        ),
        bang_bang_torque_randomization_fraction=float(
            data_config.get(
                "bang_bang_torque_randomization_fraction",
                DEFAULT_BB_TORQUE_RANDOMIZATION_FRACTION,
            )
        ),
        controller_torque_noise_std_nm=float(
            data_config.get(
                "controller_torque_noise_std_nm",
                DEFAULT_CONTROLLER_TORQUE_NOISE_STD_NM,
            )
        ),
    )


def load_flow_policy_checkpoint(path: str | Path) -> LoadedFlowPolicy:
    """Load a local flow checkpoint and reconstruct its model."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Flow checkpoint not found: {path}")
    return flow_policy_from_checkpoint(load_params(path))


def initialize_image_history(
    image: jax.Array,
    frame_stack: int,
) -> jax.Array:
    """Initialize a frame history by repeating the first observation."""
    if image.ndim != 3:
        raise ValueError("image must have shape (height, width, channels)")
    return jnp.repeat(image[None, ...], frame_stack, axis=0)


def append_image_history(
    image_history: jax.Array,
    image: jax.Array,
) -> jax.Array:
    """Drop the oldest frame and append a new observation."""
    if image_history.ndim != 4 or image.ndim != 3:
        raise ValueError(
            "image_history and image must have shapes (frames, H, W, C) and "
            "(H, W, C)"
        )
    return jnp.concatenate((image_history[1:], image[None, ...]), axis=0)


def make_flow_batch_action_function(
    policy: LoadedFlowPolicy,
    *,
    num_flow_steps: int = 5,
) -> Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """Return image-history to torque-action inference for any batch size.

    The returned pure function accepts float images shaped ``(B, F, H, W, 3)``
    and one JAX key. It returns the first torque to execute and the complete
    normalized action chunk. No underlying-controller information is an input.
    Callers can JIT this function or compose it inside a larger JAX rollout.
    """
    if num_flow_steps <= 0:
        raise ValueError("num_flow_steps must be greater than zero")

    expected_history_shape = (
        policy.model_config.frame_stack,
        *policy.image_shape,
    )

    def sample_action(
        image_histories: jax.Array,
        rng: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        if image_histories.ndim != 5:
            raise ValueError("image_histories must have shape (batch, frames, H, W, C)")
        if image_histories.shape[1:] != expected_history_shape:
            raise ValueError(
                f"Checkpoint expects image histories shaped "
                f"{expected_history_shape}, got {image_histories.shape[1:]}"
            )
        normalized_chunks = sample_action_chunks(
            policy.model,
            policy.params,
            image_histories,
            rng,
            num_steps=num_flow_steps,
        )
        normalized_chunks = jnp.clip(normalized_chunks, -1.0, 1.0)
        torque_nm = normalized_chunks[:, 0] * policy.residual_torque_limit_nm
        return torque_nm, normalized_chunks

    return sample_action


def make_flow_evaluation_env(
    policy: LoadedFlowPolicy,
    initial_level,
    static_env_params,
    env_params,
    controller: str | UnderlyingControllerType,
    *,
    pd_gain_randomization_fraction: float | None = None,
    bang_bang_torque_randomization_fraction: float | None = None,
    controller_torque_noise_std_nm: float | None = None,
    total_torque_limit_nm: float = DEFAULT_TOTAL_TORQUE_LIMIT_NM,
    energy_penalty_coefficient: float = DEFAULT_ENERGY_PENALTY_COEFFICIENT,
    goal_inside_reward_per_second: float = DEFAULT_GOAL_INSIDE_REWARD_PER_SECOND,
    goal_hold_duration_seconds: float = DEFAULT_GOAL_HOLD_DURATION_SECONDS,
    goal_linear_velocity_threshold_mps: float = (
        DEFAULT_GOAL_LINEAR_VELOCITY_THRESHOLD_MPS
    ),
):
    """Construct one pixel environment for flow-policy evaluation.

    New checkpoints predict the complete applied torque and therefore execute
    with no underlying controller. Older residual-target checkpoints retain
    their original controller-specific execution behavior.
    """
    predicts_total_torque = policy.prediction_target == "total_torque_nm"
    execution_controller = (
        UnderlyingControllerType.NONE if predicts_total_torque else controller
    )
    action_torque_limit_nm = (
        min(policy.residual_torque_limit_nm, total_torque_limit_nm)
        if predicts_total_torque
        else default_residual_torque_limit_nm(controller)
    )
    return make_residual_torque_env(
        observation_type=ObservationType.PIXELS,
        reset_fn=lambda _rng: initial_level,
        env_params=env_params,
        static_env_params=static_env_params,
        auto_reset=False,
        residual_torque_limit_nm=action_torque_limit_nm,
        total_torque_limit_nm=total_torque_limit_nm,
        energy_penalty_coefficient=energy_penalty_coefficient,
        underlying_controller=execution_controller,
        pd_gain_randomization_fraction=(
            policy.pd_gain_randomization_fraction
            if pd_gain_randomization_fraction is None
            else pd_gain_randomization_fraction
        ),
        bang_bang_torque_randomization_fraction=(
            policy.bang_bang_torque_randomization_fraction
            if bang_bang_torque_randomization_fraction is None
            else bang_bang_torque_randomization_fraction
        ),
        controller_torque_noise_std_nm=(
            policy.controller_torque_noise_std_nm
            if controller_torque_noise_std_nm is None
            else controller_torque_noise_std_nm
        ),
        goal_inside_reward_per_second=goal_inside_reward_per_second,
        goal_hold_duration_seconds=goal_hold_duration_seconds,
        goal_linear_velocity_threshold_mps=goal_linear_velocity_threshold_mps,
    )


def make_batched_flow_rollout_function(
    checkpoint: dict[str, Any] | LoadedFlowPolicy,
    initial_level,
    static_env_params,
    env_params,
    controller: str | UnderlyingControllerType,
    *,
    num_flow_steps: int = 5,
    execute_horizon: int = 1,
    pd_gain_randomization_fraction: float | None = None,
    bang_bang_torque_randomization_fraction: float | None = None,
    controller_torque_noise_std_nm: float | None = None,
    total_torque_limit_nm: float = DEFAULT_TOTAL_TORQUE_LIMIT_NM,
    energy_penalty_coefficient: float = DEFAULT_ENERGY_PENALTY_COEFFICIENT,
    goal_inside_reward_per_second: float = DEFAULT_GOAL_INSIDE_REWARD_PER_SECOND,
    goal_hold_duration_seconds: float = DEFAULT_GOAL_HOLD_DURATION_SECONDS,
    goal_linear_velocity_threshold_mps: float = (
        DEFAULT_GOAL_LINEAR_VELOCITY_THRESHOLD_MPS
    ),
) -> Callable[[jax.Array], dict[str, jax.Array]]:
    """Build a jitted, vectorized evaluator for one controller variant.

    Episode keys have shape ``(num_episodes, 2)``. The returned trajectory
    fields have leading shape ``(num_episodes, max_timesteps)`` and use
    ``valid_mask`` to distinguish real transitions from terminal padding.
    A new chunk is sampled after ``execute_horizon`` actions have been used.
    """
    policy = (
        checkpoint
        if isinstance(checkpoint, LoadedFlowPolicy)
        else flow_policy_from_checkpoint(checkpoint)
    )
    if not 1 <= execute_horizon <= policy.model_config.action_horizon:
        raise ValueError(
            "execute_horizon must be between 1 and the checkpoint action "
            f"horizon ({policy.model_config.action_horizon})"
        )
    env = make_flow_evaluation_env(
        policy,
        initial_level,
        static_env_params,
        env_params,
        controller,
        pd_gain_randomization_fraction=pd_gain_randomization_fraction,
        bang_bang_torque_randomization_fraction=(
            bang_bang_torque_randomization_fraction
        ),
        controller_torque_noise_std_nm=controller_torque_noise_std_nm,
        total_torque_limit_nm=total_torque_limit_nm,
        energy_penalty_coefficient=energy_penalty_coefficient,
        goal_inside_reward_per_second=goal_inside_reward_per_second,
        goal_hold_duration_seconds=goal_hold_duration_seconds,
        goal_linear_velocity_threshold_mps=goal_linear_velocity_threshold_mps,
    )
    sample_action = make_flow_batch_action_function(
        policy,
        num_flow_steps=num_flow_steps,
    )
    sample_observation = env.get_obs(initial_level)
    if tuple(sample_observation.image.shape) != policy.image_shape:
        raise ValueError(
            f"Level renders policy observations shaped "
            f"{tuple(sample_observation.image.shape)}, but checkpoint expects "
            f"{policy.image_shape}"
        )

    max_timesteps = int(env_params.max_timesteps)
    zero_action = jnp.zeros(policy.model_config.action_dim, dtype=jnp.float32)
    zero_chunk = jnp.zeros(
        (
            policy.model_config.action_horizon,
            policy.model_config.action_dim,
        ),
        dtype=jnp.float32,
    )
    zero_scalar = jnp.asarray(0.0, dtype=jnp.float32)

    def rollout_episode(episode_key):
        reset_key, rollout_key = jax.random.split(episode_key)
        observation, state = env.reset(reset_key, env_params)
        image_history = initialize_image_history(
            observation.image,
            policy.model_config.frame_stack,
        )

        def rollout_step(carry, unused):
            del unused
            (
                rng,
                current_observation,
                current_state,
                history,
                current_chunk,
                chunk_action_index,
                finished,
            ) = carry
            rng, action_key, step_key = jax.random.split(rng, 3)

            def active_step(active_inputs):
                (
                    observation,
                    env_state,
                    image_history,
                    previous_chunk,
                    action_index,
                ) = active_inputs

                def sample_new_chunk(unused_operand):
                    del unused_operand
                    return sample_action(
                        image_history[None, ...],
                        action_key,
                    )[
                        1
                    ][0]

                normalized_chunk = jax.lax.cond(
                    action_index == 0,
                    sample_new_chunk,
                    lambda unused_operand: previous_chunk,
                    operand=None,
                )
                torque_command_nm = (
                    normalized_chunk[action_index] * policy.residual_torque_limit_nm
                )
                next_observation, next_state, reward, done, info = env.step(
                    step_key,
                    env_state,
                    torque_command_nm,
                    env_params,
                )
                next_history = append_image_history(
                    image_history,
                    next_observation.image,
                )
                transition = {
                    "action_chunk_normalized": normalized_chunk,
                    "chunk_action_index": action_index,
                    "residual_torque_nm": info["residual_torque_nm"],
                    "underlying_torque_nm": info["controller_torque_nm"],
                    "total_torque_nm": info["total_torque_nm"],
                    "reward": reward.astype(jnp.float32),
                    "done": done,
                    "success": info["GoalR"],
                    "valid_mask": jnp.asarray(True),
                    "goal_inside": info["goal_inside"],
                    "goal_inside_reward": info["goal_inside_reward"].astype(
                        jnp.float32
                    ),
                    "goal_steady": info["goal_steady"],
                    "goal_hold_time_seconds": info["goal_hold_time_seconds"].astype(
                        jnp.float32
                    ),
                }
                next_action_index = jnp.where(
                    action_index + 1 >= execute_horizon,
                    0,
                    action_index + 1,
                ).astype(jnp.int32)
                return (
                    rng,
                    next_observation,
                    next_state,
                    next_history,
                    normalized_chunk,
                    next_action_index,
                    done,
                ), transition

            def padded_step(padded_inputs):
                (
                    observation,
                    env_state,
                    image_history,
                    previous_chunk,
                    action_index,
                ) = padded_inputs
                del previous_chunk, action_index
                transition = {
                    "action_chunk_normalized": zero_chunk,
                    "chunk_action_index": jnp.asarray(0, dtype=jnp.int32),
                    "residual_torque_nm": zero_action,
                    "underlying_torque_nm": zero_action,
                    "total_torque_nm": zero_action,
                    "reward": zero_scalar,
                    "done": jnp.asarray(False),
                    "success": jnp.asarray(False),
                    "valid_mask": jnp.asarray(False),
                    "goal_inside": jnp.asarray(False),
                    "goal_inside_reward": zero_scalar,
                    "goal_steady": jnp.asarray(False),
                    "goal_hold_time_seconds": zero_scalar,
                }
                return (
                    rng,
                    observation,
                    env_state,
                    image_history,
                    zero_chunk,
                    jnp.asarray(0, dtype=jnp.int32),
                    jnp.asarray(True),
                ), transition

            return jax.lax.cond(
                finished,
                padded_step,
                active_step,
                (
                    current_observation,
                    current_state,
                    history,
                    current_chunk,
                    chunk_action_index,
                ),
            )

        initial_carry = (
            rollout_key,
            observation,
            state,
            image_history,
            zero_chunk,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False),
        )
        _, trajectory = jax.lax.scan(
            rollout_step,
            initial_carry,
            xs=None,
            length=max_timesteps,
        )
        return trajectory

    return jax.jit(jax.vmap(rollout_episode))


def make_controller_flow_rollout_functions(
    checkpoint: dict[str, Any] | LoadedFlowPolicy,
    initial_level,
    static_env_params,
    env_params,
    *,
    controllers: Sequence[str] = RESIDUAL_CONTROLLER_NAMES,
    num_flow_steps: int = 5,
    execute_horizon: int = 1,
    **environment_kwargs,
) -> dict[str, Callable[[jax.Array], dict[str, jax.Array]]]:
    """Build reusable batched rollout functions for multiple controller labels.

    For full-torque checkpoints every returned function executes without an
    underlying controller; the labels are retained for API compatibility.
    """
    policy = (
        checkpoint
        if isinstance(checkpoint, LoadedFlowPolicy)
        else flow_policy_from_checkpoint(checkpoint)
    )
    return {
        UnderlyingControllerType.from_string(controller).value: (
            make_batched_flow_rollout_function(
                policy,
                initial_level,
                static_env_params,
                env_params,
                controller,
                num_flow_steps=num_flow_steps,
                execute_horizon=execute_horizon,
                **environment_kwargs,
            )
        )
        for controller in controllers
    }
