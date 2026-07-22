"""State helpers for the three-joint standup controller."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax2d.engine import select_shape

NUM_STANDUP_JOINTS = 3

STANDUP_KP = jnp.array([5.0, 5.0, 5.0], dtype=jnp.float32)
STANDUP_KD = jnp.array([0.3, 0.3, 0.3], dtype=jnp.float32)
STANDUP_MAX_TORQUE_NM = jnp.array([5.0, 5.0, 5.0], dtype=jnp.float32)
STANDUP_JOINT_TARGET_RAD = jnp.array([0.0, 1.57, -1.57], dtype=jnp.float32)
DEFAULT_PD_GAIN_RANDOMIZATION_FRACTION = 0.2
DEFAULT_CONTROLLER_TORQUE_NOISE_STD_NM = 0.2
PD_TORQUE_NOISE_KEY_TAG = 101


class StandupJointState(NamedTuple):
    """Joint state in torque-binding order."""

    angle_rad: jax.Array
    angular_velocity_rad_s: jax.Array


def get_standup_joint_state(state, static_env_params) -> StandupJointState:
    """Return angle and velocity for the three actuated standup joints.

    The returned arrays both have shape ``(3,)`` and are ordered by motor
    binding, so element ``i`` corresponds to torque command ``i`` for bindings
    0, 1, and 2. Angles are relative to each joint's reference orientation and
    wrapped to ``[-pi, pi]``.
    """
    actuated = state.joint.active & state.joint.motor_on
    actuated &= jnp.logical_not(state.joint.is_fixed_joint)
    joint_indices = jnp.nonzero(
        actuated,
        size=NUM_STANDUP_JOINTS,
        fill_value=0,
    )[0]

    select_all = jax.vmap(select_shape, in_axes=(None, 0, None))

    body_a = select_all(
        state,
        state.joint.a_index[joint_indices],
        static_env_params,
    )
    body_b = select_all(
        state,
        state.joint.b_index[joint_indices],
        static_env_params,
    )

    reference_rotation = state.joint.rotation[joint_indices]
    unwrapped_angle = body_b.rotation - body_a.rotation - reference_rotation

    joint_angle_rad = jnp.arctan2(
        jnp.sin(unwrapped_angle),
        jnp.cos(unwrapped_angle),
    )
    joint_velocity_rad_s = body_b.angular_velocity - body_a.angular_velocity

    motor_bindings = state.motor_bindings[joint_indices]
    binding_order = jnp.argsort(motor_bindings)

    return StandupJointState(
        angle_rad=joint_angle_rad[binding_order],
        angular_velocity_rad_s=joint_velocity_rad_s[binding_order],
    )


def stand_pd_with_gains(state, static_env_params, kp, kd):
    """Return clipped PD torque using explicit per-joint gains."""
    joint_state = get_standup_joint_state(state, static_env_params)
    joint_angle = joint_state.angle_rad
    joint_velocity = joint_state.angular_velocity_rad_s
    joint_torque = kp * (STANDUP_JOINT_TARGET_RAD - joint_angle) - kd * joint_velocity
    joint_torque = jnp.clip(
        joint_torque,
        -STANDUP_MAX_TORQUE_NM,
        STANDUP_MAX_TORQUE_NM,
    )
    return joint_torque


def stand_pd(state, static_env_params):
    """Return deterministic PD torque at the nominal gains."""
    return stand_pd_with_gains(
        state,
        static_env_params,
        STANDUP_KP,
        STANDUP_KD,
    )


def sample_pd_gains(
    episode_key,
    randomization_fraction: float = DEFAULT_PD_GAIN_RANDOMIZATION_FRACTION,
):
    """Sample independent per-joint P and D gains for one episode."""
    kp_key, kd_key = jax.random.split(episode_key)
    minimum_multiplier = 1.0 - randomization_fraction
    maximum_multiplier = 1.0 + randomization_fraction
    kp_multiplier = jax.random.uniform(
        kp_key,
        shape=STANDUP_KP.shape,
        minval=minimum_multiplier,
        maxval=maximum_multiplier,
    )
    kd_multiplier = jax.random.uniform(
        kd_key,
        shape=STANDUP_KD.shape,
        minval=minimum_multiplier,
        maxval=maximum_multiplier,
    )
    return STANDUP_KP * kp_multiplier, STANDUP_KD * kd_multiplier


def sample_controller_torque_noise_nm(
    episode_key,
    timestep,
    noise_std_nm: float = DEFAULT_CONTROLLER_TORQUE_NOISE_STD_NM,
    *,
    key_tag: int,
):
    """Sample reproducible per-joint Gaussian noise for one control step."""
    controller_key = jax.random.fold_in(episode_key, key_tag)
    step_key = jax.random.fold_in(
        controller_key,
        jnp.asarray(timestep, dtype=jnp.uint32),
    )
    return noise_std_nm * jax.random.normal(
        step_key,
        shape=(NUM_STANDUP_JOINTS,),
        dtype=jnp.float32,
    )


def stand_pd_randomized(
    state,
    static_env_params,
    episode_key,
    randomization_fraction: float = DEFAULT_PD_GAIN_RANDOMIZATION_FRACTION,
    torque_noise_std_nm: float = DEFAULT_CONTROLLER_TORQUE_NOISE_STD_NM,
):
    """Return randomized PD torque with per-step Gaussian torque noise."""
    kp, kd = sample_pd_gains(episode_key, randomization_fraction)
    controller_torque_nm = stand_pd_with_gains(state, static_env_params, kp, kd)
    torque_noise_nm = sample_controller_torque_noise_nm(
        episode_key,
        state.timestep,
        torque_noise_std_nm,
        key_tag=PD_TORQUE_NOISE_KEY_TAG,
    )
    return controller_torque_nm + torque_noise_nm
