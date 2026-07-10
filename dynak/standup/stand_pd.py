"""State helpers for the three-joint standup controller."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax2d.engine import select_shape

NUM_STANDUP_JOINTS = 3
CUTOUT_PERIOD_SECONDS = 2.0
PHYSICS_DT_SECONDS = 1.0 / 60.0


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


def stand_pd(state, static_env_params):
    kp = jnp.array([5.0, 5.0, 5.0])  # Proportional gains for each joint
    kd = jnp.array([0.3, 0.3, 0.3])  # Derivative gains for each joint
    max_torque = jnp.array([5, 5, 5])  # Maximum torque for each joint
    joint_target = jnp.array(
        [0.0, 1.57, -1.57]
    )  # Target angles for each joint (standing position)
    joint_state = get_standup_joint_state(state, static_env_params)
    joint_angle = joint_state.angle_rad
    joint_velocity = joint_state.angular_velocity_rad_s
    joint_torque = kp * (joint_target - joint_angle) - kd * joint_velocity
    joint_torque = jnp.clip(joint_torque, -max_torque, max_torque)
    return joint_torque


@jax.jit
def stand_pd_with_cutout(
    state,
    static_env_params,
    cutout_key,
):
    """Compute standup PD torque with independent two-second joint cutouts.

    ``cutout_key`` must remain fixed for the duration of an episode. A new
    per-joint Bernoulli mask is derived from it for every two-second period, so
    repeated calls within the same period return the same mask. Use a new base
    key after an episode reset when a new cutout sequence is desired.

    Args:
        state: Current Kinetix environment state.
        static_env_params: Static parameters for the current environment.
        cutout_key: Episode-level JAX PRNG key.

    Returns:
        Standup PD torque in N*m with disabled commands replaced by zero.
    """
    joint_torque = stand_pd(state, static_env_params)
    probability = jnp.clip(0.2, 0.0, 1.0)
    control_dt_seconds = PHYSICS_DT_SECONDS * static_env_params.frame_skip
    steps_per_period = jnp.maximum(
        1,
        jnp.rint(CUTOUT_PERIOD_SECONDS / control_dt_seconds).astype(jnp.int32),
    )
    period_index = (state.timestep // steps_per_period).astype(jnp.uint32)
    period_key = jax.random.fold_in(cutout_key, period_index)
    disabled = jax.random.bernoulli(
        period_key,
        p=probability,
        shape=joint_torque.shape,
    )
    return jnp.where(disabled, jnp.zeros_like(joint_torque), joint_torque)
