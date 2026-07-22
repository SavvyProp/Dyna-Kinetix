"""Bang-bang controller for the three-joint standup task."""

import jax
import jax.numpy as jnp

from dynak.standup.stand_pd import (
    DEFAULT_CONTROLLER_TORQUE_NOISE_STD_NM,
    NUM_STANDUP_JOINTS,
    STANDUP_JOINT_TARGET_RAD,
    get_standup_joint_state,
    sample_controller_torque_noise_nm,
)

BB_TORQUE = 2.5
DEFAULT_BB_TORQUE_RANDOMIZATION_FRACTION = 0.2
BB_TORQUE_NOISE_KEY_TAG = 202


def stand_bb_with_torque(state, static_env_params, torque_magnitude_nm):
    """Apply an explicit per-joint torque magnitude toward the target."""
    joint_angle = get_standup_joint_state(state, static_env_params).angle_rad
    angle_error = STANDUP_JOINT_TARGET_RAD - joint_angle
    return jnp.asarray(torque_magnitude_nm) * jnp.sign(angle_error)


def stand_bb(state, static_env_params):
    """Apply the nominal torque in the direction of each joint-angle error.

    Unlike the PD controller, the torque magnitude does not shrink near the
    target and it has no derivative term.  Exactly zero angle error produces
    zero torque rather than arbitrarily choosing a direction.
    """
    return stand_bb_with_torque(state, static_env_params, BB_TORQUE)


def sample_bang_bang_torque_nm(
    episode_key,
    randomization_fraction: float = DEFAULT_BB_TORQUE_RANDOMIZATION_FRACTION,
):
    """Sample a positive bang-bang magnitude per joint for one episode."""
    multiplier = jax.random.uniform(
        episode_key,
        shape=(NUM_STANDUP_JOINTS,),
        minval=1.0 - randomization_fraction,
        maxval=1.0 + randomization_fraction,
    )
    return BB_TORQUE * multiplier


def stand_bb_randomized(
    state,
    static_env_params,
    episode_key,
    randomization_fraction: float = DEFAULT_BB_TORQUE_RANDOMIZATION_FRACTION,
    torque_noise_std_nm: float = DEFAULT_CONTROLLER_TORQUE_NOISE_STD_NM,
):
    """Return randomized bang-bang torque with per-step Gaussian noise."""
    torque_magnitude_nm = sample_bang_bang_torque_nm(
        episode_key,
        randomization_fraction,
    )
    controller_torque_nm = stand_bb_with_torque(
        state,
        static_env_params,
        torque_magnitude_nm,
    )
    torque_noise_nm = sample_controller_torque_noise_nm(
        episode_key,
        state.timestep,
        torque_noise_std_nm,
        key_tag=BB_TORQUE_NOISE_KEY_TAG,
    )
    return controller_torque_nm + torque_noise_nm
