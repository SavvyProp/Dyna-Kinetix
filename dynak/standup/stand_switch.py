"""Per-joint switching controller for the standup task."""

import jax
import jax.numpy as jnp

from dynak.standup.stand_bb import (
    DEFAULT_BB_TORQUE_RANDOMIZATION_FRACTION,
    stand_bb_randomized,
)
from dynak.standup.stand_pd import (
    DEFAULT_CONTROLLER_TORQUE_NOISE_STD_NM,
    DEFAULT_PD_GAIN_RANDOMIZATION_FRACTION,
    NUM_STANDUP_JOINTS,
    stand_pd_randomized,
)

SWITCH_PERIOD_SECONDS = 2.0
PHYSICS_DT_SECONDS = 1.0 / 60.0
NO_CONTROLLER_PROBABILITY = 0.2

NO_CONTROLLER_INDEX = 0
PD_CONTROLLER_INDEX = 1
BANG_BANG_CONTROLLER_INDEX = 2
NUM_CONTROLLER_CHOICES = 3
ACTIVE_CONTROLLER_PROBABILITY = (1.0 - NO_CONTROLLER_PROBABILITY) / 2.0
CONTROLLER_SELECTION_KEY_TAG = 0
PD_PARAMETER_KEY_TAG = 1
BB_PARAMETER_KEY_TAG = 2
CONTROLLER_PROBABILITIES = (
    jnp.zeros(NUM_CONTROLLER_CHOICES, dtype=jnp.float32)
    .at[NO_CONTROLLER_INDEX]
    .set(NO_CONTROLLER_PROBABILITY)
    .at[PD_CONTROLLER_INDEX]
    .set(ACTIVE_CONTROLLER_PROBABILITY)
    .at[BANG_BANG_CONTROLLER_INDEX]
    .set(ACTIVE_CONTROLLER_PROBABILITY)
)


def controller_switch_steps(static_env_params):
    """Return the number of environment steps in one switching period."""
    control_dt_seconds = PHYSICS_DT_SECONDS * static_env_params.frame_skip
    return jnp.maximum(
        1,
        jnp.rint(SWITCH_PERIOD_SECONDS / control_dt_seconds).astype(jnp.int32),
    )


def get_switch_controller_indices(state, static_env_params, switch_key):
    """Select none, PD, or bang-bang independently for every joint.

    The episode key and current two-second period determine each joint's
    selection. Choices remain stable within a period and are resampled at the
    next boundary. The no-controller choice has the original 20 percent
    cutout probability, and PD and bang-bang split the remaining probability.
    """
    period_index = (
        state.timestep // controller_switch_steps(static_env_params)
    ).astype(jnp.uint32)
    selection_key = jax.random.fold_in(
        switch_key,
        CONTROLLER_SELECTION_KEY_TAG,
    )
    period_key = jax.random.fold_in(selection_key, period_index)
    return jax.random.categorical(
        period_key,
        logits=jnp.log(CONTROLLER_PROBABILITIES),
        shape=(NUM_STANDUP_JOINTS,),
    ).astype(jnp.int32)


def stand_switch(
    state,
    static_env_params,
    switch_key,
    pd_gain_randomization_fraction: float = DEFAULT_PD_GAIN_RANDOMIZATION_FRACTION,
    bang_bang_torque_randomization_fraction: float = (
        DEFAULT_BB_TORQUE_RANDOMIZATION_FRACTION
    ),
    controller_torque_noise_std_nm: float = DEFAULT_CONTROLLER_TORQUE_NOISE_STD_NM,
):
    """Switch every joint independently among none, PD, and bang-bang."""
    controller_indices = get_switch_controller_indices(
        state,
        static_env_params,
        switch_key,
    )
    pd_parameter_key = jax.random.fold_in(switch_key, PD_PARAMETER_KEY_TAG)
    bb_parameter_key = jax.random.fold_in(switch_key, BB_PARAMETER_KEY_TAG)
    controller_torques = (
        jnp.zeros(
            (NUM_CONTROLLER_CHOICES, NUM_STANDUP_JOINTS),
            dtype=jnp.float32,
        )
        .at[PD_CONTROLLER_INDEX]
        .set(
            stand_pd_randomized(
                state,
                static_env_params,
                pd_parameter_key,
                pd_gain_randomization_fraction,
                controller_torque_noise_std_nm,
            )
        )
        .at[BANG_BANG_CONTROLLER_INDEX]
        .set(
            stand_bb_randomized(
                state,
                static_env_params,
                bb_parameter_key,
                bang_bang_torque_randomization_fraction,
                controller_torque_noise_std_nm,
            )
        )
    )
    joint_indices = jnp.arange(NUM_STANDUP_JOINTS)
    return controller_torques[controller_indices, joint_indices]
