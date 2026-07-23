"""Underlying torque-controller selection for residual standup control."""

from __future__ import annotations

from enum import Enum
from functools import partial
from typing import Callable, Union

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
from dynak.standup.stand_switch import stand_switch
from kinetix.environment.env_state import EnvState, StaticEnvParams

StandupController = Callable[[EnvState, StaticEnvParams, jax.Array], jax.Array]


class UnderlyingControllerType(Enum):
    """Built-in underlying controllers available to the residual policy."""

    NONE = "none"
    PD = "pd"
    BANG_BANG = "bang_bang"
    SWITCH = "switch"

    @classmethod
    def from_string(cls, value: str) -> "UnderlyingControllerType":
        normalized = value.strip().lower().replace("-", "_")
        aliases = {
            "none": cls.NONE,
            "no_controller": cls.NONE,
            "zero": cls.NONE,
            "pd": cls.PD,
            "stand_pd": cls.PD,
            "bang_bang": cls.BANG_BANG,
            "bb": cls.BANG_BANG,
            "stand_bb": cls.BANG_BANG,
            "switch": cls.SWITCH,
            "stand_switch": cls.SWITCH,
            "random": cls.SWITCH,
            "random_switch": cls.SWITCH,
            "switching": cls.SWITCH,
            "mixed": cls.SWITCH,
        }
        try:
            return aliases[normalized]
        except KeyError as error:
            choices = ", ".join(controller.value for controller in cls)
            raise ValueError(
                f"Unknown underlying controller {value!r}; choose one of: {choices}"
            ) from error


def no_controller(state, static_env_params, controller_key):
    """Return zero baseline torque so the policy supplies all joint torque."""
    del state, static_env_params, controller_key
    return jnp.zeros(NUM_STANDUP_JOINTS, dtype=jnp.float32)


def pd_controller(
    state,
    static_env_params,
    episode_key,
    gain_randomization_fraction: float = DEFAULT_PD_GAIN_RANDOMIZATION_FRACTION,
    torque_noise_std_nm: float = DEFAULT_CONTROLLER_TORQUE_NOISE_STD_NM,
):
    """Apply PD control with episode-randomized per-joint gains."""
    return stand_pd_randomized(
        state,
        static_env_params,
        episode_key,
        gain_randomization_fraction,
        torque_noise_std_nm,
    )


def bang_bang_controller(
    state,
    static_env_params,
    episode_key,
    torque_randomization_fraction: float = (DEFAULT_BB_TORQUE_RANDOMIZATION_FRACTION),
    torque_noise_std_nm: float = DEFAULT_CONTROLLER_TORQUE_NOISE_STD_NM,
):
    """Apply bang-bang control with episode-randomized joint magnitudes."""
    return stand_bb_randomized(
        state,
        static_env_params,
        episode_key,
        torque_randomization_fraction,
        torque_noise_std_nm,
    )


def switch_controller(
    state,
    static_env_params,
    episode_key,
    pd_gain_randomization_fraction: float = (DEFAULT_PD_GAIN_RANDOMIZATION_FRACTION),
    bang_bang_torque_randomization_fraction: float = (
        DEFAULT_BB_TORQUE_RANDOMIZATION_FRACTION
    ),
    torque_noise_std_nm: float = DEFAULT_CONTROLLER_TORQUE_NOISE_STD_NM,
):
    """Apply the per-joint switching controller with randomized parameters."""
    return stand_switch(
        state,
        static_env_params,
        episode_key,
        pd_gain_randomization_fraction,
        bang_bang_torque_randomization_fraction,
        torque_noise_std_nm,
    )


BUILTIN_CONTROLLERS: dict[UnderlyingControllerType, StandupController] = {
    UnderlyingControllerType.NONE: no_controller,
    UnderlyingControllerType.PD: pd_controller,
    UnderlyingControllerType.BANG_BANG: bang_bang_controller,
    UnderlyingControllerType.SWITCH: switch_controller,
}

ControllerSpec = Union[str, UnderlyingControllerType, StandupController]


def resolve_underlying_controller(
    controller: ControllerSpec,
    *,
    pd_gain_randomization_fraction: float = (DEFAULT_PD_GAIN_RANDOMIZATION_FRACTION),
    bang_bang_torque_randomization_fraction: float = (
        DEFAULT_BB_TORQUE_RANDOMIZATION_FRACTION
    ),
    controller_torque_noise_std_nm: float = DEFAULT_CONTROLLER_TORQUE_NOISE_STD_NM,
) -> tuple[str, StandupController]:
    """Resolve a built-in name/enum or accept a custom controller callable."""
    for name, value in (
        ("pd_gain_randomization_fraction", pd_gain_randomization_fraction),
        (
            "bang_bang_torque_randomization_fraction",
            bang_bang_torque_randomization_fraction,
        ),
    ):
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must be in [0, 1)")
    if controller_torque_noise_std_nm < 0.0:
        raise ValueError("controller_torque_noise_std_nm must be non-negative")

    if isinstance(controller, str):
        controller = UnderlyingControllerType.from_string(controller)
    if isinstance(controller, UnderlyingControllerType):
        controller_function = BUILTIN_CONTROLLERS[controller]
        if controller is UnderlyingControllerType.PD:
            controller_function = partial(
                pd_controller,
                gain_randomization_fraction=pd_gain_randomization_fraction,
                torque_noise_std_nm=controller_torque_noise_std_nm,
            )
        elif controller is UnderlyingControllerType.BANG_BANG:
            controller_function = partial(
                bang_bang_controller,
                torque_randomization_fraction=(bang_bang_torque_randomization_fraction),
                torque_noise_std_nm=controller_torque_noise_std_nm,
            )
        elif controller is UnderlyingControllerType.SWITCH:
            controller_function = partial(
                switch_controller,
                pd_gain_randomization_fraction=(pd_gain_randomization_fraction),
                bang_bang_torque_randomization_fraction=(
                    bang_bang_torque_randomization_fraction
                ),
                torque_noise_std_nm=controller_torque_noise_std_nm,
            )
        return controller.value, controller_function
    if callable(controller):
        name = getattr(controller, "__name__", controller.__class__.__name__)
        return name, controller
    raise TypeError(
        "underlying_controller must be a name, UnderlyingControllerType, or callable"
    )
