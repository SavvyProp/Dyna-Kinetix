"""Kinetix environment with residual torque control for the standup task.

The public action is a three-element vector of residual joint torques in N*m.
At every environment step it is added to :func:`stand_pd_with_cutout`, clipped,
and applied directly to motor bindings 0, 1, and 2.  The regular Kinetix
velocity motors and thrusters receive zero actions.

The environment follows the same JAX/Gymnax API as ``KinetixEnv`` and can be
wrapped in Kinetix's ``LogWrapper``.  Its state is an ``EnvState`` subclass so
existing observation, rendering, logging, and action-mask code can continue to
read the normal Kinetix fields.
"""

from __future__ import annotations

from typing import Callable, Optional

import chex
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces
from jax2d.engine import PhysicsEngine

from kinetix.environment.env import KinetixEnv, make_kinetix_env
from kinetix.environment.env_state import EnvParams, EnvState, StaticEnvParams
from kinetix.environment.spaces import (
    ActionType,
    ContinuousActions,
    KinetixObservation,
    ObservationType,
)

from dynak.standup.stand_pd import (
    NUM_STANDUP_JOINTS,
    get_standup_joint_state,
    stand_pd_with_cutout,
)

DEFAULT_RESIDUAL_TORQUE_LIMIT_NM = 2.5
DEFAULT_TOTAL_TORQUE_LIMIT_NM = 20.0
DEFAULT_ENERGY_PENALTY_COEFFICIENT = 1e-3


@struct.dataclass
class ResidualTorqueEnvState(EnvState):
    """Kinetix state plus the PRNG key for this episode's cutout sequence."""

    # A default is required because EnvState has fields with defaults.  Reset
    # always replaces this with a real key before the state is returned.
    cutout_key: Optional[jax.Array] = None


class ResidualTorqueActions(ContinuousActions):
    """Three continuous residual torques expressed directly in N*m."""

    def __init__(
        self,
        env_params: EnvParams,
        static_env_params: StaticEnvParams,
        torque_limit_nm: float,
    ):
        super().__init__(env_params, static_env_params)
        self.unified_action_space_size = NUM_STANDUP_JOINTS
        self.torque_limit_nm = float(torque_limit_nm)

    def action_space(self, env_params: Optional[EnvParams] = None) -> spaces.Box:
        del env_params
        limit = jnp.full(
            (NUM_STANDUP_JOINTS,),
            self.torque_limit_nm,
            dtype=jnp.float32,
        )
        return spaces.Box(
            low=-limit,
            high=limit,
            shape=(NUM_STANDUP_JOINTS,),
        )

    def process_action(
        self,
        action: jax.Array,
        state: EnvState,
        static_env_params: StaticEnvParams,
    ) -> jax.Array:
        del state, static_env_params
        action = jnp.asarray(action, dtype=jnp.float32)
        return jnp.clip(action, -self.torque_limit_nm, self.torque_limit_nm)

    def noop_action(self) -> jax.Array:
        return jnp.zeros(NUM_STANDUP_JOINTS, dtype=jnp.float32)

    def random_action(self, rng: chex.PRNGKey) -> jax.Array:
        return jax.random.uniform(
            rng,
            shape=(NUM_STANDUP_JOINTS,),
            minval=-self.torque_limit_nm,
            maxval=self.torque_limit_nm,
        )


def apply_revolute_joint_torques_nm(
    state: EnvState,
    motor_binding_torque_nm: jax.Array,
    dt: float,
    static_env_params: StaticEnvParams,
) -> EnvState:
    """Apply motor-binding torques as equal and opposite angular impulses.

    Positive torque increases body B's angular velocity relative to body A.
    Each active, non-fixed, motor-enabled joint receives the torque associated
    with its motor binding.  ``motor_binding_torque_nm`` therefore has one
    value per Kinetix motor binding, rather than one value per physical joint.
    """
    joint_enabled = state.joint.active & state.joint.motor_on
    joint_enabled &= jnp.logical_not(state.joint.is_fixed_joint)
    joint_torque_nm = motor_binding_torque_nm[state.motor_bindings] * joint_enabled
    angular_impulse = joint_torque_nm * dt

    a_index = state.joint.a_index
    b_index = state.joint.b_index
    a_is_polygon = a_index < static_env_params.num_polygons
    b_is_polygon = b_index < static_env_params.num_polygons

    polygon_a_index = jnp.where(a_is_polygon, a_index, 0)
    polygon_b_index = jnp.where(b_is_polygon, b_index, 0)
    circle_a_index = jnp.where(
        a_is_polygon,
        0,
        a_index - static_env_params.num_polygons,
    )
    circle_b_index = jnp.where(
        b_is_polygon,
        0,
        b_index - static_env_params.num_polygons,
    )

    polygon_angular_velocity = state.polygon.angular_velocity
    polygon_angular_velocity = polygon_angular_velocity.at[polygon_a_index].add(
        -angular_impulse * a_is_polygon * state.polygon.inverse_inertia[polygon_a_index]
    )
    polygon_angular_velocity = polygon_angular_velocity.at[polygon_b_index].add(
        angular_impulse * b_is_polygon * state.polygon.inverse_inertia[polygon_b_index]
    )

    circle_angular_velocity = state.circle.angular_velocity
    circle_angular_velocity = circle_angular_velocity.at[circle_a_index].add(
        -angular_impulse
        * jnp.logical_not(a_is_polygon)
        * state.circle.inverse_inertia[circle_a_index]
    )
    circle_angular_velocity = circle_angular_velocity.at[circle_b_index].add(
        angular_impulse
        * jnp.logical_not(b_is_polygon)
        * state.circle.inverse_inertia[circle_b_index]
    )

    return state.replace(
        polygon=state.polygon.replace(
            angular_velocity=polygon_angular_velocity,
        ),
        circle=state.circle.replace(
            angular_velocity=circle_angular_velocity,
        ),
    )


class ResidualTorqueEnv(KinetixEnv):
    """Standup environment whose policy action is residual torque in N*m."""

    def __init__(
        self,
        observation_type: KinetixObservation,
        static_env_params: StaticEnvParams,
        env_params: EnvParams,
        reset_function: Optional[Callable[[chex.PRNGKey], EnvState]] = None,
        physics_engine: Optional[PhysicsEngine] = None,
        auto_reset: bool = True,
        residual_torque_limit_nm: float = DEFAULT_RESIDUAL_TORQUE_LIMIT_NM,
        total_torque_limit_nm: float = DEFAULT_TOTAL_TORQUE_LIMIT_NM,
        energy_penalty_coefficient: float = DEFAULT_ENERGY_PENALTY_COEFFICIENT,
    ):
        if static_env_params.num_motor_bindings < NUM_STANDUP_JOINTS:
            raise ValueError(
                "Residual standup control requires at least "
                f"{NUM_STANDUP_JOINTS} motor bindings; got "
                f"{static_env_params.num_motor_bindings}."
            )
        if residual_torque_limit_nm <= 0:
            raise ValueError("residual_torque_limit_nm must be greater than zero")
        if total_torque_limit_nm <= 0:
            raise ValueError("total_torque_limit_nm must be greater than zero")
        if energy_penalty_coefficient < 0:
            raise ValueError("energy_penalty_coefficient must be non-negative")

        action_type = ResidualTorqueActions(
            env_params,
            static_env_params,
            residual_torque_limit_nm,
        )
        super().__init__(
            action_type=action_type,
            observation_type=observation_type,
            static_env_params=static_env_params,
            reset_function=reset_function,
            physics_engine=physics_engine,
            auto_reset=auto_reset,
        )
        self.residual_torque_limit_nm = float(residual_torque_limit_nm)
        self.total_torque_limit_nm = float(total_torque_limit_nm)
        self.energy_penalty_coefficient = float(energy_penalty_coefficient)
        self._physics_noop_action = jnp.zeros(
            static_env_params.num_joints + static_env_params.num_thrusters,
            dtype=jnp.float32,
        )

    def reset_env(
        self,
        rng: chex.PRNGKey,
        env_params: EnvParams,
        override_reset_state: Optional[EnvState],
    ):
        del env_params
        _, reset_key, cutout_key = jax.random.split(rng, 3)
        if override_reset_state is not None:
            state = override_reset_state
        elif self.reset_function is not None:
            state = self.reset_function(reset_key)
        else:
            raise NotImplementedError("No reset function provided")

        if isinstance(state, ResidualTorqueEnvState):
            state = state.replace(cutout_key=cutout_key)
        else:
            state = ResidualTorqueEnvState(
                **state.__dict__,
                cutout_key=cutout_key,
            )

        return self.get_obs(state), state

    def step_env(
        self,
        rng: chex.PRNGKey,
        state: ResidualTorqueEnvState,
        action: jax.Array,
        env_params: EnvParams,
    ):
        del rng
        residual_torque_nm = self.action_type.process_action(
            action,
            state,
            self.static_env_params,
        )
        pd_torque_nm = stand_pd_with_cutout(
            state,
            self.static_env_params,
            state.cutout_key,
        )
        total_torque_nm = jnp.clip(
            residual_torque_nm + pd_torque_nm,
            -self.total_torque_limit_nm,
            self.total_torque_limit_nm,
        )

        motor_binding_torque_nm = jnp.zeros(
            self.static_env_params.num_motor_bindings,
            dtype=total_torque_nm.dtype,
        )
        motor_binding_torque_nm = motor_binding_torque_nm.at[:NUM_STANDUP_JOINTS].set(
            total_torque_nm
        )

        observation, state, reward, done, info = self._torque_engine_step(
            state,
            motor_binding_torque_nm,
            env_params,
        )
        info["residual_torque_nm"] = residual_torque_nm
        info["pd_torque_nm"] = pd_torque_nm
        info["total_torque_nm"] = total_torque_nm
        return observation, state, reward, done, info

    def _torque_engine_step(
        self,
        state: ResidualTorqueEnvState,
        motor_binding_torque_nm: jax.Array,
        env_params: EnvParams,
    ):
        """Run one control step, holding the commanded torque per substep."""

        def _single_step(current_state, unused):
            del unused
            joint_velocity_rad_s = get_standup_joint_state(
                current_state,
                self.static_env_params,
            ).angular_velocity_rad_s
            mechanical_energy_j = (
                jnp.sum(
                    jnp.abs(
                        motor_binding_torque_nm[:NUM_STANDUP_JOINTS]
                        * joint_velocity_rad_s
                    )
                )
                * env_params.dt
            )

            current_state = apply_revolute_joint_torques_nm(
                current_state,
                motor_binding_torque_nm,
                env_params.dt,
                self.static_env_params,
            )
            current_state, manifolds = self.physics_engine.step(
                current_state,
                env_params,
                self._physics_noop_action,
            )
            reward, info = self.compute_reward_info(current_state, manifolds)
            done = reward != 0
            return current_state, (reward, done, info, mechanical_energy_j)

        state, (rewards, dones, infos, mechanical_energy_per_substep_j) = jax.lax.scan(
            _single_step,
            state,
            xs=None,
            length=self.static_env_params.frame_skip,
        )
        state = state.replace(timestep=state.timestep + 1)

        has_at_least_one_done = dones.sum() > 0
        first_done_index = dones.argmax()
        reward = jax.lax.select(
            has_at_least_one_done,
            rewards[first_done_index],
            rewards.sum(),
        )
        done = has_at_least_one_done | jax.tree.reduce(
            jnp.logical_or,
            jax.tree.map(lambda x: jnp.isnan(x).any(), state),
            False,
        )
        done |= state.timestep >= env_params.max_timesteps

        info = jax.tree.map(
            lambda x: jax.lax.select(
                has_at_least_one_done,
                x[first_done_index],
                x[-1],
            ),
            infos,
        )

        delta_dist = (
            -(info["distance"] - state.last_distance) * env_params.dense_reward_scale
        )
        delta_dist = jnp.nan_to_num(
            delta_dist,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        reward += jax.lax.select(
            (state.last_distance == -1) | (env_params.dense_reward_scale == 0.0),
            0.0,
            delta_dist,
        )

        # Count work only up to the first terminal physics substep. Absolute
        # mechanical work treats both positive work and braking as energy use.
        final_substep = jax.lax.select(
            has_at_least_one_done,
            first_done_index,
            self.static_env_params.frame_skip - 1,
        )
        energy_substep_mask = (
            jnp.arange(self.static_env_params.frame_skip) <= final_substep
        )
        mechanical_energy_j = jnp.sum(
            mechanical_energy_per_substep_j * energy_substep_mask
        )
        energy_penalty = self.energy_penalty_coefficient * mechanical_energy_j
        reward -= energy_penalty
        info["mechanical_energy_j"] = mechanical_energy_j
        info["energy_penalty"] = energy_penalty

        distance = jax.lax.select(done, -1.0, info["distance"])
        state = state.replace(last_distance=distance)

        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def __hash__(self):
        return hash(
            (
                super().__hash__(),
                self.residual_torque_limit_nm,
                self.total_torque_limit_nm,
                self.energy_penalty_coefficient,
            )
        )

    def __eq__(self, value):
        return isinstance(value, ResidualTorqueEnv) and hash(self) == hash(value)


def make_residual_torque_env(
    observation_type: ObservationType,
    reset_fn: Optional[Callable[[chex.PRNGKey], EnvState]],
    env_params: Optional[EnvParams] = None,
    static_env_params: Optional[StaticEnvParams] = None,
    auto_reset: bool = True,
    residual_torque_limit_nm: float = DEFAULT_RESIDUAL_TORQUE_LIMIT_NM,
    total_torque_limit_nm: float = DEFAULT_TOTAL_TORQUE_LIMIT_NM,
    energy_penalty_coefficient: float = DEFAULT_ENERGY_PENALTY_COEFFICIENT,
) -> ResidualTorqueEnv:
    """Create a residual-torque standup environment.

    This mirrors ``make_kinetix_env`` except that the action type is fixed to a
    three-dimensional continuous residual torque.  Existing RL code should
    continue to set its configuration action type to ``continuous`` so the
    policy uses a Gaussian continuous-action head; the network obtains the
    new action dimension from ``env.action_space(env_params).shape``.
    """
    if env_params is None:
        env_params = EnvParams()
    if static_env_params is None:
        static_env_params = StaticEnvParams()

    # Reuse Kinetix's observation construction so every supported observation
    # type remains identical to the base environment.
    template_env = make_kinetix_env(
        action_type=ActionType.CONTINUOUS,
        observation_type=observation_type,
        reset_fn=reset_fn,
        env_params=env_params,
        static_env_params=static_env_params,
        auto_reset=auto_reset,
    )
    return ResidualTorqueEnv(
        observation_type=template_env.observation_type,
        static_env_params=static_env_params,
        env_params=env_params,
        reset_function=reset_fn,
        physics_engine=template_env.physics_engine,
        auto_reset=auto_reset,
        residual_torque_limit_nm=residual_torque_limit_nm,
        total_torque_limit_nm=total_torque_limit_nm,
        energy_penalty_coefficient=energy_penalty_coefficient,
    )
