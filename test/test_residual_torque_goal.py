"""Regression tests for the non-colliding steady-hold standup goal."""

import json
import os
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dynak-matplotlib")

import jax
import jax.numpy as jnp
import numpy as np

from dynak.standup.residual_torque_env import make_residual_torque_env
from kinetix.environment import ObservationType
from kinetix.util import load_from_json_file

LEVEL_NAME = "l/standup_goal.json"
LEVEL_PATH = Path(__file__).resolve().parents[1] / "kinetix" / "levels" / LEVEL_NAME


class TestResidualTorqueGoal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        state, static_env_params, env_params = load_from_json_file(LEVEL_NAME)
        cls.env_params = env_params
        cls.env = make_residual_torque_env(
            observation_type=ObservationType.SYMBOLIC_FLAT,
            reset_fn=lambda _rng: state,
            env_params=env_params,
            static_env_params=static_env_params,
            auto_reset=False,
            # Strictly more than two physics steps and less than three.
            goal_hold_duration_seconds=2.5 * float(env_params.dt),
        )
        _, cls.state = cls.env.reset_env(
            jax.random.PRNGKey(0),
            env_params,
            state,
        )
        cls.target_index = int(
            np.flatnonzero(np.asarray(state.polygon_shape_roles) == 1)[0]
        )
        cls.goal_index = int(
            np.flatnonzero(np.asarray(state.polygon_shape_roles) == 2)[0]
        )

    def _state_with_target_in_goal(self):
        goal_position = self.state.polygon.position[self.goal_index]
        return self.state.replace(
            polygon=self.state.polygon.replace(
                position=self.state.polygon.position.at[self.target_index].set(
                    goal_position
                )
            )
        )

    def test_goal_region_is_non_colliding_in_json(self):
        with LEVEL_PATH.open() as level_file:
            level = json.load(level_file)

        goal_regions = [
            polygon
            for polygon in level["env_state"]["polygon"]
            if polygon["active"] and polygon["role"] == 2
        ]
        self.assertEqual(len(goal_regions), 1)
        self.assertEqual(goal_regions[0]["collision_mode"], 0)

    def test_goal_uses_containment_instead_of_collision(self):
        outside, steady, _, _ = self.env._standup_goal_metrics(self.state)
        self.assertFalse(bool(outside))
        self.assertTrue(bool(steady))

        inside_state = self._state_with_target_in_goal()
        inside, steady, linear_speed, angular_speed = self.env._standup_goal_metrics(
            inside_state
        )
        self.assertTrue(bool(inside))
        self.assertTrue(bool(steady))
        self.assertEqual(float(linear_speed), 0.0)
        self.assertEqual(float(angular_speed), 0.0)

    def test_motion_resets_consecutive_hold_progress(self):
        state = self._state_with_target_in_goal()
        update_goal_progress = jax.jit(self.env._update_goal_progress)

        state, reached, _ = update_goal_progress(state, self.env_params)
        self.assertEqual(int(state.goal_hold_steps), 1)
        self.assertFalse(bool(reached))

        moving_state = state.replace(
            polygon=state.polygon.replace(
                velocity=state.polygon.velocity.at[self.target_index].set(
                    jnp.array([1.01, 0.0], dtype=jnp.float32)
                )
            )
        )
        moving_state, reached, info = update_goal_progress(
            moving_state,
            self.env_params,
        )
        self.assertEqual(int(moving_state.goal_hold_steps), 0)
        self.assertFalse(bool(reached))
        self.assertFalse(bool(info["goal_steady"]))
        self.assertAlmostEqual(
            float(info["goal_inside_reward"]),
            float(self.env_params.dt),
            places=6,
        )

    def test_default_goal_succeeds_despite_high_angular_speed(self):
        env = make_residual_torque_env(
            observation_type=ObservationType.SYMBOLIC_FLAT,
            reset_fn=lambda _rng: self.state,
            env_params=self.env_params,
            static_env_params=self.env.static_env_params,
            auto_reset=False,
        )
        state = self._state_with_target_in_goal()
        state = state.replace(
            polygon=state.polygon.replace(
                velocity=state.polygon.velocity.at[self.target_index].set(
                    jnp.array([1.0, 0.0], dtype=jnp.float32)
                ),
                angular_velocity=state.polygon.angular_velocity.at[
                    self.target_index
                ].set(10.0),
            )
        )

        state, reached, info = env._update_goal_progress(state, self.env_params)

        self.assertEqual(int(state.goal_hold_steps), 1)
        self.assertEqual(int(info["goal_required_hold_steps"]), 1)
        self.assertTrue(bool(info["goal_steady"]))
        self.assertTrue(bool(reached))
        self.assertEqual(float(info["goal_max_angular_speed_rad_s"]), 10.0)

    def test_success_requires_full_hold_duration(self):
        state = self._state_with_target_in_goal()

        for expected_steps in (1, 2):
            state, reached, info = self.env._update_goal_progress(
                state,
                self.env_params,
            )
            self.assertEqual(int(state.goal_hold_steps), expected_steps)
            self.assertFalse(bool(reached))
            self.assertEqual(int(info["goal_required_hold_steps"]), 3)

        state, reached, info = self.env._update_goal_progress(
            state,
            self.env_params,
        )
        self.assertEqual(int(state.goal_hold_steps), 3)
        self.assertTrue(bool(reached))
        self.assertAlmostEqual(
            float(info["goal_hold_time_seconds"]),
            3 * float(self.env_params.dt),
            places=6,
        )

    def test_environment_emits_success_after_steady_hold(self):
        # Keep the already-consistent initial arm pose motionless and move the
        # sensor region onto its end effector.  This isolates termination from
        # the controller while still exercising a complete physics/env step.
        target_position = self.state.polygon.position[self.target_index]
        state = self.state.replace(
            gravity=jnp.zeros_like(self.state.gravity),
            polygon=self.state.polygon.replace(
                position=self.state.polygon.position.at[self.goal_index].set(
                    target_position
                )
            ),
            joint=self.state.joint.replace(
                motor_on=jnp.zeros_like(self.state.joint.motor_on)
            ),
        )
        key = jax.random.PRNGKey(1)

        for expected_steps in (1, 2):
            key, step_key = jax.random.split(key)
            _, state, reward, done, info = self.env.step(
                step_key,
                state,
                self.env.action_type.noop_action(),
                self.env_params,
            )
            self.assertEqual(int(state.goal_hold_steps), expected_steps)
            self.assertFalse(bool(done))
            self.assertFalse(bool(info["GoalR"]))
            self.assertEqual(np.asarray(info["controller_torque_nm"]).shape, (3,))
            self.assertAlmostEqual(
                float(info["goal_inside_reward"]),
                float(self.env_params.dt),
                places=6,
            )
            self.assertAlmostEqual(
                float(reward),
                float(self.env_params.dt),
                places=5,
            )

        key, step_key = jax.random.split(key)
        _, state, reward, done, info = self.env.step(
            step_key,
            state,
            self.env.action_type.noop_action(),
            self.env_params,
        )
        self.assertEqual(int(state.goal_hold_steps), 3)
        self.assertTrue(bool(done))
        self.assertTrue(bool(info["GoalR"]))
        self.assertAlmostEqual(
            float(reward),
            1.0 + float(self.env_params.dt),
            places=5,
        )

    def test_outside_goal_has_no_inside_reward(self):
        _, reached, info = self.env._update_goal_progress(
            self.state,
            self.env_params,
        )
        self.assertFalse(bool(reached))
        self.assertFalse(bool(info["goal_inside"]))
        self.assertEqual(float(info["goal_inside_reward"]), 0.0)


if __name__ == "__main__":
    unittest.main()
