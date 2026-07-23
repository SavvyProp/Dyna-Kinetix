"""Tests for reusable flow-policy evaluation helpers."""

import unittest
from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np

from dynak.flow_action_chunking import FlowMatchingPolicy, FlowModelConfig
from dynak.imitation_rollout import (
    RESIDUAL_CONTROLLER_NAMES,
    append_image_history,
    flow_policy_from_checkpoint,
    initialize_image_history,
    make_batched_flow_rollout_function,
    make_controller_flow_rollout_functions,
    make_flow_batch_action_function,
    make_flow_evaluation_env,
)
from kinetix.util.saving import load_from_json_file


def make_checkpoint(image_shape, frame_stack=2, action_horizon=2):
    config = FlowModelConfig(
        action_horizon=action_horizon,
        frame_stack=frame_stack,
        channel_dim=16,
        token_mixing_hidden_dim=8,
        channel_mixing_hidden_dim=32,
        num_mixer_blocks=1,
        time_embedding_dim=16,
        frame_embedding_dim=8,
        observation_embedding_dim=16,
    )
    model = FlowMatchingPolicy(config)
    images = jnp.zeros((1, frame_stack, *image_shape), dtype=jnp.float32)
    actions = jnp.zeros((1, action_horizon, 3), dtype=jnp.float32)
    params = model.init(
        jax.random.PRNGKey(0),
        images,
        actions,
        jnp.zeros((1,), dtype=jnp.float32),
    )["params"]
    return {
        "params": params,
        "model_config": asdict(config),
        "data_config": {
            "observation_inputs": ["image"],
            "image_shape": list(image_shape),
            "residual_torque_limit_nm": 5.0,
        },
    }


class TestFlowEvaluation(unittest.TestCase):
    def test_checkpoint_history_and_action_inference(self):
        checkpoint = make_checkpoint((32, 32, 3))
        policy = flow_policy_from_checkpoint(checkpoint)
        first_image = jnp.zeros((32, 32, 3), dtype=jnp.float32)
        next_image = jnp.ones((32, 32, 3), dtype=jnp.float32)

        history = initialize_image_history(first_image, frame_stack=2)
        history = append_image_history(history, next_image)
        sample_action = jax.jit(
            make_flow_batch_action_function(policy, num_flow_steps=2)
        )
        residual_torque, chunks = sample_action(
            history[None, ...],
            jax.random.PRNGKey(1),
        )

        self.assertEqual(history.shape, (2, 32, 32, 3))
        self.assertEqual(policy.pd_gain_randomization_fraction, 0.2)
        self.assertEqual(policy.bang_bang_torque_randomization_fraction, 0.2)
        self.assertEqual(policy.controller_torque_noise_std_nm, 0.2)
        np.testing.assert_allclose(np.asarray(history[0]), 0.0)
        np.testing.assert_allclose(np.asarray(history[1]), 1.0)
        self.assertEqual(residual_torque.shape, (1, 3))
        self.assertEqual(chunks.shape, (1, 2, 3))
        self.assertLessEqual(float(jnp.abs(residual_torque).max()), 5.0)

    def test_controller_factory_covers_all_four_residual_environments(self):
        initial_level, static_env_params, env_params = load_from_json_file(
            "l/standup_goal.json"
        )
        image_shape = tuple(
            dimension // static_env_params.downscale
            for dimension in static_env_params.screen_dim
        ) + (3,)
        checkpoint = make_checkpoint(image_shape, frame_stack=1)

        rollout_functions = make_controller_flow_rollout_functions(
            checkpoint,
            initial_level,
            static_env_params,
            env_params.replace(max_timesteps=2),
            num_flow_steps=1,
        )

        self.assertEqual(tuple(rollout_functions), RESIDUAL_CONTROLLER_NAMES)

    def test_flow_environments_use_controller_specific_actor_limits(self):
        initial_level, static_env_params, env_params = load_from_json_file(
            "l/standup_goal.json"
        )
        image_shape = tuple(
            dimension // static_env_params.downscale
            for dimension in static_env_params.screen_dim
        ) + (3,)
        policy = flow_policy_from_checkpoint(
            make_checkpoint(image_shape, frame_stack=1)
        )

        no_controller_env = make_flow_evaluation_env(
            policy,
            initial_level,
            static_env_params,
            env_params,
            "none",
        )
        pd_env = make_flow_evaluation_env(
            policy,
            initial_level,
            static_env_params,
            env_params,
            "pd",
        )

        self.assertEqual(no_controller_env.residual_torque_limit_nm, 10.0)
        self.assertEqual(pd_env.residual_torque_limit_nm, 5.0)

    def test_batched_rollout_reuses_chunk_for_execute_horizon(self):
        initial_level, static_env_params, env_params = load_from_json_file(
            "l/standup_goal.json"
        )
        env_params = env_params.replace(max_timesteps=3)
        image_shape = tuple(
            dimension // static_env_params.downscale
            for dimension in static_env_params.screen_dim
        ) + (3,)
        checkpoint = make_checkpoint(image_shape, frame_stack=1)
        rollout = make_batched_flow_rollout_function(
            checkpoint,
            initial_level,
            static_env_params,
            env_params,
            "none",
            num_flow_steps=1,
            execute_horizon=2,
        )

        trajectory = jax.device_get(rollout(jax.random.split(jax.random.PRNGKey(2), 1)))

        self.assertEqual(trajectory["reward"].shape, (1, 3))
        np.testing.assert_array_equal(
            trajectory["chunk_action_index"][0],
            [0, 1, 0],
        )
        np.testing.assert_allclose(
            trajectory["action_chunk_normalized"][0, 0],
            trajectory["action_chunk_normalized"][0, 1],
        )
        np.testing.assert_allclose(trajectory["underlying_torque_nm"], 0.0)
        self.assertTrue(np.all(trajectory["valid_mask"]))


if __name__ == "__main__":
    unittest.main()
