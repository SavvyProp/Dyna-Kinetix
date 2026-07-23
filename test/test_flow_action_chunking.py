"""Tests for pixel action-chunk loading and rectified-flow training."""

import json
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dynak.flow_action_chunking import (
    FlowMatchingPolicy,
    FlowModelConfig,
    PixelActionChunkDataset,
    flow_matching_loss,
    sample_action_chunks,
)
from dynak.scripts.train_flow_action_chunking import parse_args, train
from kinetix.util.saving import load_params

CONTROLLERS = ("no_controller", "pd", "bang_bang")


def write_controller_dataset(
    root: Path,
    controller: str,
    residual_torque_limit_nm: float | None = None,
) -> None:
    controller_dir = root / controller
    controller_dir.mkdir(parents=True)
    num_episodes = 4
    max_timesteps = 6
    lengths = np.asarray([3, 4, 5, 6], dtype=np.int32)
    image = np.zeros((num_episodes, max_timesteps, 32, 32, 3), dtype=np.uint8)
    residual_torque = np.zeros((num_episodes, max_timesteps, 3), dtype=np.float32)
    controller_offset = CONTROLLERS.index(controller) * 0.25
    for episode_index in range(num_episodes):
        for timestep in range(max_timesteps):
            image[episode_index, timestep] = 10 * episode_index + timestep
            residual_torque[episode_index, timestep] = (
                controller_offset + 0.1 * timestep
            )

    shard_name = "shard_00000.npz"
    np.savez_compressed(
        controller_dir / shard_name,
        image=image,
        residual_torque_nm=residual_torque,
        episode_length=lengths,
    )
    manifest = {
        "schema_version": 2,
        "controller": controller,
        "observation_type": "pixels",
        "image_shape": [32, 32, 3],
        "action_dim": 3,
        "pd_gain_randomization_fraction": 0.2,
        "bang_bang_torque_randomization_fraction": 0.2,
        "controller_torque_noise_std_nm": 0.2,
        "shards": [
            {
                "file": shard_name,
                "episodes": num_episodes,
                "transitions": int(lengths.sum()),
            }
        ],
    }
    if residual_torque_limit_nm is not None:
        manifest["residual_torque_limit_nm"] = residual_torque_limit_nm
    (controller_dir / "manifest.json").write_text(json.dumps(manifest))


def write_datasets(root: Path) -> None:
    for controller in CONTROLLERS:
        write_controller_dataset(root, controller)


class TestFlowActionChunking(unittest.TestCase):
    def test_shared_normalization_accepts_smaller_controller_limits(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_controller_dataset(root, "no_controller", 10.0)
            write_controller_dataset(root, "pd", 5.0)
            write_controller_dataset(root, "bang_bang", 5.0)

            dataset = PixelActionChunkDataset(
                root,
                residual_torque_limit_nm=10.0,
            )
            self.assertEqual(dataset.residual_torque_limit_nm, 10.0)

            with self.assertRaisesRegex(ValueError, "exceeds"):
                PixelActionChunkDataset(
                    root,
                    residual_torque_limit_nm=5.0,
                )

    def test_pixel_dataset_produces_masked_normalized_chunks(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_datasets(root)
            dataset = PixelActionChunkDataset(
                root,
                split="train",
                validation_fraction=0.25,
                frame_stack=2,
                action_horizon=8,
                residual_torque_limit_nm=2.5,
            )
            batch = dataset.sample_batch(6, np.random.default_rng(0))

            self.assertEqual(batch.images.shape, (6, 2, 32, 32, 3))
            self.assertEqual(batch.images.dtype, np.float32)
            self.assertGreaterEqual(float(batch.images.min()), 0.0)
            self.assertLessEqual(float(batch.images.max()), 1.0)
            self.assertEqual(batch.actions.shape, (6, 8, 3))
            self.assertEqual(batch.action_mask.shape, (6, 8))
            self.assertTrue(np.all(batch.actions[~batch.action_mask] == 0.0))
            self.assertLessEqual(float(np.abs(batch.actions).max()), 1.0)

    def test_model_loss_gradients_and_base_sampler(self):
        config = FlowModelConfig(
            channel_dim=32,
            token_mixing_hidden_dim=16,
            channel_mixing_hidden_dim=64,
            num_mixer_blocks=2,
            time_embedding_dim=32,
            frame_embedding_dim=16,
            observation_embedding_dim=32,
        )
        model = FlowMatchingPolicy(config)
        images = jnp.ones((2, 1, 32, 32, 3), dtype=jnp.float32)
        actions = jnp.zeros((2, 8, 3), dtype=jnp.float32)
        action_mask = jnp.ones((2, 8), dtype=bool)
        params = model.init(
            jax.random.PRNGKey(0),
            images,
            actions,
            jnp.zeros((2,), dtype=jnp.float32),
        )["params"]

        loss, _ = flow_matching_loss(
            model,
            params,
            images,
            actions,
            action_mask,
            jax.random.PRNGKey(1),
        )
        gradients = jax.grad(
            lambda candidate_params: flow_matching_loss(
                model,
                candidate_params,
                images,
                actions,
                action_mask,
                jax.random.PRNGKey(1),
            )[0]
        )(params)
        token_time_velocity = model.apply(
            {"params": params},
            images,
            actions,
            jnp.zeros((2, 8), dtype=jnp.float32),
        )
        samples = sample_action_chunks(
            model,
            params,
            images,
            jax.random.PRNGKey(2),
            num_steps=2,
        )

        self.assertTrue(bool(jnp.isfinite(loss)))
        self.assertGreater(float(optax.global_norm(gradients)), 0.0)
        self.assertEqual(token_time_velocity.shape, (2, 8, 3))
        self.assertEqual(samples.shape, (2, 8, 3))
        self.assertTrue(bool(jnp.all(jnp.isfinite(samples))))

    def test_training_script_writes_self_describing_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "rollouts"
            output_dir = root / "checkpoint"
            write_datasets(dataset_root)
            args = parse_args(
                [
                    "--dataset-root",
                    str(dataset_root),
                    "--output-dir",
                    str(output_dir),
                    "--epochs",
                    "1",
                    "--steps-per-epoch",
                    "1",
                    "--batch-size",
                    "3",
                    "--validation-fraction",
                    "0.25",
                    "--validation-batches",
                    "1",
                    "--checkpoint-every",
                    "1",
                    "--warmup-steps",
                    "0",
                    "--channel-dim",
                    "32",
                    "--token-mixing-hidden-dim",
                    "16",
                    "--channel-mixing-hidden-dim",
                    "64",
                    "--num-mixer-blocks",
                    "1",
                    "--time-embedding-dim",
                    "32",
                    "--frame-embedding-dim",
                    "16",
                    "--observation-embedding-dim",
                    "32",
                ]
            )

            checkpoint_path = train(args)
            checkpoint = load_params(checkpoint_path)

            self.assertEqual(checkpoint["format_version"], 1)
            self.assertEqual(checkpoint["step"], 1)
            self.assertEqual(checkpoint["model_config"]["action_horizon"], 8)
            self.assertEqual(
                checkpoint["data_config"]["controllers"],
                list(CONTROLLERS),
            )
            self.assertEqual(
                checkpoint["data_config"]["observation_inputs"],
                ["image"],
            )
            self.assertEqual(
                checkpoint["data_config"]["residual_torque_limit_nm"],
                10.0,
            )
            self.assertEqual(
                checkpoint["data_config"]["pd_gain_randomization_fraction"],
                0.2,
            )
            self.assertEqual(
                checkpoint["data_config"]["bang_bang_torque_randomization_fraction"],
                0.2,
            )
            self.assertEqual(
                checkpoint["data_config"]["controller_torque_noise_std_nm"],
                0.2,
            )
            self.assertTrue((output_dir / "metrics.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
