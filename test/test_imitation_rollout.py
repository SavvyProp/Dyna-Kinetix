"""Unit tests for residual standup imitation dataset sharding."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from dynak.scripts.imitation_rollout_common import (
    TRANSITION_FIELDS,
    pack_shard,
    save_npz_atomic,
    successful_episode_mask,
)


def make_episode(length: int = 3, max_timesteps: int = 5):
    valid = np.arange(max_timesteps) < length
    episode = {
        "observation": np.ones((max_timesteps, 7), dtype=np.float32),
        "policy_action": np.ones((max_timesteps, 3), dtype=np.float32),
        "residual_torque_nm": np.ones((max_timesteps, 3), dtype=np.float32),
        "underlying_torque_nm": np.full((max_timesteps, 3), 2.0, dtype=np.float32),
        "total_torque_nm": np.full((max_timesteps, 3), 3.0, dtype=np.float32),
        "reward": np.where(valid, 0.25, 0.0).astype(np.float32),
        "done": np.arange(max_timesteps) == length - 1,
        "valid_mask": valid,
        "success": np.arange(max_timesteps) == length - 1,
        "goal_inside": valid,
        "goal_inside_reward": np.where(valid, 0.1, 0.0).astype(np.float32),
        "goal_steady": valid,
        "goal_hold_time_seconds": np.zeros(max_timesteps, dtype=np.float32),
        "goal_max_linear_speed_mps": np.zeros(max_timesteps, dtype=np.float32),
        "goal_max_angular_speed_rad_s": np.zeros(max_timesteps, dtype=np.float32),
    }
    return episode


def make_pixel_episode(length: int = 3, max_timesteps: int = 5):
    episode = make_episode(length, max_timesteps)
    del episode["observation"]
    episode["image"] = np.full(
        (max_timesteps, 16, 16, 3),
        127,
        dtype=np.uint8,
    )
    episode["global_info"] = np.zeros((max_timesteps, 1), dtype=np.float32)
    return episode


class TestImitationRollout(unittest.TestCase):
    def test_success_requires_goal_on_a_valid_transition(self):
        episode = {key: value[None, ...] for key, value in make_episode().items()}
        np.testing.assert_array_equal(successful_episode_mask(episode), [True])

        episode["valid_mask"][:] = False
        np.testing.assert_array_equal(successful_episode_mask(episode), [False])

    def test_pack_shard_preserves_aligned_torques_and_summaries(self):
        first = make_episode(length=3)
        second = make_episode(length=4)
        shard = pack_shard([first, second])

        for field in ("observation",) + TRANSITION_FIELDS:
            self.assertEqual(shard[field].shape[0], 2)
        np.testing.assert_array_equal(shard["episode_length"], [3, 4])
        np.testing.assert_allclose(shard["episode_return"], [0.75, 1.0])
        np.testing.assert_allclose(
            shard["total_torque_nm"],
            shard["residual_torque_nm"] + shard["underlying_torque_nm"],
        )

    def test_atomic_npz_contains_expected_arrays(self):
        shard = pack_shard([make_episode()])
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "shard_00000.npz"
            save_npz_atomic(path, shard)
            self.assertTrue(path.exists())
            with np.load(path) as stored:
                self.assertEqual(set(stored.files), set(shard))
                np.testing.assert_array_equal(
                    stored["underlying_torque_nm"],
                    shard["underlying_torque_nm"],
                )

    def test_pack_pixel_shard_keeps_uint8_images(self):
        shard = pack_shard([make_pixel_episode()])
        self.assertNotIn("observation", shard)
        self.assertEqual(shard["image"].shape, (1, 5, 16, 16, 3))
        self.assertEqual(shard["image"].dtype, np.uint8)
        self.assertEqual(shard["global_info"].shape, (1, 5, 1))


if __name__ == "__main__":
    unittest.main()
