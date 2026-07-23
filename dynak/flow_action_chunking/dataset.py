"""Balanced pixel/action-chunk sampling from imitation rollout shards."""

from __future__ import annotations

import json
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np

DEFAULT_CONTROLLERS = ("no_controller", "pd", "bang_bang")


class ActionChunkBatch(NamedTuple):
    """One behavior-cloning batch.

    Images have shape ``(B, frame_stack, height, width, 3)`` and values in
    ``[0, 1]``. Actions have shape ``(B, horizon, action_dim)`` and are
    normalized by the configured residual torque limit. ``action_mask`` is
    false for action tokens padded past episode termination.
    """

    images: np.ndarray
    actions: np.ndarray
    action_mask: np.ndarray


@dataclass
class _ShardSelection:
    path: Path
    episode_indices: np.ndarray
    episode_lengths: np.ndarray

    @property
    def num_examples(self) -> int:
        return int(self.episode_lengths.sum())


class PixelActionChunkDataset:
    """Sample complete image histories and residual-torque chunks.

    Controller directories are sampled uniformly, so one controller cannot
    dominate training merely because it produced longer or more numerous
    episodes. Within a controller, starts are sampled approximately uniformly
    over its valid transitions.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        controllers: Sequence[str] = DEFAULT_CONTROLLERS,
        *,
        split: str = "train",
        validation_fraction: float = 0.1,
        frame_stack: int = 1,
        action_horizon: int = 8,
        residual_torque_limit_nm: float = 5.0,
        cache_size: int = 3,
        shard_reuse_batches: int = 32,
    ):
        if split not in ("train", "validation"):
            raise ValueError("split must be 'train' or 'validation'")
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if frame_stack <= 0:
            raise ValueError("frame_stack must be greater than zero")
        if action_horizon <= 0:
            raise ValueError("action_horizon must be greater than zero")
        if residual_torque_limit_nm <= 0:
            raise ValueError("residual_torque_limit_nm must be greater than zero")
        if cache_size <= 0:
            raise ValueError("cache_size must be greater than zero")
        if shard_reuse_batches <= 0:
            raise ValueError("shard_reuse_batches must be greater than zero")

        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.controllers = tuple(controllers)
        self.split = split
        self.validation_fraction = float(validation_fraction)
        self.frame_stack = int(frame_stack)
        self.action_horizon = int(action_horizon)
        self.residual_torque_limit_nm = float(residual_torque_limit_nm)
        self.cache_size = max(int(cache_size), len(self.controllers))
        self.shard_reuse_batches = int(shard_reuse_batches)
        self._cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()
        self._selections: dict[str, list[_ShardSelection]] = {}
        self._active_selections: dict[str, tuple[_ShardSelection, int]] = {}
        self.pd_gain_randomization_fraction: float | None = None
        self.bang_bang_torque_randomization_fraction: float | None = None
        self.controller_torque_noise_std_nm: float | None = None

        if not self.controllers:
            raise ValueError("At least one controller dataset is required")

        image_shape = None
        action_dim = None
        for controller in self.controllers:
            (
                selections,
                controller_image_shape,
                controller_action_dim,
                pd_randomization,
                bang_bang_randomization,
                torque_noise_std_nm,
            ) = self._index_controller(controller)
            if not selections:
                raise ValueError(
                    f"No {split} episodes found for controller {controller!r}"
                )
            self._selections[controller] = selections
            if image_shape is None:
                image_shape = controller_image_shape
                action_dim = controller_action_dim
            elif image_shape != controller_image_shape:
                raise ValueError(
                    "All controller datasets must use the same image shape; "
                    f"got {image_shape} and {controller_image_shape}"
                )
            elif action_dim != controller_action_dim:
                raise ValueError(
                    "All controller datasets must use the same action dimension"
                )
            if self.pd_gain_randomization_fraction is None:
                self.pd_gain_randomization_fraction = pd_randomization
                self.bang_bang_torque_randomization_fraction = bang_bang_randomization
                self.controller_torque_noise_std_nm = torque_noise_std_nm
            elif (
                not np.isclose(
                    self.pd_gain_randomization_fraction,
                    pd_randomization,
                )
                or not np.isclose(
                    self.bang_bang_torque_randomization_fraction,
                    bang_bang_randomization,
                )
                or not np.isclose(
                    self.controller_torque_noise_std_nm,
                    torque_noise_std_nm,
                )
            ):
                raise ValueError(
                    "All controller datasets must use the same controller "
                    "randomization and noise settings"
                )

        self.image_shape = tuple(image_shape)
        self.action_dim = int(action_dim)
        self.total_examples = sum(
            selection.num_examples
            for selections in self._selections.values()
            for selection in selections
        )

    def _index_controller(
        self,
        controller: str,
    ) -> tuple[list[_ShardSelection], tuple[int, ...], int, float, float, float]:
        controller_dir = self.dataset_root / controller
        manifest_path = controller_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing rollout manifest for {controller!r}: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text())
        manifest_torque_limit_nm = float(
            manifest.get(
                "residual_torque_limit_nm",
                self.residual_torque_limit_nm,
            )
        )
        if not np.isclose(
            manifest_torque_limit_nm,
            self.residual_torque_limit_nm,
        ):
            raise ValueError(
                f"{controller_dir} was collected with residual_torque_limit_nm="
                f"{manifest_torque_limit_nm}, but the dataset loader requested "
                f"{self.residual_torque_limit_nm}"
            )
        if manifest.get("observation_type") != "pixels":
            raise ValueError(
                f"{controller_dir} contains {manifest.get('observation_type')!r} "
                "observations; retrain/collect the expert with pixel observations"
            )

        shard_names = [entry["file"] for entry in manifest.get("shards", [])]
        if not shard_names:
            shard_names = [
                path.name for path in sorted(controller_dir.glob("shard_*.npz"))
            ]
        if not shard_names:
            raise FileNotFoundError(f"No rollout shards found in {controller_dir}")

        metadata = []
        image_shape = None
        action_dim = None
        for shard_name in shard_names:
            path = controller_dir / shard_name
            if not path.is_file():
                raise FileNotFoundError(f"Manifest references missing shard: {path}")
            with np.load(path, allow_pickle=False) as shard:
                required = {"image", "residual_torque_nm", "episode_length"}
                missing = required.difference(shard.files)
                if missing:
                    raise ValueError(f"{path} is missing fields: {sorted(missing)}")
                lengths = np.asarray(shard["episode_length"], dtype=np.int32)
                current_image_shape = tuple(shard["image"].shape[2:])
                current_action_dim = int(shard["residual_torque_nm"].shape[-1])
            if np.any(lengths <= 0):
                raise ValueError(f"{path} contains an empty episode")
            if image_shape is None:
                image_shape = current_image_shape
                action_dim = current_action_dim
            elif image_shape != current_image_shape or action_dim != current_action_dim:
                raise ValueError(f"Inconsistent shapes in {path}")
            for episode_index, episode_length in enumerate(lengths):
                metadata.append((path, episode_index, int(episode_length)))

        number_of_episodes = len(metadata)
        number_of_validation_episodes = int(
            round(number_of_episodes * self.validation_fraction)
        )
        if self.validation_fraction > 0 and number_of_episodes > 1:
            number_of_validation_episodes = max(1, number_of_validation_episodes)
            number_of_validation_episodes = min(
                number_of_episodes - 1,
                number_of_validation_episodes,
            )

        split_index = number_of_episodes - number_of_validation_episodes
        selected_metadata = (
            metadata[:split_index] if self.split == "train" else metadata[split_index:]
        )
        grouped_indices = defaultdict(list)
        grouped_lengths = defaultdict(list)
        for path, episode_index, episode_length in selected_metadata:
            grouped_indices[path].append(episode_index)
            grouped_lengths[path].append(episode_length)

        selections = [
            _ShardSelection(
                path=path,
                episode_indices=np.asarray(grouped_indices[path], dtype=np.int32),
                episode_lengths=np.asarray(grouped_lengths[path], dtype=np.int32),
            )
            for path in grouped_indices
        ]
        return (
            selections,
            image_shape,
            action_dim,
            float(manifest.get("pd_gain_randomization_fraction", 0.2)),
            float(
                manifest.get(
                    "bang_bang_torque_randomization_fraction",
                    0.2,
                )
            ),
            float(manifest.get("controller_torque_noise_std_nm", 0.2)),
        )

    def _load_shard(self, path: Path) -> dict[str, np.ndarray]:
        cached = self._cache.pop(path, None)
        if cached is not None:
            self._cache[path] = cached
            return cached

        with np.load(path, allow_pickle=False) as shard:
            cached = {
                "image": np.asarray(shard["image"]),
                "residual_torque_nm": np.asarray(
                    shard["residual_torque_nm"],
                    dtype=np.float32,
                ),
            }
        self._cache[path] = cached
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return cached

    def _sample_controller(
        self,
        controller: str,
        count: int,
        rng: np.random.Generator,
    ) -> ActionChunkBatch:
        active_selection = self._active_selections.get(controller)
        if active_selection is None or active_selection[1] <= 0:
            selections = self._selections[controller]
            shard_weights = np.asarray(
                [selection.num_examples for selection in selections],
                dtype=np.float64,
            )
            shard_weights /= shard_weights.sum()
            selection = selections[int(rng.choice(len(selections), p=shard_weights))]
            remaining_uses = self.shard_reuse_batches - 1
        else:
            selection, remaining_uses = active_selection
            remaining_uses -= 1
        self._active_selections[controller] = (selection, remaining_uses)
        shard = self._load_shard(selection.path)

        episode_weights = selection.episode_lengths.astype(np.float64)
        episode_weights /= episode_weights.sum()
        local_episode_indices = rng.choice(
            len(selection.episode_indices),
            size=count,
            replace=True,
            p=episode_weights,
        )
        episode_indices = selection.episode_indices[local_episode_indices]
        episode_lengths = selection.episode_lengths[local_episode_indices]
        start_indices = np.asarray(
            [rng.integers(0, length) for length in episode_lengths],
            dtype=np.int32,
        )

        history_offsets = np.arange(
            self.frame_stack - 1,
            -1,
            -1,
            dtype=np.int32,
        )
        frame_indices = np.maximum(start_indices[:, None] - history_offsets, 0)
        images = shard["image"][episode_indices[:, None], frame_indices]
        if images.dtype == np.uint8:
            images = images.astype(np.float32) / 255.0
        else:
            images = images.astype(np.float32)

        action_indices = start_indices[:, None] + np.arange(
            self.action_horizon,
            dtype=np.int32,
        )
        action_mask = action_indices < episode_lengths[:, None]
        safe_action_indices = np.minimum(
            action_indices,
            episode_lengths[:, None] - 1,
        )
        actions = shard["residual_torque_nm"][
            episode_indices[:, None],
            safe_action_indices,
        ]
        actions = np.where(action_mask[..., None], actions, 0.0)
        actions = actions / self.residual_torque_limit_nm
        actions = np.clip(actions, -1.0, 1.0).astype(np.float32)
        return ActionChunkBatch(
            images=images,
            actions=actions,
            action_mask=action_mask,
        )

    def sample_batch(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> ActionChunkBatch:
        """Sample a controller-balanced batch using ``rng``."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        controller_ids = np.arange(batch_size) % len(self.controllers)
        rng.shuffle(controller_ids)

        batches = []
        for controller_index, controller in enumerate(self.controllers):
            count = int(np.sum(controller_ids == controller_index))
            if count:
                batches.append(self._sample_controller(controller, count, rng))

        images = np.concatenate([batch.images for batch in batches], axis=0)
        actions = np.concatenate([batch.actions for batch in batches], axis=0)
        masks = np.concatenate([batch.action_mask for batch in batches], axis=0)
        permutation = rng.permutation(batch_size)
        return ActionChunkBatch(
            images=images[permutation],
            actions=actions[permutation],
            action_mask=masks[permutation],
        )
