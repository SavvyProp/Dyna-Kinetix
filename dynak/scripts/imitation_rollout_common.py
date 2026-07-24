"""Shared CLI and persistence code for standup imitation rollout scripts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import jax
import numpy as np
from flax.serialization import to_state_dict
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dynak.imitation_rollout import make_batched_rollout_function
from dynak.standup.controllers import UnderlyingControllerType
from dynak.standup.residual_torque_env import (
    DEFAULT_GOAL_HOLD_DURATION_SECONDS,
    DEFAULT_GOAL_INSIDE_REWARD_PER_SECOND,
    DEFAULT_GOAL_LINEAR_VELOCITY_THRESHOLD_MPS,
    make_residual_torque_env,
)
from kinetix.environment.spaces import ObservationType
from kinetix.util import normalise_config
from kinetix.util.saving import load_from_json_file, load_params

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 8
DEFAULT_LEVEL = "l/standup_goal.json"
DEFAULT_OUTPUT_ROOT = Path("checkpoints/dynak/imitation_rollouts")
CONTROLLER_ORDER = ("no_controller", "pd", "bang_bang")
CONTROLLER_ENV_NAMES = {
    "no_controller": "none",
    "pd": "pd",
    "bang_bang": "bang_bang",
}
CONTROLLER_CONFIG_NAMES = {
    "no_controller": "residual_standup_no_controller_ppo",
    "pd": "residual_standup_ppo",
    "bang_bang": "residual_standup_bb_ppo",
}
DEFAULT_CHECKPOINTS = {
    "no_controller": Path(
        "checkpoints/dynak/residual_standup_no_controller/final.pbz2"
    ),
    "pd": Path("checkpoints/dynak/residual_standup/final.pbz2"),
    "bang_bang": Path("checkpoints/dynak/residual_standup_bang_bang/final.pbz2"),
}
TRANSITION_FIELDS = (
    "policy_action",
    "residual_torque_nm",
    "underlying_torque_nm",
    "total_torque_nm",
    "reward",
    "done",
    "success",
    "valid_mask",
    "goal_inside",
    "goal_inside_reward",
    "goal_steady",
    "goal_hold_time_seconds",
    "goal_max_linear_speed_mps",
    "goal_max_angular_speed_rad_s",
)


def parse_collection_args(
    controller: str,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Collect successful {controller} residual-standup demonstrations."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINTS[controller],
        help="Expert checkpoint (default: %(default)s).",
    )
    parser.add_argument(
        "--level",
        default=DEFAULT_LEVEL,
        help="Kinetix level JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Dataset root inside the checkpoint tree (default: %(default)s).",
    )
    parser.add_argument(
        "--successes",
        type=int,
        default=1000,
        help="Number of successful episodes to retain.",
    )
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=32,
        help="Number of environments stepped in parallel.",
    )
    parser.add_argument(
        "--episodes-per-shard",
        type=int,
        default=8,
        help="Successful padded episodes per compressed NPZ shard.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="Maximum attempted episodes. Zero uses 100 times --successes.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Use the Gaussian PPO policy's mode. Sampling is the default and "
            "usually gives more diverse demonstrations."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a compatible interrupted dataset from its manifest.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress after approximately this many attempted episodes.",
    )
    args = parser.parse_args(argv)

    if args.successes <= 0:
        parser.error("--successes must be greater than zero")
    if args.rollout_batch_size <= 0:
        parser.error("--rollout-batch-size must be greater than zero")
    if args.episodes_per_shard <= 0:
        parser.error("--episodes-per-shard must be greater than zero")
    if args.max_attempts < 0:
        parser.error("--max-attempts cannot be negative")
    if args.progress_every <= 0:
        parser.error("--progress-every must be greater than zero")
    return args


def resolve_checkpoint_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_dir():
        candidates = (path / "final.pbz2", path / "full_model.pbz2")
        path = next((candidate for candidate in candidates if candidate.exists()), path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path.resolve()


def load_config(controller: str, checkpoint_config: dict[str, Any] | None) -> dict:
    saved_controller = (checkpoint_config or {}).get("underlying_controller")
    expected_controller = CONTROLLER_ENV_NAMES[controller]
    if saved_controller is not None:
        if isinstance(saved_controller, UnderlyingControllerType):
            saved_controller = saved_controller.value
        else:
            saved_controller = UnderlyingControllerType.from_string(
                str(saved_controller)
            ).value
        if saved_controller != expected_controller:
            raise ValueError(
                f"The {controller} collector was given a checkpoint trained with "
                f"underlying_controller={saved_controller!r}"
            )

    with initialize_config_dir(
        version_base=None,
        config_dir=str(REPOSITORY_ROOT / "configs"),
    ):
        hydra_config = compose(config_name=CONTROLLER_CONFIG_NAMES[controller])
    default_config = normalise_config(
        OmegaConf.to_container(hydra_config, resolve=True),
        "ResidualStandupImitationRollout",
        save_config=False,
    )
    config = {**default_config, **(checkpoint_config or {})}
    # Success criteria come from the current task definition so older expert
    # checkpoints can be collected under the newly relaxed goal.
    for field in (
        "residual_torque_limit_nm",
        "goal_hold_duration_seconds",
        "goal_linear_velocity_threshold_mps",
    ):
        config[field] = default_config[field]
    config["underlying_controller"] = expected_controller
    return config


def check_level_shape(config: dict, static_env_params) -> None:
    expected = config.get("static_env_params")
    if expected is None:
        return
    for field in ("num_polygons", "num_circles", "num_joints", "num_thrusters"):
        expected_value = int(expected[field])
        actual_value = int(getattr(static_env_params, field))
        if expected_value != actual_value:
            raise ValueError(
                f"Checkpoint expects {field}={expected_value}, but the level has "
                f"{field}={actual_value}."
            )


def successful_episode_mask(trajectories: dict[str, np.ndarray]) -> np.ndarray:
    return np.any(
        trajectories["success"] & trajectories["valid_mask"],
        axis=1,
    )


def pack_shard(episodes: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not episodes:
        raise ValueError("Cannot pack an empty shard")
    observation_fields = (
        ("image", "global_info") if "image" in episodes[0] else ("observation",)
    )
    shard_fields = observation_fields + TRANSITION_FIELDS
    shard = {
        field: np.stack([episode[field] for episode in episodes])
        for field in shard_fields
    }
    valid = shard["valid_mask"]
    shard["episode_length"] = valid.sum(axis=1).astype(np.int32)
    shard["episode_return"] = (shard["reward"] * valid).sum(axis=1).astype(np.float32)
    return shard


def save_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary_path, **arrays)
    os.replace(temporary_path, path)


def save_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_path, path)


def initial_manifest(
    controller: str,
    checkpoint_path: Path,
    args: argparse.Namespace,
    config: dict[str, Any],
    env_params,
    static_env_params,
    observation_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "controller": controller,
        "environment_controller": CONTROLLER_ENV_NAMES[controller],
        "checkpoint": str(checkpoint_path),
        "level": args.level,
        "seed": args.seed,
        "stochastic_policy": not args.deterministic,
        "requested_successes": args.successes,
        "rollout_batch_size": args.rollout_batch_size,
        "episodes_per_shard": args.episodes_per_shard,
        "max_timesteps": int(env_params.max_timesteps),
        "frame_skip": int(static_env_params.frame_skip),
        "observation_type": config["observation_type_str"],
        **observation_metadata,
        "action_dim": 3,
        "residual_torque_limit_nm": float(config["residual_torque_limit_nm"]),
        "total_torque_limit_nm": float(config["total_torque_limit_nm"]),
        "pd_gain_randomization_fraction": float(
            config.get("pd_gain_randomization_fraction", 0.2)
        ),
        "bang_bang_torque_randomization_fraction": float(
            config.get("bang_bang_torque_randomization_fraction", 0.2)
        ),
        "controller_torque_noise_std_nm": float(
            config.get("controller_torque_noise_std_nm", 0.2)
        ),
        "goal_inside_reward_per_second": float(
            config.get(
                "goal_inside_reward_per_second",
                DEFAULT_GOAL_INSIDE_REWARD_PER_SECOND,
            )
        ),
        "goal_hold_duration_seconds": float(
            config.get(
                "goal_hold_duration_seconds",
                DEFAULT_GOAL_HOLD_DURATION_SECONDS,
            )
        ),
        "goal_linear_velocity_threshold_mps": float(
            config.get(
                "goal_linear_velocity_threshold_mps",
                DEFAULT_GOAL_LINEAR_VELOCITY_THRESHOLD_MPS,
            )
        ),
        "units": {
            "image": "uint8 RGB",
            "policy_action": "N*m before environment clipping",
            "residual_torque_nm": "N*m after residual clipping",
            "underlying_torque_nm": "N*m",
            "total_torque_nm": "N*m after total clipping",
        },
        "attempted_episodes": 0,
        "successful_episodes": 0,
        "successful_transitions": 0,
        "shards": [],
    }


def validate_resume_manifest(
    manifest: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    for field in (
        "schema_version",
        "controller",
        "checkpoint",
        "level",
        "seed",
        "stochastic_policy",
        "episodes_per_shard",
        "max_timesteps",
        "observation_type",
        "image_shape",
        "global_info_shape",
        "observation_shape",
        "action_dim",
        "residual_torque_limit_nm",
        "total_torque_limit_nm",
        "pd_gain_randomization_fraction",
        "bang_bang_torque_randomization_fraction",
        "controller_torque_noise_std_nm",
        "goal_inside_reward_per_second",
        "goal_hold_duration_seconds",
        "goal_linear_velocity_threshold_mps",
    ):
        if manifest.get(field) != expected.get(field):
            raise ValueError(
                f"Cannot resume: manifest {field}={manifest.get(field)!r}, "
                f"but this run requested {expected.get(field)!r}"
            )


def collect_successful_rollouts(
    controller: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    checkpoint = load_params(checkpoint_path)
    config = load_config(controller, checkpoint.get("config"))
    initial_level, static_env_params, env_params = load_from_json_file(args.level)
    check_level_shape(config, static_env_params)
    config["env_params"] = to_state_dict(env_params)
    config["static_env_params"] = to_state_dict(static_env_params)

    rollout_batch = make_batched_rollout_function(
        checkpoint,
        config,
        initial_level,
        static_env_params,
        env_params,
        deterministic=args.deterministic,
    )
    observation = make_residual_torque_env(
        observation_type=config["observation_type"],
        reset_fn=lambda _rng: initial_level,
        env_params=env_params,
        static_env_params=static_env_params,
        auto_reset=False,
        underlying_controller=CONTROLLER_ENV_NAMES[controller],
    ).get_obs(initial_level)

    output_dir = (args.output_root / controller).resolve()
    if config["observation_type"] == ObservationType.PIXELS:
        observation_metadata = {
            "image_shape": list(observation.image.shape),
            "global_info_shape": list(observation.global_info.shape),
        }
    else:
        observation_metadata = {"observation_shape": list(observation.shape)}

    expected_manifest = initial_manifest(
        controller,
        checkpoint_path,
        args,
        config,
        env_params,
        static_env_params,
        observation_metadata,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"Dataset already exists at {output_dir}; pass --resume to continue"
            )
        manifest = json.loads(manifest_path.read_text())
        validate_resume_manifest(manifest, expected_manifest)
        manifest["requested_successes"] = args.successes
        manifest["rollout_batch_size"] = args.rollout_batch_size
    else:
        if list(output_dir.glob("shard_*.npz")):
            raise FileExistsError(
                f"Found shards without a manifest in {output_dir}; refusing to overwrite"
            )
        manifest = expected_manifest
        save_json_atomic(manifest_path, manifest)

    attempted = int(manifest["attempted_episodes"])
    successful = int(manifest["successful_episodes"])
    successful_transitions = int(manifest["successful_transitions"])
    shard_index = len(manifest["shards"])
    pending_episodes: list[dict[str, np.ndarray]] = []
    max_attempts = args.max_attempts or (100 * args.successes)

    if successful >= args.successes:
        print(
            f"[{controller}] already has {successful} successful episodes; "
            "nothing to collect."
        )
        return manifest

    print(
        f"[{controller}] loading {checkpoint_path}\n"
        f"[{controller}] compiling {args.rollout_batch_size} parallel rollouts..."
    )
    controller_key = jax.random.fold_in(
        jax.random.PRNGKey(args.seed),
        CONTROLLER_ORDER.index(controller),
    )

    def persist_manifest() -> None:
        manifest["attempted_episodes"] = attempted
        manifest["successful_episodes"] = successful - len(pending_episodes)
        manifest["successful_transitions"] = successful_transitions
        save_json_atomic(manifest_path, manifest)

    def flush_pending() -> None:
        nonlocal shard_index, successful_transitions
        if not pending_episodes:
            return
        arrays = pack_shard(pending_episodes)
        filename = f"shard_{shard_index:05d}.npz"
        save_npz_atomic(output_dir / filename, arrays)
        num_episodes = len(pending_episodes)
        num_transitions = int(arrays["episode_length"].sum())
        successful_transitions += num_transitions
        manifest["shards"].append(
            {
                "file": filename,
                "episodes": num_episodes,
                "transitions": num_transitions,
            }
        )
        pending_episodes.clear()
        shard_index += 1
        persist_manifest()

    next_progress = attempted + args.progress_every
    try:
        while successful < args.successes and attempted < max_attempts:
            batch_size = min(args.rollout_batch_size, max_attempts - attempted)
            batch_start = attempted
            attempt_indices = np.arange(
                batch_start,
                batch_start + batch_size,
                dtype=np.uint32,
            )
            episode_keys = jax.vmap(
                lambda index: jax.random.fold_in(controller_key, index)
            )(attempt_indices)
            trajectories = jax.device_get(rollout_batch(episode_keys))
            success_mask = successful_episode_mask(trajectories)

            for batch_index in range(batch_size):
                attempted += 1
                if success_mask[batch_index]:
                    pending_episodes.append(
                        jax.tree.map(lambda value: value[batch_index], trajectories)
                    )
                    successful += 1
                    if len(pending_episodes) >= args.episodes_per_shard:
                        flush_pending()
                    if successful >= args.successes:
                        break

            if attempted >= next_progress or attempted == batch_size:
                persist_manifest()
                print(
                    f"[{controller}] attempts={attempted}/{max_attempts}, "
                    f"successes={successful}/{args.successes}"
                )
                next_progress = attempted + args.progress_every
    except KeyboardInterrupt:
        flush_pending()
        persist_manifest()
        print(f"\n[{controller}] interrupted; collection progress was saved.")
        raise

    flush_pending()
    persist_manifest()
    if successful < args.successes:
        raise RuntimeError(
            f"[{controller}] collected {successful} successes in {attempted} attempts, "
            f"below the requested {args.successes}. Existing shards are valid; use "
            "--resume with a larger --max-attempts to continue."
        )

    print(
        f"[{controller}] saved {successful} successful episodes "
        f"({successful_transitions} transitions) to {output_dir}"
    )
    return manifest


def run_collection(controller: str, argv: list[str] | None = None) -> None:
    if controller not in CONTROLLER_ORDER:
        raise ValueError(f"Unknown rollout controller: {controller}")
    args = parse_collection_args(controller, argv)
    collect_successful_rollouts(controller, args)
