"""Visualize a saved residual standup policy in the Pygame simulator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    repository_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository_root))
else:
    repository_root = Path(__file__).resolve().parents[2]

import jax
import jax.numpy as jnp
import numpy as np
import pygame
from flax.serialization import to_state_dict
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from kinetix.environment.spaces import ActionType
from kinetix.models import GeneralActorCriticRNN, make_network_from_config
from kinetix.render import make_render_pixels
from kinetix.util import normalise_config, rms_normalise
from kinetix.util.saving import load_from_json_file, load_params

from dynak.standup.residual_torque_env import make_residual_torque_env

DEFAULT_CHECKPOINT = Path("checkpoints/dynak/residual_standup/final.pbz2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Local full-model checkpoint (default: %(default)s).",
    )
    parser.add_argument(
        "--level",
        default="l/standup_goal.json",
        help="Kinetix level JSON to evaluate (default: %(default)s).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="Display/simulation steps per second (default: level control rate).",
    )
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--paused", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()

    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be greater than zero")
    if args.scale <= 0:
        parser.error("--scale must be greater than zero")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be greater than zero")
    return args


def load_default_config():
    """Compose the training config for older checkpoints without metadata."""
    with initialize_config_dir(
        version_base=None,
        config_dir=str(repository_root / "configs"),
    ):
        hydra_config = compose(config_name="residual_standup_ppo")
    return normalise_config(
        OmegaConf.to_container(hydra_config, resolve=True),
        "ResidualStandupPPOEval",
        save_config=False,
    )


def resolve_checkpoint_path(path: Path) -> Path:
    if path.is_dir():
        path = path / "full_model.pbz2"
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Train first with "
            "`python dynak/scripts/train_residual_standup.py`."
        )
    return path


def check_level_shape(config, static_env_params) -> None:
    expected = config.get("static_env_params")
    if expected is None:
        return
    for field in ("num_polygons", "num_circles", "num_joints", "num_thrusters"):
        expected_value = int(expected[field])
        actual_value = int(getattr(static_env_params, field))
        if expected_value != actual_value:
            raise ValueError(
                f"Checkpoint expects {field}={expected_value}, but the level has "
                f"{field}={actual_value}. Evaluate on a level with the same static shape."
            )


def pixels_for_pygame(pixels) -> np.ndarray:
    return np.asarray(pixels, dtype=np.uint8)[:, ::-1, :]


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    checkpoint = load_params(checkpoint_path)
    config = checkpoint.get("config") or load_default_config()

    if config["action_type"] != ActionType.CONTINUOUS:
        raise ValueError("Residual standup checkpoints must use a continuous policy")

    initial_level, static_env_params, env_params = load_from_json_file(args.level)
    check_level_shape(config, static_env_params)
    config["env_params"] = to_state_dict(env_params)
    config["static_env_params"] = to_state_dict(static_env_params)

    env = make_residual_torque_env(
        observation_type=config["observation_type"],
        reset_fn=lambda _rng: initial_level,
        env_params=env_params,
        static_env_params=static_env_params,
        auto_reset=False,
        residual_torque_limit_nm=config["residual_torque_limit_nm"],
        total_torque_limit_nm=config["total_torque_limit_nm"],
        energy_penalty_coefficient=config["energy_penalty_coefficient"],
        underlying_controller=config.get("underlying_controller", "pd"),
        pd_gain_randomization_fraction=config.get(
            "pd_gain_randomization_fraction",
            0.2,
        ),
        bang_bang_torque_randomization_fraction=config.get(
            "bang_bang_torque_randomization_fraction",
            0.2,
        ),
        controller_torque_noise_std_nm=config.get(
            "controller_torque_noise_std_nm",
            0.2,
        ),
        goal_hold_duration_seconds=config.get("goal_hold_duration_seconds", 1.0),
        goal_linear_velocity_threshold_mps=config.get(
            "goal_linear_velocity_threshold_mps",
            0.1,
        ),
        goal_angular_velocity_threshold_rad_s=config.get(
            "goal_angular_velocity_threshold_rad_s",
            0.1,
        ),
    )
    network = make_network_from_config(env, env_params, config)
    policy_params = checkpoint["params"]

    rms = None
    if config.get("rms_norm", False):
        rms = (checkpoint.get("extra") or {}).get("rms")
        if rms is None:
            raise ValueError(
                "Checkpoint enables RMS normalization but has no RMS state"
            )

    def _policy_step(hstate, observation, done, action_key):
        batched_observation = jax.tree.map(lambda x: x[None, ...], observation)
        if rms is not None:
            batched_observation = rms_normalise(
                rms,
                batched_observation,
                flatten="auto",
            )
        network_observation = jax.tree.map(
            lambda x: x[None, ...],
            batched_observation,
        )
        network_done = jnp.asarray(done)[None, None]
        hstate, distribution, value = network.apply(
            policy_params,
            hstate,
            (network_observation, network_done),
        )
        if args.stochastic:
            action = distribution.sample(seed=action_key)
        else:
            action = distribution.mode()
        return hstate, action[0, 0], value[0, 0]

    rng, reset_rng, policy_rng = jax.random.split(
        jax.random.PRNGKey(args.seed),
        3,
    )
    print(f"Loading policy from {checkpoint_path.resolve()}")
    print("Compiling policy, residual environment, and renderer...")

    reset = jax.jit(env.reset).lower(reset_rng, env_params).compile()
    observation, state = reset(reset_rng, env_params)
    hstate = GeneralActorCriticRNN.initialize_carry(1)
    policy = (
        jax.jit(_policy_step)
        .lower(hstate, observation, jnp.asarray(False), policy_rng)
        .compile()
    )
    noop_action = env.action_type.noop_action()
    step = jax.jit(env.step).lower(rng, state, noop_action, env_params).compile()
    render = (
        jax.jit(make_render_pixels(env_params, static_env_params))
        .lower(state)
        .compile()
    )

    pixels = pixels_for_pygame(render(state))
    window_size = (pixels.shape[0] * args.scale, pixels.shape[1] * args.scale)
    simulation_fps = args.fps or 1.0 / (
        float(env_params.dt) * static_env_params.frame_skip
    )

    pygame.init()
    screen = pygame.display.set_mode(window_size)
    clock = pygame.time.Clock()
    print(
        "Controls: Space pause/run | N or . single-step | R reset | Q/Esc quit\n"
        f"Policy mode: {'stochastic' if args.stochastic else 'deterministic'}"
    )

    running = True
    paused = args.paused
    done = False
    simulation_steps = 0
    episode_return = 0.0
    step_reward = 0.0
    value = 0.0
    residual_torque_nm = noop_action
    total_torque_nm = noop_action
    energy_penalty = 0.0
    goal_hold_time_seconds = 0.0

    while running:
        exit_after_frame = False
        single_step = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key in (pygame.K_n, pygame.K_PERIOD):
                    single_step = True
                elif event.key == pygame.K_r:
                    rng, reset_rng = jax.random.split(rng)
                    observation, state = reset(reset_rng, env_params)
                    hstate = GeneralActorCriticRNN.initialize_carry(1)
                    pixels = pixels_for_pygame(render(state))
                    done = False
                    episode_return = 0.0
                    step_reward = 0.0
                    value = 0.0
                    residual_torque_nm = noop_action
                    total_torque_nm = noop_action
                    energy_penalty = 0.0
                    goal_hold_time_seconds = 0.0

        if running and (not paused or single_step) and not done:
            rng, policy_rng, step_rng = jax.random.split(rng, 3)
            hstate, policy_action, policy_value = policy(
                hstate,
                observation,
                jnp.asarray(done),
                policy_rng,
            )
            observation, state, reward, step_done, info = step(
                step_rng,
                state,
                policy_action,
                env_params,
            )
            pixels = pixels_for_pygame(render(state))

            step_reward = float(reward)
            episode_return += step_reward
            value = float(policy_value)
            done = bool(step_done)
            residual_torque_nm = info["residual_torque_nm"]
            total_torque_nm = info["total_torque_nm"]
            energy_penalty = float(info["energy_penalty"])
            goal_hold_time_seconds = float(info["goal_hold_time_seconds"])
            simulation_steps += 1

            if done:
                paused = True
            if args.max_steps is not None and simulation_steps >= args.max_steps:
                exit_after_frame = True

        surface = pygame.surfarray.make_surface(pixels)
        if args.scale != 1:
            surface = pygame.transform.scale(surface, window_size)
        screen.blit(surface, (0, 0))

        status = "done" if done else ("paused" if paused else "running")
        residual_text = np.asarray(residual_torque_nm).round(2).tolist()
        total_text = np.asarray(total_torque_nm).round(2).tolist()
        pygame.display.set_caption(
            f"Residual standup policy | {status} | step={int(state.timestep)} | "
            f"controller={env.underlying_controller_name} | "
            f"return={episode_return:.3f} | reward={step_reward:.3f} | "
            f"value={value:.2f} | residual={residual_text} | total={total_text} | "
            f"hold={goal_hold_time_seconds:.2f}/{env.goal_hold_duration_seconds:.2f}s | "
            f"energy cost={energy_penalty:.5f}"
        )
        pygame.display.flip()
        clock.tick(max(1, round(simulation_fps)))
        if exit_after_frame:
            running = False

    pygame.quit()


if __name__ == "__main__":
    main()
