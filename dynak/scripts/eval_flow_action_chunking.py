"""Visualize the flow action-chunk standup policy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    repository_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository_root))

import jax
import numpy as np
import pygame

from dynak.imitation_rollout import (
    RESIDUAL_CONTROLLER_NAMES,
    append_image_history,
    initialize_image_history,
    load_flow_policy_checkpoint,
    make_flow_batch_action_function,
    make_flow_evaluation_env,
)
from dynak.standup.controllers import UnderlyingControllerType
from kinetix.render import make_render_pixels
from kinetix.util.saving import load_from_json_file

DEFAULT_CHECKPOINT = Path("checkpoints/dynak/flow_action_chunking/final.pbz2")


def controller_name(value: str) -> str:
    try:
        return UnderlyingControllerType.from_string(value).value
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Flow checkpoint or directory containing final.pbz2.",
    )
    parser.add_argument(
        "--level",
        default="l/standup_goal.json",
        help="Kinetix level JSON to evaluate (default: %(default)s).",
    )
    parser.add_argument(
        "--controllers",
        nargs="+",
        type=controller_name,
        default=list(RESIDUAL_CONTROLLER_NAMES),
        help=(
            "Residual environments sampled on reset. Defaults to none, PD, "
            "bang-bang, and switch."
        ),
    )
    parser.add_argument(
        "--num-flow-steps",
        type=int,
        default=5,
        help="Euler integration steps per newly sampled action chunk.",
    )
    parser.add_argument(
        "--execute-horizon",
        type=int,
        default=1,
        help=(
            "Actions to execute from each sampled chunk before replanning "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="Display/simulation steps per second (default: level control rate).",
    )
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--paused", action="store_true")
    parser.add_argument(
        "--auto-reset",
        action="store_true",
        help="Start another randomly selected controller environment after done.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Stop after this many total environment steps.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        help="Stop after this many completed episodes (most useful with --auto-reset).",
    )
    args = parser.parse_args(argv)

    args.controllers = list(dict.fromkeys(args.controllers))
    if args.num_flow_steps <= 0:
        parser.error("--num-flow-steps must be greater than zero")
    if args.execute_horizon <= 0:
        parser.error("--execute-horizon must be greater than zero")
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be greater than zero")
    if args.scale <= 0:
        parser.error("--scale must be greater than zero")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be greater than zero")
    if args.max_episodes is not None and args.max_episodes <= 0:
        parser.error("--max-episodes must be greater than zero")
    return args


def resolve_checkpoint_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_dir():
        path = path / "final.pbz2"
    if not path.is_file():
        raise FileNotFoundError(
            f"Flow checkpoint not found: {path}. Train it with "
            "`python dynak/scripts/train_flow_action_chunking.py`."
        )
    return path.resolve()


def pixels_for_pygame(pixels) -> np.ndarray:
    return np.asarray(pixels, dtype=np.uint8)[:, ::-1, :]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    policy = load_flow_policy_checkpoint(checkpoint_path)
    if policy.prediction_target == "total_torque_nm":
        if args.controllers != ["none"]:
            print(
                "This checkpoint predicts full applied torque; evaluating in "
                "the standalone no-controller environment."
            )
        args.controllers = ["none"]
    if args.execute_horizon > policy.model_config.action_horizon:
        raise ValueError(
            "--execute-horizon cannot exceed checkpoint horizon "
            f"{policy.model_config.action_horizon}"
        )
    initial_level, static_env_params, env_params = load_from_json_file(args.level)

    environments = {
        controller: make_flow_evaluation_env(
            policy,
            initial_level,
            static_env_params,
            env_params,
            controller,
        )
        for controller in args.controllers
    }
    resets = {
        controller: jax.jit(environment.reset)
        for controller, environment in environments.items()
    }
    steps = {
        controller: jax.jit(environment.step)
        for controller, environment in environments.items()
    }
    sample_action = jax.jit(
        make_flow_batch_action_function(
            policy,
            num_flow_steps=args.num_flow_steps,
        )
    )
    render = jax.jit(make_render_pixels(env_params, static_env_params))

    controller_rng = np.random.default_rng(args.seed)
    rng = jax.random.PRNGKey(args.seed)

    def random_controller() -> str:
        return str(controller_rng.choice(args.controllers))

    def reset_episode(controller: str, reset_key):
        observation, state = resets[controller](reset_key, env_params)
        if tuple(observation.image.shape) != policy.image_shape:
            raise ValueError(
                f"Level produces policy images shaped {observation.image.shape}; "
                f"checkpoint expects {policy.image_shape}"
            )
        history = initialize_image_history(
            observation.image,
            policy.model_config.frame_stack,
        )
        return observation, state, history

    controller = random_controller()
    rng, reset_key = jax.random.split(rng)
    print(f"Loading flow policy from {checkpoint_path}")
    print(
        "Compiling flow inference and the first standup environment. "
        "Other controller variants compile when first selected..."
    )
    observation, state, image_history = reset_episode(controller, reset_key)
    pixels = pixels_for_pygame(render(state))

    window_size = (pixels.shape[0] * args.scale, pixels.shape[1] * args.scale)
    simulation_fps = args.fps or 1.0 / (
        float(env_params.dt) * static_env_params.frame_skip
    )
    zero_action = np.zeros(policy.model_config.action_dim, dtype=np.float32)

    pygame.init()
    screen = pygame.display.set_mode(window_size)
    clock = pygame.time.Clock()
    print(
        "Controls: Space pause/run | N or . single-step | R random reset | "
        "C cycle controller | 1-4 select controller | Q/Esc quit\n"
        f"Controller pool: {', '.join(args.controllers)} | "
        f"flow steps: {args.num_flow_steps} | "
        f"execute horizon: {args.execute_horizon}"
    )

    running = True
    paused = args.paused
    done = False
    reset_pending = False
    total_steps = 0
    completed_episodes = 0
    episode_steps = 0
    episode_return = 0.0
    step_reward = 0.0
    success = False
    residual_torque_nm = zero_action
    underlying_torque_nm = zero_action
    total_torque_nm = zero_action
    action_chunk_normalized = np.zeros(
        (policy.model_config.action_horizon, policy.model_config.action_dim),
        dtype=np.float32,
    )
    chunk_action_index = 0
    executed_chunk_action_index = 0
    goal_hold_time_seconds = 0.0

    def start_episode(requested_controller: str | None = None) -> None:
        nonlocal controller, rng, observation, state, image_history, pixels
        nonlocal done, reset_pending, episode_steps, episode_return, step_reward
        nonlocal success, residual_torque_nm, underlying_torque_nm, total_torque_nm
        nonlocal action_chunk_normalized, chunk_action_index
        nonlocal executed_chunk_action_index, goal_hold_time_seconds

        controller = requested_controller or random_controller()
        rng, new_reset_key = jax.random.split(rng)
        observation, state, image_history = reset_episode(
            controller,
            new_reset_key,
        )
        pixels = pixels_for_pygame(render(state))
        done = False
        reset_pending = False
        episode_steps = 0
        episode_return = 0.0
        step_reward = 0.0
        success = False
        residual_torque_nm = zero_action
        underlying_torque_nm = zero_action
        total_torque_nm = zero_action
        action_chunk_normalized = np.zeros_like(action_chunk_normalized)
        chunk_action_index = 0
        executed_chunk_action_index = 0
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
                    start_episode()
                elif event.key == pygame.K_c:
                    current_index = args.controllers.index(controller)
                    next_controller = args.controllers[
                        (current_index + 1) % len(args.controllers)
                    ]
                    start_episode(next_controller)
                elif pygame.K_1 <= event.key <= pygame.K_4:
                    controller_index = event.key - pygame.K_1
                    if controller_index < len(args.controllers):
                        start_episode(args.controllers[controller_index])

        if reset_pending and running and (not paused or single_step):
            start_episode()

        if running and (not paused or single_step) and not done:
            rng, action_key, step_key = jax.random.split(rng, 3)
            if chunk_action_index == 0:
                _, action_chunk_batch = sample_action(
                    image_history[None, ...],
                    action_key,
                )
                action_chunk_normalized = np.asarray(action_chunk_batch[0])
            torque_command_nm = (
                action_chunk_normalized[chunk_action_index]
                * policy.residual_torque_limit_nm
            )
            observation, state, reward, step_done, info = steps[controller](
                step_key,
                state,
                torque_command_nm,
                env_params,
            )
            image_history = append_image_history(
                image_history,
                observation.image,
            )
            pixels = pixels_for_pygame(render(state))

            step_reward = float(reward)
            episode_return += step_reward
            done = bool(step_done)
            success = bool(info["GoalR"])
            residual_torque_nm = np.asarray(info["residual_torque_nm"])
            underlying_torque_nm = np.asarray(info["controller_torque_nm"])
            total_torque_nm = np.asarray(info["total_torque_nm"])
            executed_chunk_action_index = chunk_action_index
            chunk_action_index = (chunk_action_index + 1) % args.execute_horizon
            goal_hold_time_seconds = float(info["goal_hold_time_seconds"])
            total_steps += 1
            episode_steps += 1

            if done:
                completed_episodes += 1
                print(
                    f"episode={completed_episodes}, controller={controller}, "
                    f"steps={episode_steps}, return={episode_return:.3f}, "
                    f"success={success}"
                )
                reached_episode_limit = (
                    args.max_episodes is not None
                    and completed_episodes >= args.max_episodes
                )
                if args.auto_reset and not reached_episode_limit:
                    reset_pending = True
                else:
                    paused = True
                if reached_episode_limit:
                    exit_after_frame = True

            if args.max_steps is not None and total_steps >= args.max_steps:
                exit_after_frame = True

        surface = pygame.surfarray.make_surface(pixels)
        if args.scale != 1:
            surface = pygame.transform.scale(surface, window_size)
        screen.blit(surface, (0, 0))

        status = "done" if done else ("paused" if paused else "running")
        residual_text = np.asarray(residual_torque_nm).round(2).tolist()
        underlying_text = np.asarray(underlying_torque_nm).round(2).tolist()
        total_text = np.asarray(total_torque_nm).round(2).tolist()
        pygame.display.set_caption(
            f"Flow standup | {status} | controller={controller} | "
            f"step={int(state.timestep)} | return={episode_return:.3f} | "
            f"reward={step_reward:.3f} | residual={residual_text} | "
            f"underlying={underlying_text} | total={total_text} | "
            f"chunk index={executed_chunk_action_index}/{args.execute_horizon} | "
            f"hold={goal_hold_time_seconds:.2f}s"
        )
        pygame.display.flip()
        clock.tick(max(1, round(simulation_fps)))
        if exit_after_frame:
            running = False

    pygame.quit()


if __name__ == "__main__":
    main()
