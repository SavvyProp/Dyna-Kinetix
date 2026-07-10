"""Run and visualize the standup level without the Kinetix editor UI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# When this file is executed by path, Python adds dynak/scripts (rather than
# the repository root) to sys.path. Add only this checkout's root so absolute
# dynak imports work for both direct and ``python -m`` invocation styles.
if __package__ in (None, ""):
    repository_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository_root))

import jax
import jax.numpy as jnp
import numpy as np
import pygame

from kinetix.environment import ObservationType
from kinetix.util import load_from_json_file

from dynak.standup.residual_torque_env import (
    make_residual_torque_env,
)
from dynak.standup.stand_pd import get_standup_joint_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "level",
        nargs="?",
        default="l/simple_standup.json",
        help=(
            "Level JSON to simulate. Relative names such as 'l/my_level.json' "
            "are resolved from kinetix/levels/ (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="Display/simulation steps per second (default: use the level timestep).",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help="Integer window scale applied to the rendered pixels (default: %(default)s).",
    )
    parser.add_argument(
        "--paused",
        action="store_true",
        help="Start paused; press N or . to advance one step.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Optionally close after this many simulation steps.",
    )
    parser.add_argument(
        "--torque-limit-nm",
        type=float,
        default=20.0,
        help=(
            "Symmetric limit for the combined PD and residual torque in N*m "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be greater than zero")
    if args.scale <= 0:
        parser.error("--scale must be greater than zero")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be greater than zero")
    if args.torque_limit_nm <= 0:
        parser.error("--torque-limit-nm must be greater than zero")
    return args


def keyboard_policy_torque_nm(
    keys: pygame.key.ScancodeWrapper,
    action_size: int,
    torque_limit_nm: float,
):
    """Use the keyboard as a stand-in policy that commands torque in N*m."""
    torque_nm = np.zeros(action_size, dtype=np.float32)

    motor_keys = (
        (pygame.K_LEFT, pygame.K_RIGHT),
        (pygame.K_UP, pygame.K_DOWN),
        (pygame.K_a, pygame.K_d),
    )
    for binding, (positive_key, negative_key) in enumerate(motor_keys[:action_size]):
        if keys[positive_key]:
            torque_nm[binding] = torque_limit_nm
        elif keys[negative_key]:
            torque_nm[binding] = -torque_limit_nm

    return jnp.asarray(torque_nm)


def pixels_for_pygame(pixels) -> np.ndarray:
    """Convert Kinetix's bottom-up renderer output to Pygame coordinates."""
    return np.asarray(pixels, dtype=np.uint8)[:, ::-1, :]


def main() -> None:
    args = parse_args()
    initial_level, static_env_params, env_params = load_from_json_file(args.level)

    env = make_residual_torque_env(
        observation_type=ObservationType.PIXELS,
        reset_fn=lambda _rng: initial_level,
        env_params=env_params,
        static_env_params=static_env_params,
        auto_reset=False,
        total_torque_limit_nm=args.torque_limit_nm,
    )

    rng, reset_rng = jax.random.split(jax.random.PRNGKey(args.seed))
    residual_torque_nm = env.action_type.noop_action()
    actor_torque_limit_nm = env.action_type.torque_limit_nm

    print("Compiling the residual-torque environment...")
    reset = jax.jit(env.reset).lower(reset_rng, env_params).compile()
    observation, state = reset(reset_rng, env_params)
    step = jax.jit(env.step).lower(rng, state, residual_torque_nm, env_params).compile()

    pixels = pixels_for_pygame(observation.image * 255.0)
    reward = 0.0
    done = False
    paused = args.paused
    applied_torque_nm = residual_torque_nm

    simulation_fps = args.fps or 1.0 / (
        float(env_params.dt) * static_env_params.frame_skip
    )
    window_size = (pixels.shape[0] * args.scale, pixels.shape[1] * args.scale)

    pygame.init()
    screen = pygame.display.set_mode(window_size)
    clock = pygame.time.Clock()

    print(
        "Controls: Space pause/run | N or . single-step | R reset | Q/Esc quit\n"
        "Residual torque bindings: Left/Right, Up/Down, A/D\n"
        f"Actor/keyboard torque: +/-{actor_torque_limit_nm:g} N*m | "
        f"combined torque limit: +/-{args.torque_limit_nm:g} N*m"
    )

    running = True
    simulation_steps = 0
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
                    pixels = pixels_for_pygame(observation.image * 255.0)
                    reward = 0.0
                    done = False
                    applied_torque_nm = env.action_type.noop_action()

        if running and (not paused or single_step) and not done:
            rng, step_rng = jax.random.split(rng)
            residual_torque_nm = keyboard_policy_torque_nm(
                pygame.key.get_pressed(),
                env.action_space(env_params).shape[0],
                actor_torque_limit_nm,
            )
            observation, state, step_reward, step_done, info = step(
                step_rng,
                state,
                residual_torque_nm,
                env_params,
            )
            applied_torque_nm = info["total_torque_nm"]
            joint_state = get_standup_joint_state(state, static_env_params)
            print(joint_state)

            pixels = pixels_for_pygame(observation.image * 255.0)
            reward = float(step_reward)
            done = bool(step_done)
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
        torque_text = np.asarray(applied_torque_nm).round(1).tolist()
        pygame.display.set_caption(
            f"Dyna-Kinetix standup | {status} | step={int(state.timestep)} | "
            f"reward={reward:.3f} | torque N*m={torque_text}"
        )
        pygame.display.flip()
        clock.tick(max(1, round(simulation_fps)))
        if exit_after_frame:
            running = False

    pygame.quit()


if __name__ == "__main__":
    main()
