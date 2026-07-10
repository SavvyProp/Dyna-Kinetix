"""Render a Kinetix JSON level without starting the level editor."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np
from matplotlib import pyplot as plt

from kinetix.util import load_from_json_file
from kinetix.render import make_render_pixels

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "level",
        nargs="?",
        default="l/simple_standup.json",
        help=(
            "Level JSON to render. Relative names such as 'l/my_level.json' "
            "are resolved from kinetix/levels/ (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optionally save the rendered image (for example, level.png).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a window; useful together with --output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    level, static_env_params, env_params = load_from_json_file(args.level)
    render = jax.jit(make_render_pixels(env_params, static_env_params))
    pixels = np.asarray(render(level), dtype=np.uint8).transpose(1, 0, 2)[::-1]

    figure, axes = plt.subplots()
    axes.imshow(pixels)
    axes.set_axis_off()
    figure.tight_layout(pad=0)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, bbox_inches="tight", pad_inches=0)
        print(f"Saved render to {args.output}")

    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
