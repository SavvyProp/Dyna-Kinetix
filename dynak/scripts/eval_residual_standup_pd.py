"""Evaluate the residual standup policy with the PD underlying controller."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    repository_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository_root))

from dynak.scripts.eval_residual_standup import main as run_evaluation

DEFAULT_CHECKPOINT = Path("checkpoints/dynak/residual_standup/final.pbz2")


def main() -> None:
    run_evaluation(
        default_checkpoint=DEFAULT_CHECKPOINT,
        default_config_name="residual_standup_ppo",
    )


if __name__ == "__main__":
    main()
