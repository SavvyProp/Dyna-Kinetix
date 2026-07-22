"""Collect successful rollouts from the no-controller PPO expert."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    repository_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository_root))

from dynak.scripts.imitation_rollout_common import run_collection


def main() -> None:
    run_collection("no_controller")


if __name__ == "__main__":
    main()
