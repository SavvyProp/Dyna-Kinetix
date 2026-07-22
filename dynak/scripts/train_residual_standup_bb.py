"""Train residual standup with the bang-bang underlying controller."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    repository_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository_root))

import hydra

from dynak.scripts.train_residual_standup import run_residual_standup_training


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="residual_standup_bb_ppo",
)
def main(hydra_config) -> None:
    run_residual_standup_training(hydra_config, "ResidualStandupBangBangPPO")


if __name__ == "__main__":
    main()
