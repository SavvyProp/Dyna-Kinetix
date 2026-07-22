#!/usr/bin/env bash

python dynak/scripts/train_residual_standup_no_controller.py
python dynak/scripts/train_residual_standup_pd.py
python dynak/scripts/train_residual_standup_bb.py
python dynak/scripts/train_residual_standup_random.py

python dynak/scripts/collect_imitation_rollouts_no_controller.py --successes 200
python dynak/scripts/collect_imitation_rollouts_pd.py --successes 200
python dynak/scripts/collect_imitation_rollouts_bb.py --successes 200
