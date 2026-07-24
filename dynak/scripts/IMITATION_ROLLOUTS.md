# Residual standup imitation rollouts

Evaluate each residual PPO policy with its matching default checkpoint:

```bash
python dynak/scripts/eval_residual_standup_no_controller.py
python dynak/scripts/eval_residual_standup_pd.py
python dynak/scripts/eval_residual_standup_bb.py
python dynak/scripts/eval_residual_standup_switch.py
```

Each evaluator accepts the same options as `eval_residual_standup.py`. For
example, use `--stochastic`, `--paused`, or `--level l/standup_goal.json`. A
different checkpoint can be supplied as the first positional argument.

There is one collection script for each trained PPO expert:

```bash
conda run -n dynk python dynak/scripts/collect_imitation_rollouts_no_controller.py
conda run -n dynk python dynak/scripts/collect_imitation_rollouts_pd.py
conda run -n dynk python dynak/scripts/collect_imitation_rollouts_bb.py
```

To train all four residual PPO variants and then retain 200 successful
no-controller, PD, and bang-bang episodes, run:

```bash
./dynak/scripts/train_all_residual_and_collect.sh
```

The script contains only the four sequential training commands followed by
the three collection commands.

Each script steps 32 environments in parallel by default. A small PD smoke
test can be collected with:

```bash
conda run -n dynk python dynak/scripts/collect_imitation_rollouts_pd.py \
  --successes 4 \
  --rollout-batch-size 2 \
  --episodes-per-shard 2
```

The datasets are written beneath:

```text
checkpoints/dynak/imitation_rollouts/
  no_controller/
  pd/
  bang_bang/
```

Every compressed shard contains padded complete episodes. `valid_mask`
selects real timesteps, and `done` identifies the terminal transition. The
aligned fields include the pre-action RGB image, raw policy action, applied
residual torque, underlying-controller torque, clipped total torque, reward,
success, and goal diagnostics. Images are stored as `uint8` to keep the shards
compact and converted back to `[0, 1]` floats by the training loader.

PD proportional and derivative gains, and bang-bang torque magnitudes, are
sampled independently per joint and held fixed for an episode. Their default
uniform range is 80% to 120% of the nominal values. The switch
environment uses the same episode parameters when it selects PD or bang-bang
for a joint. These fractions are saved in dataset and flow-checkpoint metadata.
Set `pd_gain_randomization_fraction` or
`bang_bang_torque_randomization_fraction` to `0.0` in the residual PPO config
to recover deterministic controller parameters.

PD and bang-bang controller outputs also receive independent, zero-mean
Gaussian noise for every joint and control step. Its default standard deviation
is 0.2 N*m. The samples are reproducible from the episode key and timestep, and
the setting is saved in dataset and flow-checkpoint metadata. Set
`controller_torque_noise_std_nm: 0.0` to disable this noise. The no-controller
baseline remains exactly zero.

The standup reward includes `goal_inside_reward_per_second` whenever the end
effector is inside the non-colliding goal region, regardless of whether the arm
is steady yet. The default `1.0` produces `dt` reward per physics step, so the
cumulative occupancy reward grows linearly by approximately `+1` per second.
Terminal success requires one qualifying 60 Hz frame in the region. The
maximum linear speed over the movable arm must be at most 1.0 m/s. Angular
speed is reported as a rollout diagnostic but does not affect success.
Collection uses these current criteria even when an older PPO checkpoint
contains the previous stricter values.

The controller torques are retained for analysis only. The imitation policy
does not receive controller identity, underlying torque, total torque, or
symbolic simulator state. It learns to predict `residual_torque_nm` chunks
from the same rendered observation modality used by the pixel PPO experts.
Residual policy actions are bounded to +/-10 N*m without an underlying
controller and +/-5 N*m for PD, bang-bang, and switch. Each limit is stored in
the rollout manifest. Joint flow training uses +/-10 N*m as its shared
normalization range so it can represent every dataset without clipping.

Collection samples from the PPO action distribution by default. Use
`--deterministic` for evaluation or debugging. Interrupted collection can be
continued with `--resume`.

## Flow matching training

After collecting all three datasets, train the joint image-conditioned flow
policy with:

```bash
conda run -n dynk python dynak/scripts/train_flow_action_chunking.py
```

The defaults use an eight-action horizon, controller-balanced batches, a CNN
image encoder, four time-conditioned MLP-Mixer blocks, and the conditional
rectified-flow objective. Checkpoints and per-epoch metrics are written to:

```text
checkpoints/dynak/flow_action_chunking/
  epoch_0004.pbz2
  ...
  final.pbz2
  metrics.jsonl
```

For a quick pipeline smoke test, reduce the work without changing the model:

```bash
conda run -n dynk python dynak/scripts/train_flow_action_chunking.py \
  --epochs 1 \
  --steps-per-epoch 10 \
  --validation-batches 1 \
  --output-dir checkpoints/dynak/flow_action_chunking_smoke
```

This stage implements offline flow training and a basic Euler flow sampler.
It deliberately does not implement real-time chunk execution or RTC inference
guidance.

Pixel shards default to eight episodes each. During training, a loaded shard
is reused for 32 batches before another is selected; this avoids repeatedly
decompressing hundreds of megabytes of padded image data while still rotating
through the dataset.

## Visual flow evaluation

To replan an action chunk at every step and execute its first residual torque:

```bash
conda run -n dynk python dynak/scripts/eval_flow_action_chunking.py
```

Set `--execute-horizon N` to execute the first `N` actions open-loop before
sampling another chunk. The default is one-step receding-horizon evaluation;
the option is bounded by the checkpoint's action horizon.

By default, every reset randomly selects one of the four residual environments:
no controller, PD, bang-bang, or independent per-joint switch control. Use
`R` for another random environment, `C` to cycle, or keys `1` through `4` to
select directly. For an unattended visual sequence:

```bash
conda run -n dynk python dynak/scripts/eval_flow_action_chunking.py \
  --auto-reset \
  --max-episodes 20
```

The reusable checkpoint loading, image-history handling, action sampling, and
JAX-vectorized rollout functions live in
`dynak/imitation_rollout/flow_evaluation.py`. The batched helpers return padded
trajectories with `valid_mask`, success, reward, action chunks, and all three
torque components so a future statistical evaluator does not need to depend on
the Pygame loop.
