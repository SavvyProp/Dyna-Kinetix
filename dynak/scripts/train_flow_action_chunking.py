"""Train an image-conditioned flow policy on residual standup rollouts."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

if __package__ in (None, ""):
    repository_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository_root))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from dynak.flow_action_chunking import (
    FlowMatchingPolicy,
    FlowModelConfig,
    PixelActionChunkDataset,
    flow_matching_loss,
)
from kinetix.util.saving import save_params

DEFAULT_DATASET_ROOT = Path("checkpoints/dynak/imitation_rollouts")
DEFAULT_OUTPUT_DIR = Path("checkpoints/dynak/flow_action_chunking")
DEFAULT_CONTROLLERS = ("no_controller", "pd", "bang_bang")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a CNN-conditioned rectified-flow model that predicts "
            "residual-torque action chunks from rollout images."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Directory containing the per-controller rollout datasets.",
    )
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=list(DEFAULT_CONTROLLERS),
        help="Controller dataset directories to combine.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Checkpoint and metric output directory.",
    )
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=0,
        help="Optimizer steps per epoch; zero uses one pass over the train split.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument(
        "--residual-torque-limit-nm",
        type=float,
        default=10.0,
        help=(
            "Shared action normalization range. It must cover the largest "
            "controller dataset limit (default: %(default)s)."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--channel-dim", type=int, default=256)
    parser.add_argument("--token-mixing-hidden-dim", type=int, default=64)
    parser.add_argument("--channel-mixing-hidden-dim", type=int, default=512)
    parser.add_argument("--num-mixer-blocks", type=int, default=4)
    parser.add_argument("--time-embedding-dim", type=int, default=256)
    parser.add_argument("--frame-embedding-dim", type=int, default=128)
    parser.add_argument("--observation-embedding-dim", type=int, default=256)
    parser.add_argument("--cache-size", type=int, default=3)
    parser.add_argument(
        "--shard-reuse-batches",
        type=int,
        default=32,
        help="Batches to draw from a loaded shard before selecting another.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing final checkpoint and metrics file.",
    )
    args = parser.parse_args(argv)

    positive_integer_fields = (
        "epochs",
        "batch_size",
        "frame_stack",
        "action_horizon",
        "validation_batches",
        "channel_dim",
        "token_mixing_hidden_dim",
        "channel_mixing_hidden_dim",
        "num_mixer_blocks",
        "time_embedding_dim",
        "frame_embedding_dim",
        "observation_embedding_dim",
        "cache_size",
        "shard_reuse_batches",
        "checkpoint_every",
    )
    for field in positive_integer_fields:
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be greater than zero")
    if args.steps_per_epoch < 0:
        parser.error("--steps-per-epoch cannot be negative")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps cannot be negative")
    if not 0.0 <= args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be in [0, 1)")
    for field in (
        "residual_torque_limit_nm",
        "learning_rate",
        "gradient_clip",
    ):
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be greater than zero")
    if args.weight_decay < 0:
        parser.error("--weight-decay cannot be negative")
    return args


def _make_dataset(
    args: argparse.Namespace,
    split: str,
) -> PixelActionChunkDataset:
    return PixelActionChunkDataset(
        args.dataset_root,
        args.controllers,
        split=split,
        validation_fraction=args.validation_fraction,
        frame_stack=args.frame_stack,
        action_horizon=args.action_horizon,
        residual_torque_limit_nm=args.residual_torque_limit_nm,
        cache_size=args.cache_size,
        shard_reuse_batches=args.shard_reuse_batches,
    )


def _checkpoint_payload(
    state: TrainState,
    model_config: FlowModelConfig,
    args: argparse.Namespace,
    dataset: PixelActionChunkDataset,
    epoch: int,
) -> dict:
    return {
        "format_version": 1,
        "epoch": epoch,
        "step": int(jax.device_get(state.step)),
        "params": jax.device_get(state.params),
        "opt_state": jax.device_get(state.opt_state),
        "model_config": asdict(model_config),
        "data_config": {
            "dataset_root": str(dataset.dataset_root),
            "controllers": list(dataset.controllers),
            "image_shape": list(dataset.image_shape),
            "frame_stack": dataset.frame_stack,
            "action_horizon": dataset.action_horizon,
            "action_dim": dataset.action_dim,
            "residual_torque_limit_nm": dataset.residual_torque_limit_nm,
            "pd_gain_randomization_fraction": (dataset.pd_gain_randomization_fraction),
            "bang_bang_torque_randomization_fraction": (
                dataset.bang_bang_torque_randomization_fraction
            ),
            "controller_torque_noise_std_nm": (dataset.controller_torque_noise_std_nm),
            "shard_reuse_batches": dataset.shard_reuse_batches,
            "observation_inputs": ["image"],
            "prediction_target": "residual_torque_nm",
        },
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "steps_per_epoch": args.steps_per_epoch,
            "validation_fraction": args.validation_fraction,
            "validation_batches": args.validation_batches,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
            "seed": args.seed,
        },
    }


def train(args: argparse.Namespace) -> Path:
    """Train the model and return the final checkpoint path."""
    train_dataset = _make_dataset(args, "train")
    validation_dataset = (
        _make_dataset(args, "validation") if args.validation_fraction > 0.0 else None
    )
    if train_dataset.action_dim != 3:
        raise ValueError(
            "Residual standup currently expects three motor actions; "
            f"dataset contains {train_dataset.action_dim}"
        )

    output_dir = args.output_dir.expanduser().resolve()
    final_checkpoint_path = output_dir / "final.pbz2"
    metrics_path = output_dir / "metrics.jsonl"
    if not args.overwrite and (final_checkpoint_path.exists() or metrics_path.exists()):
        raise FileExistsError(
            f"Training output already exists in {output_dir}; pass --overwrite "
            "to replace it"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite and metrics_path.exists():
        metrics_path.unlink()

    steps_per_epoch = args.steps_per_epoch or math.ceil(
        train_dataset.total_examples / args.batch_size
    )
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = min(args.warmup_steps, max(total_steps - 1, 0))
    schedule = optax.warmup_constant_schedule(
        init_value=0.0,
        peak_value=args.learning_rate,
        warmup_steps=warmup_steps,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.gradient_clip),
        optax.adamw(schedule, weight_decay=args.weight_decay),
    )

    model_config = FlowModelConfig(
        action_horizon=args.action_horizon,
        action_dim=train_dataset.action_dim,
        frame_stack=args.frame_stack,
        channel_dim=args.channel_dim,
        token_mixing_hidden_dim=args.token_mixing_hidden_dim,
        channel_mixing_hidden_dim=args.channel_mixing_hidden_dim,
        num_mixer_blocks=args.num_mixer_blocks,
        time_embedding_dim=args.time_embedding_dim,
        frame_embedding_dim=args.frame_embedding_dim,
        observation_embedding_dim=args.observation_embedding_dim,
    )
    model = FlowMatchingPolicy(model_config)
    numpy_rng = np.random.default_rng(args.seed)
    initial_batch = train_dataset.sample_batch(args.batch_size, numpy_rng)
    jax_rng = jax.random.PRNGKey(args.seed)
    jax_rng, initialization_key = jax.random.split(jax_rng)
    variables = model.init(
        initialization_key,
        jnp.asarray(initial_batch.images),
        jnp.zeros_like(jnp.asarray(initial_batch.actions)),
        jnp.zeros((args.batch_size,), dtype=jnp.float32),
    )
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optimizer,
    )

    @jax.jit
    def train_step(
        current_state: TrainState,
        images: jax.Array,
        actions: jax.Array,
        action_mask: jax.Array,
        loss_key: jax.Array,
    ):
        def loss_function(params):
            return flow_matching_loss(
                model,
                params,
                images,
                actions,
                action_mask,
                loss_key,
            )

        (loss, metrics), gradients = jax.value_and_grad(
            loss_function,
            has_aux=True,
        )(current_state.params)
        metrics = {
            **metrics,
            "gradient_norm": optax.global_norm(gradients),
            "learning_rate": schedule(current_state.step),
        }
        return current_state.apply_gradients(grads=gradients), loss, metrics

    @jax.jit
    def validation_step(
        params,
        images: jax.Array,
        actions: jax.Array,
        action_mask: jax.Array,
        loss_key: jax.Array,
    ):
        return flow_matching_loss(
            model,
            params,
            images,
            actions,
            action_mask,
            loss_key,
        )[1]

    print(
        "Training image-conditioned action-chunk flow model\n"
        f"  datasets: {', '.join(train_dataset.controllers)}\n"
        f"  images: {args.frame_stack} x {train_dataset.image_shape}\n"
        f"  action chunks: {args.action_horizon} x {train_dataset.action_dim}\n"
        f"  train transitions: {train_dataset.total_examples}\n"
        f"  optimizer steps: {total_steps} ({steps_per_epoch} per epoch)"
    )

    training_start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = []
        for _ in range(steps_per_epoch):
            batch = train_dataset.sample_batch(args.batch_size, numpy_rng)
            jax_rng, loss_key = jax.random.split(jax_rng)
            state, _, metrics = train_step(
                state,
                jnp.asarray(batch.images),
                jnp.asarray(batch.actions),
                jnp.asarray(batch.action_mask),
                loss_key,
            )
            train_metrics.append(jax.device_get(metrics))

        validation_metrics = []
        if validation_dataset is not None:
            for _ in range(args.validation_batches):
                batch = validation_dataset.sample_batch(args.batch_size, numpy_rng)
                jax_rng, loss_key = jax.random.split(jax_rng)
                metrics = validation_step(
                    state.params,
                    jnp.asarray(batch.images),
                    jnp.asarray(batch.actions),
                    jnp.asarray(batch.action_mask),
                    loss_key,
                )
                validation_metrics.append(jax.device_get(metrics))

        def mean_metric(metrics: list[dict], name: str) -> float:
            return float(np.mean([np.asarray(metric[name]) for metric in metrics]))

        record = {
            "epoch": epoch,
            "step": int(jax.device_get(state.step)),
            "train_loss": mean_metric(train_metrics, "loss"),
            "train_gradient_norm": mean_metric(train_metrics, "gradient_norm"),
            "learning_rate": mean_metric(train_metrics, "learning_rate"),
            "elapsed_seconds": time.time() - training_start,
        }
        if validation_metrics:
            record["validation_loss"] = mean_metric(validation_metrics, "loss")
        with metrics_path.open("a") as metrics_file:
            metrics_file.write(json.dumps(record, sort_keys=True) + "\n")

        validation_text = (
            f", validation_loss={record['validation_loss']:.5f}"
            if "validation_loss" in record
            else ""
        )
        print(
            f"epoch={epoch:03d}/{args.epochs}, step={record['step']}, "
            f"train_loss={record['train_loss']:.5f}{validation_text}"
        )

        if epoch % args.checkpoint_every == 0:
            save_params(
                _checkpoint_payload(state, model_config, args, train_dataset, epoch),
                output_dir / f"epoch_{epoch:04d}.pbz2",
            )

    save_params(
        _checkpoint_payload(state, model_config, args, train_dataset, args.epochs),
        final_checkpoint_path,
    )
    print(f"Final flow policy saved to {final_checkpoint_path}")
    return final_checkpoint_path


def main(argv: list[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
