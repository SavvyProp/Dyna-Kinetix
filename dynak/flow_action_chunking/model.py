"""CNN-conditioned rectified-flow model for continuous action chunks.

The action backbone follows the RTC Kinetix model at a small scale: each
future action is a token, four MLP-Mixer blocks mix across time and channels,
and sinusoidal flow-time embeddings condition adaptive layer normalization.
Images replace the paper benchmark's symbolic observation encoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn


@dataclass(frozen=True)
class FlowModelConfig:
    action_horizon: int = 8
    action_dim: int = 3
    frame_stack: int = 1
    channel_dim: int = 256
    token_mixing_hidden_dim: int = 64
    channel_mixing_hidden_dim: int = 512
    num_mixer_blocks: int = 4
    time_embedding_dim: int = 256
    frame_embedding_dim: int = 128
    observation_embedding_dim: int = 256


def sinusoidal_embedding(timestep: jax.Array, dimension: int) -> jax.Array:
    """Embed scalar or per-token flow times with sinusoidal features."""
    half_dimension = dimension // 2
    periods = 4e-3 * (4.0 / 4e-3) ** jnp.linspace(
        0.0,
        1.0,
        half_dimension,
        dtype=jnp.float32,
    )
    arguments = timestep[..., None] / periods * (2.0 * jnp.pi)
    embedding = jnp.concatenate((jnp.sin(arguments), jnp.cos(arguments)), axis=-1)
    if dimension % 2:
        padding = [(0, 0)] * embedding.ndim
        padding[-1] = (0, 1)
        embedding = jnp.pad(embedding, padding)
    return embedding


class ImageHistoryEncoder(nn.Module):
    config: FlowModelConfig

    @nn.compact
    def __call__(self, images: jax.Array) -> jax.Array:
        if images.ndim != 5:
            raise ValueError(
                "images must have shape (batch, frame_stack, height, width, channels)"
            )
        batch_size, frame_stack, height, width, channels = images.shape
        if frame_stack != self.config.frame_stack:
            raise ValueError(
                f"Model expects {self.config.frame_stack} frames, got {frame_stack}"
            )

        x = images.reshape(batch_size * frame_stack, height, width, channels)
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="SAME")(x)
        x = nn.gelu(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(x)
        x = nn.gelu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.gelu(x)
        x = x.mean(axis=(1, 2))
        x = nn.Dense(self.config.frame_embedding_dim)(x)
        x = nn.gelu(x)
        x = x.reshape(batch_size, frame_stack * self.config.frame_embedding_dim)
        x = nn.Dense(self.config.observation_embedding_dim)(x)
        return nn.LayerNorm()(nn.gelu(x))


class FlowTimeEncoder(nn.Module):
    config: FlowModelConfig

    @nn.compact
    def __call__(self, flow_time: jax.Array) -> jax.Array:
        x = sinusoidal_embedding(flow_time, self.config.time_embedding_dim)
        x = nn.Dense(self.config.channel_dim)(x)
        x = nn.silu(x)
        x = nn.Dense(self.config.channel_dim)(x)
        return nn.silu(x)


class AdaptiveLayerNorm(nn.Module):
    channel_dim: int

    @nn.compact
    def __call__(
        self,
        tokens: jax.Array,
        condition: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        shift_scale_gate = nn.Dense(
            3 * self.channel_dim,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
        )(condition)
        scale, shift, gate = jnp.split(shift_scale_gate, 3, axis=-1)
        normalized = nn.LayerNorm(use_bias=False, use_scale=False)(tokens)
        normalized = normalized * (1.0 + scale) + shift
        return normalized, gate


class FinalAdaptiveLayerNorm(nn.Module):
    channel_dim: int

    @nn.compact
    def __call__(self, tokens: jax.Array, condition: jax.Array) -> jax.Array:
        scale_shift = nn.Dense(
            2 * self.channel_dim,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
        )(condition)
        scale, shift = jnp.split(scale_shift, 2, axis=-1)
        normalized = nn.LayerNorm(use_bias=False, use_scale=False)(tokens)
        return normalized * (1.0 + scale) + shift


class MLPMixerBlock(nn.Module):
    config: FlowModelConfig

    @nn.compact
    def __call__(self, tokens: jax.Array, time_condition: jax.Array) -> jax.Array:
        token_input, token_gate = AdaptiveLayerNorm(self.config.channel_dim)(
            tokens,
            time_condition,
        )
        token_update = jnp.swapaxes(token_input, 1, 2)
        token_update = nn.Dense(
            self.config.token_mixing_hidden_dim,
            use_bias=False,
        )(token_update)
        token_update = nn.gelu(token_update)
        token_update = nn.Dense(self.config.action_horizon, use_bias=False)(
            token_update
        )
        token_update = jnp.swapaxes(token_update, 1, 2)
        tokens = tokens + token_gate * token_update

        channel_input, channel_gate = AdaptiveLayerNorm(self.config.channel_dim)(
            tokens,
            time_condition,
        )
        channel_update = nn.Dense(
            self.config.channel_mixing_hidden_dim,
            use_bias=False,
        )(channel_input)
        channel_update = nn.gelu(channel_update)
        channel_update = nn.Dense(self.config.channel_dim, use_bias=False)(
            channel_update
        )
        return tokens + channel_gate * channel_update


class FlowMatchingPolicy(nn.Module):
    """Predict the rectified-flow velocity of an entire action chunk."""

    config: FlowModelConfig

    @nn.compact
    def __call__(
        self,
        images: jax.Array,
        noisy_actions: jax.Array,
        flow_time: jax.Array,
    ) -> jax.Array:
        if noisy_actions.shape[1:] != (
            self.config.action_horizon,
            self.config.action_dim,
        ):
            raise ValueError(
                "noisy_actions must have shape "
                f"(batch, {self.config.action_horizon}, {self.config.action_dim})"
            )
        if flow_time.shape not in (
            (images.shape[0],),
            (images.shape[0], self.config.action_horizon),
        ):
            raise ValueError("flow_time must have shape (batch,) or (batch, horizon)")

        observation = ImageHistoryEncoder(self.config)(images)
        observation = jnp.repeat(
            observation[:, None, :],
            self.config.action_horizon,
            axis=1,
        )
        tokens = jnp.concatenate((noisy_actions, observation), axis=-1)
        tokens = nn.Dense(self.config.channel_dim)(tokens)
        if flow_time.ndim == 1:
            flow_time = jnp.broadcast_to(
                flow_time[:, None],
                (images.shape[0], self.config.action_horizon),
            )
        time_condition = FlowTimeEncoder(self.config)(flow_time)

        for _ in range(self.config.num_mixer_blocks):
            tokens = MLPMixerBlock(self.config)(tokens, time_condition)

        tokens = FinalAdaptiveLayerNorm(self.config.channel_dim)(
            tokens,
            time_condition,
        )
        return nn.Dense(self.config.action_dim)(tokens)


def flow_matching_loss(
    model: FlowMatchingPolicy,
    params,
    images: jax.Array,
    expert_actions: jax.Array,
    action_mask: jax.Array,
    rng: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Conditional rectified-flow loss from Gaussian noise to expert chunks."""
    noise_key, time_key = jax.random.split(rng)
    noise = jax.random.normal(noise_key, expert_actions.shape)
    flow_time = jax.random.uniform(
        time_key,
        shape=(expert_actions.shape[0],),
        minval=0.0,
        maxval=1.0,
    )
    interpolation = (1.0 - flow_time[:, None, None]) * noise + flow_time[
        :, None, None
    ] * expert_actions
    target_velocity = expert_actions - noise
    predicted_velocity = model.apply(
        {"params": params},
        images,
        interpolation,
        flow_time,
    )
    token_squared_error = jnp.mean(
        jnp.square(predicted_velocity - target_velocity),
        axis=-1,
    )
    mask = action_mask.astype(token_squared_error.dtype)
    valid_tokens = jnp.maximum(mask.sum(), 1.0)
    loss = jnp.sum(token_squared_error * mask) / valid_tokens
    metrics = {
        "loss": loss,
        "valid_tokens": mask.sum(),
        "prediction_rms": jnp.sqrt(jnp.mean(jnp.square(predicted_velocity))),
        "target_rms": jnp.sqrt(jnp.mean(jnp.square(target_velocity))),
    }
    return loss, metrics


def sample_action_chunks(
    model: FlowMatchingPolicy,
    params,
    images: jax.Array,
    rng: jax.Array,
    *,
    num_steps: int = 5,
) -> jax.Array:
    """Generate action chunks with unguided Euler integration.

    This is the base flow sampler only. It intentionally contains no RTC
    prefix guidance or asynchronous execution logic.
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be greater than zero")
    batch_size = images.shape[0]
    actions = jax.random.normal(
        rng,
        (
            batch_size,
            model.config.action_horizon,
            model.config.action_dim,
        ),
    )
    step_size = 1.0 / num_steps

    def euler_step(current_actions, step_index):
        flow_time = jnp.full(
            (batch_size,),
            step_index * step_size,
            dtype=jnp.float32,
        )
        velocity = model.apply(
            {"params": params},
            images,
            current_actions,
            flow_time,
        )
        return current_actions + step_size * velocity, None

    actions, _ = jax.lax.scan(
        euler_step,
        actions,
        jnp.arange(num_steps, dtype=jnp.float32),
    )
    return actions
