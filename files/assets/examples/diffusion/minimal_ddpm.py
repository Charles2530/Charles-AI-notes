"""Minimal DDPM components used by the diffusion-model notes."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DiffusionSchedule:
    """Precomputed coefficients for one discrete DDPM noise schedule."""

    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    alpha_bars_prev: torch.Tensor
    sqrt_alpha_bars: torch.Tensor
    sqrt_one_minus_alpha_bars: torch.Tensor
    posterior_variance: torch.Tensor


def make_linear_schedule(
    num_steps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device: torch.device | str | None = None,
) -> DiffusionSchedule:
    """Create the linear beta schedule used by the teaching implementation."""
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 2:
        raise ValueError("num_steps must be an integer of at least 2")
    if not 0.0 < beta_start < beta_end < 1.0:
        raise ValueError("betas must satisfy 0 < beta_start < beta_end < 1")

    betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bars_prev = torch.cat((torch.ones_like(alpha_bars[:1]), alpha_bars[:-1]))
    posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        alpha_bars_prev=alpha_bars_prev,
        sqrt_alpha_bars=torch.sqrt(alpha_bars),
        sqrt_one_minus_alpha_bars=torch.sqrt(1.0 - alpha_bars),
        posterior_variance=posterior_variance,
    )


def _extract(values: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if t.dtype != torch.long or t.ndim != 1 or t.shape[0] != x.shape[0]:
        raise ValueError("t must provide one timestep per batch item")
    if t.device != x.device:
        raise ValueError("t and x must be on the same device")
    if values.device != x.device:
        raise ValueError("schedule and x must be on the same device")
    if torch.any(t < 0) or torch.any(t >= values.shape[0]):
        raise ValueError("timesteps must be inside the schedule")
    return values.gather(0, t).reshape(t.shape[0], *((1,) * (x.ndim - 1)))


def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    schedule: DiffusionSchedule,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample x_t directly from q(x_t | x_0)."""
    if not torch.is_floating_point(x0) or x0.ndim != 4:
        raise ValueError("x0 must be a floating BCHW tensor")
    noise = torch.randn_like(x0) if noise is None else noise
    if noise.shape != x0.shape or noise.device != x0.device:
        raise ValueError("noise must match x0 shape and device")
    return (
        _extract(schedule.sqrt_alpha_bars, t, x0) * x0
        + _extract(schedule.sqrt_one_minus_alpha_bars, t, x0) * noise
    )


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        if dimension < 2:
            raise ValueError("time embedding dimension must be at least 2")
        self.dimension = dimension

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1:
            raise ValueError("t must be a one-dimensional batch")
        half = self.dimension // 2
        exponent = torch.arange(half, device=t.device, dtype=torch.float32)
        exponent = exponent / max(half - 1, 1)
        scale = torch.tensor(10_000.0, device=t.device, dtype=torch.float32)
        frequencies = torch.exp(-torch.log(scale) * exponent)
        angles = t.float()[:, None] * frequencies[None]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
        if embedding.shape[1] < self.dimension:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class TimeResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        if out_channels % 8 != 0:
            raise ValueError("out_channels must be divisible by 8 for GroupNorm")
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.time_projection(time_embedding)[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return hidden + self.skip(x)


class TinyDenoiser(nn.Module):
    """One-level U-Net-shaped denoiser for 2D teaching examples."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        time_dim: int = 64,
    ):
        super().__init__()
        if base_channels % 8 != 0:
            raise ValueError("base_channels must be divisible by 8")
        self.in_channels = in_channels
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input_projection = nn.Conv2d(
            in_channels, base_channels, kernel_size=3, padding=1
        )
        self.encoder = TimeResidualBlock(base_channels, base_channels, time_dim)
        self.downsample = nn.Conv2d(
            base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1
        )
        self.middle = TimeResidualBlock(
            base_channels * 2, base_channels * 2, time_dim
        )
        self.upsample = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1
        )
        self.decoder = TimeResidualBlock(base_channels * 2, base_channels, time_dim)
        self.output_norm = nn.GroupNorm(8, base_channels)
        self.output_projection = nn.Conv2d(
            base_channels, in_channels, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError("x must be a BCHW tensor with the configured channels")
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError("spatial dimensions must be divisible by two")
        if t.dtype != torch.long or t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError("t must provide one integer timestep per batch item")
        time_embedding = self.time_embedding(t)
        skip = self.encoder(self.input_projection(x), time_embedding)
        hidden = self.middle(self.downsample(skip), time_embedding)
        hidden = self.upsample(hidden)
        hidden = self.decoder(torch.cat((hidden, skip), dim=1), time_embedding)
        return self.output_projection(F.silu(self.output_norm(hidden)))


def diffusion_loss(
    model: nn.Module,
    x0: torch.Tensor,
    t: torch.Tensor,
    schedule: DiffusionSchedule,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    noise = torch.randn_like(x0) if noise is None else noise
    xt = q_sample(x0, t, schedule, noise)
    prediction = model(xt, t)
    if prediction.shape != noise.shape:
        raise ValueError("model prediction must match the sampled noise")
    return F.mse_loss(prediction, noise)


@torch.no_grad()
def p_sample(
    model: nn.Module,
    xt: torch.Tensor,
    t: torch.Tensor,
    schedule: DiffusionSchedule,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one DDPM reverse transition p_theta(x_{t-1} | x_t)."""
    predicted_noise = model(xt, t)
    if predicted_noise.shape != xt.shape:
        raise ValueError("model prediction must match xt")

    alpha_t = _extract(schedule.alphas, t, xt)
    beta_t = _extract(schedule.betas, t, xt)
    one_minus_alpha_bar_t = _extract(
        schedule.sqrt_one_minus_alpha_bars, t, xt
    )
    mean = torch.rsqrt(alpha_t) * (
        xt - beta_t * predicted_noise / one_minus_alpha_bar_t
    )

    if torch.all(t == 0):
        return mean
    noise = torch.randn(
        xt.shape, device=xt.device, dtype=xt.dtype, generator=generator
    )
    nonzero_mask = (t > 0).reshape(t.shape[0], *((1,) * (xt.ndim - 1)))
    variance = _extract(schedule.posterior_variance, t, xt)
    return mean + nonzero_mask * torch.sqrt(variance) * noise


@torch.no_grad()
def sample_loop(
    model: nn.Module,
    shape: tuple[int, int, int, int],
    schedule: DiffusionSchedule,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Run the complete ancestral DDPM sampling chain from x_T to x_0."""
    if len(shape) != 4 or any(size < 1 for size in shape):
        raise ValueError("shape must contain four positive BCHW dimensions")
    target_device = torch.device(device) if device is not None else schedule.betas.device
    if schedule.betas.device != target_device:
        raise ValueError("schedule must be created on the sampling device")

    generator = torch.Generator(device=target_device).manual_seed(seed)
    xt = torch.randn(shape, device=target_device, generator=generator)
    for step in reversed(range(schedule.betas.shape[0])):
        t = torch.full((shape[0],), step, device=target_device, dtype=torch.long)
        xt = p_sample(model, xt, t, schedule, generator)
    return xt


def run_smoke_test(
    num_steps: int = 8,
    image_size: int = 8,
    batch_size: int = 2,
) -> dict[str, object]:
    """Run one optimizer update and one complete sampling chain on random data."""
    if image_size < 4 or image_size % 2:
        raise ValueError("image_size must be an even integer of at least 4")
    torch.manual_seed(7)
    schedule = make_linear_schedule(num_steps)
    model = TinyDenoiser(in_channels=1, base_channels=8, time_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x0 = torch.rand(batch_size, 1, image_size, image_size) * 2.0 - 1.0
    t = torch.randint(0, num_steps, (batch_size,), dtype=torch.long)

    loss = diffusion_loss(model, x0, t, schedule)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    samples = sample_loop(
        model,
        (batch_size, 1, image_size, image_size),
        schedule,
        seed=11,
    )
    return {
        "loss": float(loss.detach()),
        "sample_shape": list(samples.shape),
        "finite": bool(torch.isfinite(samples).all()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser(
        "smoke", help="run one update and a short sampling chain"
    )
    smoke.add_argument("--steps", type=int, default=8)
    smoke.add_argument("--image-size", type=int, default=8)
    smoke.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args(argv)

    if args.command == "smoke":
        result = run_smoke_test(args.steps, args.image_size, args.batch_size)
        print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
