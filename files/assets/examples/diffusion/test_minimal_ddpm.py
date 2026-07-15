import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

MODULE_PATH = Path(__file__).with_name("minimal_ddpm.py")
SPEC = importlib.util.spec_from_file_location("minimal_ddpm", MODULE_PATH)
minimal_ddpm = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = minimal_ddpm
assert SPEC.loader is not None
SPEC.loader.exec_module(minimal_ddpm)


def test_linear_schedule_has_valid_monotone_coefficients():
    schedule = minimal_ddpm.make_linear_schedule(8)

    assert schedule.betas.shape == (8,)
    assert torch.all((schedule.betas > 0) & (schedule.betas < 1))
    assert torch.all(schedule.alpha_bars[1:] < schedule.alpha_bars[:-1])
    assert torch.all(schedule.posterior_variance >= 0)


def test_q_sample_matches_the_closed_form_without_mutation():
    schedule = minimal_ddpm.make_linear_schedule(8)
    x0 = torch.tensor([[[[1.0, -1.0]]], [[[0.5, 0.25]]]])
    noise = torch.tensor([[[[0.2, -0.4]]], [[[1.0, -1.0]]]])
    t = torch.tensor([0, 7], dtype=torch.long)
    snapshots = (x0.clone(), noise.clone())

    actual = minimal_ddpm.q_sample(x0, t, schedule, noise)

    expected = (
        schedule.sqrt_alpha_bars[t].view(2, 1, 1, 1) * x0
        + schedule.sqrt_one_minus_alpha_bars[t].view(2, 1, 1, 1) * noise
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x0, snapshots[0])
    torch.testing.assert_close(noise, snapshots[1])


def test_q_sample_rejects_invalid_timestep_batches():
    schedule = minimal_ddpm.make_linear_schedule(8)
    x0 = torch.zeros(2, 1, 4, 4)

    with pytest.raises(ValueError, match="one timestep per batch item"):
        minimal_ddpm.q_sample(x0, torch.tensor([1]), schedule)
    with pytest.raises(ValueError, match="inside the schedule"):
        minimal_ddpm.q_sample(x0, torch.tensor([0, 8]), schedule)


def test_tiny_denoiser_preserves_image_shape():
    model = minimal_ddpm.TinyDenoiser(in_channels=1, base_channels=16, time_dim=32)
    x = torch.randn(3, 1, 28, 28)
    t = torch.tensor([0, 3, 7])

    assert model(x, t).shape == x.shape


def test_diffusion_loss_is_finite_and_updates_parameters():
    torch.manual_seed(7)
    schedule = minimal_ddpm.make_linear_schedule(8)
    model = minimal_ddpm.TinyDenoiser(in_channels=1, base_channels=16, time_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x0 = torch.rand(4, 1, 28, 28) * 2.0 - 1.0
    t = torch.tensor([0, 2, 5, 7])
    before = [parameter.detach().clone() for parameter in model.parameters()]

    loss = minimal_ddpm.diffusion_loss(model, x0, t, schedule)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, model.parameters())
    )


class ZeroNoise(torch.nn.Module):
    def forward(self, x, t):
        del t
        return torch.zeros_like(x)


def test_p_sample_is_deterministic_at_t_zero():
    schedule = minimal_ddpm.make_linear_schedule(8)
    xt = torch.randn(2, 1, 4, 4)
    t = torch.zeros(2, dtype=torch.long)

    first = minimal_ddpm.p_sample(ZeroNoise(), xt, t, schedule)
    second = minimal_ddpm.p_sample(ZeroNoise(), xt, t, schedule)

    torch.testing.assert_close(first, second)


def test_sample_loop_is_seeded_finite_and_shape_preserving():
    schedule = minimal_ddpm.make_linear_schedule(8)

    first = minimal_ddpm.sample_loop(
        ZeroNoise(), (2, 1, 4, 4), schedule, seed=11
    )
    second = minimal_ddpm.sample_loop(
        ZeroNoise(), (2, 1, 4, 4), schedule, seed=11
    )

    assert first.shape == (2, 1, 4, 4)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second)


def test_smoke_command_runs_one_update_and_sampling_chain():
    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "smoke",
            "--steps",
            "4",
            "--image-size",
            "8",
            "--batch-size",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    assert result["loss"] >= 0.0
    assert result["sample_shape"] == [2, 1, 8, 8]
    assert result["finite"] is True
