"""A tiny PTQ layer-reconstruction example.

This is a pedagogical symmetric per-row quantizer.  It is deliberately not a
GPTQ, AWQ, or runtime kernel implementation.  The example only separates the
floating-point reference from a dequantized pseudo-quantized path so that the
error accounting can be replayed without a GPU.
"""

from __future__ import annotations

import argparse
import numpy as np


def quantize_symmetric_per_row(weights: np.ndarray, bits: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Return integer weights and one positive scale per row."""

    if weights.ndim != 2:
        raise ValueError("weights must have shape [out_features, in_features]")
    if bits < 2 or bits > 8:
        raise ValueError("bits must be between 2 and 8")

    qmax = (1 << (bits - 1)) - 1
    max_abs = np.max(np.abs(weights), axis=1, keepdims=True)
    scales = np.where(max_abs == 0, 1.0, max_abs / qmax)
    quantized = np.clip(np.rint(weights / scales), -qmax, qmax).astype(np.int8)
    return quantized, scales.astype(np.float64)


def reconstruct_output(inputs: np.ndarray, weights: np.ndarray, bits: int) -> dict[str, float]:
    """Compare a floating-point layer with its dequantized approximation."""

    if inputs.ndim != 2 or weights.ndim != 2:
        raise ValueError("inputs and weights must both be matrices")
    if inputs.shape[1] != weights.shape[1]:
        raise ValueError("input features must match weight columns")

    reference = inputs @ weights.T
    qweights, scales = quantize_symmetric_per_row(weights, bits)
    dequantized = qweights.astype(np.float64) * scales
    pseudo = inputs @ dequantized.T
    denominator = max(float(np.linalg.norm(reference)), 1e-12)
    return {
        "reference_norm": float(np.linalg.norm(reference)),
        "max_abs_output_error": float(np.max(np.abs(pseudo - reference))),
        "relative_l2_output_error": float(np.linalg.norm(pseudo - reference) / denominator),
        "weight_scale_min": float(np.min(scales)),
        "weight_scale_max": float(np.max(scales)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, default=4)
    args = parser.parse_args()

    inputs = np.array([[1.0, 1.0], [1.0, -1.0], [2.0, 0.5]], dtype=np.float64)
    weights = np.array([[0.6, 0.43], [0.6, -0.37]], dtype=np.float64)
    metrics = reconstruct_output(inputs, weights, args.bits)
    for name, value in metrics.items():
        print(f"{name}={value:.8f}")


if __name__ == "__main__":
    main()
