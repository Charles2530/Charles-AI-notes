import importlib.util
from pathlib import Path
from unittest import skipUnless

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "files" / "assets" / "examples" / "quantization" / "ptq_layer_reconstruction.py"


def load_example():
    spec = importlib.util.spec_from_file_location("ptq_layer_reconstruction", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@skipUnless(np is not None, "quantization example requires NumPy")
def test_per_row_pseudo_quantization_is_finite_and_shape_preserving():
    example = load_example()
    inputs = np.array([[1.0, 1.0], [1.0, -1.0]])
    weights = np.array([[0.6, 0.43], [0.6, -0.37]])

    quantized, scales = example.quantize_symmetric_per_row(weights, bits=4)
    metrics = example.reconstruct_output(inputs, weights, bits=4)

    assert quantized.shape == weights.shape
    assert scales.shape == (weights.shape[0], 1)
    assert np.all(np.isfinite(scales))
    assert 0.0 < metrics["relative_l2_output_error"] < 0.1


@skipUnless(np is not None, "quantization example requires NumPy")
def test_zero_row_uses_a_finite_neutral_scale():
    example = load_example()
    quantized, scales = example.quantize_symmetric_per_row(
        np.array([[0.0, 0.0], [1.0, -1.0]])
    )

    np.testing.assert_array_equal(quantized[0], [0, 0])
    np.testing.assert_array_equal(scales[0], [1.0])
