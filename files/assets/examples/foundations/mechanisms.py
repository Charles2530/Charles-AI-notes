"""Small, dependency-free numerical examples for the technical handbook."""

import json
import math


def _matrix_width(matrix):
    if not matrix or not matrix[0]:
        raise ValueError("matrix must be nonempty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix rows must have the same width")
    return width


def affine(x, weights, bias):
    """Row-vector convention: [D_in] @ [D_in, D_out] + [D_out]."""
    width = _matrix_width(weights)
    if len(x) != len(weights) or len(bias) != width:
        raise ValueError("input, weight and bias dimensions must match")
    return [sum(x[i] * weights[i][j] for i in range(len(x))) + bias[j]
            for j in range(width)]


def regression_loss_and_gradients(x, y, weight, bias):
    """Return half mean squared error and its two analytic derivatives."""
    if not x or len(x) != len(y):
        raise ValueError("samples must be nonempty and paired")
    errors = [weight * value + bias - target for value, target in zip(x, y)]
    n = len(x)
    return (sum(error * error for error in errors) / (2 * n),
            sum(error * value for error, value in zip(errors, x)) / n,
            sum(errors) / n)


def softmax(logits):
    """Normalize finite logits after subtracting their maximum."""
    if not logits or not all(math.isfinite(value) for value in logits):
        raise ValueError("logits must be nonempty and finite")
    peak = max(logits)
    values = [math.exp(value - peak) for value in logits]
    denominator = sum(values)
    return [value / denominator for value in values]


def attention(queries, keys, values, *, causal=False):
    """Single head; causal=True uses aligned self-attention positions.

    Returns (outputs, attention_weights). This illustrates the operation,
    not a tensor-library implementation or an optimized attention kernel.
    """
    width = _matrix_width(queries)
    key_width = _matrix_width(keys)
    value_width = _matrix_width(values)
    if width != key_width or len(keys) != len(values):
        raise ValueError("Q/K dimensions and K/V sequence lengths must match")
    if causal and len(queries) != len(keys):
        raise ValueError("this causal example requires aligned self-attention")
    outputs, all_weights = [], []
    for position, query in enumerate(queries):
        visible = position + 1 if causal else len(keys)
        scores = [sum(q * k for q, k in zip(query, key)) / math.sqrt(width)
                  for key in keys[:visible]]
        weights = softmax(scores) + [0.0] * (len(keys) - visible)
        outputs.append([sum(weights[i] * values[i][j] for i in range(len(values)))
                        for j in range(value_width)])
        all_weights.append(weights)
    return outputs, all_weights


def main():
    loss, dw, db = regression_loss_and_gradients([1, 2], [2, 4], 1, 0)
    updated = regression_loss_and_gradients([1, 2], [2, 4], 1 - .1 * dw, -.1 * db)[0]
    output, weights = attention([[1, 0], [0, 1]], [[1, 0], [0, 1]],
                                [[1, 0], [0, 1]], causal=True)
    # Each continuation distribution sums to one. Greedy chooses A first,
    # although B followed by Y has higher probability than either A branch.
    paths = {"AX": .6 * .5, "AY": .6 * .5, "BX": .4 * .01, "BY": .4 * .99}
    print(json.dumps({
        "affine": affine([1, 2], [[1, 2], [3, 4]], [.5, -.5]),
        "regression": {"loss": loss, "dw": dw, "db": db, "updated_loss": updated},
        "causal_attention": {"output": output, "weights": weights},
        "decoding": {"sequence_probabilities": paths, "greedy_first_token": "A",
                     "most_probable_sequence": max(paths, key=paths.get)},
    }, indent=2))


if __name__ == "__main__":
    main()
