import importlib.util
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest


_SPEC = importlib.util.spec_from_file_location(
    "mlx_vlm_int8_prefill_under_test",
    Path(__file__).parents[1] / "mlx_vlm" / "int8_prefill.py",
)
int8_prefill = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(int8_prefill)


@pytest.fixture(autouse=True)
def restore_quantized_linear():
    int8_prefill.remove()
    yield
    int8_prefill.remove()


def _small_w4_linear():
    linear = nn.QuantizedLinear(
        1024, 1024, bias=False, group_size=64, bits=4
    )
    linear.scales = linear.scales.astype(mx.bfloat16)
    linear.biases = linear.biases.astype(mx.bfloat16)
    return linear


@pytest.mark.skipif(
    "M5" not in str(mx.device_info().get("device_name", "")),
    reason="the int8 prefill overlay requires M5 tensor operations",
)
def test_int8_prefill_is_bounded_reversible_and_close(monkeypatch):
    monkeypatch.setattr(int8_prefill, "SCOPE", "all")
    mx.random.seed(25)
    linear = _small_w4_linear()
    x = mx.random.normal((1, 512, 1024)).astype(mx.bfloat16)

    reference = linear(x)
    mx.eval(reference)
    assert int8_prefill.apply() is True
    assert int8_prefill.apply() is False

    actual = linear(x)
    mx.eval(actual)
    rel_l2 = mx.linalg.norm(
        actual.astype(mx.float32) - reference.astype(mx.float32)
    ) / mx.maximum(mx.linalg.norm(reference.astype(mx.float32)), 1e-8)
    assert float(rel_l2) < 0.03
    assert len(int8_prefill._int8_weights) == 0
    assert len(int8_prefill._act_cache) <= 1

    assert int8_prefill.remove() is True
    assert int8_prefill.remove() is False
    restored = linear(x)
    mx.eval(restored)
    assert bool(mx.array_equal(reference, restored).item())


@pytest.mark.skipif(
    "M5" not in str(mx.device_info().get("device_name", "")),
    reason="the int8 prefill overlay requires M5 tensor operations",
)
def test_int8_prefill_preserves_float32_semantics(monkeypatch):
    monkeypatch.setattr(int8_prefill, "SCOPE", "all")
    linear = _small_w4_linear()
    x = mx.random.normal((1, 512, 1024)).astype(mx.float32)
    reference = linear(x)
    mx.eval(reference)

    int8_prefill.apply()
    actual = linear(x)
    mx.eval(actual)

    assert actual.dtype == reference.dtype == mx.float32
    assert bool(mx.array_equal(reference, actual).item())
