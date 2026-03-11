from typing import Any


def run_noop_quantization(params: Any, calibration_tokens) -> Any:
    _ = calibration_tokens
    return params
