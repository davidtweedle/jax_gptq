from dataclasses import dataclass
from typing import Literal


SolverMode = Literal["chol", "eigh"]
OrderingMode = Literal[
    "none",
    "diag_h",
    "pivot_sqrt_h",
    "pivot_sqrt_hinv_smallest",
]
TruncationMode = Literal["abs", "percent", "mean_top32", "energy"]


@dataclass(frozen=True)
class QuantConfig:
    w_bits: int = 4
    group_size: int = 128
    sym: bool = False
    damp_percent: float = 0.01
    rank_eps: float = 1e-5
    truncation_mode: TruncationMode = "abs"
    truncation_value: float = 1e-5
    solver_mode: SolverMode = "chol"
    ordering_mode: OrderingMode = "none"
