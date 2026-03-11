from dataclasses import dataclass


@dataclass(frozen=True)
class QuantConfig:
    w_bits: int = 4
    group_size: int = 128
    sym: bool = False
    rank_eps: float = 1e-5
