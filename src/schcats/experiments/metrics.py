from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Tuple


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.
    Returns (low, high).
    """
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054  # ~95%
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    half = (z / denom) * math.sqrt((phat * (1 - phat) / n) + (z * z) / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


@dataclass
class WinStats:
    wins: int
    losses: int

    @property
    def n(self) -> int:
        return self.wins + self.losses

    @property
    def winrate(self) -> float:
        return self.wins / self.n if self.n else 0.0

    def ci95(self) -> Tuple[float, float]:
        return wilson_ci(self.wins, self.n)