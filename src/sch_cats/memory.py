from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from .rules import Claim


@dataclass
class RoundMemory:
    """
    Explicit memory of (at least) the previous round.
    Must influence ToM0 decisions per project variant.
    """
    opponent_doubted: bool
    opponent_evidence_revealed: int
    opponent_last_claim: Optional[Claim]