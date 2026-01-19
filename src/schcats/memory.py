from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from .rules import Claim


@dataclass
class RoundMemory:
    """
    Explicit memory of at least the previous round.
    Used by ToM0 (required) and informs ToM1 (required by variant text).
    """
    opponent_doubted: bool
    opponent_evidence_revealed: int
    opponent_last_claim: Optional[Claim]
    opponent_last_claim_was_true: Optional[bool]