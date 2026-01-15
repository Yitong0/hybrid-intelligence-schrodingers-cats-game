from dataclasses import dataclass
from typing import Optional, List, Tuple
from .rules import Claim
from .cards import Card

@dataclass(frozen=True)
class PublicEvent:
    pid: int
    kind: str  # "claim" or "doubt"
    claim: Optional[Claim] = None
    revealed_count: int = 0

@dataclass
class PublicState:
    current_claim: Optional[Claim]
    revealed: dict[int, list[Card]]
    turn: int
    round_idx: int
    history: List[PublicEvent]      

@dataclass(frozen=True)
class Observation:
    """
    What a player can observe:
    - their own hand (private)
    - the public state (claim, evidence, history, etc.)
    """
    my_id: int
    my_hand: List[Card]
    public: PublicState   