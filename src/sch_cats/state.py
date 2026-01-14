from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Set
from .rules import Claim
from .cards import Card


@dataclass
class PublicState:
    current_claim: Optional[Claim]
    # evidence revealed so far: player_id -> list of revealed cards (public)
    revealed: Dict[int, List[Card]]
    # whose turn
    turn: int
    # round index inside a match (for explicit memory)
    round_idx: int


@dataclass
class Observation:
    """
    What an agent can use.
    Must NOT contain hidden info about opponent hand.
    Your own hand is private information you do observe.
    """
    my_id: int
    my_hand: List[Card]
    public: PublicState