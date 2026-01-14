from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import random

from .base import Agent
from ..state import Observation
from ..env import Action, MakeClaim, Doubt
from ..rules import Claim, QState, check_claim_is_true
from ..memory import RoundMemory
from ..cards import Card


class ToM0MemoryAgent(Agent):
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.last_memory: Optional[RoundMemory] = None

    def act(self, obs: Observation) -> Action:
        pub = obs.public
        my_hand = obs.my_hand

        # Very simple “statistical” proxy:
        # estimate how many of target exist by (my count + expected opponent count)
        def my_count(q: QState) -> int:
            return sum(1 for c in my_hand if c.value == q.value) + sum(1 for c in my_hand if c == Card.HUP)

        # aggressiveness adjusted by explicit memory:
        # if opponent doubted often last round -> be more conservative
        conservative = False
        if self.last_memory is not None:
            conservative = self.last_memory.opponent_doubted

        # If no claim yet: open with something you can strongly support
        if pub.current_claim is None:
            # choose best qstate by my_count
            choices = [QState.DEAD, QState.ALIVE, QState.EMPTY]
            q = max(choices, key=my_count)
            qty = max(1, my_count(q) - (1 if conservative else 0))
            reveal = tuple(i for i, c in enumerate(my_hand) if c == Card.HUP or c.value == q.value)
            return MakeClaim(Claim(qty, q), reveal if not conservative else ())

        # If there is a claim: decide doubt vs raise
        current = pub.current_claim
        assert current is not None

        # crude estimate: assume opponent contributes ~2 matching cards on average
        est_total = my_count(current.qstate) + 2

        # doubt if claim seems too high (more conservative = doubt earlier)
        margin = 0 if conservative else 1
        if est_total + margin < current.qty:
            return Doubt()

        # otherwise raise slightly
        new_qty = min(12, current.qty + 1)
        new_q = current.qstate
        reveal = tuple(i for i, c in enumerate(my_hand) if c == Card.HUP or c.value == new_q.value)
        return MakeClaim(Claim(new_qty, new_q), reveal if not conservative else ())

    def observe_round_end(self, *, my_id: int, winner: int, last_obs: Observation) -> None:
        # Build explicit memory from last round’s public info
        pub = last_obs.public
        opp = 1 - my_id

        # Did opponent doubt? We can infer: if round ended on opponent turn with Doubt
        # (In a fuller implementation you’d log actions; here we approximate with “round ended and it was my turn next”)
        opponent_doubted = (winner == my_id and pub.turn == my_id)

        opponent_evidence = len(pub.revealed.get(opp, []))
        opponent_last_claim = pub.current_claim

        self.last_memory = RoundMemory(
            opponent_doubted=opponent_doubted,
            opponent_evidence_revealed=opponent_evidence,
            opponent_last_claim=opponent_last_claim,
        )