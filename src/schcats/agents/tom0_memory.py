from __future__ import annotations
from typing import Optional
import random

from .base import Agent
from ..state import Observation
from ..env import Action, MakeClaim, Doubt
from ..memory import RoundMemory
from ..cards import Card
from ..rules import Claim, QState


class ToM0MemoryAgent(Agent):
    """
    Zero-order Theory of Mind agent with explicit memory.

    - Uses only:
        * own hand
        * current public state
        * explicit memory of opponent's actions from the PREVIOUS round
    - Does NOT reason about opponent beliefs or intentions.
    """

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.last_memory: Optional[RoundMemory] = None

    def act(self, obs: Observation) -> Action:
        pub = obs.public
        my_hand = obs.my_hand

        # --- helper: how many cards I personally support for a state ---
        def my_count(q: QState) -> int:
            return sum(
                1 for c in my_hand
                if c == Card.HUP or c.value == q.value
            )

        # --- adjust aggressiveness using explicit memory ---
        # If opponent doubted last round -> play more conservatively
        conservative = False
        if self.last_memory is not None:
            conservative = self.last_memory.opponent_doubted

        # --- initial claim ---
        if pub.current_claim is None:
            choices = [QState.DEAD, QState.ALIVE, QState.EMPTY]
            q = max(choices, key=my_count)
            qty = max(1, my_count(q) - (1 if conservative else 0))

            reveal = tuple(
                i for i, c in enumerate(my_hand)
                if c == Card.HUP or c.value == q.value
            )

            return MakeClaim(Claim(qty, q), reveal if not conservative else ())

        # --- respond to existing claim ---
        current = pub.current_claim
        assert current is not None

        # crude likelihood proxy: my support + rough prior for opponent
        est_total = my_count(current.qstate) + 2

        # doubt earlier if conservative
        margin = 0 if conservative else 1
        if est_total + margin < current.qty:
            return Doubt()

        # otherwise raise slightly
        new_qty = min(12, current.qty + 1)
        new_q = current.qstate

        reveal = tuple(
            i for i, c in enumerate(my_hand)
            if c == Card.HUP or c.value == new_q.value
        )

        return MakeClaim(Claim(new_qty, new_q), reveal if not conservative else ())

    def observe_round_end(
        self,
        *,
        my_id: int,
        winner: int,
        last_obs: Observation
    ) -> None:
        """
        Store explicit memory from the FINISHED previous round.
        This memory is used in the NEXT round's decision-making.
        """
        pub = last_obs.public
        opp = 1 - my_id

        # --- find opponent’s last action in the round ---
        opp_last = None
        for ev in reversed(pub.history):
            if ev.pid == opp:
                opp_last = ev
                break

        opponent_doubted = (
            opp_last is not None and opp_last.kind == "doubt"
        )

        opponent_evidence = sum(
            ev.revealed_count
            for ev in pub.history
            if ev.pid == opp and ev.kind == "claim"
        )

        opponent_last_claim = pub.current_claim

        # --- store explicit memory ---
        self.last_memory = RoundMemory(
            opponent_doubted=opponent_doubted,
            opponent_evidence_revealed=opponent_evidence,
            opponent_last_claim=opponent_last_claim,
        )