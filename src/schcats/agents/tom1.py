from __future__ import annotations
import random
from typing import Optional

from .base import Agent
from ..state import Observation
from ..env import Action, MakeClaim, Doubt
from ..rules import Claim, QState, is_stronger
from ..cards import Card


class ToM1Agent(Agent):
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        # ToM1 also maintains an internal guess about opponent cautiousness (memory-driven)
        self.opponent_is_conservative: bool = False

    def act(self, obs: Observation) -> Action:
        pub = obs.public
        my_hand = obs.my_hand
        my_id = obs.my_id
        opp = 1 - my_id

        def my_count(q: QState) -> int:
            return sum(1 for c in my_hand if c.value == q.value) + sum(1 for c in my_hand if c == Card.HUP)

        # Interpretative ToM: use revealed opponent evidence as a clue
        opp_evidence = pub.revealed.get(opp, [])
        # how much opponent has “signaled” matching the current claim (plus HUP)
        # (coarse, but it’s genuinely using opponent actions to infer hidden state)
        def evidence_supports(q: QState) -> int:
            return sum(1 for c in opp_evidence if c == Card.HUP or c.value == q.value)

        # If no claim yet, open with a claim that you can support strongly (and optionally reveal evidence)
        if pub.current_claim is None:
            q = max([QState.DEAD, QState.ALIVE, QState.EMPTY], key=my_count)
            qty = max(1, my_count(q))  # more confident opening than ToM0
            reveal = tuple(i for i, c in enumerate(my_hand) if c == Card.HUP or c.value == q.value)
            return MakeClaim(Claim(qty, q), reveal)

        current = pub.current_claim
        assert current is not None

        # ToM1 belief heuristic:
        # opponent likely has at least as many matching cards as they have revealed support for
        opp_min = evidence_supports(current.qstate)
        est_total = my_count(current.qstate) + opp_min + 1  # +1 as weak prior

        # Predictive ToM: if opponent seems conservative, they will doubt earlier, so avoid over-raising
        conservative_penalty = 1 if self.opponent_is_conservative else 0

        if est_total - conservative_penalty < current.qty:
            return Doubt()

        # otherwise we must make a STRICTLY stronger claim

        # Candidate 1: same qstate, qty + 1
        cand1 = Claim(min(12, current.qty + 1), current.qstate)

        # Candidate 2: same qty but stronger qstate ordering if possible (rarely works but valid)
        # ordering for same qty: DEAD < ALIVE < EMPTY (EMPTY uses double-effective so often stronger)
        q_order = [QState.DEAD, QState.ALIVE, QState.EMPTY]
        idx = q_order.index(current.qstate)
        cand2 = None
        if idx < len(q_order) - 1:
            cand2 = Claim(current.qty, q_order[idx + 1])

        # Choose a legal stronger candidate (prefer "safer" under conservative opponents)
        chosen = None
        if cand2 is not None and is_stronger(cand2, current):
            chosen = cand2
        elif is_stronger(cand1, current):
            chosen = cand1
        else:
            # absolute fallback: increase qty until stronger (should always succeed before 12)
            q = current.qstate
            for qty in range(current.qty + 1, 13):
                c = Claim(qty, q)
                if is_stronger(c, current):
                    chosen = c
                    break

        assert chosen is not None, "No stronger claim found, this should not happen."

        new_q = chosen.qstate
        reveal = tuple(i for i, c in enumerate(my_hand) if c == Card.HUP or c.value == new_q.value)
        
        return MakeClaim(chosen, reveal)


        

    def observe_round_end(self, *, my_id: int, winner: int, last_obs: Observation) -> None:
        pub = last_obs.public
        opp = 1 - my_id

        opp_doubts = sum(1 for ev in pub.history if ev.pid == opp and ev.kind == "doubt")
        opp_claims = sum(1 for ev in pub.history if ev.pid == opp and ev.kind == "claim")
        opp_evidence = sum(ev.revealed_count for ev in pub.history if ev.pid == opp and ev.kind == "claim")

        # conservative if opponent doubts often and reveals little evidence
        doubt_rate = opp_doubts / max(1, opp_doubts + opp_claims)
        self.opponent_is_conservative = (doubt_rate > 0.45 and opp_evidence < 2)