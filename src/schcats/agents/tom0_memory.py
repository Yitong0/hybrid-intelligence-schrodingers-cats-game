from __future__ import annotations
from typing import Optional, List
import random
import math

from .base import Agent
from ..state import Observation
from ..env import Action, MakeClaim, Doubt
from ..memory import RoundMemory
from ..cards import Card
from ..rules import Claim, QState, is_stronger


class ToM0MemoryAgent(Agent):
    """
    Zero-order Theory of Mind agent with explicit memory.

    Uses:
    - own hand
    - current public state (claim + evidence)
    - explicit memory from previous round
    - statistical likelihood of opponent unknown cards (hypergeometric)
    """

    DECK_COUNTS = {
        Card.ALIVE: 20,
        Card.DEAD: 20,
        Card.EMPTY: 8,
        Card.HUP: 4,
    }
    DECK_SIZE = 52
    HAND_SIZE = 6  

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.last_memory: Optional[RoundMemory] = None

    @staticmethod
    def _nCk(n: int, k: int) -> int:
        if k < 0 or k > n:
            return 0
        return math.comb(n, k)

    def _hypergeom_p_geq(self, N: int, K: int, n: int, kmin: int) -> float:
        """
        X ~ Hypergeom(N population, K successes, n draws).
        Return P(X >= kmin).
        """
        if kmin <= 0:
            return 1.0
        if kmin > n:
            return 0.0
        denom = self._nCk(N, n)
        if denom == 0:
            return 0.0
        p = 0.0
        for k in range(kmin, n + 1):
            p += (self._nCk(K, k) * self._nCk(N - K, n - k)) / denom
        return float(p)

    def _support_counts_in_known(self, known: List[Card], q: QState) -> int:
        """How many known cards support q (q cards + HUP)."""
        target = q.value
        return sum(1 for c in known if c == Card.HUP or c.value == target)

    def _deck_support_total(self, q: QState) -> int:
        """Total supporting cards in full deck for claim truth (q + HUP)."""
        if q == QState.ALIVE:
            return self.DECK_COUNTS[Card.ALIVE] + self.DECK_COUNTS[Card.HUP]
        if q == QState.DEAD:
            return self.DECK_COUNTS[Card.DEAD] + self.DECK_COUNTS[Card.HUP]
        # EMPTY
        return self.DECK_COUNTS[Card.EMPTY] + self.DECK_COUNTS[Card.HUP]

    def _prob_claim_true(self, obs: Observation, claim: Claim) -> float:
        """
        P(claim is true) from *my* perspective.
        In 2-player: unknowns are opponent unrevealed cards.
        We approximate opponent unknown cards as drawn without replacement from the remaining deck
        after removing my hand and opponent revealed evidence (identity-correct).
        """
        pub = obs.public
        my_id = obs.my_id
        opp = 1 - my_id

        my_hand = obs.my_hand
        opp_revealed_cards = list(pub.revealed.get(opp, {}).values())

        known_cards = list(my_hand) + opp_revealed_cards

        known_support = self._support_counts_in_known(known_cards, claim.qstate)
        remaining_needed = claim.qty - known_support
        if remaining_needed <= 0:
            return 1.0

        n_unknown = self.HAND_SIZE - len(opp_revealed_cards)
        if n_unknown <= 0:
            return 0.0

        N = self.DECK_SIZE - len(known_cards)
        if N <= 0:
            return 0.0

        total_support = self._deck_support_total(claim.qstate)
        support_in_known = self._support_counts_in_known(known_cards, claim.qstate)
        K = total_support - support_in_known
        K = max(0, min(K, N))

        return self._hypergeom_p_geq(N=N, K=K, n=n_unknown, kmin=remaining_needed)

    def act(self, obs: Observation) -> Action:
        pub = obs.public
        my_hand = obs.my_hand

        # memory-driven conservatism
        conservative = False
        if self.last_memory is not None:
            conservative = self.last_memory.opponent_doubted

        # thresholds
        doubt_threshold = 0.55 if conservative else 0.45

        if pub.current_claim is None:
            candidates = [Claim(qty=q, qstate=s) for q in range(1, 7) for s in (QState.DEAD, QState.ALIVE, QState.EMPTY)]
            best = None
            best_score = -1e9
            for c in candidates:
                p = self._prob_claim_true(obs, c)
                score = p - 0.02 * c.qty
                if score > best_score:
                    best_score = score
                    best = c
            assert best is not None

            reveal = tuple(
                i for i, card in enumerate(my_hand)
                if (card == Card.HUP or card.value == best.qstate.value)
            )
            return MakeClaim(best, reveal if not conservative else ())

        # responding to existing claim
        current = pub.current_claim
        assert current is not None

        p_true = self._prob_claim_true(obs, current)

        if p_true < doubt_threshold:
            return Doubt()

        legal_stronger: List[Claim] = []
        for qty in range(1, 13):
            for q in (QState.DEAD, QState.ALIVE, QState.EMPTY):
                c = Claim(qty=qty, qstate=q)
                if is_stronger(c, current):
                    legal_stronger.append(c)

        best = None
        best_score = -1e9
        for c in legal_stronger:
            p = self._prob_claim_true(obs, c)
            jump_penalty = 0.03 * (c.qty - current.qty) if c.qty >= current.qty else 0.01
            size_penalty = (0.03 if conservative else 0.02) * c.qty
            score = p - jump_penalty - size_penalty
            if score > best_score:
                best_score = score
                best = c

        # If every stronger claim is too risky, doubt instead (this happens late game)
        if best is None or (best_score < (0.10 if conservative else 0.05)):
            return Doubt()

        reveal = tuple(
            i for i, card in enumerate(my_hand)
            if (card == Card.HUP or card.value == best.qstate.value)
        )
        # conservative reveals less
        return MakeClaim(best, reveal if not conservative else ())

    def observe_round_end(self, *, my_id: int, winner: int, last_obs: Observation) -> None:
        pub = last_obs.public
        opp = 1 - my_id

        # opponent last action
        opp_last = None
        for ev in reversed(pub.history):
            if ev.pid == opp:
                opp_last = ev
                break

        opponent_doubted = (opp_last is not None and opp_last.kind == "doubt")

        opponent_evidence = sum(
            ev.revealed_count
            for ev in pub.history
            if ev.pid == opp and ev.kind == "claim"
        )

        opponent_last_claim = pub.current_claim

        last_doubt = None
        for ev in reversed(pub.history):
            if ev.kind == "doubt":
                last_doubt = ev
                break
        opponent_last_claim_was_true = None
        if last_doubt is not None and opponent_last_claim is not None:
            doubter = last_doubt.pid
            claimant = 1 - doubter
            opponent_last_claim_was_true = (winner == claimant)

        self.last_memory = RoundMemory(
            opponent_doubted=opponent_doubted,
            opponent_evidence_revealed=opponent_evidence,
            opponent_last_claim=opponent_last_claim,
            opponent_last_claim_was_true=opponent_last_claim_was_true,
        )