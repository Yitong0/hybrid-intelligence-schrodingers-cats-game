from __future__ import annotations
import random
import math
from typing import List

from .base import Agent
from ..state import Observation
from ..env import Action, MakeClaim, Doubt
from ..rules import Claim, QState, is_stronger, claim_strength
from ..cards import Card


class ToM1Agent(Agent):
    """
    First-order ToM agent.

    - Interpretative ToM:
        Uses opponent revealed evidence as information about opponent hidden hand.
    - Predictive ToM:
        Predicts opponent doubt probability for candidate claims using inferred traits.
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

        self.opponent_is_conservative: bool = False
        self.opponent_bluffiness: float = 0.0  

        self.fallback_rounds_left: int = 0

        self._used_fallback_this_round: bool = False

        self.opp_doubted_last_round: bool = False
        self.opp_doubted_two_rounds_ago: bool = False

        self.fallback_actions_taken: int = 0
        self.fallback_rounds_started: int = 0
        self.fallback_trigger_events: int = 0

    @staticmethod
    def _nCk(n: int, k: int) -> int:
        if k < 0 or k > n:
            return 0
        return math.comb(n, k)

    def _hypergeom_p_geq(self, N: int, K: int, n: int, kmin: int) -> float:
        """P(X >= kmin) for X ~ Hypergeom(N, K, n)."""
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

    def _deck_support_total(self, q: QState) -> int:
        """Total supporting cards in the full deck for q (q + HUP)."""
        if q == QState.ALIVE:
            return self.DECK_COUNTS[Card.ALIVE] + self.DECK_COUNTS[Card.HUP]
        if q == QState.DEAD:
            return self.DECK_COUNTS[Card.DEAD] + self.DECK_COUNTS[Card.HUP]
        return self.DECK_COUNTS[Card.EMPTY] + self.DECK_COUNTS[Card.HUP]

    def _support_in_cards(self, cards: List[Card], q: QState) -> int:
        """How many cards support q (q + HUP)."""
        target = q.value
        return sum(1 for c in cards if (c == Card.HUP or c.value == target))


    def _prob_claim_true_my_view(self, obs: Observation, claim: Claim) -> float:
        """
        P(claim true) from my perspective:
        - known cards: my hand + opponent revealed evidence (public)
        - unknown cards: opponent unrevealed hand slots
        Model unknowns as draws without replacement from remaining deck.
        """
        pub = obs.public
        my_id = obs.my_id
        opp = 1 - my_id

        my_hand = obs.my_hand

        opp_revealed_map = pub.revealed.get(opp, {})
        opp_revealed_cards = list(opp_revealed_map.values())

        known = list(my_hand) + opp_revealed_cards

        known_support = self._support_in_cards(known, claim.qstate)
        remaining_needed = claim.qty - known_support
        if remaining_needed <= 0:
            return 1.0

        n_unknown = self.HAND_SIZE - len(opp_revealed_cards)
        if n_unknown <= 0:
            return 0.0

        N = self.DECK_SIZE - len(known)
        if N <= 0:
            return 0.0

        total_support = self._deck_support_total(claim.qstate)

        support_in_known = self._support_in_cards(known, claim.qstate)

        K = total_support - support_in_known
        if K < 0:
            K = 0
        if K > N:
            K = N

        return self._hypergeom_p_geq(N=N, K=K, n=n_unknown, kmin=remaining_needed)

    def _predict_opponent_doubt_prob(self, obs: Observation, claim: Claim) -> float:
        """
        Predict probability opponent doubts this claim.
        Higher if claim seems risky, higher if opponent is conservative,
        lower if opponent is bluff-prone.
        """
        p_true = self._prob_claim_true_my_view(obs, claim)

        base = 0.45
        if self.opponent_is_conservative:
            base += 0.15
        base -= 0.20 * self.opponent_bluffiness

        risk_adj = (0.60 - p_true)  # positive if p_true < 0.60
        p_doubt = base + risk_adj
        return float(max(0.05, min(0.95, p_doubt)))

    def _act_like_tom0(self, obs: Observation) -> Action:
        """
        A ToM0-style policy:
        - Decide doubt vs continue from claim plausibility only (no opponent-reaction model).
        - Choose stronger claim maximizing plausibility minus mild aggression penalties.
        """
        pub = obs.public
        my_hand = obs.my_hand

        doubt_threshold = 0.52

        if pub.current_claim is None:
            candidates = [
                Claim(qty=q, qstate=s)
                for q in range(1, 8)
                for s in (QState.DEAD, QState.ALIVE, QState.EMPTY)
            ]
            best = None
            best_score = -1e9
            for c in candidates:
                p = self._prob_claim_true_my_view(obs, c)
                score = p - 0.02 * c.qty
                if score > best_score:
                    best_score = score
                    best = c
            assert best is not None
            return MakeClaim(best, ())

        current = pub.current_claim
        assert current is not None

        p_true_current = self._prob_claim_true_my_view(obs, current)
        if p_true_current < doubt_threshold:
            return Doubt()

        legal_stronger: List[Claim] = []
        for qty in range(1, 13):
            for q in (QState.DEAD, QState.ALIVE, QState.EMPTY):
                c = Claim(qty=qty, qstate=q)
                if is_stronger(c, current):
                    legal_stronger.append(c)

        best = None
        best_score = -1e9
        base_strength = claim_strength(current)[0]

        for c in legal_stronger:
            p = self._prob_claim_true_my_view(obs, c)
            gap = claim_strength(c)[0] - base_strength
            score = p - 0.015 * gap - 0.02 * c.qty
            if score > best_score:
                best_score = score
                best = c

        if best is None or best_score < 0.08:
            return Doubt()

        return MakeClaim(best, ())

    def act(self, obs: Observation) -> Action:
        pub = obs.public

        if pub.current_claim is None:
            self._used_fallback_this_round = False

        if self.fallback_rounds_left > 0:
            if pub.current_claim is None:
                self.fallback_rounds_started += 1
            self.fallback_actions_taken += 1
            self._used_fallback_this_round = True
            return self._act_like_tom0(obs)

        my_hand = obs.my_hand

        if pub.current_claim is None:
            candidates = [
                Claim(qty=q, qstate=s)
                for q in range(1, 8)
                for s in (QState.DEAD, QState.ALIVE, QState.EMPTY)
            ]
            best = None
            best_score = -1e9
            for c in candidates:
                p = self._prob_claim_true_my_view(obs, c)
                score = p - 0.02 * c.qty
                if score > best_score:
                    best_score = score
                    best = c
            assert best is not None

            reveal = tuple(
                i for i, card in enumerate(my_hand)
                if (card == Card.HUP or card.value == best.qstate.value)
            )
            return MakeClaim(best, reveal)

        current = pub.current_claim
        assert current is not None

        p_true_current = self._prob_claim_true_my_view(obs, current)

        accept_threshold = 0.50
        accept_threshold += 0.05 * self.opponent_bluffiness
        if self.opponent_is_conservative:
            accept_threshold -= 0.03

        if p_true_current < accept_threshold:
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
            p_true = self._prob_claim_true_my_view(obs, c)
            p_doubt = self._predict_opponent_doubt_prob(obs, c)

            immediate_ev = p_doubt * p_true
            strength_gap = claim_strength(c)[0] - claim_strength(current)[0]
            continuation_bonus = (1.0 - p_doubt) * (p_true - 0.015 * strength_gap)

            score = immediate_ev + continuation_bonus
            if score > best_score:
                best_score = score
                best = c

        if best is None or best_score < 0.05:
            return Doubt()

        reveal = tuple(
            i for i, card in enumerate(my_hand)
            if (card == Card.HUP or card.value == best.qstate.value)
        )
        return MakeClaim(best, reveal)

    def observe_round_end(self, *, my_id: int, winner: int, last_obs: Observation) -> None:
        # Decrement fallback ONCE per finished round (if we used it)
        if self._used_fallback_this_round and self.fallback_rounds_left > 0:
            self.fallback_rounds_left -= 1

        pub = last_obs.public
        opp = 1 - my_id

        opp_doubts = sum(1 for ev in pub.history if ev.pid == opp and ev.kind == "doubt")
        opp_claims = sum(1 for ev in pub.history if ev.pid == opp and ev.kind == "claim")
        opp_evidence = sum(ev.revealed_count for ev in pub.history if ev.pid == opp and ev.kind == "claim")

        doubt_rate = opp_doubts / max(1, opp_doubts + opp_claims)

        self.opponent_is_conservative = (doubt_rate > 0.45 and opp_evidence < 2)

        last_doubt = None
        for ev in reversed(pub.history):
            if ev.kind == "doubt":
                last_doubt = ev
                break

        if last_doubt is not None and pub.current_claim is not None:
            doubter = last_doubt.pid
            claimant = 1 - doubter
            claim_true = (winner == claimant)

            if claimant == opp and not claim_true:
                self.opponent_bluffiness = min(1.0, self.opponent_bluffiness + 0.15)
            elif claimant == opp and claim_true:
                self.opponent_bluffiness = max(0.0, self.opponent_bluffiness - 0.08)

        opp_doubted_this_round = (opp_doubts >= 1)
        self.opp_doubted_two_rounds_ago = self.opp_doubted_last_round
        self.opp_doubted_last_round = opp_doubted_this_round

        new_fallback = 0
        if self.opp_doubted_last_round and self.opp_doubted_two_rounds_ago:
            new_fallback = 1

        if new_fallback > 0:
            self.fallback_trigger_events += 1

        self.fallback_rounds_left = max(self.fallback_rounds_left, new_fallback)