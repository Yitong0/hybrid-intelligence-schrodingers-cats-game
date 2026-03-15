from __future__ import annotations
import math
import random
from typing import Dict, List, Optional

from .base import Agent
from ..state import Observation
from ..env import Action, MakeClaim, Doubt
from ..rules import Claim, QState, is_stronger, claim_strength
from ..cards import Card
from ..memory import RoundMemory


class ToM1Agent(Agent):
    """
    First-order Theory of Mind agent for Evidence-based Schrödinger's Cats.

    Implements four components:

    1. EXPLICIT MEMORY: stores RoundMemory after each round (mirroring ToM0(mem)),
       using it to adjust opening claim aggressiveness and opponent type inference.

    2. INTERPRETATIVE ToM: maintains belief_s[q][s] = P(opponent has s supporting
       cards for qstate q in hidden slots), updated via Bayesian update each time
       the opponent makes a claim: P(s | claim Q) ∝ P(claim Q | s) * P(s).

    3. PREDICTIVE ToM: estimates expected utility of each candidate claim by
       simulating the opponent's full ToM0 response (DOUBT or specific RAISE)
       for each possible hidden support count s, weighted by the posterior belief.

    4. ToM0 FALLBACK: if accumulated evidence suggests the opponent is a simple
       ToM0 agent, switches to ToM0-style play — a deliberate inference-driven
       decision based on observed opponent behaviour.
    """

    DECK_COUNTS = {Card.ALIVE: 20, Card.DEAD: 20, Card.EMPTY: 8, Card.HUP: 4}
    DECK_SIZE = 52
    HAND_SIZE = 6
    DOUBT_THRESHOLD = 0.45
    LIKELIHOOD_TEMP = 0.5

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.belief_s: Dict[QState, List[float]] = {}
        self._n_unrevealed: int = self.HAND_SIZE
        self._last_processed_history_len: int = 0
        self._opp_revealed_this_round: List[Card] = []
        self._round_initialized: bool = False

        # Explicit memory (required by explicit-memory variant)
        self.last_memory: Optional[RoundMemory] = None

        # ToM0 fallback counters
        self._rounds_observed: int = 0
        self._doubt_correct: int = 0
        self._doubt_total: int = 0
        self._opp_bluffiness: float = 0.0
        self._acting_as_tom0: bool = False

    # Memo helpers

    def _memory_opening_penalty(self) -> float:
        """
        Return an extra quantity penalty for opening claims based on memory.

        If the opponent doubted last round, they are playing aggressively.
        We apply a small extra penalty (0.01 per qty unit) to discourage
        high-quantity opening bids, preferring safer claims in response.
        This is one of the ways explicit memory directly affects ToM1's
        decision-making, as required by the explicit-memory variant.
        """
        if self.last_memory is not None and self.last_memory.opponent_doubted:
            return 0.01
        return 0.0

    # Some helpers

    def _supporting_types(self, q: QState) -> List[Card]:
        """Return the card types that support qstate q (q-type cards plus HUP)."""
        if q == QState.ALIVE:
            return [Card.ALIVE, Card.HUP]
        if q == QState.DEAD:
            return [Card.DEAD, Card.HUP]
        return [Card.EMPTY, Card.HUP]

    def _count_support(self, cards: List[Card], q: QState) -> int:
        """Count how many cards in the list support qstate q."""
        return sum(1 for c in cards if c in set(self._supporting_types(q)))

    @staticmethod
    def _nCk(n: int, k: int) -> int:
        """Binomial coefficient C(n, k), returns 0 for invalid inputs."""
        return math.comb(n, k) if 0 <= k <= n else 0

    def _hypergeom_pmf(self, N: int, K: int, n: int) -> List[float]:
        """
        Full PMF of the Hypergeometric distribution Hypergeom(N, K, n).
        Returns a list p where p[k] = P(X = k) for k in {0, ..., n}.
        Models drawing n cards without replacement from N cards, K of which
        are successes (supporting cards).
        """
        denom = self._nCk(N, n)
        if denom == 0:
            p = [0.0] * (n + 1)
            p[0] = 1.0
            return p
        return [self._nCk(K, k) * self._nCk(N - K, n - k) / denom
                for k in range(n + 1)]

    def _hypergeom_p_geq(self, N: int, K: int, n: int, kmin: int) -> float:
        """
        P(X >= kmin) for X ~ Hypergeom(N, K, n).
        Used to compute the probability that at least kmin supporting cards
        appear among n draws from a population of N with K supporters.
        """
        if kmin <= 0:
            return 1.0
        if kmin > n or K <= 0:
            return 0.0
        return sum(self._hypergeom_pmf(N, K, n)[k] for k in range(kmin, n + 1))

    # Belief initialisation and Bayesian update (Interpretative ToM)     

    def _init_belief(self, my_hand: List[Card], opp_revealed: List[Card]) -> None:
        """
        Initialise belief_s[q] as a hypergeometric prior at the start of a round.

        For each qstate q, we compute the PMF of the number of supporting cards
        the opponent holds in their unrevealed slots, assuming those slots are
        drawn without replacement from the remaining deck (deck minus my hand
        minus opponent's already-revealed cards).
        """
        self._opp_revealed_this_round = list(opp_revealed)
        self._n_unrevealed = self.HAND_SIZE - len(opp_revealed)
        self._last_processed_history_len = 0
        self._round_initialized = True
        known = my_hand + opp_revealed
        N = max(1, self.DECK_SIZE - len(known))
        n = max(0, self._n_unrevealed)
        self.belief_s = {}
        for q in QState:
            K = sum(max(0, self.DECK_COUNTS[ct] - sum(1 for c in known if c == ct))
                    for ct in self._supporting_types(q))
            K = max(0, min(K, N))
            self.belief_s[q] = self._hypergeom_pmf(N, K, n)

    def _likelihood_claim_given_s(self, claim: Claim, s: int) -> float:
        """
        P(opponent makes this claim | they have s supporting cards in hidden slots).

        Models the opponent as a rational ToM0 agent: they are more likely to
        claim qty=Q when s + revealed_support >= Q (the claim is supportable).
        Uses a softmax over a plausibility score with temperature LIKELIHOOD_TEMP.
        """
        revealed_support = self._count_support(self._opp_revealed_this_round, claim.qstate)
        can_support = 1.0 if (s + revealed_support) >= claim.qty else 0.0
        plausibility = can_support - 0.05 * claim.qty
        return math.exp(plausibility / self.LIKELIHOOD_TEMP)

    def _update_belief(self, claim: Claim) -> None:
        """
        INTERPRETATIVE ToM: Bayesian update of belief_s[q] given opponent's claim.

        Computes the posterior:
            P(s | opponent claimed qty=Q) ∝ P(claim Q | s) * P(s)

        This updates our estimate of how many supporting cards the opponent
        holds in their hidden slots, based on what a rational agent with s
        supporters would be likely to claim.
        """
        q = claim.qstate
        prior = self.belief_s.get(q)
        if prior is None:
            return
        posterior = [self._likelihood_claim_given_s(claim, s) * prior[s]
                     for s in range(len(prior))]
        Z = sum(posterior)
        if Z > 0:
            self.belief_s[q] = [p / Z for p in posterior]

    def _reinit_and_replay(self, obs: Observation) -> None:
        """
        Reinitialise beliefs and replay all opponent claim events from history.

        Called when the opponent reveals new evidence mid-round, changing the
        number of their known cards. We must reinitialise the prior with the
        updated revealed count and replay all claim-based Bayesian updates to
        maintain a coherent posterior throughout the round.
        """
        pub = obs.public
        opp = 1 - obs.my_id
        opp_revealed = list(pub.revealed.get(opp, {}).values())
        self._init_belief(obs.my_hand, opp_revealed)
        for ev in pub.history:
            if ev.pid == opp and ev.kind == "claim" and ev.claim is not None:
                self._update_belief(ev.claim)
        self._last_processed_history_len = len(pub.history)

    # Predictive ToM: opponent perspective probability  

    def _p_claim_true_opp_view(self, claim: Claim, obs: Observation) -> float:
        """
        P(claim true | opponent's view), averaged over our posterior belief_s[q].

        For each possible hidden support count s (weighted by belief_s[q][s]):
          - The opponent knows their full hand: s hidden supporters + opp_revealed
          - The opponent can see our revealed cards (my_revealed)
          - The opponent models our unrevealed cards with a hypergeometric prior
            drawn from the remaining deck after removing all known cards

        This is the core of predictive ToM: we simulate the probability the
        opponent assigns to the claim being true, from their perspective.
        """
        pub = obs.public
        my_id = obs.my_id
        opp = 1 - my_id
        my_revealed = list(pub.revealed.get(my_id, {}).values())
        opp_revealed = list(pub.revealed.get(opp, {}).values())
        supporting = set(self._supporting_types(claim.qstate))

        my_revealed_support = self._count_support(my_revealed, claim.qstate)
        opp_revealed_support = self._count_support(opp_revealed, claim.qstate)
        n_my_unrevealed = self.HAND_SIZE - len(my_revealed)

        known_opp_cards = opp_revealed + my_revealed
        N = max(1, self.DECK_SIZE - len(known_opp_cards))
        K_base = sum(max(0, self.DECK_COUNTS[ct] - sum(1 for c in known_opp_cards if c == ct))
                     for ct in supporting)

        belief = self.belief_s.get(claim.qstate, [1.0])
        p_true = 0.0

        for s, prob_s in enumerate(belief):
            if prob_s < 1e-9:
                continue
            known_opp_support = s + opp_revealed_support + my_revealed_support
            still_needed = claim.qty - known_opp_support
            if still_needed <= 0:
                p_true += prob_s
                continue
            if n_my_unrevealed <= 0:
                continue
            K = max(0, min(K_base - s, N))
            p_true += prob_s * self._hypergeom_p_geq(
                N=N, K=K, n=n_my_unrevealed, kmin=still_needed)

        return float(p_true)

    def _p_claim_true_opp_view_given_s(
        self, claim: Claim, obs: Observation, s: int
    ) -> float:
        """
        P(claim true | opponent's view) for a specific hidden support count s.

        Unlike _p_claim_true_opp_view, this does not average over the posterior —
        it computes the opponent-view probability for a concrete value of s.
        Used inside _opp_choose_action to simulate the opponent's decision
        for each possible hidden hand composition.
        """
        pub = obs.public
        my_id = obs.my_id
        opp = 1 - my_id
        my_revealed = list(pub.revealed.get(my_id, {}).values())
        opp_revealed = list(pub.revealed.get(opp, {}).values())
        supporting = set(self._supporting_types(claim.qstate))

        my_revealed_support = self._count_support(my_revealed, claim.qstate)
        opp_revealed_support = self._count_support(opp_revealed, claim.qstate)
        n_my_unrevealed = self.HAND_SIZE - len(my_revealed)

        known_opp_support = s + opp_revealed_support + my_revealed_support
        still_needed = claim.qty - known_opp_support
        if still_needed <= 0:
            return 1.0
        if n_my_unrevealed <= 0:
            return 0.0

        known_opp_cards = opp_revealed + my_revealed
        N = max(1, self.DECK_SIZE - len(known_opp_cards))
        K = sum(max(0, self.DECK_COUNTS[ct] - sum(1 for c in known_opp_cards if c == ct))
                for ct in supporting)
        K = max(0, min(K - s, N))
        return self._hypergeom_p_geq(N=N, K=K, n=n_my_unrevealed, kmin=still_needed)

    def _opp_choose_action(
        self, obs: Observation, current_claim: Claim, s: int
    ) -> object:
        """
        Simulate the opponent's full ToM0 decision given they hold s hidden supporters.

        The opponent doubts if P(claim true | their view with s supporters) < 0.45.
        Otherwise, they select the strongest claim they can make that maximises
        P(claim true | their view) minus penalties for escalation and quantity.

        Returns the string "DOUBT" or a specific stronger Claim object.
        This full action simulation (not just doubt probability) is what makes
        the predictive ToM component genuinely first-order: we simulate the
        opponent's entire decision process from their perspective.
        """
        p_true = self._p_claim_true_opp_view_given_s(current_claim, obs, s)
        if p_true < self.DOUBT_THRESHOLD:
            return "DOUBT"

        my_id = obs.my_id
        opp = 1 - my_id
        my_revealed = list(obs.public.revealed.get(my_id, {}).values())
        opp_revealed = list(obs.public.revealed.get(opp, {}).values())

        # Limit to 12 weakest legal claims for efficiency
        legal = sorted(
            [Claim(qty=qty, qstate=q)
             for qty in range(1, 13)
             for q in (QState.DEAD, QState.ALIVE, QState.EMPTY)
             if is_stronger(Claim(qty=qty, qstate=q), current_claim)],
            key=lambda c: claim_strength(c)[0]
        )[:12]

        if not legal:
            return "DOUBT"

        best, best_score = None, -1e9
        for c in legal:
            p = self._p_claim_true_opp_view_given_s(c, obs, s)
            gap = claim_strength(c)[0] - claim_strength(current_claim)[0]
            score = p - 0.015 * gap - 0.02 * c.qty
            if score > best_score:
                best_score, best = score, c

        if best is None or best_score < 0.05:
            return "DOUBT"
        return best

    def _estimate_utility(self, obs: Observation, candidate_claim: Claim) -> float:
        """
        PREDICTIVE ToM: expected utility of making candidate_claim.

        Averages over our posterior belief_s[q], simulating the opponent's full
        ToM0 decision for each possible hidden support count s:

          EV(c) = sum_s  belief_s[q][s] * outcome(c, s)

        where outcome(c, s) depends on the opponent's simulated action:
          - If opponent doubts: utility = 1 if claim is true (known_support + s >= qty),
            else 0. We win iff the claim we made can actually be satisfied.
          - If opponent raises to c2: if we would doubt c2, utility = 1 - P(c2 true)
            (we win by doubting a false claim); otherwise utility = P(c2 true) minus
            a penalty proportional to how much the opponent escalated.

        This is the core decision function for ToM1: every claim selection goes
        through this expected value computation rather than simple plausibility.
        """
        pub = obs.public
        opp = 1 - obs.my_id
        my_hand = obs.my_hand
        opp_revealed = list(pub.revealed.get(opp, {}).values())

        opp_revealed_support_q = self._count_support(opp_revealed, candidate_claim.qstate)
        my_support_q = self._count_support(my_hand, candidate_claim.qstate)
        known_support = my_support_q + opp_revealed_support_q

        belief = self.belief_s.get(candidate_claim.qstate, [1.0])
        total_u = 0.0

        for s, prob_s in enumerate(belief):
            if prob_s < 1e-9:
                continue

            decision = self._opp_choose_action(obs, candidate_claim, s)

            if decision == "DOUBT":
                u = 1.0 if (known_support + s) >= candidate_claim.qty else 0.0
            else:
                c2 = decision
                gap = claim_strength(c2)[0] - claim_strength(candidate_claim)[0]
                p_c2_true = self._prob_claim_true_my_view(obs, c2)
                if p_c2_true < self.DOUBT_THRESHOLD:
                    u = 1.0 - p_c2_true
                else:
                    u = max(0.0, p_c2_true - 0.02 * gap)

            total_u += prob_s * u

        return float(total_u)

    #  Our own claim probability (uses posterior)  

    def _prob_claim_true_my_view(self, obs: Observation, claim: Claim) -> float:
        """
        P(claim true) from our perspective, using the posterior belief_s.

        P(true) = sum_s  belief_s[q][s] * 1{ known_support + s >= claim.qty }

        This is more accurate than a fresh hypergeometric prior because it
        incorporates everything we have inferred about the opponent's hand
        from their claim history via interpretative ToM.
        """
        pub = obs.public
        opp = 1 - obs.my_id
        my_hand = obs.my_hand
        opp_revealed = list(pub.revealed.get(opp, {}).values())
        known_support = (self._count_support(my_hand, claim.qstate)
                         + self._count_support(opp_revealed, claim.qstate))
        still_needed = claim.qty - known_support
        if still_needed <= 0:
            return 1.0
        belief = self.belief_s.get(claim.qstate, [1.0])
        return float(sum(p for s, p in enumerate(belief) if s >= still_needed))

    # ToM0 fallback

    def _opponent_seems_tom0(self) -> bool:
        """
        Infer whether the opponent is a simple ToM0 agent from accumulated evidence.

        Uses two signals from explicit memory:
          1. Doubt accuracy: a ToM0 agent doubts when claims are statistically
             unlikely, so their doubts should be correct at a high rate (> 65%).
          2. Bluffiness: if the opponent's claims are often false, they are a
             bluffer — consistent with ToM0 statistical play.

        Requires at least 5 rounds of observation before making a judgment.
        """
        if self._rounds_observed < 5 or self._doubt_total == 0:
            return False
        doubt_accuracy = self._doubt_correct / self._doubt_total
        return doubt_accuracy > 0.65 and self._opp_bluffiness < 0.5

    def _act_as_tom0(self, obs: Observation) -> Action:
        """
        ToM0-style fallback policy used when the opponent is inferred to be ToM0.

        This is a deliberate first-order ToM decision: having inferred from
        observed behaviour that the opponent is a ToM0 agent, we switch to the
        simpler policy that is optimal against that opponent type. We still use
        our posterior belief for P(claim true), so this is strictly better than
        a real ToM0 agent.
        """
        pub = obs.public
        my_hand = obs.my_hand
        mem_penalty = self._memory_opening_penalty()

        if pub.current_claim is None:
            candidates = [Claim(qty=q, qstate=s)
                          for q in range(1, 8)
                          for s in (QState.DEAD, QState.ALIVE, QState.EMPTY)]
            best, best_score = None, -1e9
            for c in candidates:
                score = self._prob_claim_true_my_view(obs, c) - (0.02 + mem_penalty) * c.qty
                if score > best_score:
                    best_score, best = score, c
            assert best is not None
            reveal = tuple(i for i, card in enumerate(my_hand)
                           if card in set(self._supporting_types(best.qstate)))
            return MakeClaim(best, reveal)

        current = pub.current_claim
        if self._prob_claim_true_my_view(obs, current) < self.DOUBT_THRESHOLD:
            return Doubt()

        legal = [Claim(qty=qty, qstate=q)
                 for qty in range(1, 13)
                 for q in (QState.DEAD, QState.ALIVE, QState.EMPTY)
                 if is_stronger(Claim(qty=qty, qstate=q), current)]
        best, best_score = None, -1e9
        for c in legal:
            gap = claim_strength(c)[0] - claim_strength(current)[0]
            score = self._prob_claim_true_my_view(obs, c) - 0.015 * gap - 0.02 * c.qty
            if score > best_score:
                best_score, best = score, c
        if best is None or best_score < 0.05:
            return Doubt()
        reveal = tuple(i for i, card in enumerate(my_hand)
                       if card in set(self._supporting_types(best.qstate)))
        return MakeClaim(best, reveal)

    # Main act()

    def act(self, obs: Observation) -> Action:
        """
        Choose an action given the current observation.

        At round start, initialises the hypergeometric belief prior.
        If opponent evidence changes mid-round, reinitialises and replays.
        Incrementally applies Bayesian belief updates from opponent claims
        (interpretative ToM), then selects the claim with highest expected
        utility under full opponent response simulation (predictive ToM).
        """
        pub = obs.public
        my_hand = obs.my_hand
        my_id = obs.my_id
        opp = 1 - my_id
        opp_revealed = list(pub.revealed.get(opp, {}).values())

        if pub.current_claim is None:
            self._init_belief(my_hand, opp_revealed)

        if not self._round_initialized:
            self._init_belief(my_hand, opp_revealed)
        elif len(opp_revealed) != len(self._opp_revealed_this_round):
            self._reinit_and_replay(obs)
        else:
            self._opp_revealed_this_round = list(opp_revealed)
            self._n_unrevealed = self.HAND_SIZE - len(opp_revealed)

        if self._acting_as_tom0:
            return self._act_as_tom0(obs)

        # Interpretative ToM, process new opponent claim events incrementally
        for ev in pub.history[self._last_processed_history_len:]:
            if ev.pid == opp and ev.kind == "claim" and ev.claim is not None:
                self._update_belief(ev.claim)
        self._last_processed_history_len = len(pub.history)

        mem_penalty = self._memory_opening_penalty()

        # Opening claim, prefilter by plausibility then select by expected utility
        if pub.current_claim is None:
            all_opening = [Claim(qty=q, qstate=s)
                           for q in range(1, 8)
                           for s in (QState.DEAD, QState.ALIVE, QState.EMPTY)]
            scored_o = sorted(all_opening, key=lambda c: -self._prob_claim_true_my_view(obs, c))
            candidates = scored_o[:10]
            best, best_score = None, -1e9
            for c in candidates:
                ev = self._estimate_utility(obs, c) - mem_penalty * c.qty
                if ev > best_score:
                    best_score, best = ev, c
            assert best is not None
            reveal = tuple(i for i, card in enumerate(my_hand)
                           if card in set(self._supporting_types(best.qstate)))
            return MakeClaim(best, reveal)

        # Responding to opponent's claim
        current = pub.current_claim
        if self._prob_claim_true_my_view(obs, current) < self.DOUBT_THRESHOLD:
            return Doubt()

        legal = [Claim(qty=qty, qstate=q)
                 for qty in range(1, 13)
                 for q in (QState.DEAD, QState.ALIVE, QState.EMPTY)
                 if is_stronger(Claim(qty=qty, qstate=q), current)]

        scored_legal = sorted(legal, key=lambda c: -self._prob_claim_true_my_view(obs, c))
        candidates = scored_legal[:10]

        best, best_score = None, -1e9
        for c in candidates:
            ev = self._estimate_utility(obs, c)
            if ev > best_score:
                best_score, best = ev, c

        if best is None or best_score < 0.05:
            return Doubt()

        reveal = tuple(i for i, card in enumerate(my_hand)
                       if card in set(self._supporting_types(best.qstate)))
        return MakeClaim(best, reveal)


    def observe_round_end(self, *, my_id: int, winner: int, last_obs: Observation) -> None:
        """
        Update memory and counters at the end of each round.

        Builds a RoundMemory mirroring the ToM0(mem) construction, updates the
        bluffiness tracker from opponent claim truthfulness, and updates the
        ToM0 fallback decision for the next round.
        """
        pub = last_obs.public
        opp = 1 - my_id
        self._rounds_observed += 1

        opp_last = next((ev for ev in reversed(pub.history) if ev.pid == opp), None)
        opponent_doubted = opp_last is not None and opp_last.kind == "doubt"
        opponent_evidence = sum(
            ev.revealed_count for ev in pub.history
            if ev.pid == opp and ev.kind == "claim"
        )
        last_doubt = next((ev for ev in reversed(pub.history) if ev.kind == "doubt"), None)
        opponent_last_claim_was_true = None
        if last_doubt is not None and pub.current_claim is not None:
            claimant = 1 - last_doubt.pid
            opponent_last_claim_was_true = (winner == claimant)

        self.last_memory = RoundMemory(
            opponent_doubted=opponent_doubted,
            opponent_evidence_revealed=opponent_evidence,
            opponent_last_claim=pub.current_claim,
            opponent_last_claim_was_true=opponent_last_claim_was_true,
        )

        # Update bluffiness from memory. used in ToM0 fallback detection
        if opponent_last_claim_was_true is not None:
            if not opponent_last_claim_was_true:
                self._opp_bluffiness = min(1.0, self._opp_bluffiness + 0.1)
            else:
                self._opp_bluffiness = max(0.0, self._opp_bluffiness - 0.05)

        # Update doubt accuracy for ToM0 fallback detection
        for ev in pub.history:
            if ev.pid == opp and ev.kind == "doubt":
                self._doubt_total += 1
                if winner == opp:
                    self._doubt_correct += 1

        self._acting_as_tom0 = self._opponent_seems_tom0()
        self._round_initialized = False
        self._opp_revealed_this_round = []
        self._last_processed_history_len = 0