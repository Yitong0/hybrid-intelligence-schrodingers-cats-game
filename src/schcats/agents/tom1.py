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
from .tom0_memory import ToM0MemoryAgent

from ..state import Observation as Obs
from ..state import PublicState
from ..env import Doubt as DoubtAction, MakeClaim as MakeClaimAction


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

        # Explicit memory
        self.last_memory: Optional[RoundMemory] = None

        # ToM0 fallback counters
        self._rounds_observed: int = 0
        self._doubt_correct: int = 0
        self._doubt_total: int = 0
        self._opp_bluffiness: float = 0.0
        self._acting_as_tom0: bool = False

        # Diagnostics
        self._total_turns: int = 0
        self._fallback_turns: int = 0

        # ToM0 decision generator
        self._tom0_simulator: ToM0MemoryAgent = ToM0MemoryAgent(seed=0)

    @property
    def fallback_rate(self) -> float:
        """Fraction of turns where ToM1 fell back to ToM0-style play."""
        if self._total_turns == 0:
            return 0.0
        return self._fallback_turns / self._total_turns

    # Memory helpers

    def _memory_opening_penalty(self) -> float:
        """
        Return an extra quantity penalty for opening claims based on memory.
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
        support = set(self._supporting_types(q))
        return sum(1 for c in cards if c in support)

    @staticmethod
    def _nCk(n: int, k: int) -> int:
        """Binomial coefficient C(n, k), returns 0 for invalid inputs."""
        return math.comb(n, k) if 0 <= k <= n else 0

    def _hypergeom_pmf(self, N: int, K: int, n: int) -> List[float]:
        """
        Full PMF of the Hypergeometric distribution Hypergeom(N, K, n).
        """
        denom = self._nCk(N, n)
        if denom == 0:
            p = [0.0] * (n + 1)
            p[0] = 1.0
            return p
        return [
            self._nCk(K, k) * self._nCk(N - K, n - k) / denom
            for k in range(n + 1)
        ]

    def _hypergeom_p_geq(self, N: int, K: int, n: int, kmin: int) -> float:
        """
        P(X >= kmin) for X ~ Hypergeom(N, K, n).
        """
        if kmin <= 0:
            return 1.0
        if kmin > n or K <= 0:
            return 0.0
        pmf = self._hypergeom_pmf(N, K, n)
        return sum(pmf[k] for k in range(kmin, n + 1))

    # Belief initialisation and Bayesian update (Interpretative ToM)     

    def _init_belief(self, my_hand: List[Card], opp_revealed: List[Card]) -> None:
        """
        Initialise belief_s[q] as a hypergeometric prior at the start of a round.
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
            K = sum(
                max(0, self.DECK_COUNTS[ct] - sum(1 for c in known if c == ct))
                for ct in self._supporting_types(q)
            )
            K = max(0, min(K, N))
            self.belief_s[q] = self._hypergeom_pmf(N, K, n)

    def _likelihood_claim_given_s(self, claim: Claim, s: int) -> float:
        """
        P(opponent makes this claim | they have s supporting cards in hidden slots).
        """
        revealed_support = self._count_support(self._opp_revealed_this_round, claim.qstate)
        can_support = 1.0 if (s + revealed_support) >= claim.qty else 0.0
        plausibility = can_support - 0.05 * claim.qty
        return math.exp(plausibility / self.LIKELIHOOD_TEMP)

    def _update_belief(self, claim: Claim) -> None:
        """
        Bayesian update of belief_s[q] given opponent's claim.
        """
        q = claim.qstate
        prior = self.belief_s.get(q)
        if prior is None:
            return

        posterior = [
            self._likelihood_claim_given_s(claim, s) * prior[s]
            for s in range(len(prior))
        ]
        Z = sum(posterior)
        if Z > 0:
            self.belief_s[q] = [p / Z for p in posterior]

    def _reinit_and_replay(self, obs: Observation) -> None:
        """
        Reinitialise beliefs and replay opponent claim events from history.
        """
        pub = obs.public
        opp = 1 - obs.my_id
        opp_revealed = list(pub.revealed.get(opp, {}).values())
        self._init_belief(obs.my_hand, opp_revealed)

        for ev in pub.history:
            if ev.pid == opp and ev.kind == "claim" and ev.claim is not None:
                self._update_belief(ev.claim)

        self._last_processed_history_len = len(pub.history)

    # Predictive ToM probabilities

    def _p_claim_true_opp_view(self, claim: Claim, obs: Observation) -> float:
        """
        P(claim true | opponent's view), averaged over our posterior belief_s[q].
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

        known_cards = opp_revealed + my_revealed
        N = max(1, self.DECK_SIZE - len(known_cards))
        K_base = sum(
            max(0, self.DECK_COUNTS[ct] - sum(1 for c in known_cards if c == ct))
            for ct in supporting
        )

        belief = self.belief_s.get(claim.qstate, [1.0])
        p_true = 0.0

        for s, prob_s in enumerate(belief):
            if prob_s < 1e-9:
                continue

            known_support = s + opp_revealed_support + my_revealed_support
            still_needed = claim.qty - known_support

            if still_needed <= 0:
                p_true += prob_s
                continue
            if n_my_unrevealed <= 0:
                continue

            K = max(0, min(K_base - s, N))
            p_true += prob_s * self._hypergeom_p_geq(N, K, n_my_unrevealed, still_needed)

        return float(p_true)

    def _p_claim_true_opp_view_given_s(
        self, claim: Claim, obs: Observation, s: int
    ) -> float:
        """
        P(claim true | opponent's view) for a specific hidden support count s.
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

        known_support = s + opp_revealed_support + my_revealed_support
        still_needed = claim.qty - known_support

        if still_needed <= 0:
            return 1.0
        if n_my_unrevealed <= 0:
            return 0.0

        known_cards = opp_revealed + my_revealed
        N = max(1, self.DECK_SIZE - len(known_cards))
        K = sum(
            max(0, self.DECK_COUNTS[ct] - sum(1 for c in known_cards if c == ct))
            for ct in supporting
        )
        K = max(0, min(K - s, N))
        return self._hypergeom_p_geq(N, K, n_my_unrevealed, still_needed)

    def _sample_opp_hand(
        self, my_hand: List[Card], opp_revealed: List[Card], q: QState, s: int
    ) -> List[Card]:
        """
        Build a hypothetical opponent hand with exactly s hidden supporters for q.
        """
        supporting_set = set(self._supporting_types(q))

        known = my_hand + opp_revealed
        counts = dict(self.DECK_COUNTS)
        for c in known:
            counts[c] = max(0, counts.get(c, 0) - 1)

        remaining: List[Card] = []
        for ct, cnt in counts.items():
            remaining.extend([ct] * cnt)

        n = min(self._n_unrevealed, len(remaining))
        if n == 0:
            return list(opp_revealed)

        supp_pool = [c for c in remaining if c in supporting_set]
        non_supp_pool = [c for c in remaining if c not in supporting_set]

        self.rng.shuffle(supp_pool)
        self.rng.shuffle(non_supp_pool)

        s_actual = min(s, n, len(supp_pool))
        n_non = min(n - s_actual, len(non_supp_pool))

        hidden = supp_pool[:s_actual] + non_supp_pool[:n_non]

        extra = supp_pool[s_actual:] + non_supp_pool[n_non:]
        self.rng.shuffle(extra)
        while len(hidden) < n and extra:
            hidden.append(extra.pop())

        self.rng.shuffle(hidden)
        return list(opp_revealed) + hidden[:n]

    def _opp_choose_action(
        self, obs: Observation, current_claim: Claim, s: int
    ) -> object:
        """
        Simulate opponent action using an actual ToM0 agent as decision generator.
        """

        pub = obs.public
        my_id = obs.my_id
        opp = 1 - my_id
        my_hand = obs.my_hand
        opp_revealed = list(pub.revealed.get(opp, {}).values())

        # hypothetical opponent hand
        hyp_hand = self._sample_opp_hand(my_hand, opp_revealed, current_claim.qstate, s)

        # reuse a ToM0 object as decision generator
        # we only pass memory already available to ToM1 itself
        self._tom0_simulator = ToM0MemoryAgent(seed=0)
        self._tom0_simulator.last_memory = self.last_memory

        # post-claim public state the opponent would see
        my_revealed_idxs = tuple(
            i for i, card in enumerate(my_hand)
            if card in set(self._supporting_types(current_claim.qstate))
        )

        new_my_revealed = dict(pub.revealed.get(my_id, {}))
        for idx in my_revealed_idxs:
            if idx not in new_my_revealed:
                new_my_revealed[idx] = my_hand[idx]

        opp_public = PublicState(
            current_claim=current_claim,
            revealed={
                my_id: new_my_revealed,
                opp: dict(pub.revealed.get(opp, {})),
            },
            turn=opp,
            round_idx=pub.round_idx,
            history=list(pub.history),
        )

        opp_obs = Obs(my_id=opp, my_hand=hyp_hand, public=opp_public)
        action = self._tom0_simulator.act(opp_obs)

        if isinstance(action, DoubtAction):
            return "DOUBT"
        if isinstance(action, MakeClaimAction):
            if is_stronger(action.claim, current_claim):
                return action.claim
        return "DOUBT"

    def _estimate_utility(self, obs: Observation, candidate_claim: Claim) -> float:
        """
        Expected utility of making candidate_claim.
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

    # Our own claim probability

    def _prob_claim_true_my_view(self, obs: Observation, claim: Claim) -> float:
        """
        P(claim true) from our perspective, using the posterior belief_s.

        P(true) = sum_s  belief_s[q][s] * 1{ known_support + s >= claim.qty }

        """
        pub = obs.public
        opp = 1 - obs.my_id
        my_hand = obs.my_hand
        opp_revealed = list(pub.revealed.get(opp, {}).values())

        known_support = (
            self._count_support(my_hand, claim.qstate)
            + self._count_support(opp_revealed, claim.qstate)
        )
        still_needed = claim.qty - known_support
        if still_needed <= 0:
            return 1.0

        belief = self.belief_s.get(claim.qstate, [1.0])
        return float(sum(p for s, p in enumerate(belief) if s >= still_needed))

    # Tom0 fallback

    def _opponent_seems_tom0(self) -> bool:
        """
        Infer whether the opponent seems to be a simple ToM0 agent.
        """
        if self._rounds_observed < 10 or self._doubt_total < 3:
            return False
        doubt_accuracy = self._doubt_correct / self._doubt_total
        return doubt_accuracy > 0.58 and self._opp_bluffiness < 0.7

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
            candidates = [
                Claim(qty=q, qstate=s)
                for q in range(1, 8)
                for s in (QState.DEAD, QState.ALIVE, QState.EMPTY)
            ]
            best, best_score = None, -1e9
            for c in candidates:
                score = self._prob_claim_true_my_view(obs, c) - (0.02 + mem_penalty) * c.qty
                if score > best_score:
                    best_score, best = score, c
            assert best is not None
            reveal = tuple(
                i for i, card in enumerate(my_hand)
                if card in set(self._supporting_types(best.qstate))
            )
            return MakeClaim(best, reveal)

        current = pub.current_claim
        if self._prob_claim_true_my_view(obs, current) < self.DOUBT_THRESHOLD:
            return Doubt()

        legal = [
            Claim(qty=qty, qstate=q)
            for qty in range(1, 13)
            for q in (QState.DEAD, QState.ALIVE, QState.EMPTY)
            if is_stronger(Claim(qty=qty, qstate=q), current)
        ]

        best, best_score = None, -1e9
        for c in legal:
            gap = claim_strength(c)[0] - claim_strength(current)[0]
            score = self._prob_claim_true_my_view(obs, c) - 0.015 * gap - 0.02 * c.qty
            if score > best_score:
                best_score, best = score, c

        if best is None or best_score < 0.05:
            return Doubt()

        reveal = tuple(
            i for i, card in enumerate(my_hand)
            if card in set(self._supporting_types(best.qstate))
        )
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

        self._total_turns += 1

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
            self._fallback_turns += 1
            return self._act_as_tom0(obs)

        # Process new opponent claim events
        for ev in pub.history[self._last_processed_history_len:]:
            if ev.pid == opp and ev.kind == "claim" and ev.claim is not None:
                self._update_belief(ev.claim)
        self._last_processed_history_len = len(pub.history)

        mem_penalty = self._memory_opening_penalty()

        # Opening move
        if pub.current_claim is None:
            all_opening = [
                Claim(qty=q, qstate=s)
                for q in range(1, 8)
                for s in (QState.DEAD, QState.ALIVE, QState.EMPTY)
            ]
            scored_o = sorted(
                all_opening,
                key=lambda c: -self._prob_claim_true_my_view(obs, c)
            )
            candidates = scored_o[:10]

            best, best_score = None, -1e9
            for c in candidates:
                ev = self._estimate_utility(obs, c) - mem_penalty * c.qty
                if ev > best_score:
                    best_score, best = ev, c

            assert best is not None
            reveal = tuple(
                i for i, card in enumerate(my_hand)
                if card in set(self._supporting_types(best.qstate))
            )
            return MakeClaim(best, reveal)

        # Responding to opponent claim
        current = pub.current_claim
        if self._prob_claim_true_my_view(obs, current) < self.DOUBT_THRESHOLD:
            return Doubt()

        legal = [
            Claim(qty=qty, qstate=q)
            for qty in range(1, 13)
            for q in (QState.DEAD, QState.ALIVE, QState.EMPTY)
            if is_stronger(Claim(qty=qty, qstate=q), current)
        ]

        scored_legal = sorted(
            legal,
            key=lambda c: -self._prob_claim_true_my_view(obs, c)
        )
        candidates = scored_legal[:10]

        best, best_score = None, -1e9
        for c in candidates:
            ev = self._estimate_utility(obs, c)
            if ev > best_score:
                best_score, best = ev, c

        if best is None or best_score < 0.05:
            return Doubt()

        reveal = tuple(
            i for i, card in enumerate(my_hand)
            if card in set(self._supporting_types(best.qstate))
        )
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
            ev.revealed_count
            for ev in pub.history
            if ev.pid == opp and ev.kind == "claim"
        )

        opp_last_claim = None
        for ev in reversed(pub.history):
            if ev.pid == opp and ev.kind == "claim" and ev.claim is not None:
                opp_last_claim = ev.claim
                break

        last_doubt = next((ev for ev in reversed(pub.history) if ev.kind == "doubt"), None)
        opponent_last_claim_was_true = None
        if last_doubt is not None:
            claimant = 1 - last_doubt.pid
            if claimant == opp:
                opponent_last_claim_was_true = (winner == opp)

        self.last_memory = RoundMemory(
            opponent_doubted=opponent_doubted,
            opponent_evidence_revealed=opponent_evidence,
            opponent_last_claim=opp_last_claim,
            opponent_last_claim_was_true=opponent_last_claim_was_true,
        )

        if opponent_last_claim_was_true is not None:
            if not opponent_last_claim_was_true:
                self._opp_bluffiness = min(1.0, self._opp_bluffiness + 0.1)
            else:
                self._opp_bluffiness = max(0.0, self._opp_bluffiness - 0.05)

        for ev in pub.history:
            if ev.pid == opp and ev.kind == "doubt":
                self._doubt_total += 1
                if winner == opp:
                    self._doubt_correct += 1

        self._acting_as_tom0 = self._opponent_seems_tom0()
        self._round_initialized = False
        self._opp_revealed_this_round = []
        self._last_processed_history_len = 0