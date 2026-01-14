from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import random

from .cards import Card, make_deck
from .rules import Claim, QState, is_stronger, check_claim_is_true
from .state import PublicState, Observation


@dataclass(frozen=True)
class Action:
    pass


@dataclass(frozen=True)
class MakeClaim(Action):
    claim: Claim
    # indices of cards in my hand to reveal as evidence (can include HUP)
    reveal_idxs: Tuple[int, ...] = ()


@dataclass(frozen=True)
class Doubt(Action):
    pass


class SchCatsEnv:
    """
    Two-player match consisting of multiple rounds (so explicit memory is meaningful).
    Each round ends when someone doubts, then we redeal for next round.
    """
    def __init__(self, rounds_per_match: int = 20, seed: int = 0):
        self.rng = random.Random(seed)
        self.rounds_per_match = rounds_per_match

        self.hands: List[List[Card]] = []
        self.public: Optional[PublicState] = None
        self.scores = [0, 0]  # rounds won

    def reset_round(self, round_idx: int, starting_player: int):
        deck = make_deck(self.rng)
        self.hands = [deck[:6], deck[6:12]]  # 2 players, 6 cards each  [oai_citation:13‡HybridIntelligence_project.pdf](file-service://file-JkWz1xDKUBgnMr327uyr5J)

        self.public = PublicState(
            current_claim=None,
            revealed={0: [], 1: []},
            turn=starting_player,
            round_idx=round_idx,
        )

    def reset_match(self):
        self.scores = [0, 0]
        self.reset_round(round_idx=0, starting_player=0)
        return self._obs(0), self._obs(1)

    def _obs(self, pid: int) -> Observation:
        assert self.public is not None
        return Observation(my_id=pid, my_hand=list(self.hands[pid]), public=self.public)

    def legal_actions(self, pid: int) -> List[Action]:
        assert self.public is not None
        actions: List[Action] = []

        # If no claim yet, must make a claim (cannot doubt)
        if self.public.current_claim is None:
            actions.extend(self._all_legal_claims(pid, base_claim=None))
            return actions

        # Otherwise can doubt or raise
        actions.append(Doubt())
        actions.extend(self._all_legal_claims(pid, base_claim=self.public.current_claim))
        return actions

    def _all_legal_claims(self, pid: int, base_claim: Optional[Claim]) -> List[MakeClaim]:
        """
        Generates a *reasonable* set of possible claims.
        You can expand this later, but this is enough to run experiments.
        """
        candidates: List[Claim] = []
        # quantities from 1..12 (max cards across both hands is 12)
        for qty in range(1, 13):
            for q in [QState.DEAD, QState.ALIVE, QState.EMPTY]:
                c = Claim(qty=qty, qstate=q)
                if base_claim is None or is_stronger(c, base_claim):
                    candidates.append(c)

        # Evidence indices: allow reveal nothing or reveal all matching
        out: List[MakeClaim] = []
        for c in candidates:
            out.append(MakeClaim(c, ()))
            # reveal all matching cards (including HUP allowed)  [oai_citation:14‡HybridIntelligence_project.pdf](file-service://file-JkWz1xDKUBgnMr327uyr5J)
            reveal = tuple(i for i, card in enumerate(self.hands[pid])
                           if card == Card.HUP or card.value == c.qstate.value)
            out.append(MakeClaim(c, reveal))
        return out

    def step(self, pid: int, action: Action) -> Tuple[Optional[int], bool]:
        """
        Returns (winner_pid_if_round_ended_else_None, match_done_bool)
        """
        assert self.public is not None
        assert pid == self.public.turn, "Not your turn."

        if isinstance(action, Doubt):
            assert self.public.current_claim is not None, "Cannot doubt before any claim."
            claim = self.public.current_claim
            truth = check_claim_is_true(claim, self.hands) 
            # If claim true -> claimant wins; else doubter wins
            claimant = 1 - pid
            winner = claimant if truth else pid
            self.scores[winner] += 1

            # next round or end match
            next_round = self.public.round_idx + 1
            if next_round >= self.rounds_per_match:
                return winner, True
            # alternate starting player for fairness
            self.reset_round(round_idx=next_round, starting_player=next_round % 2)
            return winner, False

        if isinstance(action, MakeClaim):
            # validate claim legality
            if self.public.current_claim is not None:
                if not is_stronger(action.claim, self.public.current_claim):
                    raise ValueError("Claim not stronger than current claim.")

            # cannot claim HUP (guaranteed by QState enum)  [oai_citation:16‡HybridIntelligence_project.pdf](file-service://file-JkWz1xDKUBgnMr327uyr5J)

            # apply evidence-based reveal: reveal chosen indices if legal
            revealed_cards: List[Card] = []
            for idx in action.reveal_idxs:
                if idx < 0 or idx >= len(self.hands[pid]):
                    raise ValueError("Bad reveal index.")
                card = self.hands[pid][idx]
                # must match claimed state OR be HUP  [oai_citation:17‡HybridIntelligence_project.pdf](file-service://file-JkWz1xDKUBgnMr327uyr5J)
                if not (card == Card.HUP or card.value == action.claim.qstate.value):
                    raise ValueError("Revealed card does not support claim.")
                revealed_cards.append(card)

            self.public.revealed[pid].extend(revealed_cards)
            self.public.current_claim = action.claim
            self.public.turn = 1 - pid
            return None, False

        raise ValueError("Unknown action type.")