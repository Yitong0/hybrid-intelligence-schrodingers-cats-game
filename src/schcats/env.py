from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import random

from .cards import Card, make_deck
from .rules import Claim, QState, is_stronger, check_claim_is_true
from .state import PublicState, Observation, PublicEvent


@dataclass(frozen=True)
class Action:
    pass


@dataclass(frozen=True)
class MakeClaim(Action):
    claim: Claim
    reveal_idxs: Tuple[int, ...] = ()


@dataclass(frozen=True)
class Doubt(Action):
    pass


class SchCatsEnv:
    """
    Two-player match consisting of multiple rounds.
    Evidence-based variant: when making a claim, a player may reveal any number
    of matching cards (and/or HUP) as evidence.

    Revealed cards are tracked by hand index (identity), not by Card value.
    public.revealed: pid -> {hand_idx: Card}
    """

    def __init__(self, rounds_per_match: int = 20, seed: int = 0):
        self.rng = random.Random(seed)
        self.rounds_per_match = rounds_per_match

        self.hands: List[List[Card]] = []
        self.public: Optional[PublicState] = None
        self.scores = [0, 0]  # rounds won

        self.last_round_public: Optional[PublicState] = None
        self.last_round_hands: Optional[List[List[Card]]] = None

    def reset_round(self, round_idx: int, starting_player: int):
        deck = make_deck(self.rng)
        self.hands = [deck[:6], deck[6:12]]  

        self.public = PublicState(
            current_claim=None,
            revealed={0: {}, 1: {}},  
            turn=starting_player,
            round_idx=round_idx,
            history=[],
        )

    def reset_match(self):
        self.scores = [0, 0]
        self.last_round_public = None
        self.last_round_hands = None
        self.reset_round(round_idx=0, starting_player=0)
        return self._obs(0), self._obs(1)

    def _obs(self, pid: int) -> Observation:
        assert self.public is not None
        return Observation(my_id=pid, my_hand=list(self.hands[pid]), public=self.public)

    def last_round_obs(self, pid: int) -> Observation:
        """
        Observation for the most recently FINISHED round.
        """
        assert self.last_round_public is not None, "No finished round snapshot available yet."
        assert self.last_round_hands is not None, "No finished round snapshot available yet."
        return Observation(
            my_id=pid,
            my_hand=list(self.last_round_hands[pid]),
            public=self.last_round_public,
        )

    def legal_actions(self, pid: int) -> List[Action]:
        assert self.public is not None
        actions: List[Action] = []

        if self.public.current_claim is None:
            actions.extend(self._all_legal_claims(pid, base_claim=None))
            return actions

        actions.append(Doubt())
        actions.extend(self._all_legal_claims(pid, base_claim=self.public.current_claim))
        return actions

    def _all_legal_claims(self, pid: int, base_claim: Optional[Claim]) -> List[MakeClaim]:
        candidates: List[Claim] = []
        for qty in range(1, 13):
            for q in (QState.DEAD, QState.ALIVE, QState.EMPTY):
                c = Claim(qty=qty, qstate=q)
                if base_claim is None or is_stronger(c, base_claim):
                    candidates.append(c)

        out: List[MakeClaim] = []
        for c in candidates:
            out.append(MakeClaim(c, ()))
            reveal = tuple(
                i for i, card in enumerate(self.hands[pid])
                if card == Card.HUP or card.value == c.qstate.value
            )
            out.append(MakeClaim(c, reveal))
        return out

    def _snapshot_finished_round(self):
        """Store a safe snapshot of the just-finished round for memory updates."""
        assert self.public is not None
        self.last_round_hands = [list(self.hands[0]), list(self.hands[1])]
        self.last_round_public = PublicState(
            current_claim=self.public.current_claim,
            revealed={0: dict(self.public.revealed[0]), 1: dict(self.public.revealed[1])},
            turn=self.public.turn,
            round_idx=self.public.round_idx,
            history=list(self.public.history),
        )

    def step(self, pid: int, action: Action) -> Tuple[Optional[int], bool]:
        """
        Returns (winner_pid_if_round_ended_else_None, match_done_bool)
        """
        assert self.public is not None
        assert pid == self.public.turn, "Not your turn."

        if isinstance(action, Doubt):
            assert self.public.current_claim is not None, "Cannot doubt before any claim."

            self.public.history.append(
                PublicEvent(
                    pid=pid,
                    kind="doubt",
                    claim=self.public.current_claim,
                    revealed_count=0,
                )
            )

            claim = self.public.current_claim
            truth = check_claim_is_true(claim, self.hands)

            claimant = 1 - pid
            winner = claimant if truth else pid
            self.scores[winner] += 1

            self._snapshot_finished_round()

            next_round = self.public.round_idx + 1
            if next_round >= self.rounds_per_match:
                return winner, True

            self.reset_round(round_idx=next_round, starting_player=next_round % 2)
            return winner, False

        if isinstance(action, MakeClaim):
            if self.public.current_claim is not None:
                if not is_stronger(action.claim, self.public.current_claim):
                    raise ValueError("Claim not stronger than current claim.")

            revealed_cards: List[Card] = []
            revealed_map: Dict[int, Card] = self.public.revealed[pid]

            for idx in action.reveal_idxs:
                if idx < 0 or idx >= len(self.hands[pid]):
                    raise ValueError("Bad reveal index.")

                if idx in revealed_map:
                    continue

                card = self.hands[pid][idx]

                if not (card == Card.HUP or card.value == action.claim.qstate.value):
                    raise ValueError("Revealed card does not support claim.")

                revealed_map[idx] = card
                revealed_cards.append(card)

            self.public.history.append(
                PublicEvent(
                    pid=pid,
                    kind="claim",
                    claim=action.claim,
                    revealed_count=len(revealed_cards),
                )
            )

            self.public.current_claim = action.claim
            self.public.turn = 1 - pid
            return None, False

        raise ValueError("Unknown action type.")