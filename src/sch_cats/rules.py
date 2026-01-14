from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Tuple, List
from .cards import Card


class QState(Enum):
    ALIVE = "alive"
    DEAD = "dead"
    EMPTY = "empty"


@dataclass(frozen=True)
class Claim:
    qty: int
    qstate: QState


def claim_strength(c: Claim) -> Tuple[int, int]:
    """
    Implements ordering rules as described in the project pdf on brightspace:
    - higher number is stronger
    - for same number: alive stronger than dead
    - empty 'counts double' and is stronger than alive
      (e.g., 2 empty ~ effective 4, stronger than 4 alive, weaker than 5 dead)
    """
    if c.qstate == QState.EMPTY:
        effective = 2 * c.qty
        tier = 2  # strongest tie-break
    elif c.qstate == QState.ALIVE:
        effective = c.qty
        tier = 1
    else:  # DEAD
        effective = c.qty
        tier = 0
    return (effective, tier)


def is_stronger(new: Claim, old: Claim) -> bool:
    return claim_strength(new) > claim_strength(old)


def check_claim_is_true(
    claim: Claim,
    all_hands: List[List[Card]],
) -> bool:
    """
    When doubting, all hands are revealed.
    HUP counts as any quantum state for satisfying the claim.
     [oai_citation:9‡HybridIntelligence_project.pdf](file-service://file-JkWz1xDKUBgnMr327uyr5J)
    """
    needed = claim.qty
    target = claim.qstate.value

    cnt_target = 0
    cnt_hup = 0

    for hand in all_hands:
        for card in hand:
            if card == Card.HUP:
                cnt_hup += 1
            elif card.value == target:
                cnt_target += 1

    return (cnt_target + cnt_hup) >= needed