from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import random
from typing import List


class Card(Enum):
    ALIVE = "alive"
    DEAD = "dead"
    EMPTY = "empty"
    HUP = "hup"  # wild for checking claims, cannot be claimed


def make_deck(rng: random.Random) -> List[Card]:
    # 20 alive, 20 dead, 8 empty, 4 HUP
    deck = (
        [Card.ALIVE] * 20 +
        [Card.DEAD] * 20 +
        [Card.EMPTY] * 8 +
        [Card.HUP] * 4
    )
    rng.shuffle(deck)
    return deck