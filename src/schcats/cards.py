from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import random
from typing import List


class Card(Enum):
    ALIVE = "alive"
    DEAD = "dead"
    EMPTY = "empty"
    HUP = "hup"  


def make_deck(rng: random.Random) -> List[Card]:
    deck = (
        [Card.ALIVE] * 20 +
        [Card.DEAD] * 20 +
        [Card.EMPTY] * 8 +
        [Card.HUP] * 4
    )
    rng.shuffle(deck)
    return deck