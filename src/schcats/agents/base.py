from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from ..state import Observation
from ..env import Action


class Agent(ABC):
    @abstractmethod
    def act(self, obs: Observation) -> Action:
        ...

    def observe_round_end(self, *, my_id: int, winner: int, last_obs: Observation) -> None:
        """Hook for explicit memory updates."""
        return