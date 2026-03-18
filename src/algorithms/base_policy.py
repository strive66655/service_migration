from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from src.env.entities import User
from src.env.mec_env import MECEnvironment


class BasePolicy(ABC):
    @abstractmethod
    def select_node(self, env: MECEnvironment, user: User) -> Optional[int]:
        raise NotImplementedError

    def debug_snapshot(self) -> dict:
        return {}
