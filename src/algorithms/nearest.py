from __future__ import annotations

from typing import Optional

from src.algorithms.base_policy import BasePolicy
from src.env.entities import User
from src.env.mec_env import MECEnvironment


class NearestPolicy(BasePolicy):
    def select_node(self, env: MECEnvironment, user: User) -> Optional[int]:
        candidates = env.get_candidates(user)
        if not candidates:
            return None
        return candidates[0].node_id