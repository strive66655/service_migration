from __future__ import annotations

from typing import Optional

from src.algorithms.base_policy import BasePolicy
from src.env.entities import User
from src.env.mec_env import MECEnvironment


class NeverMigratePolicy(BasePolicy):
    def select_node(self, env: MECEnvironment, user: User) -> Optional[int]:
        if user.current_node_id is not None:
            current_node = env.nodes[user.current_node_id]
            if env.can_keep_assignment(user, current_node):
                return current_node.node_id
            return None

        candidates = env.get_candidates(user)
        if not candidates:
            return None
        return candidates[0].node_id
