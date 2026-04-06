from __future__ import annotations

from typing import Optional

from src.algorithms.base_policy import BasePolicy
from src.algorithms.policy_params import PolicyParams
from src.env.entities import User
from src.env.mec_env import MECEnvironment


class MyopicPolicy(BasePolicy):
    def __init__(self, params: PolicyParams) -> None:
        self.params = params

    def _assignment_cost(self, env: MECEnvironment, user: User, node_id: int) -> float:
        node = env.nodes[node_id]
        return env.assignment_cost(
            user,
            node,
            user.current_node_id,
            params=self.params,
        )

    def select_node(self, env: MECEnvironment, user: User) -> Optional[int]:
        candidates = env.get_candidates_with_params(user, params=self.params)
        if not candidates:
            return None

        best_node = min(
            candidates,
            key=lambda node: self._assignment_cost(env, user, node.node_id),
        )
        return best_node.node_id