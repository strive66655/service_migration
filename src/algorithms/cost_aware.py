from __future__ import annotations

from typing import Optional

from src.algorithms.base_policy import BasePolicy
from src.algorithms.policy_params import PolicyParams
from src.env.entities import User
from src.env.mec_env import MECEnvironment


class CostAwarePolicy(BasePolicy):
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
        candidates = env.get_candidates(user)
        if not candidates:
            return None

        best_node = min(
            candidates,
            key=lambda n: self._assignment_cost(env, user, n.node_id),
        )

        if user.current_node_id is None:
            return best_node.node_id

        current_node = env.nodes[user.current_node_id]
        if not env.can_allocate(user, current_node):
            return best_node.node_id

        if user.cooldown_left > 0:
            return current_node.node_id

        current_cost = self._assignment_cost(env, user, current_node.node_id)
        best_cost = self._assignment_cost(env, user, best_node.node_id)

        if (
            best_node.node_id != current_node.node_id
            and best_cost + self.params.migrate_threshold < current_cost
        ):
            return best_node.node_id

        return current_node.node_id
