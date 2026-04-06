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

    def _business_sensitivity(self, user: User) -> float:
        budget_ref = max(self.params.enhanced_delay_budget_ref, 1e-6)
        priority_pressure = max(user.workload.priority, 0.0)
        delay_pressure = max(1.0 - (user.workload.delay_budget / budget_ref), 0.0)
        return self.params.enhanced_business_sensitivity * (
            self.params.enhanced_priority_boost * priority_pressure + delay_pressure
        )

    def _adjusted_stay_bias(self, user: User) -> float:
        factor = max(0.0, 1.0 - self._business_sensitivity(user))
        return self.params.enhanced_stay_bias * factor

    def _relative_gain_threshold(self, user: User) -> float:
        base_threshold = self.params.enhanced_relative_gain_threshold
        factor = max(0.0, 1.0 - self._business_sensitivity(user))
        return base_threshold * factor

    def select_node(self, env: MECEnvironment, user: User) -> Optional[int]:
        candidates = env.get_candidates_with_params(user, params=self.params)
        if not candidates:
            return None

        best_node = min(
            candidates,
            key=lambda n: self._assignment_cost(env, user, n.node_id),
        )

        if user.current_node_id is None:
            return best_node.node_id

        current_node = env.nodes[user.current_node_id]
        if not env.can_keep_assignment(user, current_node, params=self.params):
            return best_node.node_id

        if user.cooldown_left > 0:
            return current_node.node_id

        current_cost = self._assignment_cost(env, user, current_node.node_id)
        best_cost = self._assignment_cost(env, user, best_node.node_id)

        if best_node.node_id == current_node.node_id:
            return current_node.node_id

        adjusted_current_cost = current_cost - self._adjusted_stay_bias(user)
        absolute_gain = adjusted_current_cost - best_cost
        relative_gain = absolute_gain / max(adjusted_current_cost, 1e-6)

        if (
            best_cost + self.params.migrate_threshold < adjusted_current_cost
            and relative_gain >= self._relative_gain_threshold(user)
        ):
            return best_node.node_id

        return current_node.node_id
