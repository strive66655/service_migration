from __future__ import annotations

from typing import Optional, Tuple

from src.algorithms.base_policy import BasePolicy
from src.algorithms.policy_params import PolicyParams
from src.env.entities import EdgeNode, User
from src.env.mec_env import MECEnvironment


class CostAwarePolicy(BasePolicy):
    def __init__(self, params: PolicyParams) -> None:
        self.params = params

    def _predict_position(
        self,
        user: User,
        width: float,
        height: float,
        steps_ahead: int,
    ) -> Tuple[float, float, float, float]:
        x, y = user.x, user.y
        vx, vy = user.vx, user.vy

        for _ in range(steps_ahead):
            x += vx
            y += vy

            if x < 0 or x > width:
                vx *= -1
                x = min(max(x, 0), width)

            if y < 0 or y > height:
                vy *= -1
                y = min(max(y, 0), height)

        return x, y, vx, vy

    def _future_delay_cost(
        self,
        env: MECEnvironment,
        user: User,
        node: EdgeNode,
        steps_ahead: int,
        projected_load_ratio: float,
    ) -> float:
        x, y, _, _ = self._predict_position(user, env.width, env.height, steps_ahead)
        distance = ((x - node.x) ** 2 + (y - node.y) ** 2) ** 0.5
        queue_penalty = self.params.queue_penalty_coeff * projected_load_ratio
        sensitivity = 1.0 + self.params.sensitivity_coeff * user.workload.latency_sensitivity
        return self.params.lambda_delay * sensitivity * (distance + queue_penalty)

    def _score(self, env: MECEnvironment, user: User, node: EdgeNode) -> float:
        immediate_cost = env.assignment_cost(user, node, user.current_node_id)
        projected_load_ratio = env.projected_load_ratio(user, node)

        horizon = max(1, self.params.cooldown_steps)
        future_cost = 0.0
        for step in range(1, horizon + 1):
            future_cost += self._future_delay_cost(
                env,
                user,
                node,
                step,
                projected_load_ratio,
            ) / (step + 1)

        return immediate_cost + future_cost

    def select_node(self, env: MECEnvironment, user: User) -> Optional[int]:
        candidates = env.get_candidates(user)
        if not candidates:
            return None

        best_node = min(candidates, key=lambda n: self._score(env, user, n))

        if user.current_node_id is None:
            return best_node.node_id

        current_node = env.nodes[user.current_node_id]
        if not env.can_allocate(user, current_node):
            return best_node.node_id

        if user.cooldown_left > 0:
            return current_node.node_id

        current_score = self._score(env, user, current_node)
        best_score = self._score(env, user, best_node)

        if (
            best_node.node_id != current_node.node_id
            and best_score + self.params.migrate_threshold < current_score
        ):
            return best_node.node_id

        return current_node.node_id
