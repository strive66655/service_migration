from __future__ import annotations

from typing import Optional

from src.algorithms.base_policy import BasePolicy
from src.algorithms.policy_params import PolicyParams
from src.env.entities import EdgeNode, User
from src.env.mec_env import MECEnvironment


class CostAwarePolicy(BasePolicy):
    def __init__(self, params: PolicyParams) -> None:
        self.params = params

    def _score(self, env: MECEnvironment, user: User, node: EdgeNode) -> float:
        # 1. 距离/时延项
        delay_cost = env.transmission_delay(user, node) / max(self.params.d_max, 1e-6)

        # 2. 迁移代价项
        migration_cost = env.migration_cost(user, user.current_node_id, node.node_id) / 20.0

        # 3. 资源紧张项
        resource_cost = min(env.resource_tension(user, node), 5.0) / 5.0

        # 4. 负载均衡项
        balance_cost = node.load_ratio

        return (
            self.params.lambda_delay * delay_cost
            + self.params.lambda_migration * migration_cost
            + self.params.lambda_resource * resource_cost
            + self.params.lambda_balance * balance_cost
        )

    def select_node(self, env: MECEnvironment, user: User) -> Optional[int]:
        candidates = env.get_candidates(user)
        if not candidates:
            return None

        # 候选节点中选 score 最小的
        best_node = min(candidates, key=lambda n: self._score(env, user, n))

        # 如果之前没有部署，直接部署到 best_node
        if user.current_node_id is None:
            return best_node.node_id

        current_node = env.nodes[user.current_node_id]

        # 当前节点已经无法承载，则必须迁移
        if not env.can_allocate(user, current_node):
            return best_node.node_id

        # 冷却期内，不迁移
        if user.cooldown_left > 0:
            return current_node.node_id

        current_score = self._score(env, user, current_node)
        best_score = self._score(env, user, best_node)

        # 只有当新节点显著更优时才迁移，避免抖动
        if (
            best_node.node_id != current_node.node_id
            and best_score + self.params.migrate_threshold < current_score
        ):
            return best_node.node_id

        return current_node.node_id