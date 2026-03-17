from __future__ import annotations

from typing import Optional

from src.algorithms.base_policy import BasePolicy
from src.env.entities import User
from src.env.mec_env import MECEnvironment


class NeverMigratePolicy(BasePolicy):
    def select_node(self, env: MECEnvironment, user: User) -> Optional[int]:
        # 如果当前已经有部署节点，并且还能容纳当前用户，就一直不迁移
        if user.current_node_id is not None:
            current_node = env.nodes[user.current_node_id]
            if env.can_allocate(user, current_node):
                return current_node.node_id

        # 首次部署或当前节点不可用时，选最近的可行节点
        candidates = env.get_candidates(user)
        if not candidates:
            return None
        return candidates[0].node_id