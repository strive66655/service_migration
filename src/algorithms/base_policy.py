from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from src.env.entities import User
from src.env.mec_env import MECEnvironment


class BasePolicy(ABC):
    @abstractmethod
    def select_node(self, env: MECEnvironment, user: User) -> Optional[int]:
        """
        返回为当前用户选择的目标节点 node_id。
        如果没有可用节点，返回 None。
        """
        raise NotImplementedError