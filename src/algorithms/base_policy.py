from abc import ABC, abstractmethod
from typing import Dict


class BasePolicy(ABC):
    @abstractmethod
    def decide(self, env) -> Dict[int, int]:
        """
        返回:
            {user_id: target_server_id}
        """
        raise NotImplementedError