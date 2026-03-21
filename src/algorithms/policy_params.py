from dataclasses import asdict, dataclass
from typing import Any, Dict

@dataclass
class PolicyParams:
    # cost-aware 权重
    lambda_delay: float
    lambda_migration: float
    lambda_resource: float
    lambda_balance: float

    # 冷却与候选节点
    migrate_threshold: float
    cooldown_steps: int
    max_candidates: int
    d_max: float

    # transmission_delay 系数
    queue_penalty_coeff: float
    sensitivity_coeff: float

    # migration_cost 系数
    migration_state_coeff: float
    migration_hops_coeff: float

    # resource_tension 上限
    resource_tension_max: float

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PolicyParams":
        return cls(
            lambda_delay=float(config_dict["lambda_delay"]),
            lambda_migration=float(config_dict["lambda_migration"]),
            lambda_resource=float(config_dict["lambda_resource"]),
            lambda_balance=float(config_dict["lambda_balance"]),
            migrate_threshold=float(config_dict["migrate_threshold"]),
            cooldown_steps=int(config_dict["cooldown_steps"]),
            max_candidates=int(config_dict["max_candidates"]),
            d_max=float(config_dict["d_max"]),
            queue_penalty_coeff=float(config_dict["queue_penalty_coeff"]),
            sensitivity_coeff=float(config_dict["sensitivity_coeff"]),
            migration_state_coeff=float(config_dict["migration_state_coeff"]),
            migration_hops_coeff=float(config_dict["migration_hops_coeff"]),
            resource_tension_max=float(config_dict["resource_tension_max"]),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def merged_with(self, partial: Dict[str, Any]) -> "PolicyParams":
        merged = self.to_dict()
        merged.update(partial)
        return PolicyParams.from_dict(merged)
    
    def normalized(self) -> "PolicyParams":
        weights = [
            max(0.0, float(self.lambda_delay)),
            max(0.0, float(self.lambda_migration)),
            max(0.0, float(self.lambda_resource)),
            max(0.0, float(self.lambda_balance)),
        ]
        total = sum(weights)
        if total <= 1e-8:
            weights = [0.25, 0.25, 0.25, 0.25]
            total = 1.0
        
        return PolicyParams(
            lambda_delay=weights[0] / total,
            lambda_migration=weights[1] / total,
            lambda_resource=weights[2] / total,
            lambda_balance=weights[3] / total,
            migrate_threshold=min(max(float(self.migrate_threshold), 0.0), 1.0),
            cooldown_steps=min(max(int(self.cooldown_steps), 0), 20),
            max_candidates=max(int(self.max_candidates), 1),
            d_max=max(float(self.d_max), 1e-6),
            queue_penalty_coeff=max(float(self.queue_penalty_coeff), 0.0),
            sensitivity_coeff=max(float(self.sensitivity_coeff), 0.0),
            migration_state_coeff=max(float(self.migration_state_coeff), 0.0),
            migration_hops_coeff=max(float(self.migration_hops_coeff), 0.0),
            resource_tension_max=max(float(self.resource_tension_max), 1e-6),
        )    
