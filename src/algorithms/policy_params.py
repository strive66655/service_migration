from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class PolicyParams:
    lambda_delay: float
    lambda_migration: float
    lambda_resource: float
    lambda_balance: float

    migrate_threshold: float
    cooldown_steps: int
    max_candidates: int
    d_max: float

    queue_penalty_coeff: float
    sensitivity_coeff: float

    migration_state_coeff: float
    migration_hops_coeff: float
    migration_norm_factor: float

    resource_tension_max: float
    allocation_failure_penalty: float

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
            migration_norm_factor=float(config_dict["migration_norm_factor"]),
            resource_tension_max=float(config_dict["resource_tension_max"]),
            allocation_failure_penalty=float(config_dict["allocation_failure_penalty"]),
        )
