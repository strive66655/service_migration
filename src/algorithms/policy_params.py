from dataclasses import asdict, dataclass
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

    enhanced_relative_gain_threshold: float = 0.04
    enhanced_stay_bias: float = 0.08
    enhanced_priority_boost: float = 0.35
    enhanced_delay_budget_ref: float = 40.0
    enhanced_business_sensitivity: float = 0.45

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
            enhanced_relative_gain_threshold=float(
                config_dict["enhanced_relative_gain_threshold"]
            ),
            enhanced_stay_bias=float(config_dict["enhanced_stay_bias"]),
            enhanced_priority_boost=float(config_dict["enhanced_priority_boost"]),
            enhanced_delay_budget_ref=float(config_dict["enhanced_delay_budget_ref"]),
            enhanced_business_sensitivity=float(
                config_dict["enhanced_business_sensitivity"]
            ),
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
            migration_norm_factor=max(float(self.migration_norm_factor), 1e-6),
            resource_tension_max=max(float(self.resource_tension_max), 1e-6),
            allocation_failure_penalty=max(float(self.allocation_failure_penalty), 0.0),
            enhanced_relative_gain_threshold=min(
                max(float(self.enhanced_relative_gain_threshold), 0.0),
                1.0,
            ),
            enhanced_stay_bias=max(float(self.enhanced_stay_bias), 0.0),
            enhanced_priority_boost=max(float(self.enhanced_priority_boost), 0.0),
            enhanced_delay_budget_ref=max(
                float(self.enhanced_delay_budget_ref),
                1e-6,
            ),
            enhanced_business_sensitivity=max(
                float(self.enhanced_business_sensitivity),
                0.0,
            ),
        )
