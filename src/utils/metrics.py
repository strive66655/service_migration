from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
import statistics

def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))

def compute_qos_index(
    avg_delay: float,
    avg_failed_allocations: float,
    user_count: int,
    avg_load_ratio: float,
    load_std: float,
    delay_ref: float = 80.0,
    load_std_ref: float = 0.30,
    w_delay: float = 0.50,
    w_failure: float = 0.30,
    w_balance: float = 0.20,
) -> Dict[str, float]:
    
    safe_user_count = max(user_count, 1)

    delay_score = _clamp(1.0 - avg_delay / delay_ref)

    failure_rate = avg_failed_allocations / safe_user_count
    failure_rate_score = _clamp(1.0 - failure_rate)

    load_health_score = _clamp(1.0 - avg_load_ratio)
    load_distribution_score = _clamp(1.0 - load_std / load_std_ref)
    balance_score = 0.5 * load_health_score + 0.5 * load_distribution_score
    
    qos_index = (
        w_delay * delay_score
        + w_failure * failure_rate_score
        + w_balance * balance_score
    )

    return {
        "qos_index": qos_index,
        "delay_score": delay_score,
        "failure_rate": failure_rate,
        "failure_rate_score": failure_rate_score,
        "load_health_score": load_health_score,
        "load_distribution_score": load_distribution_score,
        "balance_score": balance_score,
    }

@dataclass
class StepMetrics:
    avg_delay: float = 0.0
    migration_count: int = 0
    failed_allocations: int = 0
    total_cost: float = 0.0
    avg_load_ratio: float = 0.0
    load_std: float = 0.0
    user_count: int = 0

    def qos_summary(self) -> Dict[str, float]:
        return compute_qos_index(
            avg_delay=self.avg_delay,
            avg_failed_allocations=float(self.failed_allocations),
            user_count=self.user_count,
            avg_load_ratio=self.avg_load_ratio,
            load_std=self.load_std,
        )
    
@dataclass
class SimulationMetrics:
    step_metrics: List[StepMetrics] = field(default_factory=list)

    def summary(self) -> Dict[str, float]:
        if not self.step_metrics:
            return {}

        avg_delay = statistics.mean(m.avg_delay for m in self.step_metrics)
        avg_total_cost = statistics.mean(m.total_cost for m in self.step_metrics)
        avg_migrations = statistics.mean(m.migration_count for m in self.step_metrics)
        avg_failed_allocations = statistics.mean(
            m.failed_allocations for m in self.step_metrics
        )
        avg_load_ratio = statistics.mean(m.avg_load_ratio for m in self.step_metrics)
        avg_load_std = statistics.mean(m.load_std for m in self.step_metrics)

        avg_user_count = round(statistics.mean(m.user_count for m in self.step_metrics))

        result = {
            "avg_delay": avg_delay,
            "avg_total_cost": avg_total_cost,
            "avg_migrations": avg_migrations,
            "avg_failed_allocations": avg_failed_allocations,
            "avg_load_ratio": avg_load_ratio,
            "avg_load_std": avg_load_std,
        }

        result.update(
            compute_qos_index(
                avg_delay=avg_delay,
                avg_failed_allocations=avg_failed_allocations,
                user_count=avg_user_count,
                avg_load_ratio=avg_load_ratio,
                load_std=avg_load_std,
            )
        )

        return result
