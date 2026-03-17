from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
import statistics


@dataclass
class StepMetrics:
    avg_delay: float = 0.0
    migration_count: int = 0
    failed_allocations: int = 0
    total_cost: float = 0.0
    avg_load_ratio: float = 0.0


@dataclass
class SimulationMetrics:
    step_metrics: List[StepMetrics] = field(default_factory=list)

    def summary(self) -> Dict[str, float]:
        if not self.step_metrics:
            return {}

        return {
            "avg_delay": statistics.mean(m.avg_delay for m in self.step_metrics),
            "avg_total_cost": statistics.mean(m.total_cost for m in self.step_metrics),
            "avg_migrations": statistics.mean(m.migration_count for m in self.step_metrics),
            "avg_failed_allocations": statistics.mean(m.failed_allocations for m in self.step_metrics),
            "avg_load_ratio": statistics.mean(m.avg_load_ratio for m in self.step_metrics),
        }