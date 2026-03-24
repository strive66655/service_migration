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
    qos_records: List[dict] = field(default_factory=list)

@dataclass
class SimulationMetrics:
    step_metrics: List[StepMetrics] = field(default_factory=list)

    def summary(self) -> Dict[str, float]:
        if not self.step_metrics:
            return {}

        result = {
            "avg_delay": statistics.mean(m.avg_delay for m in self.step_metrics),
            "avg_total_cost": statistics.mean(m.total_cost for m in self.step_metrics),
            "avg_migrations": statistics.mean(m.migration_count for m in self.step_metrics),
            "avg_failed_allocations": statistics.mean(m.failed_allocations for m in self.step_metrics),
            "avg_load_ratio": statistics.mean(m.avg_load_ratio for m in self.step_metrics),
        }

        qos_records: List[dict] = []
        for m in self.step_metrics:
            qos_records.extend(m.qos_records)

        if not qos_records:
            return result
        
        result["avg_qos_score"] = statistics.mean(r["qos_score"] for r in qos_records)
        result["intent_satisfaction_rate"] = statistics.mean(r["qos_satisfied"] for r in qos_records)
        result["sla_violation_rate"] = statistics.mean(r["sla_violated"] for r in qos_records)

        service_types = sorted(set(r["service_type"] for r in qos_records))
        for service_type in service_types:
            subset = [r for r in qos_records if r["service_type"] == service_type]

            result[f"{service_type}_intent_satisfaction_rate"] = statistics.mean(
                r["qos_satisfied"] for r in subset
            )
            result[f"{service_type}_sla_violated_rate"] = statistics.mean(
                r["sla_violated"] for r in subset
            )

            delays = [r["delay"] for r in subset if r["delay"] is not None]
            if delays:
                result[f"{service_type}_avg_delay"] = statistics.mean(delays)
                delays_sorted = sorted(delays)
                idx = min(len(delays_sorted) - 1, int(0.95 * len(delays_sorted)))
                result[f"{service_type}_p95_delay"] = delays_sorted[idx]
            else:
                result[f"{service_type}_avg_delay"] = 0.0
                result[f"{service_type}_p95_delay"] = 0.0

        return result



