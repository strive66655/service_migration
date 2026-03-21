from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from  src.algorithms.policy_params import PolicyParams


@dataclass
class LLMDecision:
    params: PolicyParams
    reason: str
    provider: str
    model: str
    used_fallback: bool
    raw_text: str
    parsed_payload: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SceneSummary:
    step: int
    node_count: int
    user_count: int
    avg_node_load: float
    max_node_load: float
    failed_allocations_recent: int 
    migrations_recent: int
    service_counts: Dict[str, int]
    intent_summary: str
    snapshot: Dict[str, Any]