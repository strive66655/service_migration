from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_experiment import resolve_instruction_phase
from src.algorithms.policy_params import PolicyParams
from src.llm.prompt_builder import PromptBuilder
from src.llm.response_parser import ResponseParser
from src.llm.schemas import SceneSummary


def make_policy_params() -> PolicyParams:
    return PolicyParams(
        lambda_delay=0.25,
        lambda_migration=0.25,
        lambda_resource=0.25,
        lambda_balance=0.25,
        migrate_threshold=0.05,
        cooldown_steps=2,
        max_candidates=4,
        d_max=32.0,
        queue_penalty_coeff=24.0,
        sensitivity_coeff=0.25,
        migration_state_coeff=0.15,
        migration_hops_coeff=8.0,
        migration_norm_factor=20.0,
        resource_tension_max=4.0,
        allocation_failure_penalty=50.0,
    )


def make_scene_summary(operator_instruction: str = "优先降低时延") -> SceneSummary:
    return SceneSummary(
        step=10,
        node_count=5,
        user_count=15,
        avg_node_load=0.4,
        max_node_load=0.8,
        failed_allocations_recent=1,
        migrations_recent=3,
        avg_delay_recent=12.5,
        migration_rate_recent=0.04,
        failed_allocation_rate_recent=0.01,
        users_in_cooldown_ratio=0.2,
        avg_migrations_per_user_recent=0.2,
        service_counts={"video": 8, "ar": 7},
        user_context_summary="video: 8 users ; ar: 7 users",
        operator_instruction=operator_instruction,
        snapshot={"time_step": 10, "users": {}, "nodes": {}},
    )


def test_prompt_builder_keeps_readable_instruction_text() -> None:
    builder = PromptBuilder()
    system_prompt, user_prompt = builder.build(
        make_scene_summary(),
        make_policy_params(),
    )

    assert "你是移动边缘计算中的运维策略参数优化器" in system_prompt
    payload = json.loads(user_prompt)
    assert payload["scene_summary"]["operator_instruction"] == "优先降低时延"


def test_response_parser_marks_empty_tunable_payload_as_fallback() -> None:
    parser = ResponseParser("mock", "mock-model")
    decision = parser.parse('{"reason": "only explanation"}', make_policy_params())

    assert decision.used_fallback is True
    assert "No tunable parameters parsed" in decision.reason


def test_response_parser_uses_model_output_when_tunable_fields_exist() -> None:
    parser = ResponseParser("mock", "mock-model")
    decision = parser.parse(
        '{"lambda_delay": 0.7, "cooldown_steps": 4, "reason": "adjusted"}',
        make_policy_params(),
    )

    assert decision.used_fallback is False
    assert decision.params.cooldown_steps == 4
    assert abs(decision.params.lambda_delay - 0.48275862068965514) < 1e-9


def test_instruction_phase_reflects_whether_instruction_was_updated() -> None:
    assert resolve_instruction_phase(3, 5, False) == "default_instruction"
    assert resolve_instruction_phase(6, 5, False) == "default_instruction"
    assert resolve_instruction_phase(6, 5, True) == "updated_instruction"
