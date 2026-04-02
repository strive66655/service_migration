from __future__ import annotations

from src.algorithms.cost_aware import CostAwarePolicy
from src.algorithms.llm_policy import LLMCostAwarePolicy
from src.algorithms.policy_params import PolicyParams
from src.env.entities import EdgeNode, ServiceType, User, WorkloadProfile
from src.env.mec_env import MECEnvironment
from src.llm.base import BaseLLMProvider
from src.llm.controller import LLMPolicyController
from src.llm.response_parser import ResponseParser
from src.runners.simulation_runner import SimulationRunner


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


def make_two_node_env(env_params: PolicyParams) -> MECEnvironment:
    nodes = [
        EdgeNode(node_id=0, x=0.0, y=0.0, cpu_capacity=20.0, bandwidth_capacity=20.0),
        EdgeNode(node_id=1, x=10.0, y=0.0, cpu_capacity=20.0, bandwidth_capacity=20.0),
    ]
    users = [
        User(
            user_id=0,
            x=9.0,
            y=0.0,
            vx=0.0,
            vy=0.0,
            service_type=ServiceType.AR,
            workload=WorkloadProfile(
                cpu_demand=2.0,
                bandwidth_demand=2.0,
                state_size=10.0,
                latency_sensitivity=1.0,
            ),
            current_node_id=0,
        )
    ]
    return MECEnvironment(
        width=100.0,
        height=100.0,
        nodes=nodes,
        users=users,
        params=env_params,
    )


def test_response_parser_only_accepts_threshold_in_extended_mode() -> None:
    parser = ResponseParser("mock", "mock-model")
    decision = parser.parse(
        '{"migrate_threshold": 0.8, "cooldown_steps": 4, "reason": "adjusted"}',
        make_policy_params(),
    )

    assert decision.params.cooldown_steps == 4
    assert decision.params.migrate_threshold == make_policy_params().migrate_threshold

    extended_decision = parser.parse(
        '{"migrate_threshold": 0.8, "cooldown_steps": 4, "reason": "adjusted"}',
        make_policy_params(),
        experiment_mode="extended",
    )

    assert extended_decision.params.cooldown_steps == 4
    assert extended_decision.params.migrate_threshold == 0.8


def test_cost_aware_policy_uses_runtime_params_for_candidates_and_threshold() -> None:
    env_params = make_policy_params().merged_with(
        {
            "d_max": 5.0,
            "max_candidates": 1,
            "migrate_threshold": 1.0,
        }
    ).normalized()
    runtime_params = make_policy_params().merged_with(
        {
            "d_max": 20.0,
            "max_candidates": 2,
            "migrate_threshold": 0.0,
            "lambda_delay": 0.8,
            "lambda_migration": 0.05,
            "lambda_resource": 0.1,
            "lambda_balance": 0.05,
        }
    ).normalized()
    env = make_two_node_env(env_params)
    user = env.users[0]
    policy = CostAwarePolicy(runtime_params)

    selected_node = policy.select_node(env, user)

    assert selected_node == 1
    assert [node.node_id for node in env.get_candidates(user)] == [1]
    assert [node.node_id for node in env.get_candidates_with_params(user, params=runtime_params)] == [1, 0]
    assert env.assignment_cost(
        user,
        env.nodes[1],
        user.current_node_id,
        params=runtime_params,
    ) < env.assignment_cost(
        user,
        env.nodes[0],
        user.current_node_id,
        params=runtime_params,
    )


class StaticJSONProvider(BaseLLMProvider):
    def __init__(self, payload: str) -> None:
        self.payload = payload

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self.payload


def test_llm_policy_refresh_keeps_runtime_params_consistent() -> None:
    default_params = make_policy_params().merged_with(
        {
            "d_max": 20.0,
            "max_candidates": 2,
            "migrate_threshold": 0.6,
            "cooldown_steps": 1,
        }
    ).normalized()
    env = make_two_node_env(default_params)
    provider = StaticJSONProvider(
        '{"lambda_delay": 0.8, "lambda_migration": 0.05, '
        '"lambda_resource": 0.1, "lambda_balance": 0.05, '
        '"migrate_threshold": 0.0, "cooldown_steps": 4, "reason": "refresh"}'
    )
    controller = LLMPolicyController(
        provider=provider,
        provider_name="static",
        model_name="test-model",
        default_params=default_params,
        experiment_mode="extended",
    )
    policy = LLMCostAwarePolicy(
        default_params=default_params,
        controller=controller,
        update_interval=1,
    )
    runner = SimulationRunner(env, policy)

    metrics = runner.step()
    user = env.users[0]

    assert metrics.migration_count == 1
    assert user.current_node_id == 1
    assert user.cooldown_left == 4
    assert policy.current_params == policy.inner_policy.params
    assert policy.last_decision_meta["cooldown_steps"] == 4
    assert policy.last_decision_meta["migrate_threshold"] == 0.0
