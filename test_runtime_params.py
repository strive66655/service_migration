from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from experiments.run_experiment import resolve_policy_params, run_policy_suite
from src.algorithms.cost_aware import CostAwarePolicy
from src.algorithms.llm_policy import LLMCostAwarePolicy
from src.algorithms.myopic import MyopicPolicy
from src.algorithms.policy_params import PolicyParams
from src.env.entities import EdgeNode, ServiceType, User, WorkloadProfile
from src.env.mec_env import MECEnvironment
from src.utils.config_loader import load_config
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
        enhanced_relative_gain_threshold=0.04,
        enhanced_stay_bias=0.08,
        enhanced_priority_boost=0.35,
        enhanced_delay_budget_ref=40.0,
        enhanced_business_sensitivity=0.45,
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


class FakeEnv:
    def __init__(
        self,
        costs: dict[int, float],
        candidate_ids: list[int] | None = None,
        keep_assignment: bool = True,
    ) -> None:
        self.nodes = {
            node_id: SimpleNamespace(node_id=node_id)
            for node_id in costs
        }
        self._costs = costs
        self._candidate_ids = candidate_ids or list(costs.keys())
        self._keep_assignment = keep_assignment

    def get_candidates_with_params(self, user: User, params: PolicyParams):
        return [self.nodes[node_id] for node_id in self._candidate_ids]

    def assignment_cost(
        self,
        user: User,
        node,
        src_node_id: int | None,
        params: PolicyParams,
    ) -> float:
        return self._costs[node.node_id]

    def can_keep_assignment(self, user: User, node, params: PolicyParams) -> bool:
        return self._keep_assignment


def make_mock_user(
    current_node_id: int = 0,
    cooldown_left: int = 0,
    priority: float = 1.0,
    delay_budget: float = 40.0,
) -> User:
    return User(
        user_id=0,
        x=0.0,
        y=0.0,
        vx=0.0,
        vy=0.0,
        service_type=ServiceType.AR,
        workload=WorkloadProfile(
            cpu_demand=1.0,
            bandwidth_demand=1.0,
            state_size=10.0,
            latency_sensitivity=1.0,
            delay_budget=delay_budget,
            priority=priority,
        ),
        current_node_id=current_node_id,
        cooldown_left=cooldown_left,
    )


def test_myopic_policy_migrates_to_lowest_cost_even_in_cooldown() -> None:
    env = FakeEnv(costs={0: 10.0, 1: 5.0}, candidate_ids=[0, 1], keep_assignment=True)
    user = make_mock_user(cooldown_left=3)

    selected_node = MyopicPolicy(make_policy_params()).select_node(env, user)

    assert selected_node == 1


def test_cost_aware_policy_stays_during_cooldown() -> None:
    env = FakeEnv(costs={0: 10.0, 1: 5.0}, candidate_ids=[0, 1], keep_assignment=True)
    user = make_mock_user(cooldown_left=2)

    selected_node = CostAwarePolicy(make_policy_params()).select_node(env, user)

    assert selected_node == 0


def test_enhanced_policy_requires_relative_gain_threshold() -> None:
    params = make_policy_params().merged_with(
        {
            "migrate_threshold": 0.05,
            "enhanced_stay_bias": 0.0,
            "enhanced_relative_gain_threshold": 0.04,
        }
    ).normalized()
    env = FakeEnv(costs={0: 10.0, 1: 9.7}, candidate_ids=[0, 1], keep_assignment=True)

    selected_node = CostAwarePolicy(params).select_node(env, make_mock_user())

    assert selected_node == 0


def test_enhanced_policy_is_more_willing_to_migrate_for_urgent_users() -> None:
    params = make_policy_params().normalized()
    env = FakeEnv(costs={0: 10.0, 1: 9.68}, candidate_ids=[0, 1], keep_assignment=True)
    policy = CostAwarePolicy(params)

    assert policy.select_node(env, make_mock_user(priority=0.3, delay_budget=120.0)) == 0
    assert policy.select_node(env, make_mock_user(priority=1.5, delay_budget=10.0)) == 1


def test_enhanced_policy_must_migrate_when_current_assignment_cannot_be_kept() -> None:
    env = FakeEnv(costs={0: 10.0, 1: 9.95}, candidate_ids=[0, 1], keep_assignment=False)

    selected_node = CostAwarePolicy(make_policy_params()).select_node(
        env,
        make_mock_user(),
    )

    assert selected_node == 1


def test_enhanced_policy_cooldown_takes_priority() -> None:
    env = FakeEnv(costs={0: 10.0, 1: 9.0}, candidate_ids=[0, 1], keep_assignment=True)

    selected_node = CostAwarePolicy(make_policy_params()).select_node(
        env,
        make_mock_user(cooldown_left=1, priority=1.5, delay_budget=10.0),
    )

    assert selected_node == 0


def test_run_policy_suite_includes_new_baselines() -> None:
    env_config = load_config("config/env.yaml")
    policy_config = load_config("config/policy.yaml")
    experiment_config = deepcopy(load_config("config/experiment.yaml"))
    experiment_config["steps"] = 1
    experiment_config["observe_steps"] = 0
    experiment_config["interactive_operator_input"] = False

    summary_df, _, _ = run_policy_suite(
        env_config=env_config,
        policy_config=policy_config,
        experiment_config=experiment_config,
        include_llm=False,
        verbose=False,
        scenario_name="test_baselines",
        seed=42,
    )

    policies = set(summary_df["Policy"])

    assert "myopic" in policies
    assert "cost_aware" in policies


def test_policy_resolution_uses_cost_aware_overrides() -> None:
    policy_config = {
        **make_policy_params().to_dict(),
        "cost_aware": {
            "migrate_threshold": 0.02,
            "cooldown_steps": 1,
            "enhanced_stay_bias": 0.3,
        },
        "llm": {"provider": "mock"},
    }

    cost_aware_params = resolve_policy_params(policy_config, "cost_aware")

    assert cost_aware_params.migrate_threshold == 0.02
    assert cost_aware_params.cooldown_steps == 1
    assert cost_aware_params.enhanced_stay_bias == 0.3
