from __future__ import annotations

from src.algorithms.cost_aware import CostAwarePolicy
from src.algorithms.nearest import NearestPolicy
from src.algorithms.policy_params import PolicyParams
from src.env.entities import EdgeNode, ServiceType, User, WorkloadProfile
from src.env.mec_env import MECEnvironment
from src.runners.simulation_runner import SimulationRunner


def make_params(**overrides) -> PolicyParams:
    base = {
        "lambda_delay": 0.35,
        "lambda_migration": 0.2,
        "lambda_resource": 0.3,
        "lambda_balance": 0.15,
        "migrate_threshold": 0.05,
        "cooldown_steps": 2,
        "max_candidates": 4,
        "d_max": 50.0,
        "queue_penalty_coeff": 24.0,
        "sensitivity_coeff": 0.25,
        "migration_state_coeff": 0.15,
        "migration_hops_coeff": 8.0,
        "migration_norm_factor": 20.0,
        "resource_tension_max": 4.0,
        "allocation_failure_penalty": 120.0,
    }
    base.update(overrides)
    return PolicyParams.from_dict(base).normalized()


def make_user(
    *,
    user_id: int = 0,
    x: float = 10.0,
    y: float = 10.0,
    vx: float = 0.0,
    vy: float = 0.0,
    current_node_id: int | None = None,
    cooldown_left: int = 0,
    workload: WorkloadProfile | None = None,
) -> User:
    return User(
        user_id=user_id,
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        service_type=ServiceType.AR,
        workload=workload
        or WorkloadProfile(
            cpu_demand=4.0,
            bandwidth_demand=3.0,
            state_size=60.0,
            latency_sensitivity=1.0,
        ),
        current_node_id=current_node_id,
        cooldown_left=cooldown_left,
    )


def make_env(
    nodes: list[EdgeNode],
    users: list[User],
    params: PolicyParams | None = None,
) -> MECEnvironment:
    return MECEnvironment(
        width=100.0,
        height=100.0,
        nodes=nodes,
        users=users,
        params=params or make_params(),
    )


def test_failed_allocation_penalty_is_added_to_total_cost() -> None:
    params = make_params()
    nodes = [
        EdgeNode(node_id=0, x=0.0, y=0.0, cpu_capacity=1.0, bandwidth_capacity=1.0),
    ]
    users = [
        make_user(
            workload=WorkloadProfile(
                cpu_demand=5.0,
                bandwidth_demand=5.0,
                state_size=20.0,
                latency_sensitivity=0.5,
            )
        )
    ]
    env = make_env(nodes, users, params)

    metrics = SimulationRunner(env, NearestPolicy()).step()

    assert metrics.failed_allocations == 1
    assert metrics.total_cost == params.allocation_failure_penalty


def test_cost_aware_uses_assignment_cost_for_node_selection() -> None:
    params = make_params(
        lambda_delay=0.2,
        lambda_migration=0.1,
        lambda_resource=0.5,
        lambda_balance=0.2,
    )
    nodes = [
        EdgeNode(
            node_id=0,
            x=0.0,
            y=0.0,
            cpu_capacity=10.0,
            bandwidth_capacity=10.0,
            used_cpu=6.0,
            used_bandwidth=6.0,
        ),
        EdgeNode(node_id=1, x=12.0, y=0.0, cpu_capacity=20.0, bandwidth_capacity=20.0),
    ]
    user = make_user(x=1.0, y=0.0)
    env = make_env(nodes, [user], params)

    expected_node_id = min(
        env.get_candidates(user),
        key=lambda node: env.assignment_cost(user, node, user.current_node_id),
    ).node_id

    actual_node_id = CostAwarePolicy(params).select_node(env, user)

    assert actual_node_id == expected_node_id


def test_cooldown_prevents_migration_even_if_better_node_exists() -> None:
    params = make_params(migrate_threshold=0.0, cooldown_steps=3)
    nodes = [
        EdgeNode(node_id=0, x=40.0, y=40.0, cpu_capacity=20.0, bandwidth_capacity=20.0),
        EdgeNode(node_id=1, x=11.0, y=10.0, cpu_capacity=20.0, bandwidth_capacity=20.0),
    ]
    user = make_user(x=10.0, y=10.0, current_node_id=0, cooldown_left=2)
    env = make_env(nodes, [user], params)

    target_node_id = CostAwarePolicy(params).select_node(env, user)

    assert target_node_id == 0


def test_forced_migration_when_current_node_cannot_allocate() -> None:
    params = make_params()
    nodes = [
        EdgeNode(node_id=0, x=0.0, y=0.0, cpu_capacity=2.0, bandwidth_capacity=2.0),
        EdgeNode(node_id=1, x=5.0, y=0.0, cpu_capacity=20.0, bandwidth_capacity=20.0),
    ]
    user = make_user(
        x=1.0,
        y=0.0,
        current_node_id=0,
        workload=WorkloadProfile(
            cpu_demand=4.0,
            bandwidth_demand=3.0,
            state_size=60.0,
            latency_sensitivity=1.0,
        ),
    )
    env = make_env(nodes, [user], params)

    target_node_id = CostAwarePolicy(params).select_node(env, user)

    assert target_node_id == 1


def test_runner_uses_same_assignment_cost_as_policy_evaluation() -> None:
    params = make_params()
    nodes = [
        EdgeNode(node_id=0, x=8.0, y=8.0, cpu_capacity=20.0, bandwidth_capacity=20.0),
    ]
    user = make_user(x=10.0, y=10.0)
    env = make_env(nodes, [user], params)

    expected_cost = env.assignment_cost(user, env.nodes[0], user.current_node_id)
    metrics = SimulationRunner(env, CostAwarePolicy(params)).step()

    assert metrics.failed_allocations == 0
    assert metrics.total_cost == expected_cost
