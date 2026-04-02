from __future__ import annotations

import statistics
from typing import List

from src.algorithms.base_policy import BasePolicy
from src.env.mec_env import MECEnvironment
from src.utils.metrics import SimulationMetrics, StepMetrics


class SimulationRunner:
    def __init__(self, env: MECEnvironment, policy: BasePolicy) -> None:
        self.env = env
        self.policy = policy

    def _resolve_policy_params(self):
        if hasattr(self.policy, "current_params"):
            return self.policy.current_params
        if hasattr(self.policy, "params"):
            return self.policy.params
        return self.env.params

    def step(self) -> StepMetrics:
        self.env.time_step += 1
        self.env.move_users()
        if hasattr(self.policy, "before_step"):
            self.policy.before_step(self.env)
        self.env.reset()

        metrics = StepMetrics()
        delay_list: List[float] = []

        metrics.user_count = len(self.env.users)

        for user in self.env.users.values():
            prev_node_id = user.current_node_id
            target_node_id = self.policy.select_node(self.env, user)
            policy_params = self._resolve_policy_params()

            if target_node_id is None:
                metrics.failed_allocations += 1
                metrics.total_cost += policy_params.allocation_failure_penalty
                continue

            target_node = self.env.nodes[target_node_id]
            if not self.env.can_allocate(user, target_node):
                metrics.failed_allocations += 1
                metrics.total_cost += policy_params.allocation_failure_penalty
                continue

            total_cost = self.env.assignment_cost(
                user,
                target_node,
                prev_node_id,
                params=policy_params,
            )

            migration_happened = int(
                prev_node_id is not None and prev_node_id != target_node_id
            )
            if migration_happened:
                user.migration_history.append(self.env.time_step)

            delay = self.env.projected_transmission_delay(user, target_node, policy_params)
            self.env.allocate(user, target_node_id)

            if migration_happened:
                metrics.migration_count += 1
                user.cooldown_left = policy_params.cooldown_steps

            delay_list.append(delay)
            metrics.total_cost += total_cost

        metrics.avg_delay = statistics.mean(delay_list) if delay_list else 0.0
        load_ratios = [node.load_ratio for node in self.env.nodes.values()]
        metrics.avg_load_ratio = statistics.mean(load_ratios) if load_ratios else 0.0
        metrics.load_std = statistics.pstdev(load_ratios) if len(load_ratios) > 1 else 0.0

        if hasattr(self.policy, "record_step_metrics"):
            self.policy.record_step_metrics(metrics)

        return metrics

    def run(self, steps: int) -> SimulationMetrics:
        sim_metrics = SimulationMetrics()
        for _ in range(steps):
            sim_metrics.step_metrics.append(self.step())
        return sim_metrics
