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

    def step(self) -> StepMetrics:
        self.env.time_step += 1
        self.env.move_users()
        self.env.reset()

        metrics = StepMetrics()
        delay_list: List[float] = []

        for user in self.env.users.values():
            prev_node_id = user.current_node_id
            target_node_id = self.policy.select_node(self.env, user)

            if target_node_id is None:
                metrics.failed_allocations += 1
                metrics.total_cost += self.env.params.allocation_failure_penalty
                continue

            target_node = self.env.nodes[target_node_id]

            if not self.env.can_allocate(user, target_node):
                metrics.failed_allocations += 1
                metrics.total_cost += self.env.params.allocation_failure_penalty
                continue

            total_cost = self.env.assignment_cost(user, target_node, prev_node_id)
            self.env.allocate(user, target_node_id)

            delay = self.env.transmission_delay(user, target_node)

            if prev_node_id is not None and prev_node_id != target_node_id:
                metrics.migration_count += 1
                user.cooldown_left = self.env.params.cooldown_steps

            delay_list.append(delay)
            metrics.total_cost += total_cost

        metrics.avg_delay = statistics.mean(delay_list) if delay_list else 0.0
        metrics.avg_load_ratio = statistics.mean(node.load_ratio for node in self.env.nodes.values())
        metrics.policy_debug = self.policy.debug_snapshot()
        return metrics

    def run(self, steps: int) -> SimulationMetrics:
        sim_metrics = SimulationMetrics()
        for _ in range(steps):
            sim_metrics.step_metrics.append(self.step())
        return sim_metrics
