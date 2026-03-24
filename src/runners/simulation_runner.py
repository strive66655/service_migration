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

            # 确保成本统计使用这次决策对应的实际参数
            policy_params = getattr(self.policy, "current_params", self.env.params)

            if target_node_id is None:
                metrics.failed_allocations += 1
                metrics.total_cost += policy_params.allocation_failure_penalty
                metrics.qos_records.append(
                    {
                        "service_type": user.service_type.value,
                        "priority_level": user.qos_profile.priority_level if user.qos_profile else 1,
                        "delay": None,
                        "allocation_failed": 1,
                        "migration_happened": 0,
                        "qos_score": 0.0,
                        "qos_satisfied": 0,
                        "sla_violated": 1,
                    }
                )
                continue

            target_node = self.env.nodes[target_node_id]

            if not self.env.can_allocate(user, target_node):
                metrics.failed_allocations += 1
                metrics.total_cost += policy_params.allocation_failure_penalty
                metrics.qos_records.append(
                    {
                        "service_type": user.service_type.value,
                        "priority_level": user.qos_profile.priority_level if user.qos_profile else 1,
                        "delay": None,
                        "allocation_failed": 1,
                        "migration_happened": 0,
                        "qos_score": 0.0,
                        "qos_satisfied": 0,
                        "sla_violated": 1,
                    }
                )
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
            qos_score = self.env.compute_qos_score(user, target_node, self.env.time_step)
            qos_satisfied = int(
                self.env.check_qos_satisfaction(user, target_node, self.env.time_step)
            )
            sla_violated = int(
                self.env.check_sla_violation(
                    user,
                    target_node,
                    self.env.time_step,
                    allocation_failed=False
                )
            )

            self.env.allocate(user, target_node_id)

            if migration_happened:
                metrics.migration_count += 1
                user.cooldown_left = policy_params.cooldown_steps

            delay_list.append(delay)
            metrics.total_cost += total_cost

            metrics.qos_records.append(
                {
                    "service_type": user.service_type.value,
                    "priority_level": user.qos_profile.priority_level if user.qos_profile else 1,
                    "delay": delay,
                    "allocation_failed": 0,
                    "migration_happened": migration_happened,
                    "qos_score": qos_score,
                    "qos_satisfied": qos_satisfied,
                    "sla_violated": sla_violated,
                }
            )

        metrics.avg_delay = statistics.mean(delay_list) if delay_list else 0.0
        metrics.avg_load_ratio = statistics.mean(
            node.load_ratio for node in self.env.nodes.values()
        )

        if hasattr(self.policy, "record_step_metrics"):
            self.policy.record_step_metrics(metrics)

        return metrics

    def run(self, steps: int) -> SimulationMetrics:
        sim_metrics = SimulationMetrics()
        for _ in range(steps):
            sim_metrics.step_metrics.append(self.step())
        return sim_metrics
