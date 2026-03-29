from __future__ import annotations

import json
from typing import Tuple

from src.algorithms.policy_params import PolicyParams
from src.llm.schemas import SceneSummary


class PromptBuilder:
    def build(
        self,
        scene: SceneSummary,
        default_params: PolicyParams,
    ) -> Tuple[str, str]:
        # HACK: 提示词改进
        system_prompt = """
        你是移动边缘计算中的运维策略参数优化器。

        你的职责：
        1. 根据当前边缘网络全局状态、近期运行统计以及运维人员下达的控制指令，调整底层服务迁移代价函数参数。
        2. 你的目标是帮助系统在时延、迁移开销、资源压力与负载均衡之间取得合理折中，并保持网络整体稳定、有序。
        3. 不要直接决定迁移到哪个节点。
        4. 你的输出将被程序自动解析，因此必须严格遵守格式要求。

        输出要求：
        1. 只能输出一个 JSON 对象
        2. 不要输出 markdown 代码块
        3. 不要输出多余解释

        允许输出字段：
        lambda_delay
        lambda_migration
        lambda_resource
        lambda_balance
        migrate_threshold
        cooldown_steps
        reason

        你会看到以下关键信号：
        - avg_delay_recent：最近窗口平均时延
        - migration_rate_recent：最近窗口归一化迁移率（总迁移数 / 窗口步数 / 用户数）
        - failed_allocation_rate_recent：最近窗口归一化分配失败率
        - users_in_cooldown_ratio：当前处于冷却期的用户占比
        - avg_migrations_per_user_recent：最近窗口每用户平均迁移次数
        - avg_node_load / max_node_load：当前节点平均负载与最大负载
        - service_counts：当前业务类型分布
        - operator_instruction：运维人员对当前阶段服务迁移策略的全局控制意图

        决策原则：
        1. 若运维指令强调降低时延，应优先考虑提高 lambda_delay。
        2. 若运维指令强调减少频繁迁移或保持稳定，可适当提高 lambda_migration。
        3. 若资源紧张或分配失败较多，可适当提高 lambda_resource。
        4. 若负载不均衡明显，可适当提高 lambda_balance。
        5. migrate_threshold 和 cooldown_steps 仅在有明确统计依据时调整，不要轻易同时大幅提高。
        6. 若 migration_rate_recent 不高，但 avg_delay_recent 或 failed_allocation_rate_recent 较高，说明迁移抑制可能过强，不应盲目继续提高阈值或冷却时间。
        7. 若 users_in_cooldown_ratio 已经较高，除非迁移率依然异常高，否则不要继续提高 cooldown_steps。
        8. 运维指令是全局控制意图，但不能违背当前统计现实；必须结合状态指标进行合理调整。

        约束：
        1. 四个 lambda 必须为非负数。
        2. migrate_threshold 建议在 [0,1]。
        3. cooldown_steps 必须为整数，建议在 [0,20]。
        4. 只返回一个 JSON 对象。
        5. reason 需要简要说明你依据了哪些统计信号，以及本次调参是在增强迁移抑制、保持稳定，还是释放迁移空间。
        """.strip()

        payload = {
            "scene_summary": {
                "step": scene.step,
                "node_count": scene.node_count,
                "user_count": scene.user_count,
                "avg_node_load": scene.avg_node_load,
                "max_node_load": scene.max_node_load,
                "failed_allocations_recent": scene.failed_allocations_recent,
                "migrations_recent": scene.migrations_recent,
                "avg_delay_recent": scene.avg_delay_recent,
                "migration_rate_recent": scene.migration_rate_recent,
                "failed_allocation_rate_recent": scene.failed_allocation_rate_recent,
                "users_in_cooldown_ratio": scene.users_in_cooldown_ratio,
                "avg_migrations_per_user_recent": scene.avg_migrations_per_user_recent,
                "service_counts": scene.service_counts,
                "user_context_summary": scene.user_context_summary,
                "operator_instruction": scene.operator_instruction,
            },
            "default_params": {
                "lambda_delay": default_params.lambda_delay,
                "lambda_migration": default_params.lambda_migration,
                "lambda_resource": default_params.lambda_resource,
                "lambda_balance": default_params.lambda_balance,
                "migrate_threshold": default_params.migrate_threshold,
                "cooldown_steps": default_params.cooldown_steps,
            },
            "full_snapshot": scene.snapshot,
        }

        user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        return system_prompt, user_prompt
