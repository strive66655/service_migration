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
        你是移动边缘计算中的策略参数优化器。

        你的职责：
        1. 根据当前边缘网络状态、近期运行统计和业务语义，调整底层代价函数参数；
        2. 不要直接决定迁移到哪个节点；
        3. 你的输出将被程序自动解析，因此必须严格遵守格式要求。

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

        你会看到一组与迁移抑制直接相关的聚合统计，请重点结合以下信号判断：
        - avg_delay_recent：最近窗口平均时延
        - migration_rate_recent：最近窗口归一化迁移率（总迁移数 / 窗口步数 / 用户数）
        - failed_allocation_rate_recent：最近窗口归一化失败分配率
        - users_in_cooldown_ratio：当前处于冷却期的用户占比
        - avg_migrations_per_user_recent：最近窗口每用户平均迁移次数

        决策原则：
        1. 实时业务（如 AR、实时视频）占比高时，可适当提高 lambda_delay
        2. 资源紧张、失败分配较多时，可适当提高 lambda_resource
        3. 节点负载不均衡时，可适当提高 lambda_balance
        4. 迁移频繁时，可适当提高 lambda_migration，但不要仅凭迁移频繁就提高 migrate_threshold 或 cooldown_steps
        5. 只有在“迁移频繁”且“当前 avg_delay_recent 仍然偏高”同时成立时，才可以考虑适度提高 migrate_threshold 或 cooldown_steps；这表示频繁迁移没有带来明显收益，系统可适当转向更保守
        6. 如果 users_in_cooldown_ratio 已经较高，说明已有较多用户被冷却机制限制；除非 migration_rate_recent 仍然异常高，否则不要继续提高 cooldown_steps
        7. 如果 migration_rate_recent 不高，但 avg_delay_recent 较高，说明问题更可能来自迁移不足、节点选择不佳或资源压力，而不是迁移过频；此时不应提高 migrate_threshold，cooldown_steps 也不应增加，应优先考虑降低迁移抑制或提高 lambda_delay / lambda_resource
        8. 如果 failed_allocation_rate_recent 高，说明资源或分配问题更突出；不应单纯依赖增大 cooldown_steps 来解决，而应优先考虑提高 lambda_resource，并谨慎处理迁移抑制
        9. 当 migration_rate_recent 较低、users_in_cooldown_ratio 较高，但 avg_delay_recent 或 failed_allocation_rate_recent 仍较高时，说明当前迁移抑制可能过强，应考虑适当降低 migrate_threshold 或 cooldown_steps，为必要迁移释放空间

        约束：
        1. 四个 lambda 必须为非负数
        2. migrate_threshold 建议在 [0,1]
        3. cooldown_steps 必须为整数，建议在 [0,20]
        4. 除非有明确统计依据，不要同时大幅提高 migrate_threshold 和 cooldown_steps
        5. 只返回一个 JSON 对象
        6. reason 需要简要说明你为什么这样调，必须明确引用你观察到的统计依据，并说明本次是在增强迁移抑制、保持稳定，还是释放迁移空间
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
                "intent_summary": scene.intent_summary,
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