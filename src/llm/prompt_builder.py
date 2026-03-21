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

        system_prompt = """
        你是移动边缘计算中的策略参数优化器。

        你的职责：
        1. 根据当前边缘网络状态和业务语义，调整底层代价函数参数；
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

        决策原则：
        1. 实时业务(如 AR、实时视频)占比高时，可适当提高 lambda_delay
        2. 资源紧张或失败分配较多时，可适当提高 lambda_resource
        3. 迁移过于频繁时，可适当提高 lambda_migration，并增大 cooldown_steps
        4. 节点负载不均衡时，可适当提高 lambda_balance

        约束：
        1. 四个 lambda 必须为非负数
        2. migrate_threshold 建议在 [0,1]
        3. cooldown_steps 必须为整数，建议在 [0,20]
        4. 只返回一个 JSON 对象
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