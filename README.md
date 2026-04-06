# Service Migration

一个面向边缘计算场景的服务迁移仿真项目，用于比较多种迁移策略，并支持使用 LLM 动态调整成本感知策略参数。

当前仓库重点覆盖：

- MEC/边缘节点与移动用户的仿真环境
- 基线迁移策略对比
- 成本感知策略参数搜索
- 基于 LLM 的在线参数调节
- 实验结果导出与可视化

## 项目结构

```text
service_migration/
├─ config/                     # 环境、策略、实验配置
├─ data/                       # 预留数据目录
├─ experiments/                # 实验脚本、调参脚本、结果与图表
├─ src/
│  ├─ algorithms/             # 迁移策略实现
│  ├─ env/                    # MEC 环境、实体、移动模型
│  ├─ llm/                    # Prompt、解析器、Provider、控制器
│  ├─ runners/                # 仿真执行器
│  └─ utils/                  # 配置加载、指标统计、日志等
├─ README.md
└─ requirements.txt
```

## 策略说明

当前代码包含以下策略：

- `never_migrate`: 用户首次分配后尽量不迁移
- `nearest`: 优先选择最近的候选节点
- `cost_aware`: 基于综合成本函数进行迁移决策
- `llm_cost_aware_*`: 在 `cost_aware` 基础上，由 LLM 周期性调整策略参数

其中 `cost_aware` / `llm_cost_aware_*` 关注的核心维度包括：

- 时延
- 迁移代价
- 资源紧张度
- 负载均衡

这些权重与阈值由 [`config/policy.yaml`](/f:/service_migration/config/policy.yaml) 控制。

## 仿真环境

默认场景定义在 [`config/env.yaml`](/f:/service_migration/config/env.yaml)：

- 100 x 100 的二维空间
- 5 个边缘节点
- 15 个移动用户
- 4 类业务类型：`ar`、`video`、`compute`、`background`

每类业务会绑定不同的工作负载与 QoS 画像，包括：

- 时延阈值
- 资源压力阈值
- 迁移容忍度
- 优先级

若用户配置了 `intent_text`，系统会进一步修正 QoS 偏好。

## 指标输出

实验会统计并导出以下指标：

- 平均时延 `avg_delay`
- 平均总成本 `avg_total_cost`
- 平均迁移次数 `avg_migrations`
- 平均分配失败数 `avg_failed_allocations`
- 平均节点负载率 `avg_load_ratio`
- QoS 分数、意图满足率、SLA 违约率
- 分业务类型的满意率、SLA 违约率与 P95 时延

输出文件默认位于 `experiments/results/`，包括：

- `baseline_results.csv`
- `baseline_step_results.csv`
- `qos_results.csv`
- `qos_step_results.csv`
- `llm_decisions.csv`
- `figures/*.png`

## 安装

当前 [`requirements.txt`](/f:/service_migration/requirements.txt) 还是空文件，仓库里实际使用到的核心依赖至少包括：

- `pandas`
- `matplotlib`
- `PyYAML`
- `openai`

建议先创建虚拟环境，再手动安装：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install pandas matplotlib PyYAML openai
```

如果后续补全 `requirements.txt`，则可改为：

```bash
pip install -r requirements.txt
```

## 配置

### 1. 环境配置

[`config/env.yaml`](/f:/service_migration/config/env.yaml) 用于定义：

- 地图尺寸
- 节点位置与容量
- 用户初始位置、速度与业务类型
- 可选扰动参数，如 `user_position_jitter`、`user_velocity_jitter`

### 2. 策略配置

[`config/policy.yaml`](/f:/service_migration/config/policy.yaml) 用于定义：

- 候选范围与迁移阈值
- 冷却时间
- 多目标权重
- 时延、迁移、资源相关成本系数
- LLM provider / model / 刷新周期

### 3. 实验配置

[`config/experiment.yaml`](/f:/service_migration/config/experiment.yaml) 用于定义：

- 随机种子 `seed`
- 仿真步数 `steps`
- 预留策略名字段 `policy_name`

## 运行方式

### 运行完整实验

```bash
python experiments/run_experiment.py
```

该脚本会：

- 运行基线策略对比
- 可选运行 LLM 策略
- 导出 CSV 结果
- 生成图表

### 调优成本感知策略参数

```bash
python experiments/tune_cost_aware.py
```

该脚本会搜索一组权重和阈值组合，并将结果保存到：

```text
experiments/results/cost_aware_tuning.csv
```

### 仅重新生成图表

```bash
python experiments/visualize_baseline.py
```

## LLM 接入

项目支持以下 Provider：

- `mock`
- `qwen`
- `openrouter`

配置位置在 [`config/policy.yaml`](/f:/service_migration/config/policy.yaml) 的 `llm` 段。

### OpenRouter

当 `provider: openrouter` 时，需要配置环境变量：

```bash
set OPENROUTER_API_KEY=your_key
```

可选：

```bash
set OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
set OPENROUTER_SITE_URL=https://your-site.example
set OPENROUTER_APP_NAME=service-migration
```

### Qwen / DashScope

当 `provider: qwen` 时，需要配置：

```bash
set DASHSCOPE_API_KEY=your_key
```

或：

```bash
set QWEN_API_KEY=your_key
```

可选：

```bash
set QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

如果未配置可用密钥，代码会回退到 `mock` provider 或在 provider 初始化阶段报出明确错误。

## 结果说明

执行 [`experiments/run_experiment.py`](/f:/service_migration/experiments/run_experiment.py) 后，通常会看到两类结果：

- 汇总结果：用于横向比较不同策略的平均表现
- Step 级结果：用于观察随时间演化的趋势

图表默认输出为：

- `baseline_overview.png`
- `cost_trend.png`
- `qos_overview.png`
- `qos_trend.png`

## 适用场景

这个项目适合用于：

- 服务迁移/边缘调度策略课程或论文原型
- 对比基线策略与 LLM 调参策略
- 验证 QoS 约束下的迁移成本权衡
- 快速搭建可视化实验流水线

## 当前状态

仓库目前可以作为实验原型使用，但有几个现实约束需要注意：

- `requirements.txt` 尚未补全
- 部分源码注释存在编码问题，不影响主流程理解与运行
- 默认配置更偏向单机场景复现实验，不是生产级调度系统

## License

## Baseline Policy Notes

- `myopic`: picks the feasible node with the minimum current `assignment_cost`, with no cooldown, threshold, or stay bias.
- `cost_aware`: the main cost-aware baseline in this repo, using cooldown, stay bias, relative-gain migration gating, and lightweight business awareness based on `delay_budget` and `priority`.
- `llm_cost_aware_*`: continues to use `cost_aware` as its inner policy.

许可证见 [`LICENSE`](/f:/service_migration/LICENSE)。
