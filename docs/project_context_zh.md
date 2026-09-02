# DFB Project Context

状态：Active Reference

这份文档现在作为项目总索引与高层边界说明。  
各部分的长期上下文已拆分到独立文档中维护。

## 当前活动计划

- `docs/dfb_game_context_zh.md`
- `docs/part2_context_zh.md`
- `docs/part3_context_zh.md`
- `docs/part3_policy_contract_v1_zh.md`
- `docs/plans/part2_visual_recovery_plan_zh.md`
- `docs/plans/part2_temporal_training_plan_zh.md`
- `docs/plans/part2_audio_next_step_plan_zh.md`
- `docs/plans/part2_single_step_improvement_plan_zh.md`
- `docs/plans/part3_scene_and_opponent_pool_plan_zh.md`
- `docs/plans/part3_reward_safety_refinement_plan_zh.md`
- `docs/plans/part3_speed_control_refinement_plan_zh.md`
- `docs/plans/part3_episode_diagnosis_plan_zh.md`
- `docs/plans/part3_reward_performance_optimization_plan_zh.md`
- `docs/plans/part3_policy_contract_reset_plan_zh.md`
- `docs/plans/part3_training_failure_and_performance_recovery_plan_zh.md`
- `docs/plans/part3_teacher_distillation_plan_zh.md`
- `docs/plans/part3_dagger_plan_zh.md`
- `docs/plans/part3_dagger_to_ppo_plan_zh.md`
- `docs/plans/part3_shot_hit_curriculum_plan_zh.md`
- `docs/plans/part3_bc_dataset_reorganization_plan_zh.md`
- `docs/plans/part3_rule_ai_action_coverage_plan_zh.md`
- `docs/plans/part3_long_bc_retraining_plan_zh.md`
- `docs/plans/part3_teacher_v3_scaleup_plan_zh.md`
- `docs/plans/part3_teacher_v3_dagger_round1_plan_zh.md`
- `docs/plans/part3_ppo_design_regression_diagnosis_zh.md`
- `docs/plans/part3_advantage_retention_curriculum_plan_zh.md`
- `docs/plans/part3_general_scene_pool_v2_plan_zh.md`
- `docs/plans/part3_policy_architecture_scaling_migration_plan_zh.md`
- `docs/plans/part3_collision_and_mixed_opponent_training_plan_zh.md`
- `docs/thesis/part3_undergraduate_thesis_plan_zh.md`

近期已完成并归档：

- `docs/plans/archive/part2_visual_rework_plan_zh.md`
- `docs/plans/archive/part2_experiment_plan_zh.md`
- `docs/plans/archive/part2_audio_refactor_plan_zh.md`
- `docs/plans/archive/part2_training_eval_plan_zh.md`
- `docs/plans/archive/part3_policy_contract_reset_step0_baseline_zh.md` 至 `part3_policy_contract_reset_step7_cleanup_zh.md`
- `docs/plans/archive/part3_action_semantics_migration_plan_zh.md`
- `docs/plans/archive/part3_obs_v4_dual_state_plan_zh.md`
- `docs/plans/archive/part3_actor_critic_architecture_upgrade_plan_zh.md`

## 项目目标

`DFB` 所在的是一个更大的工作空间，而不是孤立的单项目。  
完整目标由三部分组成：

1. 狗斗环境
2. 基于图像/音频信息的时序状态观察预测模型
3. DRL 狗斗决策智能体

当前主阶段判断：

- `dfb_game` 已进入冻结基线
- Part 2 主架构与工程迁移已完成，当前进入真实数据实验阶段
- Part 3 teacher v3 已完成 500 个独立初始场景的无重复录制、独立 episode 数据划分与 120 epoch BC。保守的 DAgger round 1 已完成并通过自动闭环与 gameplay 人工验收。首轮 PPO 长训暴露出的实验谱系污染、裁剪动作概率不一致和奖励衰减失效已经修复；尾追专项策略已经形成，但固定种子能力门确认单场景优势保持训练会在 100 至 500 个 update 内造成灾难性遗忘，并将近距离对头收益导向双方同毁，当前进入保留既有能力的课程重构阶段

## 分部分上下文

### DFB Game: 狗斗环境

详细上下文统一参考：

- `docs/dfb_game_context_zh.md`

当前定位：

- 已冻结主契约
- 只做明确 bug 修复、低风险工具增强和文档维护
- 默认不再主动重写模拟、联机同步和录制主语义

### Part 2: 多模态时序状态估计

详细上下文统一参考：

- `docs/part2_context_zh.md`

当前定位：

- 主架构与工程主链已落地
- 当前重点转为：
  - 重建新的 binaural pack
  - 建立真实数据 baseline
  - 做实验对比与结果沉淀
- Part 2 应输出可供 Part 3 消费的 belief state，而不是只输出黑箱 embedding

### Part 3: DRL 狗斗智能体

详细上下文统一参考：

- `docs/part3_context_zh.md`

当前高层方向保持不变：

- 先成形
- 再优化

即：

- 先用少量专家经验做行为克隆
  - 规则控制器
  - 简单自动驾驶
  - 人类操作数据
- 再用 PPO 等算法做策略优化

训练阶段预期分成：

1. 飞行稳定与基本控制
2. 接近与跟踪
3. 攻防、自博弈与空战策略

## 当前阶段的总重点

当前已经不再是“继续收口 Part 1 主环境契约”的阶段。  
当前重点转为：

1. 保持 Part 1 稳定
2. 推进 Part 2 的完整设计与实现
3. 用 authoritative environment / recording / reconstruction / dataset 支撑 Part 2/3
4. 避免无必要破坏性修改

## 共享约束

以下约束当前仍然对整个项目成立：

- 默认不主动修改：
  - 飞行模拟主模型
  - server 权威 / client 预测同步主链
  - bridge 协议
  - recording schema
  - replay / modalities / datapacker 的录制消费语义
- 对外环境命令仍序列化为：
  - `throttle / pitch / roll / yaw ∈ [-1, 1]`
  - `brake / fire_gun / repair`
- 其中环境命令字段 `throttle` 在 Part 3 策略契约中明确映射为油门变化率命令 `throttle_delta`，不是绝对油门位置
- Part 3 唯一目标动作契约为 `dfb_part3_policy_action_v1`，其姿态语义为：
  - `+pitch`：绕飞机局部 `+X` 的右手系正旋转，表现为机头下沉
  - `+yaw`：绕飞机局部 `+Y` 的右手系正旋转，表现为机头向左
  - `+roll`：绕飞机局部 `+Z` 的右手系正旋转，表现为右翼下沉
- 策略动作第 0 维明确为油门变化率命令 `throttle_delta`，不是绝对油门位置
- Part 3 唯一观察契约保持 `dfb_part3_policy_observation_v1` 标识，当前固定为 69 维；双方修复与越界状态均采用“二值活动量 + 连续计时量”，不使用负数哨兵值
- Part 3 新训练默认采用两层共享 MLP 干路，以及 actor、critic 各两个基础残差块；容量扩展使用 checkpoint 中显式记录、恒等初始化的 shared/actor/critic 门控扩展块，旧 checkpoint 缺失扩展字段时统一解释为零；实际网络超参数必须随 checkpoint 保存并严格加载
- Part 3 行为克隆同时保留双方视角；fighter1 默认为权重 `2.0` 的人类主要示范，fighter2 默认为权重 `1.0` 的非人类辅助示范，来源和权重必须写入数据集元数据
- 活动链路不识别旧动作、观察、dataset 或 checkpoint 契约；历史实现仅存在于 Git 与归档文档中
- authoritative recording 仍是 expert data 的主来源
- `reward` 默认不由环境定义和计算，交给训练侧
- Part 2 正式实现前，必须先统一环境中的坐标语义，尤其是飞机对象空间中的前向与左右机翼标记约定
- 当前冻结的飞机对象空间语义采用右手系：
  - `forward = +Z`
  - `left = +X`
  - `up = +Y`
- 相机局部空间保留图形相机自身约定，不再要求与飞机机体系同轴同向

## 训练访问主线

当前第 2/3 部分可依赖的训练访问主线已收口完成，稳定存在四条路线：

- `Environment`
- `EpisodeRecording`
- `EpisodeReconstructor`
- 预构建 derived / dataset

这意味着：

- Part 2 的监督与样本生成应优先复用现有 authoritative 资产
- 不应再为训练访问额外引入破坏性环境改造

## 版本规范

- patch 修复：增加第三位
- 功能更新：增加第二位
- 变革级修改：增加第一位

## 文档规范

- `docs/project_context_zh.md`
  - 总索引与高层边界
- `docs/dfb_game_context_zh.md`
  - Part 1 长期上下文
- `docs/part2_context_zh.md`
  - Part 2 长期上下文
- `docs/part3_context_zh.md`
  - Part 3 长期上下文
- `docs/plans/`
  - 当前活动计划
- `docs/plans/archive/`
  - 已完成或废弃计划
- `docs/guides/`
  - 使用说明和工作流文档

## 实现偏好

- 优先统一语义，而不是补丁堆砌
- 新活动路径必须 fail closed，不提供运行时旧契约兼容
- Part 3 训练只使用已重建的原生新契约 recording 资产，不得在训练或推理时回退旧语义
- 能通过 server 权威单向决定的东西，不让 client 自行决定
- 能录语义，不录原始多模态波形

## 协作规范

- 对明显较大的修改：
  - 先写计划
  - 拆成有限、可逐步实现的步骤
  - 再落地
- 当需求不够清晰时：
  - 先追问关键约束
  - 不在目标不明确时直接做大改
- 当前协作默认偏好是：
  - 根源性方案优先
  - 明确记录方向和上下文
  - 让后续迭代可持续

## 当前开放问题

- Part 2 的完整输出状态与训练标签如何定义
- Part 2 与 Part 3 的接口如何冻结
- 是否需要在 expert dataset 之外额外保留更原始的人类输入层信息
  - 当前默认不做
  - 只有训练侧明确证明 `normalized command` 不足时再重新讨论
- recording 后续是否还需要继续推进更紧凑的非文本容器
  - 当前默认不做
  - 现有 chunked text bundle + lightweight index 足够作为冻结基线
