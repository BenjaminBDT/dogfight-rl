# Part 3 Context

状态：Active Reference

本文档用于单独管理第 3 部分“DRL 狗斗决策智能体”的长期上下文。  
它不取代 [project_context_zh.md](/run/media/ayano/SharedProjects/general_projects/dfb/docs/project_context_zh.md)，而是作为其 Part 3 细化文档存在。

## 1. Part 3 的任务定义

Part 3 的目标不是复现环境动力学，也不是替代 Part 2 的观测建模。  
Part 3 的职责是：

- 接收 Part 2 输出的状态估计结果
- 在不完全可观测的条件下做连续控制决策
- 最终形成可执行的空战策略

当前默认链路：

```text
simulation/environment
-> recording / reconstruction / dataset
-> Part 2 belief state / policy view
-> Part 3 policy
-> normalized action command
```

因此 Part 3 默认不是“直接吃权威真值状态”的方案，而是：

- 训练早期允许用更强监督或 teacher 信号加速收敛
- 正式主线应收口到使用 Part 2 输出

## 2. 当前总方向

当前主线已经确定为三阶段：

1. 行为克隆 warm start
2. PPO 类策略优化
3. 自博弈训练

简化表述为：

- `BC -> PPO -> self-play`

当前状态：teacher v3 已完成 500 个独立初始场景的 exhaustive 无重复录制。新数据集按独立 episode 随机划分为 400/50/50 局，并由 train split 重建 normalizer；fighter1 train、val、test 样本数分别为 683,539、81,599 和 85,631。120 epoch `2x512 + 2x2x512` BC 已完成，最佳 checkpoint 位于 epoch 107，验证与测试连续动作损失分别为 0.003999 和 0.002024。

teacher v3 固定短闭环未出现自毁或出界，平均绝对 roll 为 0.1194，优于 v2 长训模型的 0.1401，但仍不及 v2 旧最佳模型的 0.0979；20 秒末平均机距约为 403 m，表明追踪闭环能力仍需 gameplay 人工验收。当前不应只依据离线损失将该模型晋升到 PPO。

gameplay 初步验收认为 teacher v3 BC 行为尚可。保守 DAgger round 1 已使用 200 个全新独立初始场景完成纯学生执行、teacher 只读标注与 0.5 权重聚合，共追加 211,684 个 fighter1 student-state step；父 val/test 划分和 normalizer 统计保持不变。

DAgger round 1 候选从父 `best.pt` 以 `3e-5` 学习率微调 20 epoch，按父验证集连续动作损失选择 epoch 2。候选在 DAgger 状态上的连续动作平均绝对误差由 0.13015 降至 0.07326，二值动作失配率由 7.98% 降至 1.14%。固定 `open_ho` 三局短闭环中无自毁或越界，平均绝对 roll 由 0.1194 降至 0.0594，20 秒末平均机距由 403.0 m 降至 221.3 m，且未发生枪械过热。

该候选在父 test 上的连续动作损失由 0.00202 上升至 0.00308，说明单步专家分布拟合与 student-state 纠错之间存在预期权衡。自动闸门与 gameplay 人工验收均已通过，该模型继续作为 PPO 的正式初始化基线。

首轮 PPO 长训目录混入多次旧 checkpoint 分叉和中途奖励修改，不能作为单条实验轨迹解释；连续动作概率不一致和奖励衰减失效已经修复。优势保持课程的固定种子能力门进一步确认：不同初始距离下的训练日志不可直接横向比较；在同一尾追场景上，单独对头训练会在 100 至 500 个 update 内覆盖已有专项能力；当前公平近距离对头训练还主要将命中提升导向双方同毁。详细结果见 `docs/plans/part3_advantage_retention_curriculum_plan_zh.md`。

PPO 训练语义现已升级到 `dfb.part3.ppo-training-semantics.v2`。每次新训练都会把基础场景、场景池中的命名场景和生成场景物化到输出目录并记录 SHA-256；精确续训只复用已冻结场景，不重新读取可能已经变化的项目场景文件。旧训练语义 checkpoint 必须通过 `--init-checkpoint` 分叉，不能原地 `--resume`。

PPO 指标中的 `reward_diagnostics_by_outcome` 按最近 episode 窗口的终局原因拆分奖励。每个 component 同时提供平均每局累计值 `mean_episode_sum`、平均每局绝对累计值 `mean_episode_abs_sum` 和平均每步值 `mean_step`。worker 在进程内累计当前对局，只在终局时传回一次紧凑 payload，避免逐 step 传输奖励明细。

优势保持受控验证使用 `part3_advantage_retention_control_v1.json`，按 75% 公平对头和 25% fighter1 尾追回放采样。`dfb_reinforcement_learning.eval.advantage_retention_gate` 离线扫描每 100 update checkpoint，并在训练目录冻结场景上执行统一能力门；详细命令和选择约束见 `docs/plans/part3_advantage_retention_curriculum_plan_zh.md`。

2026-08-03 的首轮提高学习率实验确认 `shared=2e-6`、`actor=5e-6`、`critic=1e-4` 在当前批量设置下优化稳定。update 100 虽未在固定能力门中产生仅敌机损毁，但尾追命中数显著提高，gameplay 人工验收也确认 `v3/v4 best` 存在小幅动作改进。因此当前决策是从 `v4 best` 创建长期训练分支，只调整 rollout、GAE、batch 和评估频率等超参数，暂不修改课程与奖励；详细配置见 `docs/plans/part3_advantage_retention_curriculum_plan_zh.md`。

确定性规则 teacher 首版蒸馏已经完成：环境模式 `built_in_ai_teacher` 不使用随机 focus-fire、imperfection 或边界保持隐状态，`built_in_ai_imperfect` 仅作为交互对手。固定 `standard_open` arena 下已生成 85 局权威录制和 fighter1-only BC 数据；`2 x 1024` 共享干路、actor/critic 各两个 1024 残差块的 30-epoch BC 已完成，最佳验证 checkpoint 为运行目录中的 `checkpoints/best.pt`。

规则式 AI 当前已补充可观察状态驱动的开火热量控制、战术制动和保守修复决策。开火使用 `30°` 半锥角与 `0.8` 热量上限；制动与负油门变化率和 50 m 近距意图对齐；修复要求至少一个部件耐久不高于 `50%`、完整剩余修复窗口边界安全、低失速风险、预测最近机距安全且敌方威胁较低。一次修复周期恢复全部部件，总机体耐久不单独触发规则式 AI 修复；precise 与 imperfect 路径每帧整体生成动作，避免修复命令跨帧残留，因而部件恢复后能够回归战斗。

环境与 PPO 命令行同时支持 `built_in_ai_passive_bounce`：该被动对手常态保持零操纵、不开火、不修复，仅在边界预测触发时复用规则 AI 的边界恢复，恢复后继续直飞。它用于补充规则战术 AI 很少覆盖的非主动机动状态。

`part3_general_training_opponents_pool_v2.json` 固定采用 `0.75` imperfect 规则对手、`0.15` 当前策略 self-play、`0.10` 被动边界弹球对手。对手池中的 `self_play` 表示由当前训练策略同时控制另一架飞机，不引用或冻结历史 checkpoint。

PPO 在每次 episode reset 时分别采样 scene 与 opponent，并将该组合保持到本局终止或截断。只要 opponent pool 可能采到 `self_play`，rollout 就为每个物理环境预留 ego/opponent 两个策略槽：ego 始终有效，只有当前 self-play 局的 opponent 槽有效。双方各自构造 observation、reward、value、done 与截断 bootstrap，并共同进入 PPO batch；规则、被动与 checkpoint 对手的第二槽通过有效掩码从 GAE 后的优化、PopArt 与动作诊断中排除。日志 `[Pool]` 同时报告已完成局的 self-play episode 比例和当前 rollout 的 self-play 有效样本比例，因此可以直接观察“双边入 buffer”带来的实际权重膨胀。

在 `subproc + worker reward` 路径中，self-play 对手视角 observation 与第二侧 reward 均由对应环境 worker 构造。主进程每步只接收紧凑 observation/reward payload，不再为 self-play 传输完整权威状态，也不再按 self-play 环境数串行重算第二侧 reward；只有 checkpoint 对手仍因主进程推理需要显式请求完整状态。该执行位置调整不改变 observation、reward、GAE 或 PPO 更新语义。

基于 v2 teacher 场景池已重新采集 100 局权威录制并生成 `part3_policy_dataset_teacher_v2`。fighter1 train split 共 98,663 帧，brake、fire_gun、repair 正样本率分别为 6.21%、27.12% 和 5.47%，normalizer 已由新 train split 重建。新的 fighter1-only `2x512 + actor/critic 2x512 residual` BC 已完成 30 epoch，最佳 checkpoint 位于 `part3_bc_teacher_v2_f1_2x512_2x2x512_run1/checkpoints/best.pt`；闭环 smoke 可稳定运行，但俯仰输出偏大，当前下一步是 gameplay 人工验收，而不是直接进入 PPO。

- BC、PPO 和 self-play 的算法入口只接受新契约资产。
- 当前重新录制的策略数据遵循 fighter1 由人类操作、fighter2 由非人类控制器操作的来源约定。
- dataset、normalizer 和 checkpoint 必须由新契约链路重新生成，不迁移旧模型。
- 唯一契约参考 `docs/part3_policy_contract_v1_zh.md`。

## 3. 训练阶段划分

### 3.1 Phase A：行为克隆

目标：

- 学出基本稳定控制
- 学出“接近目标、保持可攻击态势”的初级策略

行为克隆同时保留双方视角，但不再把两类示范视为等质量来源：

- fighter1 默认为人类操作，是主要示范对象，默认样本权重为 `2.0`
- fighter2 默认为非人类控制器操作，用于补充基础飞行与对手状态分布，默认样本权重为 `1.0`
- 示范来源与权重必须写入 dataset chunk metadata；训练侧只读取数据资产中的显式值，不按角色隐式推断
- observation normalizer 仍按训练 split 的全部状态等权统计；示范权重只作用于 BC 和 BC distillation 的监督损失

训练输入默认应包含：

- Part 2 输出的 `policy_view`
- 必要的动作历史或短时上下文

训练目标固定为唯一 action schema：

- `throttle_delta`
- `pitch`
- `roll`
- `yaw`
- `brake`
- `fire_gun`
- `repair`

其中连续控制保持：

- `throttle_delta / pitch / roll / yaw ∈ [-1, 1]`

Part 3 唯一动作 ID 为 `dfb_part3_policy_action_v1`。`throttle_delta` 表示油门变化率命令；连续姿态控制按飞机局部坐标轴的右手系解释：

- `+pitch`：绕局部 `+X` 正旋转，机头下沉
- `+yaw`：绕局部 `+Y` 正旋转，机头向左
- `+roll`：绕局部 `+Z` 正旋转，右翼下沉

活动策略录制已显式声明唯一 policy contract 与 action schema。exporter 必须验证 recording schema 12 和完整契约 metadata，并按 `S_t -> A_t -> S_(t+1)` 对齐 BC 样本。旧 checkpoint 不再识别或迁移。若未来录制控制来源发生变化，必须在打包时显式覆盖对应的 `demonstration_source` 和 `sample_weight`，不得沿用错误默认值。

### 3.2 Phase B：PPO 策略优化

目标：

- 在已有可飞行、可接敌的基础上进一步优化策略
- 让策略真正对 reward 敏感，而不是只模仿专家

当前默认算法方向：

- PPO 类 on-policy 方法

选择原因：

- 连续动作控制成熟
- 工程实现路径清楚
- 对行为克隆 warm start 兼容性较好

这一阶段默认先不直接进入自博弈，而是：

- 先对固定 opponent 做策略优化
- 让策略把基础攻防闭环跑通
- 每次训练使用输出目录中的冻结场景快照；命名场景和场景池配置的源文件修改不得隐式改变已开始的训练

### 3.3 Phase C：自博弈

实现入口已完成双角色 rollout 状态修复；在 Step 9 完成长时间基线验证前，仅用于短时训练和自博弈冒烟测试。

当前自博弈设计采用“对称配对”模式：同一对局中，fighter1 和 fighter2 都由同一个 PPO 模型控制。每步将双方归一化观察组成同一批次完成 model forward，并将两个视角的 `(obs, action, reward)` 写入共享 buffer。

关键实现细节：

- buffer 维度翻倍：`effective_envs = num_envs * 2`，`buf_i = env_index * 2`（F1），`buf_i + 1`（F2）
- 双方观察在进入模型和 PPO buffer 前使用同一 normalizer
- rollout 结束时分别根据双方当前观察计算 bootstrap value，不复制 ego value
- previous action、reward history 和几何 cache 按 `(environment, role)` 隔离
- 截断（opening-shot-window + `--max-episode-seconds`）只追踪 ego-role（默认 fighter1），防止防守方过早触发
- done 信号同时作用于双方 buffer slot
- 对手方 reward 通过同一 `reward_composer` 以翻转视角计算

命令行用法：

```bash
--opponent-mode self_play --ego-role fighter1   # F1 为主追踪角色
--opponent-mode self_play --ego-role fighter2   # F2 为主追踪角色
```

后续方向：

- opponent pool / checkpoint league（冻结旧策略作为对手）
- 避免当前策略 vs 当前策略可能引入的非平稳奖励问题

## 4. 唯一观察 Schema

唯一目标 observation ID 为 `dfb_part3_policy_observation_v1`，固定 69 维：

- 敌机基础状态 31 维
- 敌机对偶攻防状态 3 维
- episode 时间 1 维
- 自机基础状态 31 维
- 自机对偶攻防状态 3 维

修复与越界状态均拆分为独立二值活动量和连续计时量。未激活时连续计时固定为 0，不使用负数哨兵值。

对偶攻防状态为双方各自的 `tracking_quality`、`tail_hold_score` 和 `shot_feasibility`。完整字段、坐标、公式、归一化和 fail-closed 规则统一参考 `docs/part3_policy_contract_v1_zh.md`。

Rust 与 Python 必须通过共享 fixture 的逐字段 parity test。任何一侧不得使用简化 shot 公式，也不得通过 observation 维度推断兼容性。

当前新训练默认使用两层共享 MLP 干路，并为 actor 与 critic 分别配置两个基础残差块。该默认值用于降低小规模高质量示范数据上的优化难度；checkpoint 仍必须记录实际宽度和塔深度，加载时不得用当前默认值覆盖其架构。容量扩展路径额外记录 `shared_extension_blocks`、`actor_extension_blocks` 与 `critic_extension_blocks`，使用零门控残差块保证参数迁移时初始函数不变；旧 checkpoint 缺少这些字段时按零处理。架构迁移 checkpoint 不继承 optimizer，必须通过 `--init-checkpoint` 启动新运行，不能在旧架构目录中直接 `--resume`。

## 5. Part 3 与 Part 2 的接口

当前默认接口方向：

- Part 2 应输出可供 Part 3 直接消费的 `belief_state`
- 同时保留更接近策略输入语义的 `policy_view`

当前 `policy_view` 至少应包含：

- `relative_position`
- `relative_orientation`
- `position_confidence`
- `orientation_confidence`
- `linear_velocity`
- `angular_velocity`
- `track_confidence`

Part 3 在正式主线中优先消费：

- `policy_view`

而不是直接依赖 Part 2 内部中间特征。

原因：

- `policy_view` 更贴近决策语义
- 更便于冻结 Part 2/Part 3 边界
- 更利于后续调试和替换

## 6. 专家数据主线

Part 3 的行为克隆默认应优先复用 authoritative 资产。  
也就是：

- authoritative environment
- authoritative recording
- reconstruction / dataset

当前不建议为 Part 3 单独引入绕开 recording 主链的新数据采集方案。

当前默认 expert dataset 语义应以：

- `observation / belief-like input`
- `normalized command action`
- episode metadata

为核心，而不是依赖临时 runtime hook。

## 7. Reward 原则

继续沿用项目总原则：

- reward 在训练侧定义
- 环境默认不内置 reward

Part 3 的 reward 设计应服务于分阶段训练，而不是一次性追求“终极 reward”。

建议分层：

1. 飞行稳定与姿态保持
2. 接近、跟踪、保持相对态势
3. 交战收益
   - 命中
   - 生存
   - 能量态势
   - 攻防结果

当前不建议把所有 reward 在第一天一次性耦合到一起。

当前冻结的时间与修复语义为：

- 时间压力从 `0.5/s` 的初始生存奖励平滑过渡到 `-1.0/s`，参考时长为 600 秒，约在 232 秒转为负值。
- 修复命令承担 `3.0/s` 的固定机会成本；威胁、边界、失速和完成安全门继续决定修复收益与额外风险惩罚。
- 不保留未实际参与 reward 计算的修复动作幅度门槛。

当前已完成射击命中专项课程的工程准备：

- 使用 `part3_aim_fire_pool_v1.json` 提高真实命中奖励密度。
- 使用射击窗口丢失与短时上限截断无关机动过程。
- PPO 对 `truncated` 的最终状态执行 value bootstrap，但不让 GAE 跨 reset 传播。
- 命中日志与 reward 统一只统计权威 `Hit` 事件。
- `fire_command_bonus_weight` 提供默认关闭的无条件开火奖励率；启用后只依赖 `fire_gun` 命令，不受射击窗口、命中预测、边界门控、枪热或目标几何影响，但仍遵循持续正奖励的 `dt` 缩放和全局时间衰减。
- 首轮只验证“利用已有窗口”，有效后再用近距对头场景学习主动形成窗口。

防对撞课程新增 `collision_course` 场景模板：在场地内采样公共碰撞点、两机速度方向与速度，并按合并碰撞球半径逆推出生位置，使球体首次接触时间落在 `2-5s`。该模板除出生与直线路径不得接触场地边界外不附加战术约束，也不人为生成近失碰撞。`part3_train_scene_pool_v3.json` 保持 `50% tactical / 50% recovery`，并在 tactical 内为该模板保留全局 `10%` 权重。

## 8. 训练入口与终端日志

### 8.1 主要 CLI 入口

| 入口 | 命令 |
|------|------|
| BC 训练 | `python -m dfb_reinforcement_learning.train.train_bc` |
| PPO 训练 | `python -m dfb_reinforcement_learning.train.train_ppo` |
| PPO 自博弈 | 加 `--opponent-mode self_play` |
| BC checkpoint 初始化 PPO | `--init-checkpoint <bc_dir>/checkpoints/best.pt` |
| 恢复训练 | `--resume <dir>/checkpoints/latest.pt` |
| 闭环能力闸门 | `python -m dfb_reinforcement_learning.eval.policy_capability_gate` |

关键 PPO 参数：

```bash
--num-envs 32 --rollout-steps 2048 --ppo-epochs 4 \
--clip-eps 0.2 --binary-entropy-coef 0.0 --continuous-action-std 0.2 \
--opponent-mode self_play --ego-role fighter1 \
--max-episode-seconds 5.0 --eval-interval 5 \
--eval-max-seconds 120 --target-kl 0.03
```

### 8.2 终端日志

每 update 输出分行显示，分类如下：

```
[Update] #0001  step=0002048  reward_mean=X.XXXX
[Rollout] return=X.XXXX  eps=N  len=X.X  dur=X.XXs  self_destroy=X.XXX  enemy_destroy=X.XXX  trunc=X.XXX  hit_self=X.XX  hit_enemy=X.XX
[Loss] policy=X.XXXX  value=X.XXXX  cont_entropy=X.XXXX  bin_entropy=X.XXXX  approx_kl=X.XXXX  clip_frac=X.XXXX
[PPO] max_kl=X.XXXX  kl_stop=N  epochs=N/M  minibatches=N
[LR] shared=X.XXe-XX  actor=X.XXe-XX  critic=X.XXe-XX
[Time] rollout=X.XXs  policy=X.XXs  opp=X.XXs  t_env=X.XXs  reward=X.XXs  reset=X.XXs  ppo=X.XXs  eval=X.XXs  checkpoint=X.XXs  total=X.XXs  sps=X.X
[Eval] return=X.XXXX  duration=X.XXs  self_destroy=X.XXXX  enemy_destroy=X.XXXX  timeout=X.XXXX  truncated=X.XXXX
```

命中统计（`hit_self` / `hit_enemy`）从每步 `events_since_last_step` 中的 `Hit` / `Damage` / `SubsystemHit` 等事件统计。

完整诊断还写入 `metrics.jsonl`，包括 100-episode 滚动窗口、详细终止原因、连续/二值动作统计、按终局拆分的 reward 子项贡献、value/return 误差、PopArt 状态和相对初始化策略的 action drift。终端只保留适合实时观察的摘要。连续动作探索仅由 `continuous_action_std` 控制；`binary_entropy_coef` 只正则化制动、开火和修复三个 Bernoulli 动作。

### 8.3 metrics.jsonl

每个 update 的完整指标以 JSONL 格式写入 `output_dir/metrics.jsonl`，同时保留逐 update 快照到 `output_dir/evals/update_NNNN.json`。

## 9. 当前工程共识

当前 Part 3 的工程原则已经明确：

1. 先成形，再优化  
2. 先专家模仿，再强化学习  
3. 先固定对手，再自博弈  
4. 优先消费 Part 2 的显式状态输出，而不是黑箱 latent  
5. 数据和 reward 都尽量保持可解释、可回滚

## 10. 当前未决问题

以下问题当前仍未冻结：

1. 行为克隆阶段使用单步 policy 还是带短时记忆的 policy
2. PPO 主干是否直接引入 recurrent / transformer memory
3. 自博弈 opponent pool 的维护策略
4. Part 3 训练时是否需要联合使用 Part 2 confidence / validity 作为决策 mask
5. expert dataset 是否需要额外保留更细粒度的高层意图标签

这些问题后续应在独立的 Part 3 训练计划文档中继续细化。  
当前这份 context 只固定方向与边界，不提前锁死具体实现。
