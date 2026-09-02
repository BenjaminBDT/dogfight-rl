# Part 3 Policy Contract v1

状态：Frozen Specification

本文档定义 Part 3 唯一活动策略契约。实现、数据集、normalizer、checkpoint、训练、评估和 live inference 都必须服从本文档；旧 observation/action 约定不得作为当前实现依据。

## 1. 契约身份

唯一活动 ID：

```text
policy_contract_id       = dfb_part3_policy_contract_v1
observation_schema_id    = dfb_part3_policy_observation_v1
action_schema_id         = dfb_part3_policy_action_v1
normalizer_schema_id     = dfb_part3_policy_normalizer_v1
dataset_schema_id        = dfb_part3_policy_dataset_v1
checkpoint_schema_id     = dfb_part3_policy_checkpoint_v1
model_family_id          = dfb_part3_stateless_hybrid_actor_critic_v1
```

活动代码不得再提供 `current`、`legacy`、`v2`、`v3`、`v4` 等策略契约别名。未来若需要破坏性修改，必须新建 policy contract，不得原地改变上述 ID 的语义。

## 2. 权威输入与时序

### 2.1 权威来源

观察只能由 server 权威状态构建。client 预测状态、渲染 Transform 和非权威 UI 状态不得进入观察。

每个策略决策遵循：

```text
authoritative state S_t
-> observation O_t
-> policy action A_t
-> environment transition
-> authoritative state S_(t+1)
```

在线 PPO、评估和 live inference 必须使用相同的 `S_t -> O_t` 构建器。

### 2.2 录制到 BC 的对齐

当前权威录制中的 `RecordedStep` 在 simulation step 结束后保存：

- 本 step 使用的 command
- command 执行后的权威 state

因此 BC 样本不得直接使用同一 `RecordedStep` 中的 state 和 command。正确对齐为：

```text
sample 0:
  observation = initial_snapshot.state
  action      = recorded_step[0].command
  next_state  = recorded_step[0].state

sample i, i > 0:
  observation = recorded_step[i - 1].state
  action      = recorded_step[i].command
  next_state  = recorded_step[i].state
```

`fighter1` 和 `fighter2` 分别从同一 transition 构建各自视角。两个角色必须进入同一个 dataset split，避免同一 episode 跨 train/val/test 泄漏。

## 3. 坐标与旋转约定

### 3.1 世界与飞机机体系

游戏和策略统一使用右手系。飞机局部坐标定义：

- `+X`：左翼方向
- `+Y`：机背上方
- `+Z`：机头前向

四元数固定为：

```text
[x, y, z, w]
```

四元数在构建旋转矩阵前必须归一化。零长度、非有限或缺失四元数是数据错误，必须拒绝，不得回退为单位矩阵。

定义：

```text
R_self  : self body -> world
R_enemy : enemy body -> world
```

### 3.2 6D rotation

6D rotation 固定取旋转矩阵前两列并按列展开：

```text
rotation_6d(R) = [
  R[0,0], R[1,0], R[2,0],
  R[0,1], R[1,1], R[2,1],
]
```

不得使用 NumPy 对 `R[:, :2]` 的默认行优先 flatten。

敌机相对姿态为：

```text
R_enemy_in_self = transpose(R_self) * R_enemy
```

## 4. Observation Schema

### 4.1 总体布局

原始策略观察固定为 `float32[69]`。这里的“原始”是指已经完成物理尺度缩放、但尚未执行 dataset mean/std 标准化的向量。

```text
enemy base state        31
enemy tactical state     3
episode time             1
self base state         31
self tactical state      3
total                   69
```

### 4.2 逐字段布局

| Offset | 字段 | Dim | 定义与缩放 |
|---:|---|---:|---|
| 0 | `enemy_relative_position_body` | 3 | `transpose(R_self) * (p_enemy - p_self) / 3000` |
| 3 | `enemy_relative_orientation_6d` | 6 | `rotation_6d(transpose(R_self) * R_enemy)` |
| 9 | `enemy_linear_velocity_body` | 3 | `transpose(R_enemy) * v_enemy_world / 120` |
| 12 | `enemy_angular_velocity_body` | 3 | enemy body angular velocity，`deg/s -> rad/s -> /pi` |
| 15 | `enemy_health_state_norm` | 6 | `[total, LeftWing, RightWing, PitchTail, YawTail, Engine]` 耐久比例；total 上限固定为 100 |
| 21 | `enemy_throttle_norm` | 1 | 实际油门状态，clip 到 `[0,1]` |
| 22 | `enemy_brake_active` | 1 | 实际制动状态，`0/1` |
| 23 | `enemy_stall_factor` | 1 | clip 到 `[0,1]` |
| 24 | `enemy_gun_overheated` | 1 | `0/1` |
| 25 | `enemy_gun_heat_norm` | 1 | `gun_heat / 1.0`，clip 到 `[0,4]` |
| 26 | `enemy_fire_gun_active` | 1 | 权威实际开火状态，`0/1` |
| 27 | `enemy_repair_active` | 1 | 权威修复状态，`0/1` |
| 28 | `enemy_repair_seconds_norm` | 1 | `repair_elapsed / 10`，clip 到 `[0,4]`；未修复时为 `0` |
| 29 | `enemy_out_of_bounds_active` | 1 | 权威累计越界时间大于 0，或水平位置达到 arena radius 时为 `1` |
| 30 | `enemy_out_of_bounds_seconds_norm` | 1 | `oob_elapsed / 20`，clip 到 `[0,4]`；未越界时为 `0` |
| 31 | `enemy_tracking_quality` | 1 | 第 5 节公式，`[0,1]` |
| 32 | `enemy_tail_hold_score` | 1 | 第 5 节公式，`[0,1]` |
| 33 | `enemy_shot_feasibility` | 1 | 第 5 节公式，`[0,1]` |
| 34 | `episode_time_norm` | 1 | `clip(episode_elapsed_seconds / 180, 0, 4)` |
| 35 | `self_position_world_norm` | 3 | `p_self_world / 3000` |
| 38 | `self_orientation_world_6d` | 6 | `rotation_6d(R_self)` |
| 44 | `self_throttle_norm` | 1 | 实际油门状态，clip 到 `[0,1]` |
| 45 | `self_brake_active` | 1 | 实际制动状态，`0/1` |
| 46 | `self_stall_factor` | 1 | clip 到 `[0,1]` |
| 47 | `self_linear_velocity_body` | 3 | `transpose(R_self) * v_self_world / 120` |
| 50 | `self_angular_velocity_body` | 3 | self body angular velocity，`deg/s -> rad/s -> /pi` |
| 53 | `self_health_state_norm` | 6 | `[total, LeftWing, RightWing, PitchTail, YawTail, Engine]` 耐久比例；total 上限固定为 100 |
| 59 | `self_gun_overheated` | 1 | `0/1` |
| 60 | `self_gun_heat_norm` | 1 | `gun_heat / 1.0`，clip 到 `[0,4]` |
| 61 | `self_fire_gun_active` | 1 | 权威实际开火状态，`0/1` |
| 62 | `self_repair_active` | 1 | 权威修复状态，`0/1` |
| 63 | `self_repair_seconds_norm` | 1 | `repair_elapsed / 10`，clip 到 `[0,4]`；未修复时为 `0` |
| 64 | `self_out_of_bounds_active` | 1 | 权威累计越界时间大于 0，或水平位置达到 arena radius 时为 `1` |
| 65 | `self_out_of_bounds_seconds_norm` | 1 | `oob_elapsed / 20`，clip 到 `[0,4]`；未越界时为 `0` |
| 66 | `self_tracking_quality` | 1 | 第 5 节公式，`[0,1]` |
| 67 | `self_tail_hold_score` | 1 | 第 5 节公式，`[0,1]` |
| 68 | `self_shot_feasibility` | 1 | 第 5 节公式，`[0,1]` |

总耐久上限由模拟契约固定为 100；固定子系统使用录制中的各自最大耐久。健康状态缺少任一固定子系统、子系统最大耐久非正、角色重复或缺失、关键状态缺失时必须报错，不得填充默认值。

修复与越界分别使用独立的二值活动状态和连续计时。连续计时未激活时固定为 `0`，不得再使用 `-1` 哨兵值。`out_of_bounds_active` 的活动状态定义为：权威累计越界时间大于 0，或飞机水平位置半径已经达到 arena radius，因此进入边界的首帧会输出 `active=1, seconds_norm=0`。

### 4.3 时间语义

`episode_elapsed_seconds` 必须相对本 episode 初始状态计算：

```text
episode_elapsed_seconds = max(state.sim_time_seconds - episode_start_sim_time_seconds, 0)
```

不得直接使用跨 episode 累计的进程运行时间。录制数据使用 initial snapshot 的 `sim_time_seconds` 作为起点；在线环境在 reset 时保存起点。

### 4.4 二值字段

二值观察索引固定为：

```text
[22, 24, 26, 27, 29, 45, 59, 61, 62, 64]
```

分别对应双方的 brake、gun_overheated、fire_gun_active、repair_active 和 out_of_bounds_active。

## 5. 对偶攻防状态

### 5.1 公共量

```text
relative_position = p_enemy - p_self
distance          = norm(relative_position)
los               = relative_position / max(distance, 1e-6)
self_forward      = R_self  * [0,0,1]
enemy_forward     = R_enemy * [0,0,1]
heading_cos       = clip(dot(self_forward, enemy_forward), -1, 1)
```

双方位置重合或 `distance <= 1e-6` 视为无效状态，观察构建失败，不使用任意 LOS 回退。

### 5.2 Tracking quality

```text
self_tracking_raw  = clip(dot(self_forward,  los), -1, 1)
enemy_tracking_raw = clip(dot(enemy_forward, -los), -1, 1)

tracking_quality(raw) = clip(0.5 * (raw + 1), 0, 1)
```

不得用 `-self_tracking_raw` 代替 `enemy_tracking_raw`。

### 5.3 Tail hold score

```text
self_tail_cos  = clip(dot(enemy_forward,  los), -1, 1)
enemy_tail_cos = clip(dot(self_forward,  -los), -1, 1)

tail_exposure(tail_cos) = clip(0.5 * (tail_cos + 1), 0, 1)
heading_alignment       = clip(0.5 * (heading_cos + 1), 0, 1)

self_tail_hold_score  = tail_exposure(self_tail_cos)  * heading_alignment
enemy_tail_hold_score = tail_exposure(enemy_tail_cos) * heading_alignment
```

### 5.4 Shot feasibility

shot feasibility 使用固定、角色无关的精确函数。攻击方和防守方交换后调用同一函数。

常量：

```text
projectile_speed                  = 1200 m/s
projectile_max_range              = 1400 m
muzzle_forward_offset             = 11.4 m
attack_tau_reference              = 0.75 s
fire_alignment_threshold_cos      = 0.25
projectile_aircraft_hit_radius     = 0.8 m
projectile_subsystem_hit_radius    = 0.4 m
shot_outer_radius                 = 2.4 m
shot_core_radius                  = 0.9 m
shot_outer_weight                 = 0.2
shot_core_weight                  = 0.8
```

弹道：

```text
muzzle_position = attacker_position + attacker_forward * 11.4
bullet_velocity = attacker_velocity_world + attacker_forward * 1200
tau_max         = 1400 / norm(bullet_velocity)
```

枪线门控：

```text
aim_cos = clip(dot(attacker_forward, normalize(defender_position - attacker_position)), -1, 1)

if aim_cos <= 0.25:
    fire_alignment = 0
else:
    alpha = (aim_cos - 0.25) / (1 - 0.25)
    fire_alignment = alpha^2
```

对每个未被摧毁的 collision OBB，使用防守方相对弹丸速度计算有限射程内最近接时间：

```text
relative_position = box_center - muzzle_position
relative_velocity = defender_velocity_world - bullet_velocity

if dot(relative_velocity, relative_velocity) <= 1e-6:
    tau_box = 0
else:
    tau_box = clip(-dot(relative_position, relative_velocity)
                   / dot(relative_velocity, relative_velocity),
                   0,
                   tau_max)
```

通过 OBB 在最近接方向上的 support radius 得到非负 clearance。每个 box 的分数只施加一次时间门控：

```text
tau_gate = exp(-tau_box / 0.75)

outer_box_score = tau_gate
                * exp(-(aircraft_clearance / max(2.4, 0.8))^2)

core_box_score  = tau_gate
                * exp(-(subsystem_clearance / max(0.9, 0.4))^2)
```

`outer_score` 取全部活动 collision boxes 最大值；`core_score` 只在 LeftWing、RightWing、PitchTail、YawTail、Engine 中取最大值。已经摧毁的子系统对应 box 不参与。

最终定义：

```text
shot_feasibility = clip(
    fire_alignment * (0.2 * outer_score + 0.8 * core_score),
    0,
    1,
)
```

明确禁止：

- Rust 使用无碰撞盒简化公式。
- observation 与 reward 各自实现不同 shot 公式。
- 在 box score 已包含 `tau_gate` 后再次乘全局 `tau_gate`。
- 仅按距离或枪线夹角伪造 shot feasibility。

碰撞盒尺寸以 simulation 的权威 collision geometry 为唯一来源。Step 2/3 实现时应抽取共享定义或建立强制 parity test，不能在文档、Rust 和 Python 各维护一套不可校验常量。

## 6. Action Schema

### 6.1 布局

动作固定为：

```text
continuous float32[4]:
  [throttle_delta, pitch, roll, yaw]

binary float32/bool[3]:
  [brake, fire_gun, repair]
```

所有连续值在进入环境前 clip 到 `[-1,1]`。二值策略输出为 Bernoulli logits；传给环境时必须是明确的 bool。

### 6.2 Throttle

`throttle_delta` 是油门变化率命令，不是绝对油门位置：

- `+1`：以最大允许速率增加油门。
- `-1`：以最大允许速率减小油门。
- `0`：保持当前油门。

观察中的 `self_throttle_norm` 和 `enemy_throttle_norm` 才是当前实际油门位置。活动策略代码不得继续把 action 第 0 维描述为绝对 throttle。

### 6.3 姿态动作

姿态动作严格采用飞机局部右手系：

- `+pitch`：绕局部 `+X` 正旋转，机头下沉。
- `+yaw`：绕局部 `+Y` 正旋转，机头向左。
- `+roll`：绕局部 `+Z` 正旋转，右翼下沉。

内置 AI、人类输入和 policy adapter 可以为保持操控直觉执行上层映射，但送入 simulation 的最终动作必须服从以上符号。

### 6.4 二值动作

- `brake`：启用制动/减速控制。
- `fire_gun`：请求开火；是否实际开火仍由过热、冷却和弹道系统决定。
- `repair`：请求修复；是否执行仍由环境规则决定。

策略动作的 `fire_gun` 是请求，观察中的 `fire_gun_active` 是权威实际开火状态，两者不得混淆。

## 7. Normalizer Schema

策略输入为：

```text
normalized_obs = (raw_obs - mean) / max(std, epsilon)
epsilon = 1e-6
```

mean/std 只使用 train split 统计，禁止使用 validation/test 数据。统计使用 float64 累积，输出存为 float32。

二值索引 `[22,24,26,27,29,45,59,61,62,64]` 不执行统计标准化，固定：

```text
mean = 0
std  = 1
```

normalizer 必须包含：

```text
normalizer_schema_id
policy_contract_id
observation_schema_id
contract_sha256
obs_dim
epsilon
mean
std
train_row_count
source_dataset_id
```

禁止在线更新 observation normalizer。PPO、eval 和 live 必须读取同一个冻结 normalizer。

## 8. Dataset Schema

新 BC 数据集只能从保留的权威录制重建。dataset 必须包含：

```text
dataset_schema_id
dataset_id
policy_contract_id
observation_schema_id
action_schema_id
normalizer_schema_id
contract_sha256
obs_dim = 69
action_cont_dim = 4
action_bin_dim = 3
source_recording_schema
source_episode_manifest_sha256
split_seed
```

policy arrays：

```text
obs         float32 [N,69]
action_cont float32 [N,4]
action_bin  float32 [N,3]
```

每个 chunk 还必须包含：

```text
observed_role
demonstration_source
sample_weight
```

单一来源数据集可以继续在 `constants` 中使用按角色定义的旧式约定：

```text
demonstration_sources[role] = source
bc_sample_weights[role] = weight
```

同一角色包含多种合法示范来源时，必须改用：

```text
bc_demonstration_conventions[role][source] = weight
```

loader 必须逐 chunk 校验 `observed_role`、`demonstration_source` 和
`sample_weight`。不允许为了合并数据而把 human、teacher 或 DAgger 来源静默改成同一个
模糊标签。

当前默认来源约定为 fighter1 人类操作、fighter2 非人类控制器操作；默认权重分别为 `2.0` 和 `1.0`。BC 连续动作损失、二值动作损失及 BC distillation 损失均先按样本计算，再以 `sample_weight` 做归一化加权平均。normalizer 不使用该权重。

同一 episode 的两个角色必须进入同一 split。dataset loader 必须 fail closed：metadata 缺失、示范来源为空、样本权重非有限或非正、ID 不一致、指纹不一致或数组 shape 不一致时直接报错。

## 9. Model 与 Checkpoint Schema

当前唯一模型家族为 stateless hybrid actor-critic：

```text
input: normalized_obs float32[...,69]
actor continuous output: mean float32[...,4]
actor binary output: logits float32[...,3]
critic output: value float32[...,1]
```

新训练的默认结构为两层共享 MLP 干路，以及 actor、critic 各两个残差块。`num_layers` 表示每个独立塔的残差块数量，不包含共享干路；模型宽度和塔深度仍作为 checkpoint 超参数显式保存。

网络宽度、层数、dropout、连续动作标准差和 PopArt 参数属于 checkpoint 超参数，必须记录但不改变 policy contract。

checkpoint 必须包含：

```text
checkpoint_schema_id
policy_contract_id
observation_schema_id
action_schema_id
normalizer_schema_id
model_family_id
contract_sha256
obs_dim
action_cont_dim
action_bin_dim
model_hyperparameters
model_state_dict
training_stage
global_step
update_index
```

optimizer state 可选，但 `--resume` 必须要求存在且契约完全匹配。`--init-checkpoint` 可以不包含 optimizer state，但仍必须是原生新契约 checkpoint。

本次重置不迁移旧 checkpoint。所有新 checkpoint 从新 BC 数据集或新契约随机初始化开始生成。

## 10. 契约指纹

Step 2 必须新增版本控制内的机器可读文件：

```text
config/dfb_reinforcement_learning/part3_policy_contract_v1.json
```

该文件使用：

- UTF-8，无 BOM
- LF 换行
- 文件末尾一个换行
- 不允许 NaN/Infinity
- 字段和常量按规范固定顺序

`contract_sha256` 定义为该文件完整字节的 SHA-256 小写十六进制值。所有新 dataset、normalizer 和 checkpoint 保存同一指纹。任何语义修改都必须创建新契约文件和新 ID，不得仅更新指纹后沿用 v1。

## 11. 一致性与验收

### 11.1 跨语言

Rust 与 Python 必须对同一权威 fixture 输出：

- 相同字段顺序和 shape。
- 二值字段完全一致。
- 连续字段在规定容差内一致。
- 两个角色交换后满足对偶关系。

默认浮点验收容差：

```text
absolute <= 1e-5
relative <= 1e-5
```

### 11.2 跨运行路径

同一权威状态、同一 normalizer 和同一 checkpoint 必须在 train、eval、opponent pool 和 live 中得到一致观察与确定性动作。

### 11.3 Fail closed

以下情况必须拒绝：

- metadata 缺失契约 ID。
- 只根据 `obs_dim` 判断兼容。
- 契约 ID 相同但 SHA-256 不同。
- normalizer 来源 observation 不一致。
- checkpoint 与 dataset action schema 不一致。
- 旧 schema 被自动推断或静默转换。

## 12. 与 Part 2 的边界

本契约是 Current-GT 训练与验证契约，不冒充最终 Part 2 belief-state 接口。未来接入 Part 2 时，应定义新的 policy contract 或经过明确验证的输入映射，不得原地改变 `dfb_part3_policy_observation_v1`。
