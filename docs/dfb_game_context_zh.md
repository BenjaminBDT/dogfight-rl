# Part 1 Context

状态：Active Reference

本文档用于单独管理第 1 部分“狗斗环境”的长期上下文。  
它不取代 [project_context_zh.md](/run/media/ayano/SharedProjects/general_projects/dfb/docs/project_context_zh.md)，而是作为其 Part 1 细化文档存在。

## 1. Part 1 的目标

Part 1 的目标是一个：

- 可游玩
- 可联机
- 可录制
- 可回放
- 可重建视觉/音频模态
- 可作为训练环境提供给后续程序

的双机空战环境。

其定位不是完整真实飞行仿真，而是：

- 近似合理
- 手感可控
- 行为稳定
- 可支撑后续感知模型与强化学习训练

## 2. 当前阶段定位

截至当前阶段，Part 1 已进入冻结基线状态。

冻结并不表示完全禁止修改，而是表示默认不再主动做破坏性重构。  
后续 Part 1 的主要任务是：

- 明确 bug 修复
- 低风险工具性增强
- 文档整理
- 为 Part 2 / Part 3 提供非破坏性接口补充

默认不再主动修改：

- 飞行模拟主模型
- server 权威 / client 预测同步主链
- bridge 协议
- recording schema
- replay / modalities / datapacker 的主消费语义

## 3. 当前冻结主契约

### 3.1 飞行模拟

当前飞行模型的冻结要求是：

- 不追求复杂、严格真实的空气动力学
- 但也不允许明显违背基本飞行直觉
- 非失速区速度变化应能对机动性能产生可感知影响

当前已冻结的方向包括：

- 非失速区速度差异优先体现到：
  - `roll`
  - `pitch`
  - `yaw`
  的有效速率
- 标准参考工作点、trim 语义、重量参考链已经收口
- 失速语义已向 `capacity / demand` 收敛
- 当前模型定位是“可玩、稳定、近似合理”的工程仿真

### 3.2 动作语义

当前对外稳定主契约为：

- `throttle / pitch / roll / yaw ∈ [-1, 1]`
- `brake / fire_gun / repair`

说明：

- 项目内部采用：
  - `intent -> normalized control command -> simulation input`
- 本地人类输入链中的：
  - `keyboard weight`
  - `mouse weight`
  - `mouse smoothing`
  只属于输入适配层，不属于对外环境动作语义

### 3.3 联机与权威

当前主联机语义已经冻结为：

- `server` 是唯一权威
- `client` 负责输入、本机预测、远端插值和 presentation-only 即时反馈
- authoritative fire / hit 与 presentation muzzle 分层
- 第一版服务器 lag compensation 是当前命中语义基线

### 3.4 录制与离线数据

当前 authoritative recording 是 expert data 的主来源。

当前录制主资产包括：

- world state
- dynamic world
- command
- recorded audio semantics

当前 authoritative 录制与训练数据主动作字段已冻结为：

- `fighter1_command`
- `fighter2_command`

### 3.5 replay / modalities / datapacker

当前这条主链已冻结为：

- 统一消费 authoritative recording
- modalities 基于 recorded world + recorded audio semantics 重建视觉/音频
- 不把 client prediction / presentation smoothing 当作权威数据源

## 4. 当前架构理解

### 4.1 运行角色

- `dfb_server`
  - 唯一权威仿真源
  - 负责权威状态、权威 projectile、权威伤害与录制
- `dfb_launcher`
  - live 联机入口
  - 负责地址/会话输入、角色选择、ready 与拉起 child
- `dfb_client_gameplay`
  - 玩家游玩客户端
  - 负责输入采集、本机预测、表现层与音频
- `dfb_client_observer`
  - 观察客户端
  - 支持 live observer 与 recorded observer
- `dfb_tool_dataset`
  - 统一数据工具
  - `extract / reconstruct`
  - `label`
  - `pack`
- `dfb_tool_keybindings`
  - 输入绑定配置工具

### 4.2 联机结构

- `server` 是唯一权威
- `launcher` 是唯一 session owner
- `gameplay/observer child` 是 launcher 当前会话阶段的运行子客户端
- live scene 由 `server` 单向决定：
  - `server -> lobby/start -> launcher -> child`

### 4.3 同步结构

当前同步主链已经收敛为：

- 本机输入历史
- 权威基线
- 未确认输入重演
- 远端对象插值
- presentation-only 即时反馈
- authoritative fire / hit 与 presentation muzzle 分层
- 服务器 lag compensation 第一版

当前这条主链已达到可用冻结水平。  
后续若继续优化，更多应是高阶体验增强，而不是主链重写。

## 5. 当前已知稳定方向

### 5.1 飞行物理

当前已具备：

- 基础升力 / 阻力 / 推力 / 重力
- 失速因子
- 基于 rate limit 的姿态控制
- trim 参考点
- 重量参考链
- 非失速区速度-机动性能关系
- 角运动层级拆分

### 5.2 音频

当前 live / replay / modalities 的主音频语义已完成统一：

- step 级 observation 音频固定为 2 通道双耳/agent-hearing 观测
- validation 视频当前输出 2 通道音轨
- 第一版稳定覆盖：
  - engine
  - gunfire
  - flyby
  - hit

补充结论：

- 2026-04 的长时间引擎声退化问题已定位完成
- 根因是共享 engine synthesis helper 中 `phase: f32` 长时间累加不回卷导致精度退化
- 修复已经落地在共享音频语义层，因此同时覆盖：
  - live runtime audio
  - replay
  - `EpisodeReconstructor`
  - `dfb_tool_dataset`

### 5.3 录制格式

当前 recording 格式已经完成一轮压缩收口：

- manifest 显式声明 artifact convention
- visual/audio capture 元信息已上提到 manifest
- `RecordedStep` 收紧为 transition core
- step 主体采用：
  - chunked text bundle
  - lightweight index
- recorded observer 和通用 reconstruction 允许读取未声明 `policy_contract_id` 的历史录制，并将其显式标记为 `unspecified_legacy_policy_contract`
- 该读取兼容仅用于回放与检查；Part 3 dataset 构建仍要求当前 policy/action/recording 契约完全匹配，历史录制不会被隐式接入活动训练链路
- validation 产物统一放在：
  - 开发环境：`datasets/dfb_game/recordings/<episode>/validations/<role>/`
  - 分发环境：`recordings/<episode>/validations/<role>/`

### 5.4 接口预留

当前 Part 1 已具备：

- Rust 原生环境 API
- `PyO3` Python 绑定
- recording access / replay / dataset 的统一主语义

当前明确结论：

- `reward` 默认不由环境定义和计算，交给训练侧
- `events` 保留为辅助事实流，而不是主 observation 契约
- 后续语言绑定扩展，应以当前冻结语义为基线

### 5.5 Python 训练访问与视觉重建

当前 Python 训练访问主线已收口完成，稳定存在四条路线：

- `Environment`
- `EpisodeRecording`
- `EpisodeReconstructor`
- 预构建 derived / dataset

当前视觉重建主线也已收口为：

- 无窗口 offscreen render
- GPU readback
- `VisualCaptureFrames` 统一 observation 契约

### 5.6 规则式 AI 战术动作

三个活动规则式 AI profile 共享基础战术动作门控：

- 开火要求目标处于前方 `30°` 半锥角和 `95%` 最大射程内，枪热达到 `0.8` 或过热时停火。
- 减油门、50 m 近距和原有近距速度控制可以触发制动，低速和失速状态抑制战术制动。
- 修复只在至少一个部件耐久不高于 `50%`、剩余修复窗口边界安全、预测机距安全且敌方威胁较低时触发；总机体耐久不单独触发规则式 AI 修复。
- 一次完整修复周期恢复全部五个部件，包含轻度受损部件；总机体耐久仍按配置比例恢复。权威修复状态在恢复完成的同一帧结束，规则式 AI 在下一次输入采集时恢复战斗。
- 边界恢复优先于修复，修复优先于开火与战术制动。
- 规则式 AI 每个固定帧从默认值生成完整动作并整体写回，不允许沿用上一帧的修复、开火或制动字段。

环境 API 另提供 `built_in_ai_passive_bounce` 被动飞行 profile。它不追踪、不射击、不主动变向，常态输出零操纵；只有现有边界预测判定存在风险时才复用统一边界恢复控制，恢复安全后继续按当前运动状态直飞。该 profile 用于构造非战术、可移动的空间感与运动感训练对手，不属于上述战术 AI。

这些决策由权威当前状态生成，并通过正常 `ControlInput` 主线执行和录制，不存在仅用于训练标签的旁路。

### 5.7 客户端飞行辅助显示

客户端 presentation 层支持通过 `F4` 切换飞行辅助信息：

- 为每架未损毁飞机绘制冻结对象空间中的 `+X` 左轴、`+Y` 上轴与 `+Z` 前轴
- 从表现层枪口沿机体 `+Z` 前向绘制超长瞄准射线
- 辅助显示只消费当前客户端表现状态，不进入权威模拟、bridge 协议、recording 或训练 observation
- 受击方位优先使用权威事件中的攻击者角色定位；机体 `+X` 左侧映射到屏幕左侧，只有无法解析攻击者时才退回弹着点或接触点

## 6. 当前允许继续做的修改

- 明确 bug 修复
- 小型 HUD / 展示调整
- 工具链易用性改进
- 文档整理
- 第 2/3 部分实际接入时发现的非破坏性接口补充

## 7. 当前不建议继续做的修改

- 重写飞行模拟主方程
- 重写 bridge 协议或 recording schema
- 重新定义动作/观测字段
- 为局部体验再次引入 client-side 权威混淆
- 在没有明确收益验证前继续压缩 recording 容器格式

## 8. 当前残留 Bug

以下问题是 Part 1 在冻结基线下仍保留的明确缺陷：

- 飞机仍可能出现频闪式抽搐
  - 当前不显著破坏整体游玩
  - 但会影响体验
  - 这仍属于权威 / 预测 / 表现层同步残留问题

- 坐标语义仍存在历史不统一
  - 当前实现事实：
    - 世界系采用 `+Y` 向上，水平面为 `XZ`
    - 飞机对象空间当前采用 `+Z` 前向、`+X` 左向、`+Y` 上向
    - 相机局部前向则遵循图形相机习惯，等价于沿 `-Z` 朝屏幕内
  - 这一不统一已经导致过：
    - 左右机翼标签与受伤语义难以直觉对应
    - Part 2 中相机系 / 机体系 / 图像系设计容易混乱
  - 已明确结论：
    - 在 Part 2 实现前，必须先统一飞机对象空间、资产朝向与左右机翼标记语义
    - 当前冻结的飞机对象空间语义为：
      - `forward = +Z`
      - `left = +X`
      - `up = +Y`
    - 相机局部空间继续保留图形相机约定，不再要求与飞机对象空间同轴同向
    - 在这一收口完成前，不应开始依赖该语义的 Part 2 坐标实现

这类问题当前应被视为：

- 第 1 部分后续仅保留的明确 bug 修复目标

而不是扩大范围重写模拟、网络协议或录制格式的理由。
