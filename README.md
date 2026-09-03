# dogfight-rl

近距离空战（狗斗）仿真与深度强化学习项目：Rust/Bevy 驱动 Python DRL 训练管线，通过 PPO、行为克隆、DAgger 与自博弈训练自主空战智能体。

本项目探索自训练智能体在高保真六自由度飞行仿真中能达到的水平——从模仿脚本教师到完全自博弈的策略，全部集成在单一代码库中，并通过契约固定的观察/动作 schema 保证一致性。

## 演示

红色战机为训练后的智能体（RL 策略），两个场景：

**被咬尾时的防御机动** —— 智能体自身视角：
![防御机动演示](docs/demo-defense.gif)

<video controls src="docs/demo-defense.mp4" width="100%"></video>

**对头交战** —— 对手（黄色战机）视角：
![对头交战演示](docs/demo-head-on.gif)

<video controls src="docs/demo-head-on.mp4" width="100%"></video>

## 功能特性

- **六自由度飞行仿真**：基于 Rust + Bevy 0.18 构建，通过 pyo3 Python 绑定实现紧密的 RL 循环集成
- **69 维契约驱动观察空间**（37 个字段：敌机/本机在机体坐标系下的相对几何与运动学，以及回合时间），由带 SHA-256 校验的版本化 JSON 策略契约固定
- **混合动作空间**：4 个连续（油门/俯仰/滚转/偏航）+ 3 个二值（刹车/开火/维修）
- **多阶段训练课程**：行为克隆（BC）热启动 → DAgger 迭代 → PPO 优化 → 自博弈（模型控制双方战机）
- **完整 ML 工具链**：从录制数据打包数据集、观察归一化器、checkpoint 架构迁移、实时模型试飞

## 技术栈

- **Rust / Bevy 0.18** —— 仿真引擎、物理、渲染、录制
- **pyo3 / maturin** —— 连接仿真与训练的 Python 绑定
- **PyTorch** —— 神经网络与 PPO/BC/DAgger 训练
- **69 维观察空间**、混合连续+二值动作头（clipped-normal 策略）
- 自博弈与对手池（内置 AI 变体 + 模型）

## 快速开始

构建 Rust 仿真引擎：

```bash
cargo build --release
```

查看 PPO 训练入口（模块布局：`project_src/dfb_reinforcement_learning`）：

```bash
PYTHONPATH=project_src python -m dfb_reinforcement_learning.train.train_ppo --help
```

## 结果

在 1000 局评估窗口内，训练后的智能体达到：

- **敌机损毁率：0.78**
- **自机损毁率：0.42**
- **共同损毁率：0.20**

## Vibe Coding 说明

核心算法（观察/奖励设计、自博弈 rollout 架构、截断策略）与整体系统架构由作者自主设计。工程实现由 AI 编程智能体（Hermes、Codex 等）辅助。

## License

MIT —— 见 [LICENSE](LICENSE)。
