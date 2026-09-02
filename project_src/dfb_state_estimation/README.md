# DFB State Estimation Workspace

本目录承载 DFB 的 Part 2 正式子项目 `dfb_state_estimation`，当前主要是 Python 侧实现。

当前目标：

- Part 2 数据读取
- Part 2 预处理
- Part 2 多模态模型
- 训练与评估逻辑

当前进度：

- `Phase 2.1` 已完成最小数据层骨架
- `Phase 2.2` 已完成 step 级真实字段读取
- `Phase 2.3` 已完成连续窗口字段装配
- `Phase 2.4` 已完成可复现的非均匀 stride 采样
- 已可加载 `meta.json / schema.json`
- `StepDataset` 已可读取 `chunked npz` 并返回 `StepSample`
- `WindowDataset` 已可返回固定长度窗口、时间字段与可配置 stride 采样
- `Phase 3.1` 已落地单步视觉模块骨架：
  - shared backbone
  - segmentation head
  - keypoint head
  - visual embedding head
  - 对应的最小监督 loss 定义
- `Phase 3.2` 已落地视觉几何验证链：
  - PnP / reprojection error
  - `v_sup`
  - `v_rep`
  - `raw_visual_evidence_strength`
- `Phase 3.3` 已落地单步音频模块骨架：
  - audio backbone
  - audio embedding head
  - `a_energy`
  - `a_cue`
  - `raw_audio_evidence_strength`
- `Phase 3.4` 已落地单步 evidence 组装：
  - `evidence_state_t`
  - `evidence_t`
  - `position_confidence / orientation_confidence` 监督
- `Phase 3.5` 已完成单步模块训练闭环：
  - 单步视觉 loss
  - 单步 evidence loss
  - 小规模 overfit / smoke train
- `Phase 4.1` 已落地第一轮时序 token/projection 骨架：
  - `state_token_t`
  - `visual_token_t`
  - `audio_token_t`
- `Phase 4.2` 已落地第一轮 temporal modality transformer 主干：
  - 2-layer transformer encoder
  - `hidden_tokens`
  - `state_hidden / visual_hidden / audio_hidden`
- `Phase 4.3` 已落地第一轮解码 heads：
  - `coarse_state_t`
  - `visual_evidence_strength`
  - `audio_evidence_strength`
  - 当前已改为“整窗编码、只解码最后一步”
- `Phase 4.4` 已完成第一轮训练 smoke：
  - `coarse_state_t` 监督 loss
  - 小规模 overfit / smoke train
- `Phase 5.1` 已落地第二轮 `belief_update_token_t` 组装：
  - `coarse_state_t`
  - calibrated `visual/audio_evidence_strength`
  - `delta_position / delta_orientation`
  - `linear_velocity / angular_velocity`
  - 小 MLP token projector
- `Phase 5.2` 已落地第二轮时序主干：
  - 3-layer belief transformer
  - `belief_hidden_states`
- `Phase 5.3` 已落地 `belief_state_t` heads：
  - `relative_position`
  - `relative_orientation`
  - `position_confidence`
  - `orientation_confidence`
  - `track_confidence`
  - 符号派生 `linear_velocity / angular_velocity`
- `Phase 5.4` 已落地 `policy_view_t` adapter：
  - 显式字段挑选
  - 保留一阶派生量
- `Phase 5.5` 已完成第二轮训练 smoke：
  - `belief_state_t` 监督 loss
  - 小规模 overfit / smoke train
- `Phase 6.2` 已完成统一 eval runner：
  - `single_step`
  - `temporal_modality`
  - `temporal_belief`
  - 结构化 JSON 指标输出
- `Phase 6.3` 已完成可视化与样本抽查导出：
  - segmentation overlay
  - keypoints overlay
  - reprojection overlay
  - window curve export
  - `summary.json`
- `Phase 6.4` 已完成评估结果格式固定：
  - `metrics.json`
  - `summary.txt`
  - `summary.json`
  - `index.html`
  - overlay / curve PNG
  - `comparison.json`
  - `comparison.csv`
  - `metrics_log.jsonl`
  - `metrics_log.csv`
- `Phase 7.1` 已完成训练配置冻结：
  - canonical train config 模板
  - 最小 train config loader
- `Phase 7.2` 已完成统一 trainer 入口：
  - `train.py --config ...`
  - 三个 stage 的统一 dispatch
- 当前 unified trainer 已进一步收口到新的双耳音频主链：
  - temporal 路径中的临时 audio bridge 已移除
  - 第一轮/第二轮训练都会真实调用新的单步音频模块
  - 当前 `single_step` 活动默认训练配置已固定：
    - `single_step_loss_weights.vision = 1.0`
    - `single_step_loss_weights.audio = 1.0`
    - `single_step_loss_weights.evidence = 0.01`
  - 训练过程中现在统一支持：
    - `logs/train_log.jsonl`
    - `checkpoints/step_*.pt`
    - `checkpoints/latest.pt`
    - `eval/eval_step_*.json`
    - `visuals/summary.json`
    - `visuals/index.html`
    - `visuals/*.png`
  - 训练可视化脚本：
    - `train/visualize_training_run.py`
    - 用于从：
      - `train_log.jsonl`
      - `eval/eval_step_*.json`
      自动生成分组曲线与可浏览页面

目录边界：

- `datasets/`, `preprocess/`, `models/`, `losses/`, `train/`, `utils/`
  - Part 2 Python 包下的子模块
- `config/dfb_state_estimation/`
  - repo 级 canonical 配置、schema、训练相关配置资产

工作目录约定：

- Python 开发默认以仓库根目录作为工作目录
- 典型导入方式使用：
  - `PYTHONPATH=project_src`
  - `from dfb_state_estimation.datasets import StepDataset, WindowDataset`

版本记录：

- `dfb_state_estimation` 当前独立版本记录位于：
  - `VERSION`
- `dfb_game` 只在其自身 Rust 代码发生实际变更时更新 patch 版本

不应放入：

- dataset 产物
- tensorboard 日志
- checkpoint
- 临时实验草稿脚本

当前 `Phase 3.1` 说明：

- 视觉模型代码已位于：
  - `models/vision/backbone.py`
  - `models/vision/heads.py`
  - `models/vision/module.py`
- 当前实现默认使用 PyTorch 接口
- `torch` 已加入 `train` 依赖组
- 最小 smoke 验证脚本位于：
  - `train/inspect_single_step_vision.py`
- 当前已完成最小验证：
  - segmentation / keypoint / embedding 前向 shape 正常
  - 监督 loss 可计算
  - `backward()` 可通过

当前 `Phase 3.2` 说明：

- 视觉几何 helper 位于：
  - `models/vision/geometry.py`
- 当前 `e_reproj` 定义为：
  - 单相机 PnP 解出的姿态将 canonical 3D keypoints 重投影回图像后，
    与预测 2D keypoints 的可见性加权平均像素误差
- 当前运行时量：
  - `v_sup`
  - `v_rep = exp(-ln2 * (e_reproj / e_half)^2)`
  - `raw_visual_evidence_strength = 0.5 * v_sup + 0.5 * v_rep`
- 当前不使用 `ADD / ADD-S` 作为运行时几何验证量；它们只保留给后续离线评估

当前 `Phase 3.3` 说明：

- 音频模型代码已位于：
  - `models/audio/backbone.py`
  - `models/audio/heads.py`
  - `models/audio/module.py`
- 当前单步音频模块输入：
  - `audio_window_binaural`
  - `binaural_energy_t`
  - `binaural_cue_vector_t`
- 当前运行时量：
  - `a_energy = 1 - exp(- E_sum / energy_scale)`
  - `a_cue` 由 `gcc_peak_value / interaural_coherence / reverb_ratio_proxy / directness_proxy` 平滑映射组合得到
  - `raw_audio_evidence_strength = 0.5 * a_energy + 0.5 * a_cue`
- 当前单步音频模块公共输出：
  - `doa_unit_vector_body`
  - `doa_conf`
  - `log_distance_scalar`
  - `dist_conf`
  - `binaural_energy_t`
  - `binaural_cue_vector_t`
  - `raw_audio_evidence_strength`
  - `audio_embedding`
- 当前实现细节：
  - 波形分支先由双耳短窗构造 `L / R / (L+R) / (L-R)` 四路输入
  - 显式 cue 分支将 `binaural_energy_t + binaural_cue_vector_t` 编码为 `cue_embedding`
  - 二者融合成共享 `audio_latent`
  - `audio_embedding` 与后续几何 heads 都从 `audio_latent` 读取
- 最小 smoke 验证脚本位于：
  - `train/inspect_single_step_audio.py`
- 当前已完成最小验证：
  - `audio_embedding` 前向 shape 正常
  - `raw_audio_evidence_strength` 可输出
  - `backward()` 可通过
- 当前评估/可视化链也已同步到新的音频契约：
  - 单步 eval 会额外统计：
    - `doa_conf_mean`
    - `dist_conf_mean`
    - `log_distance_mean`
    - `audio_position_l1`
  - 样本导出会在 `summary.json / index.html` 中展示：
    - `audio_state_t`
    - `evidence_state_t`
    - `coarse_state_t`
    - `belief_state_t`
    - `policy_view_t`

当前 `Phase 3.4` 说明：

- evidence 组装代码已位于：
  - `models/evidence/module.py`
- 当前单步 evidence 模块输入：
  - `SingleStepVisionOutput`
  - `SingleStepAudioOutput`
- 当前输出：
  - `evidence_state_t`
    - `relative_position`
    - `relative_orientation`
    - `position_confidence`
    - `orientation_confidence`
  - `evidence_t`
    - `visual_embedding`
    - `audio_embedding`
    - `raw_visual_evidence_strength`
    - `raw_audio_evidence_strength`
- 当前实现细节：
  - evidence 模块先从视觉分支内部解码出：
    - `visual_relative_position`
    - `visual_relative_orientation`
    - `visual_position_confidence`
    - `visual_orientation_confidence`
  - 再把音频分支的：
    - `doa_unit_vector_body`
    - `log_distance_scalar`
    - `doa_conf`
    - `dist_conf`
    解释为粗音频位置观测：
    - `audio_relative_position = doa_unit_vector_body * exp(log_distance_scalar)`
    - `audio_position_confidence = sqrt(doa_conf * dist_conf)`
  - 最后仅对位置与位置置信度做单步融合，写入：
    - `evidence_state_t.relative_position`
    - `evidence_state_t.position_confidence`
  - `relative_orientation / orientation_confidence` 第一版仍由视觉分支主导
- 最小监督 loss 位于：
  - `losses/evidence_supervision.py`
- 最小 smoke 验证脚本位于：

数据集抽查工具：

- 当前已提供：
  - `train/audit_dataset.py`
- 用途：
  - 对 packed dataset 做存储层全量扫描
  - 抽查 `StepDataset / WindowDataset` 视图层
  - 检查：
    - schema/group/field 完整性
    - chunk leading dimension
    - segmentation 类别范围
    - keypoint visibility 取值
    - binaural 音频字段 shape 与有限值
    - rule target 范围
- 典型用法：
  ```bash
  PYTHONPATH=project_src .venv/bin/python \
    project_src/dfb_state_estimation/train/audit_dataset.py \
    --dataset-root runs/dfb_state_estimation/open-20260409-173309-test_pack
  ```
  - `train/inspect_single_step_evidence.py`
- 当前已完成最小验证：
  - 真实数据集样本可前向
  - `position_confidence / orientation_confidence` loss 可计算
  - `backward()` 可通过

当前 `Phase 3.5` 说明：

- 最小训练闭环脚本位于：
  - `train/smoke_train_single_step.py`
- 当前训练闭环包含：
  - 单步视觉监督 loss
  - 单步 evidence 监督 loss
  - 共享优化器联训视觉、音频、evidence 模块
- 当前已完成最小验证：
  - 基于真实数据集样本的 20 步 overfit smoke train 可运行
  - 总 loss 可稳定下降

当前 `Phase 4.1` 说明：

- 时序模态 token projection 已位于：
  - `models/temporal/modality.py`
- 当前输入：
  - `evidence_state_t` 对应的状态字段
  - `evidence_t` 对应的 embedding / raw evidence strength
  - `binaural_energy_t`
  - `binaural_cue_vector_t`
  - `delta_binaural_cue_t`
  - `dt_to_prev`
  - `time_from_now`
- 当前输出：
  - `state_tokens`
  - `visual_tokens`
  - `audio_tokens`
  - `stacked_tokens`
- 最小 smoke 验证脚本位于：
  - `train/inspect_temporal_modality.py`
- 当前时序约束：
  - 每步 `state/visual/audio` 三子 token 允许同一步内互相注意
  - 未来步 token 通过 step-level block-causal mask 被严格屏蔽

当前 `Phase 4.2` 说明：

- 时序主干已位于：
  - `models/temporal/modality.py`
- 当前第一轮主干实现为：
  - `TemporalModalityTransformer`
  - 2-layer `TransformerEncoder`
- 当前输出：
  - `hidden_tokens`
  - `state_hidden`
  - `visual_hidden`
  - `audio_hidden`
- 已通过最小 smoke 验证：
  - 真实 `WindowDataset` 样本可前向 through backbone

当前 `Phase 4.3` 说明：

- 第一轮 heads 已位于：
  - `models/temporal/modality.py`
- 当前已落地：
  - `TemporalModalityCalibrationHeads`
  - `TemporalModalityCalibrationStage`
- 当前输出：
  - `coarse_state_t`
    - `relative_position`
    - `relative_orientation`
    - `position_confidence`
    - `orientation_confidence`
  - `visual_evidence_strength`
  - `audio_evidence_strength`
- 已通过最小 smoke 验证：
  - 真实 `WindowDataset` 样本可前向 through 第一轮完整模块

当前 `Phase 4.4` 说明：

- 时序监督 loss 已位于：
  - `losses/temporal_supervision.py`
- 最小训练闭环脚本位于：
  - `train/smoke_train_temporal_modality.py`
- 当前训练闭环监督：
  - `coarse_state_t.relative_position`
  - `coarse_state_t.relative_orientation`
  - `coarse_state_t.position_confidence`

当前 `Phase 5.1` 说明：

- 第二轮 token 组装代码已位于：
  - `models/temporal/belief.py`
- 当前已落地：
  - `BeliefUpdateInputs`
  - `BeliefUpdateTokenBuilder`
  - `BeliefUpdateTokenOutput`
- 当前 `belief_update_token_t` 由以下信息拼接后经小 MLP projector 投影到隐藏空间：
  - `coarse_state_t`
  - calibrated `visual_evidence_strength`
  - calibrated `audio_evidence_strength`
  - `delta_position`
  - `delta_orientation`
  - `linear_velocity`
  - `angular_velocity`
  - `dt_to_prev`
  - `time_from_now`
- 最小 smoke 验证脚本位于：
  - `train/inspect_belief_update_token.py`
- 当前已完成最小验证：
  - 真实 `WindowDataset` 样本可稳定输出 `belief_update_token_t`

时间特征说明：

- 当前第二轮仍同时保留：
  - `dt_to_prev`
  - `time_from_now`
- 它们在信息上存在轻度冗余，但当前被视为**有意保留的显式时间归纳偏置**：
  - `dt_to_prev` 提供局部采样间隔
  - `time_from_now` 提供全局相对时距
- 后续可单独做 ablation，再决定是否删减其中一项

当前 `Phase 5.2` 说明：

- 第二轮时序主干代码已位于：
  - `models/temporal/belief.py`
- 当前已落地：
  - `TemporalBeliefUpdateTransformer`
  - 3-layer `TransformerEncoder`
  - 标准 token-level causal mask
- 当前输出：
  - `belief_hidden_states`
- 最小 smoke 验证脚本位于：
  - `train/inspect_belief_update_token.py`
- 当前已完成最小验证：
  - 真实 `WindowDataset` 样本可稳定输出第二轮 hidden states

当前 `Phase 5.3` 说明：

- 第二轮 heads 已位于：
  - `models/temporal/belief.py`
- 当前已落地：
  - `BeliefStateHeads`
  - `TemporalBeliefUpdateStage`
- 当前输出：
  - `belief_state_t`
    - `relative_position`
    - `relative_orientation`
    - `position_confidence`
    - `orientation_confidence`
    - `track_confidence`
    - `linear_velocity`
    - `angular_velocity`
- 设计约定：
  - `coarse_state_t` 不补一阶导项
  - 第二轮 token 里使用“历史上下文序列 + 当前 `coarse_state_t` 替换最后一步”的方式组装
  - `belief_state_t.linear_velocity / angular_velocity` 则由当前步 belief 与前一上下文步做符号差分派生
- 最小 smoke 验证脚本位于：
  - `train/inspect_belief_update_token.py`
- 当前已完成最小验证：
  - 真实 `WindowDataset` 样本可稳定输出当前步 `belief_state_t`

当前 `Phase 5.4` 说明：

- `policy_view_t` adapter 已位于：
  - `models/temporal/belief.py`
- 当前已落地：
  - `PolicyViewAdapter`
  - `PolicyViewOutput`
- 当前 `policy_view_t` 第一版保留：
  - `relative_position`
  - `relative_orientation`
  - `linear_velocity`
  - `angular_velocity`
  - `track_confidence`
- 当前 adapter 为显式规则型：
  - 不引入额外可学习参数
  - 直接从 `belief_state_t` 挑选/转发字段
- 最小 smoke 验证脚本位于：
  - `train/inspect_belief_update_token.py`
- 当前已完成最小验证：
  - 真实 `WindowDataset` 样本可稳定构造当前步 `policy_view_t`

当前 `Phase 5.5` 说明：

- 第二轮监督 loss 已位于：
  - `losses/belief_supervision.py`
- 最小训练闭环脚本已位于：
  - `train/smoke_train_temporal_belief.py`
- 当前监督：
  - `belief_state_t.relative_position`
  - `belief_state_t.relative_orientation`
  - `belief_state_t.linear_velocity`
  - `belief_state_t.angular_velocity`
  - `belief_state_t.position_confidence`
  - `belief_state_t.orientation_confidence`
- 当前不对 `track_confidence` 构造精确离线 target
- 当前已完成最小验证：
  - 基于真实 `WindowDataset` 样本的 20 步 overfit smoke train 可运行
  - 总 loss 可稳定下降

下一阶段重点：

- 统一评估基础设施
- 统一训练入口与训练工作流

当前 `Phase 6.2` 说明：

- 统一评估入口已位于：
  - `train/eval_runner.py`
- 当前支持阶段：
  - `single_step`
  - `temporal_modality`
  - `temporal_belief`
- 当前输出：
  - 结构化 JSON 指标结果
- 当前已完成最小验证：
  - 三种阶段都可在真实导出数据集上完成一次 eval

当前 `Phase 6.3 / 6.4` 说明：

- 可视化导出脚本已位于：
  - `train/export_eval_visuals.py`
- 当前固定结果目录格式为：
  - `runs/dfb_state_estimation/eval/<run_name>/`
- 当前固定结果文件：
  - `metrics.json`
  - `summary.txt`
  - `summary.json`
  - `index.html`
  - overlay / curve PNG
- 其中：
- `metrics.json` 来自统一 `eval_runner.py`
- `summary.txt` 为指标快速扫读文本
- `summary.json` 为单样本详细状态摘要
- `index.html` 为最小可视化入口，直接引用导出的 PNG 与摘要
- 多阶段快照日志工具已位于：
  - `train/export_eval_logbook.py`
- 当前可直接把不同阶段/不同时间点的评估结果追加成：
  - `metrics_log.jsonl`
  - `metrics_log.csv`

当前 `Phase 7.1` 说明：

- canonical 训练配置模板已位于：
  - `config/dfb_state_estimation/train/default_train_config.json`
- 最小配置加载器已位于：
  - `train/config.py`
- 当前冻结的训练配置范围包括：
  - 顶层运行参数
  - optimizer
  - schedule
  - `single_step`
  - `temporal_modality`
  - `temporal_belief`

当前 `Phase 7.2` 说明：

- 统一训练入口已位于：
  - `train/train.py`
- 当前支持命令行：
  - `--config`
  - 可选 `--stage`
  - 可选 `--num-steps`
  - 可选 `--output-root`
- 当前支持的统一训练 stage：
  - `single_step`
  - `temporal_modality`
  - `temporal_belief`
- 当前训练输出：
  - `resolved_train_config.json`
  - `train_summary.json`
- 当前 trainer 已显式消费：
  - `single_step.step_index`
  - `temporal_modality.window_index`
  - `temporal_belief.window_index`

当前 `Phase 7.3` 说明：

- 统一训练入口当前已支持：
  - `--resume`
- 当前训练目录下的固定产物：
  - `checkpoints/latest.pt`
  - `checkpoints/step_*.pt`
  - `logs/train_log.jsonl`
  - `eval/eval_step_*.json`
  - 可选可视化产物：
    - `visuals/summary.json`
    - `visuals/index.html`
    - `visuals/*.png`
- 当前 checkpoint 内容包括：
  - stage
  - `step / global_step`
  - modules state dict
  - optimizer state dict
  - torch RNG state
  - 可用时的 CUDA RNG state
- 当前 resume 语义：
  - 校验 checkpoint stage 与当前 stage 一致
  - 恢复 modules / optimizer / RNG state
  - 从 `checkpoint.step + 1` 继续训练

当前 `Phase 7.4` 说明：

- 原有 smoke train 脚本当前只保留为快速调试包装：
  - `train/smoke_train_single_step.py`
  - `train/smoke_train_temporal_modality.py`
  - `train/smoke_train_temporal_belief.py`
- 当前 smoke wrapper 不再维护独立训练逻辑
- 它们现在只负责：
  - 覆盖少量配置
  - 调用统一 `train.run_training(...)`
  - 检查 `train_summary.json` 中的 loss 是否下降
- 当前正式训练主入口已经完全收口到：
  - `train/train.py`
- 当前 stage dispatch：
  - `single_step`
  - `temporal_modality`
  - `temporal_belief`
- 当前每次训练都会写出：
  - `resolved_train_config.json`
  - `train_summary.json`
- 当前 stage 训练策略：
  - `single_step`
    - 联合训练 `vision / audio / evidence`
  - `temporal_modality`
    - 冻结单步模块
    - 训练第一轮时序模块
  - `temporal_belief`
    - 冻结单步模块
    - 联合训练第一轮与第二轮时序模块
