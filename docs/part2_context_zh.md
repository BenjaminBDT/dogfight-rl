# Part 2 Context

状态：Active Reference

本文档用于单独管理第 2 部分“多模态时序状态估计”的长期上下文。  
它不取代 [project_context_zh.md](/run/media/ayano/SharedProjects/general_projects/dfb/docs/project_context_zh.md)，而是作为其 Part 2 细化文档存在。

当前基础来源：

- [part_2_overall_model_architecture.md](/run/media/ayano/SharedProjects/general_projects/dfb/tmp/part_2_overall_model_architecture.md)
- [pre_temporal_modules_spec_updated.md](/run/media/ayano/SharedProjects/general_projects/dfb/tmp/pre_temporal_modules_spec_updated.md)
- [temporal_belief_module_spec.md](/run/media/ayano/SharedProjects/general_projects/dfb/tmp/temporal_belief_module_spec.md)

当前与音频专项重构并行参考：

- [part2_audio_refactor_plan_zh.md](/run/media/ayano/SharedProjects/general_projects/dfb/docs/plans/archive/part2_audio_refactor_plan_zh.md)

当前实验主线参考：

- [part2_visual_recovery_plan_zh.md](/run/media/ayano/SharedProjects/general_projects/dfb/docs/plans/part2_visual_recovery_plan_zh.md)
- [part2_temporal_training_plan_zh.md](/run/media/ayano/SharedProjects/general_projects/dfb/docs/plans/part2_temporal_training_plan_zh.md)
- [part2_audio_next_step_plan_zh.md](/run/media/ayano/SharedProjects/general_projects/dfb/docs/plans/part2_audio_next_step_plan_zh.md)

当前新的视觉恢复数据线补充：

- 2026-04-28 synthetic 数据线已执行一次强制重置：
  - 已确认旧版 synthetic generator 在单样本导出时先后做了两次独立 offscreen flush
  - `RGB` 与 `semantic` 可能来自不同渲染帧，因此旧 synthetic 数据集整体失效
  - 当前修正已切到：
    - 单次 offscreen flush 同时提取 `RGB + semantic`
  - 因此此前生成的 synthetic 数据集、基于其训练得到的 synthetic segmentation run、以及相关抽检产物均已清理
  - 后续 synthetic 数据必须在修正后的生成链上重建
  - 当前重建后的 synthetic 采样基线已调整为：
    - `front_only / rear_only`
    - `[2r,50], [50,100], [100,150], ...` 的 `50m` 半径 band 保底
    - `[2r,max_range]` 球扇形体积均匀采样补量
    - `min_selected_target_area >= 5`
    - 相机到目标中心距离约束：
      - `>= near_plane + AIRCRAFT_COLLISION_RADIUS`
  - synthetic offscreen capture 主链又补了一轮时序修正：
    - 已引入显式 capture generation
    - 当前 sample 只接受本代 `RGB/semantic` readback
    - 不再接受 render app 侧旧 generation 的残留读回
    - 单样本等待窗口已提升到 `24` update
  - 但当前继续审计后，问题尚未彻底归零：
    - 现已确认根因不应再被理解为单纯 `msaa_trial` 边缘偏差
    - 当前更可能的框架级问题是：
      - synthetic sample 的结果仍通过全局 `front/rear × rgb/semantic` 槽回收
      - 而不是按 sample 自身的唯一身份回收
    - 因此后续 synthetic capture 修正方向已切换为：
      - `session/request` 身份隔离
      - one-shot capture request
      - 冻结 world state 后的顺序采集
    - 不再继续把 `generation/clear/wait` 视为最终方案
    - 当前已完成其中第二步：
      - synthetic sample 先在固定 world state 上采 `RGB`
      - 再在同一逻辑状态上采 `semantic`
      - 不再依赖“一轮 flush 同时收齐四路结果”的假设
    - 当前又完成了第三步：
      - synthetic capture 不再从全局 `VisualCaptureFrames` 读活动结果
      - 当前改为：
        - 显式发起 per-session request
        - 按 `session_id/request_id` 回收 frame
      - 因此 synthetic 主线已与旧的全局活动 frame 槽解耦
    - 当前又完成了第四步：
      - 已在新 capture 架构上重建 `100` 样本审计集
      - 当前审计集：
        - `synthetic_visual_dataset_fix_v6_100_msaa_trial_overlay`
      - 后续是否进入 `5k` 重建，以人工 overlay 抽检结果为准
    - 当前又完成了第五步：
      - 已在同一新 capture 架构上重建 `5k` synthetic 主数据集
      - 当前主数据集：
        - `synthetic_visual_dataset_fix_v6_5k`
    - 在 `v6_20k -> target_identity clean` 之后又确认：
      - 当前 clean 根已经足够稳定，可继续作为 synthetic 训练主基线
      - 但它当前只覆盖 segmentation 监督
      - 还不包含完整单步视觉模块训练所需的：
        - `keypoints_2d_front / rear`
        - `keypoint_visibility_front / rear`
        - `keypoint_voting_pixels_front / rear`
        - `keypoint_voting_unit_vectors_front / rear`
        - `keypoint_voting_mask_front / rear`
      - 已补一套 synthetic single-step 审计导出：
        - segmentation / keypoint / `PnP` 重投影 / sparse voting 可视化
      - 因此下一阶段的重点不再是重做 segmentation 数据线
      - 而是：
        - 从 `v6` clean 根派生完整 synthetic single-step `PnP` 数据集
        - 直接服务完整单步视觉模块训练
      - 该派生数据集的冻结语义是：
        - 原始 `RGB + segmentation` 保持在 `target_identity clean`
        - 新增 `labels/*.json + voting/*.bin` 存放 keypoint / visibility / sparse voting / geometry 监督
        - Python 训练侧通过 `dataset_format = synthetic_single_step` 直接复用现有 `single_step` stage
  - 已完成 `750~800m` 经验 probe：
    - 在 `400x300`、`msaa_trial` 下仍可稳定保持 `>=5px`
    - 当前活动 `max_range` 暂定为 `800m`
- 已新增 `dfb_tool_dataset synthetic-visual` 原型工具
- 该工具用于在受控环境下生成单步视觉恢复样本：
  - front / rear RGB
  - front / rear semantic segmentation
  - 每样本的 bucket 与位姿 metadata
- 当前定位：
  - 合成视觉原型集生成器
  - 服务 segmentation / selected-view / dense voting / PnP smoke
- 当前状态：
  - 命令路径与 GPU 渲染链已跑通
  - 小规模原型集已可生成真实目标类 semantic mask
  - 第一轮采样校准后，`front_only / rear_only / both` 均已能实际生成
  - synthetic 样本目录已直接导出彩色 segmentation 预览图，便于人工审计
  - 但当前样本分布仍明显偏向极小目标与极大目标
  - bucket 命中率与覆盖度仍未达到正式训练要求
  - 因此当前 synthetic line 仍定位为 calibration branch，尚未并入正式视觉恢复主训练线
  - 当前新的采样重构方向已经冻结为：
    - 自机 6D 合法随机采样
    - 视锥屏幕网格采样
    - 相机射线距离采样
    - 敌机姿态与局部位移扰动
    - 渲染后按可视性 / 面积 / 网格覆盖验收
  - 旧的 `area bucket -> distance range` 逻辑后续仅保留为验收统计语义，不再作为 synthetic 主采样变量
  - 当前已完成的第一步是：
    - 自机位置改为 arena 柱状空间内均匀采样
    - 自机姿态改为 `SO(3)` 均匀采样
    - synthetic 生成已支持同时导出 `fighter1` / `fighter2` 两套观察机角色数据
  - 当前已完成的第二步是：
    - 目标采样主路径改为：
      - `front/rear` 相机锚点
      - `半径 500m` 的球扇形体积空间均匀采样
      - 再按投影结果验收到屏幕网格 cell
    - `front/rear` 使用全屏 `6 x 4` 网格
  - 当前已完成的第三步是：
    - 活动 synthetic 主线已收口为：
      - `front_only / rear_only`
      - `0~500m`
      - 球扇形体积均匀采样
    - `>500m` 远距补线当前不再进入活动主线
  - 当前已完成的第四步是：
    - `grid cell id` 已写入 synthetic metadata / manifest
    - manifest 已新增 `coverage_summary`
    - 当前可直接按：
      - `sampling_regime`
      - `requested_visibility_bucket`
      - `grid cell`
      查看覆盖情况
  - 当前 synthetic segmentation loader 已支持：
    - mixed-role merged manifest
    - 一个训练集内同时混合 `fighter1` / `fighter2` 观察机样本
    - 按 entry 级 observed_role 做 `binary_target` remap
  - 当前新的主采样收口方向已改为：
    - `front_only / rear_only`
    - `0~500m` 球扇形体积均匀采样
    - 屏幕网格 + 投影验收联合采样
  - `both` 与 `>500m` 暂时不再作为活动主集的优先目标
  - 下一步 synthetic 数据处理主线已切换为：
    - 先将“分半径段生成数据”和“空间均匀采样生成数据”聚合成新的 mixed-role 主集
    - 再从聚合主集中切出三阶段 curriculum 子集
    - 先跑 Stage 1 的中大目标启动训练
  - 当前这条线的最新结果是：
    - 聚合主集和三阶段子集均已生成
    - Stage 1 已用更高 batch (`16`) 跑过一轮
    - 但聚合评估仍然回到：
      - `pred_selected_view_none = 1.0`
      - `target_iou = 0.0`
    - 说明 synthetic curriculum 第一阶段本身还不足以救活当前 segmentation
  - 之后又做了一轮视觉结构修正：
    - segmentation 分支改为轻量多尺度 decoder
    - keypoint / support / PnP 主线未动
  - 但在相同 Stage 1 synthetic curriculum 上，聚合评估仍未改善
    - 当前可以确认：问题并不只来自“低分辨率最终特征直接上采样”
  - 又补做了一轮成熟语义分割模型试作：
    - `DeepLabV3-ResNet50 pretrained`
    - `batch_size = 4`
    - `300` step synthetic Stage 1 短训
  - 工程链路已跑通，但 `256` 样本聚合评估仍然回到：
    - `target_iou = 0.0`
    - `target_precision = 0.0`
    - `target_recall = 0.0`
    - `pred_selected_view_none = 1.0`
  - 当前可以继续确认：问题不只是 backbone / decoder 选择本身
  - 之后又补做了一轮 target-centered patch 训练试作：
    - 固定 `96x96` 局部 patch
    - 仅对含目标的视图计算 segmentation loss
  - 该分支已经显著提高了训练 batch 中的正类像素比例：
    - 从约 `0.27%~1.0%` 提升到约 `3.5%~11.5%`
  - 但 `300` step 短训后的 `256` 样本聚合评估仍然为：
    - `target_iou = 0.0`
    - `target_precision = 0.0`
    - `target_recall = 0.0`
    - `pred_selected_view_none = 1.0`
  - 当前可以继续确认：问题也不只是“整图监督里正类像素过于稀薄”
  - 同时已经确认当前旧的 mixed aggregated curriculum 在目标身份上不适合作为主训练集：
    - 训练 `target=fighter2` 时，不应再混入 `observer=fighter2` 的数据
    - 因为该子集里 `fighter2` 实际上是自机，视角分布会产生固定偏置
  - 已完成一轮新的 synthetic clean / curriculum 数据集重建：
    - 先按 `target_role` 固定切分
    - 再分别构建新的阶段数据集
  - 基于这套新数据，已经补做了一轮：
    - `target_fighter2/stage1_seed_large`
    - `ResNet18 pretrained + multi-scale decoder`
    - `binary_target + focal + dice + target-centered patch`
    - `batch_size = 16` 的 warm-start
  - 但 `256` 样本聚合评估仍然回到：
    - `target_iou = 0.0`
    - `target_precision = 0.0`
    - `target_recall = 0.0`
    - `pred_selected_view_none = 1.0`

当前新的 segmentation 恢复共识补充：

- 继续坚持绝对三分类并不是当前第一优先级
- 当前恢复线应优先采用：
  - `binary_target segmentation`
- 其语义是：
  - 当前观察机所关心的敌机为正类
  - 其余全部并入负类
- 设计要求：
  1. 不改 backbone
  2. segmentation head 支持可配置输出类别数
  3. config 中显式声明 segmentation label mode
  4. dataset 侧按 mode 生成训练 target
- 这样做的目标不是永久放弃多类语义，而是先恢复：
  - 目标是否存在
  - 目标轮廓是否可稳定分出
- 后续若要回到：
  - `multiclass_absolute`
  仍可通过保持 backbone / trunk、替换最后 head 与 label mode 平滑升级
- 当前最小工程实现已落地：
  - `segmentation_label_mode` 已接入训练配置
  - `segmentation_only` 已支持：
    - `multiclass_absolute`
    - `binary_target`
  - `binary_target` 当前只接入 `segmentation_only`
  - 其余 stage 暂不切换语义
- 当前进一步确认的工程边界是：
  - `segmentation_only` 不再继续沿用“伪单视图双输入”兼容路径
  - 已新增**真单视图** synthetic segmentation 训练主线：
    - dataset 每个样本只暴露一个 `active_view`
    - model 前向只接收一张图
    - loss / eval 只围绕该单视图 mask 计算
  - 旧双视图单步视觉主链继续保留给：
    - `single_step`
    - `temporal_modality`
    - `temporal_belief`
  - 当前 segmentation 恢复实验应统一在真单视图主线下继续推进

当前新的视觉 backbone 恢复共识补充：

- 当前视觉输入不再需要使用 `RGBA`
- 后续视觉主线统一按：
  - `RGB-only`
  进入视觉 backbone
- 当前新增恢复方向是：
  - 保持现有 head / loss / routing 主链不变
  - 将视觉 backbone 改为可配置
  - 第一版优先引入：
    - `ResNet18 pretrained`
- 这样做的目标是：
  - 为极小目标 segmentation 提供更强的视觉先验
  - 与当前 `binary_target segmentation` 路线组合验证
  - 不直接破坏旧 conv backbone 配置

## 1. Part 2 的任务定义

Part 2 不是“直接做 RL 决策”，而是先做一个在部分可观测条件下运行的：

- 多模态
- 时序
- 带不确定性

的敌机状态估计模块。

它的目标不是复原完整世界，而是为 Part 3 提供一个更稳定、语义化、更接近 belief 的敌机状态表示。

当前问题定义可收口为：

输入：

- 当前时刻双相机图像
- 一段双耳/agent-hearing 双通道音频窗口
- 自机状态
- 最近若干步历史

输出：

- 对敌机状态的 belief estimate
- 对该 estimate 的不确定性/置信度
- 视觉是否可靠、当前主要证据来源等辅助量

## 2. 当前已形成的总体架构共识

当前主线已经比较明确，整体结构是：

1. `Single-Step Evidence Extraction`
2. `Hierarchical Temporal Belief Transformer`
   - intermediate: `coarse_state_t`
   - final: `belief_state_t`
3. `policy_view_t adapter`

即：

- 单步先做模态内编码与单步证据提取
- 在单步层完成当前时刻的多模态一致性校正，形成：
  - `evidence_state_t`
  - `evidence_t`
- 再交给分层时序模块做两轮时序校正：
  - 第一轮：多模态时序校正
  - 第二轮：运动状态一致性时序校正
- 最后由 `policy_view_t adapter` 产出 RL 消费视图

这比“直接把视觉/音频 embedding 拼起来丢给 Transformer”更合理，因为：

- 单步几何与声学证据的职责不同
- 时序层不应该从第一层开始承担全部模态对齐负担
- 多模态校正与状态一致性校正并不是同一种时序任务
- 视觉优先、音频辅助的 inductive bias 可以在单步层和第一轮时序校正里明确写死

## 3. 当前模块职责共识

### 3.1 Visual Encoder

当前共识：

- 双相机分别做单相机几何估计
- 单相机内部采用：
  - 外观编码
  - 关键点检测
  - 语义分割
  - PnP 粗姿态估计
  - 几何验证
  - residual refinement
- 双相机之间当前采用择优，而不是软融合

当前推荐方向：

- 视觉分支可沿 `Pixel-Voting Net` 风格路线组织
- 这样可以在一个统一框架内同时支撑：
  - 关键点预测
  - 目标/背景语义分割
  - 几何姿态恢复
- 语义分割第一版可先采用低类别数设置：
  - 飞机（物体）/ 背景
- 但需评估是否应进一步区分：
  - 本机 / 敌机
  - 或至少在多目标/遮挡情况下保留敌机目标识别能力

视觉分支当前定位：

- 主导几何精确信息
- 在目标可见时提供最强证据
- 输出姿态/位置候选及其证据质量
- 语义分割结果还可辅助判断：
  - visibility
  - `raw_visual_evidence_strength`
  - 与音频侧结构证据的时序对齐关系

补充说明：

- 当前活动数据集中的 segmentation mask 已通过 `label / pack` 的防御性清洗保持干净
- 但 renderer 侧的 semantic capture 根修仍未完成
- 后续方向已经冻结为：
  - semantic variant 改为离散 class-id render path
  - 不再让普通颜色图像承担 segmentation label 语义
- 当前已落地的第一步是：
  - semantic activity export 主路径已改为直接导出 `Gray8` class-id artifact
  - 不再把颜色图像直接写入活动 segmentation artifact
- 当前单步视觉链还存在一个独立已知风险：
  - `keypoint_visibility` 的离线标签规则在低分辨率下会系统性失效
  - 在 `96x72` pack 上，`keypoint_visibility_front/rear` 可全量为 `0`
  - 在 `400x300` pack 上，该问题虽已缓解，但标签仍然偏稀疏
- 当前判断：
  - 该问题首先来自 `label.rs` 中：
    - `segmentation_matches && depth_matches`
    的 visibility 定义过严
  - 其中固定绝对深度阈值：
    - `KEYPOINT_DEPTH_EPSILON = 0.05`
    与低分辨率、远距离目标不成比例
  - `geometry.py` 中当前 `pnp_success_rate` 的定义偏宽，是第二层独立问题
  - 当前已开始收紧为：
    - 可见点数充足
    - 数值求解成功
    - 重投影误差达标
  - 两者需要分步修正：
    - 先修 visibility 标签规则
    - 再收紧几何成功定义
  - 在 `9.6.4` 之后，新的阶段判断已经明确：
    - 需要继续停留在单步链
    - 先把 visibility 失效拆成：
      - `segmentation_match`
      - `depth_match`
      可诊断的量
    - 再决定第二版 visibility 规则与训练稳定性修正
  - 当前 `9.7.1` 审计已确认：
    - visibility 失效的主来源更偏 `segmentation_match` 稀疏
    - 而不是 `depth_match` 系统性失败
  - 当前 `9.7.2` 已进一步确认：
    - `segmentation` 的单像素中心匹配确实过脆
    - `3x3` 邻域匹配能显著提升 `segmentation_match` 与最终 visibility
  - 当前 `9.7.3` 短训练复测已进一步确认：
    - `3x3` 邻域匹配能显著抬升 `raw_visual_evidence_mean`
    - 但严格 `pnp_success_rate` 仍为 `0.0`
    - `vision_keypoint_loss` 在最终 eval 样本上仍可能回到 `0`
    - 单步位置误差依旧很大，视觉几何链尚未恢复到可用状态
  - 当前 `9.7.4` 审计已进一步确认：
    - 现有 `single_step` 训练把 `vision_loss + audio_loss + evidence_loss` 直接相加
    - `evidence_loss` 长期在量级上压制视觉与音频分支
    - `doa_conf / dist_conf` 后段走低，首先应解释为几何误差过大与总损失失衡，而不是公式本身错误
    - 因此下一步应优先引入单步阶段级 loss 配重，再做短训练复测
  - 当前 `9.7.4` 第一轮配重实验已进一步确认：
    - 下调 `evidence_loss_weight` 到 `0.01` 后，视觉与音频分支不再被状态损失完全压制
    - `raw_visual_evidence_mean`、`dist_conf_mean`、`audio_position_l1`、`evidence_position_l1` 都出现了正向改善
    - 但严格 `pnp_success_rate` 仍为 `0.0`
    - 因此当前应保留阶段级 loss 配重，并继续围绕视觉几何监督与配重细化
  - 当前进一步的粗粒度扫描已确认：
    - `evidence_loss_weight = 0.01` 在 `0.005 / 0.01 / 0.02 / 0.05` 中综合最好
    - `0.005` 会把状态主任务压得过头
    - `0.02 / 0.05` 会让 `weighted_evidence_loss` 再次偏重
    - 因此后续默认应围绕 `0.01` 附近细化，而不再继续大跨度扫描
  - 当前新的活动主线已切到：
    - `Phase 9.8`
    - 即在保留 `evidence_loss_weight = 0.01` 的前提下，继续修正单步视觉监督链本身
  - 当前 `9.8.1` 已进一步确认：
    - 关键点监督失效并不是单一原因
    - 在 `400x300` 的活动 pack 可见样例上，标签侧仍然只提供极少量 visible keypoints
    - 这会直接导致：
      - `gt_pnp_usable = 0`
    - 在这种稀疏标签前提下，模型预测的 visible keypoints 还会进一步塌缩为 `0`
    - 因此当前视觉链的主问题已可拆成：
      - 标签侧可监督关键点数不足
      - 模型侧二值 visibility 预测进一步塌缩
    - 后续 `9.8.2` 应优先围绕：
      - 关键点位置监督如何不再完全依赖极少量最终 visible 点
      展开
  - 当前 `9.8.2` 已冻结的设计判断：
    - 当前 keypoint schema 虽然只有 `8 + center = 9` 个点，但“先增加关键点数量”不应作为第一刀
    - 现阶段更本质的问题是：
      - 当前单步视觉头仍是：
        - `segmentation + direct keypoint regression + binary visibility gating + PnP`
      - 关键点位置监督和 PnP 入口都被 strict visible 点数卡死
    - 因此当前正确顺序应是：
      - 先把单步视觉头改成更接近 PVNet 的：
        - `segmentation + dense pixel-wise voting head`
      - 再用聚合出的所有 keypoints 做 confidence-aware PnP
      - 最后再评估是否需要更密的 keypoint schema
    - 当前对 PVNet 论文 supervision 的收口是：
      - 只在目标前景像素上监督 voting field
      - 对每个前景像素、每个 keypoint 回归指向该 keypoint 的 2D unit vector
      - 先聚合 keypoint，再做 PnP
    - 当前实现边界进一步冻结为：
      - dense voting supervision 不在线构造
      - 而是在 `tool_dataset` 中离线预计算
      - 这样训练、评估、抽检与可视化可以共用同一份确定性 target
    - 当前 `9.8.2a` 已完成：
      - 单步视觉头已从 direct keypoint regression 切到 dense voting head
      - 当前 `keypoints_xy` 由 voting field 聚合得到
    - 当前仍未完成的是：
      - sparse foreground-only voting supervision 到 Python 数据链的接入
      - 以及基于该 supervision 的训练闭环
    - 当前执行顺序已进一步冻结为：
      1. 先补 `tool_dataset` 离线 dense voting supervision
      2. 再把当前简化聚合器升级到 RANSAC-style voting aggregation
    - 当前还新增了一个更上游的阻塞项：
      - `step 784` 的 raw segmentation bundle 已确认在小目标附近存在：
        - `fighter1 / fighter2` 混标
      - 该问题在：
        - `derived/fighter1/segmentation/front.frames`
        中就已经存在
      - 因此当前可以明确排除：
        - `label` / `pack`
        - inspect/export 工具颜色映射
      - 当前优先级应先提升：
        - semantic segmentation 生成链审计与修复
      - 当前第一版根因判断已收口为：
        - semantic camera 之前仍携带普通视觉相机的：
          - `DistanceFog`
          - 默认 `Msaa::Sample4`
        - 远距离小目标在 `0 / 2` 边缘会被渲染成中间值
        - 再经 `snapshot` 的最近类映射后，易落成伪 `class 1`
      - 当前第一版修复已落地并验证：
        - semantic camera 显式关闭 `Msaa`
        - semantic camera 移除 `DistanceFog`
      - 在真实 GPU extract 复测中：
        - `step 783/784/785` 的 raw segmentation 已不再在小目标附近产生伪 `class 1` 碎块
        - `class 2` 小目标面积恢复到更合理的 `15~19 px` 量级
      - 因此：
        - 这条上游 semantic 混标问题已解除
        - `selected-view-only embedding` 不再因该数据生成问题被阻塞
      - 后续仍需注意：
        - `selected-view` 当前依赖预测 segmentation
        - 其 early-stage instability 仍是模型结构风险，而非数据生成 bug
      3. 最后再评估是否继续做 uncertainty-driven PnP
    - 当前 `9.8.2b.1` 已完成：
      - dense voting supervision 的 schema / view contract 已补齐
      - `audit_dataset.py` 与 `inspect_dataset.py` 已能识别这些字段
    - 当前 `9.8.2b.2` 已完成：
      - `dfb_game label` 已离线写出 sparse foreground-only voting artifacts
      - 当前不再尝试把 dense `[H, W, K, 2]` tensor 直接写进单 episode pack
    - 当前 `9.8.2b.3` 已完成：
      - `pack` 已将 sparse artifact 整理为 chunk 内 padded sparse tensors
      - 当前活动契约改为：
        - `keypoint_voting_pixels_*`: `[P, 2]`
        - `keypoint_voting_unit_vectors_*`: `[P, K, 2]`
        - `keypoint_voting_mask_*`: `[P]`
      - `audit_dataset.py` / `inspect_dataset.py` 已切到 sparse 语义
    - 当前 `9.8.2b.4` 已完成：
      - `vision_supervision` 已接入 sparse voting loss
      - 训练时会在 sparse pixel slots 上从 dense voting field 采样，并对 unit-vector target 做监督
      - 当前 `keypoint xy` 与 `keypoint_visibility` 监督仍作为辅助项保留
    - 当前 `9.8.2c` 已完成：
      - 当前聚合器已从最小二乘聚合升级到 RANSAC-style voting aggregation
      - 每个 keypoint 现在都有：
        - 聚合后的 `keypoints_xy`
        - `keypoint_support`
      - 当前 PnP 入口已改成：
        - strict visible threshold 后再做 top-k 截断
      - 当前仍未进入：
        - covariance / uncertainty-driven PnP
      - 当前真实样例上：
        - `pnp_success` 仍为 `0.0`
      - 但 `v_sup` 与 `raw_visual_evidence_strength` 已保持非零
      - 说明当前过渡版视觉链已跑通：
        - dense voting
        - sparse supervision
        - RANSAC-style aggregation
        - simplified top-k PnP
    - 当前下一步已收口为：
      - 不立即进入 uncertainty-driven PnP
      - 先做 support-led 几何接口收口：
        - 移除 `confidence_head`
        - 让 `support` 直接接管 PnP 选点
        - 将：
          - `v_kp -> v_sup`
          - `v_geo -> v_rep`
      - 当前冻结语义为：
        - `v_sup = mean(support of selected PnP points)`
        - `v_rep` 继续表示 reprojection-based post-PnP 几何一致性
        - `raw_visual_evidence_strength = 0.5 * v_sup + 0.5 * v_rep`
    - 当前 `9.8.2d.1` 已完成：
      - 视觉头中的旧全局 `confidence_head` 已移除
      - `keypoint_visibility_loss` 已退出训练主线
    - 当前 `9.8.2d.2` 已完成：
      - `geometry.py` 已改成直接消费 `keypoint_support`
      - `PnP` 选点已改成：
        - support top-k
      - `front/rear_keypoint_visibility_logits` 这层过渡兼容壳已移除
      - 当前视觉几何主线已经不再依赖旧 visibility gating
    - 当前 `9.8.2d.3` 已完成：
      - `v_kp` 已改名为：
        - `v_sup`
      - `v_geo` 已改名为：
        - `v_rep`
      - 当前 support-led 几何中间量命名已经与主语义对齐
    - 当前 `9.8.2d.4` 已完成：
      - 已在真实 CUDA 环境下完成 `400x300` `single_step` `400 step` 短训练复测
      - `support-led` 几何接口已让 `raw_visual_evidence_strength` 保持稳定非零
      - 但最终：
        - `pnp_success_rate` 仍为 `0.0`
        - `audio_position_l1 / evidence_position_l1` 仍然很高
        - 后半程训练稳定性仍然一般
      - 因此当前结论是：
        - support-led 收口已完成
        - 但单步视觉主线下一步应转向 `selected-view` 与视觉时序差分设计
        - “救活 PnP” 已显式转为后续待办，而不再和当前 support-led 收口混在一起
    - 当前 `9.8.3.1` 已完成，`selected-view` 第一版语义冻结为：
      - 三态：
        - `front`
        - `rear`
        - `none`
      - 选择依据：
        - segmentation foreground 面积
      - `front <-> rear` 切换采用“惯性”规则：
        - 候选侧需连续 `k=3` 步面积更大才允许切换
      - `none -> front/rear`：
        - 不引入延迟
        - 有前景即进入具体视图
      - 当前输出只保留：
        - `selected_view`
        - `selected_view_onehot`
        - `selected_view_changed`
      - 第一版不引入：
        - `selected_view_score`
      - `selected_view = none` 允许主路径直接跳过：
        - voting
        - aggregation
        - PnP
    - 当前 `9.8.3.2` 已完成第一版实现：
      - `selected_view` 已显式输出：
        - `selected_view_index`
        - `selected_view_onehot`
        - `selected_view_changed`
      - `front/rear foreground area` 已显式输出，供后续审计使用
      - 上述面积语义现已修正为：
        - `front_pred_target_area`
        - `rear_pred_target_area`
      - 它们表示：
        - observed-role 对应的对手飞机语义类面积
      - 不再把整个非背景区域误算成目标面积
      - 同时需要明确区分：
        - `pred_target_area`
        - `gt_target_area`
      - 当前 `raw_visual_evidence_strength`
        - 已切成 selected-view 主语义
        - 不再简单取 front/rear 最大值
      - 但为了保留诊断能力：
        - front/rear 双路 voting / aggregation / geometry 仍然继续计算
      - 另外：
        - `k=3` 惯性切换由于当前模块无跨步状态，尚未在代码层实现
        - 当前代码只实现了当前 step 的三态 selection
    - 当前 `9.8.3.3` 已完成：
      - `visual_embedding` 不再由：
        - `front pooled + rear pooled`
        直接拼接得到
      - 当前改为：
        - `selected-view pooled feature + selected_view_onehot`
        再投影回固定 `128d`
      - 因此 selected-view 主语义已经真正进入视觉主 embedding
      - 同时保持：
        - evidence / temporal 模块的输入维度契约不变
    - 当前 `9.8.3.4` 已完成：
      - selected-view 版本的 smoke / 样例审计已通过
      - `visual_embedding`、`v_sup`、`v_rep`、`raw_visual_evidence_strength` 均可稳定产出
      - 但真实样例 `step 784` 也明确暴露：
        - `selected_view` 当前仍依赖预测 target-area
        - 当分割预测仍然很差时，selected-view 可能偏离 GT 直觉
      - 因此当前 selected-view 线的风险不再是“接口没接通”，而是：
        - prediction-driven routing 在早期训练阶段可能不稳
      - 这也解释了为什么下一步应继续补：
        - segmentation difference
        - keypoint delta
        再做短训练复测
    - 当前 `9.8.4.1` 已完成：
      - selected-view segmentation difference 已接入时序模态模块
      - 仅在当前步与上一时刻：
        - 都选中了 `front/rear`
        - 且 view 相同
        时产出非零差分
      - 输出包含：
        - `selected_segmentation_difference_t [B,T,2,H,W]`
        - `selected_segmentation_diff_valid_t [B,T]`
      - 该差分当前只作为视觉时序辅助 cue
      - 不进入单步 evidence 主链
    - 当前 `9.8.4.2` 已完成：
      - selected-view keypoint delta 已接入时序模态模块
      - 仅在当前步与上一时刻：
        - 都选中了 `front/rear`
        - 且 view 相同
        时产出非零 keypoint delta
      - 输出包含：
        - `selected_keypoint_delta_t [B,T,K,2]`
        - `selected_keypoint_delta_support_summary_t [B,T]`
        - `selected_keypoint_delta_valid_t [B,T]`
      - 该差分当前也只作为视觉时序辅助 cue
      - 不进入单步 evidence 主链
    - 当前 `9.8.5` 已完成一轮 clean-pack 短训练复测：
      - selected-view
      - segmentation difference
      - keypoint delta
      都已在 `temporal_modality` 主线上跑通
      - 但短训结果尚未显示稳定收益
      - `position_loss` 仍大且在中后段回弹
      - 因此下一步不应继续扩展时序结构，而应优先回到：
        - segmentation-only 训练 / 审计
        - `PnP` 复活
        - 或 selected-view prediction-driven routing 稳定性审计
    - 当前对 `PnP` 主阻塞的判断已修正为：
      - segmentation 不稳定是更上游的问题
      - 在 prediction-driven routing 条件下：
        - selected-view 会漂移
        - voting foreground 会失真
        - support 难以稳定
      - 因此更合理的顺序是：
        - 先单独训练 segmentation
        - 再回头救活 voting / PnP
    - 当前已新增 segmentation-only 恢复线：
      - 第一版只训练 `front/rear` segmentation
      - 先不训练 keypoint / voting / audio / evidence / temporal
      - 训练入口已实现为独立 `segmentation_only` stage
      - 已完成 clean pack 上的最小 smoke，日志 / checkpoint / eval snapshot 正常
      - 已完成 `200 step` 短训：
        - segmentation CE 明显下降
        - 但目标类 `precision/recall` 仍为 `0`
        - `selected_view_agreement` 仍为 `0`
      - 当前判断：
        - 仅靠 segmentation-only 继续拉长训练，不足以救活 routing / PnP
      - 当前执行策略已调整为：
        - 单步视觉前端优先走合成视觉数据线
        - 真实飞行数据暂时退到后续筛选 / 清洗 / 分布校正
      - 合成视觉数据线第一版采样语义已冻结为：
        - 以目标像素面积、图像位置、视角可见性、目标姿态为主变量
        - 明确向远处小目标倾斜
        - 目标姿态改为连续均匀采样，而不是固定离散姿态桶
        - 小目标更多，但中目标、大目标仍需保留稳定配额，避免只学极难分布
      - 先看 segmentation 本身是否足以显著改善：
        - target-class IoU
        - selected-view 一致率
        - `front/rear/none` 切换稳定性

### 3.2 Audio Encoder

当前共识：

- 输入将从“内部空间音频渲染结果”中导出的智能体可听觉观测
- 输出不追求完整几何姿态
- 更偏：
  - 位置/方向粗提示
  - 油门/推力状态提示
  - 音频质量

音频分支当前定位：

- 精度低于视觉
- 但在视觉缺失时具连续性
- 更像 motion capability / rough localization 证据

当前与 `dfb_game` 的音频边界先冻结为三层：

1. `dfb_game` 内部空间音频场景
- 负责：
  - listener pose
  - emitter
  - 声源空间关系
  - 衰减 / 方位 / 运动相关语义
- 不把内部主语义绑定为固定 7.1 布局

2. 播放设备渲染层
- 负责把内部空间音频场景映射到具体设备输出：
  - headphones / binaural
  - stereo
  - 7.1
- 它是渲染适配层，不是训练观测契约

3. Part 2 数据集导出观测层
- 负责导出给智能体使用的“可听觉观测”
- 它默认不再与播放设备布局同义
- 当前专项重构方向已经冻结为：
  - 不再把 7.1 全通道直接作为 Part 2 主观测
 - 当前新的 canonical 主观测命名已冻结为：
   - `audio_window_binaural`
 - 内部主语义以“飞行员双耳”解释为准
 - 对外写作时允许扩展解释为：
   - 低成本双通道采样系统
 - 但实现与字段命名不再使用 `stereo` 作为 canonical 语义
 - 当前通道顺序固定为：
   - `left`
   - `right`
 - canonical 采样率当前冻结为：
   - `48_000 Hz`
 - 当前明确约束：
   - `audio_window_binaural` 来自内部空间音频场景的双耳渲染结果
   - 它不是播放设备格式本身
   - 也不是把内部 7.1 简单下混后的训练观测

这意味着当前之后的设计与实现都应遵守：

- `dfb_game` 可以保留完整空间音频语义
- 训练数据集不必直接暴露完整播放布局
- Part 2 音频模块的输入契约应由“智能体可听觉观测”定义，而不是由游戏内播放设备布局定义

补充方向：

- 音频分支后续可考虑引入“音频语义分割”或等价的结构化分解
- 其目的不是复原完整语义标签，而是形成：
  - 可随时间追踪的声源结构证据
  - 可与视觉分割结果做时序对齐/重叠比较的摘要
- 这类结构证据未来可在多模态时序校正阶段辅助判断跨模态对齐与冲突

### 3.3 Single-Step Evidence Extraction

当前共识：

- 单步层负责模态内编码、几何/声学粗估计与 evidence extraction
- 在进入时序层前，先完成当前时刻的单步多模态一致性校正
- 权重估计必须内置“视觉优先”偏置
- 还要显式做跨模态一致性检查

这一层的输出不应再是原始 observation，而应是：

- `evidence_state_t`
  - 单步多模态一致性校正后的粗估计
- `evidence_t`
  - embeddings、质量分数与其他供时序模块使用的证据块

这一层的职责是：

- 产出适合进入时序模块的当前单步状态粗估计与证据块
- 但不直接承担完整的时序 belief 更新职责

当前第一版字段冻结结果：

`evidence_state_t`

- `relative_position: Vec3`
- `relative_orientation: rotation_6d`
- `position_confidence: Scalar`
- `orientation_confidence: Scalar`

`evidence_t`

- `visual_embedding`
- `audio_embedding`
- `raw_visual_evidence_strength: Scalar`
- `raw_audio_evidence_strength: Scalar`

当前设计说明：

- `evidence_state_t` 负责承载单步层已经完成一致性校正后的粗状态骨架
- `evidence_t` 负责承载第一轮时序模态校正所需的主要证据块
- `track_confidence` 不再属于单步显式监督字段，它是最终 belief 层的高层运行时信号
- 第一版不把过多手工几何摘要、单步 cross-modal consistency 标量或 modality dominance 标量塞进 `evidence_t`
- 这部分更适合保留给时序层自己从 embeddings 与 reliability 中学习或在后续扩展中引入
- 当前新增收口：
  - 单步视觉分支提供：
    - 完整 `relative_position`
    - 完整 `relative_orientation`
    - `position_confidence`
    - `orientation_confidence`
  - 单步音频分支提供：
    - `doa_unit_vector_body`
    - `log_distance_scalar`
    - `doa_conf`
    - `dist_conf`
  - 音频这套输出不被视为完整状态，而被视为“位置几何的弱观测”
  - 单步融合层第一版应先将音频极坐标解释到与视觉位置一致的坐标语义下，例如：
    - `audio_relative_position = doa_unit_vector_body * exp(log_distance_scalar)`
  - 然后用：
    - 音频 `audio_relative_position + doa_conf/dist_conf`
    - 视觉 `relative_position + position_confidence`
    做一轮位置级融合
  - 最终：
    - 融合后的 `relative_position + position_confidence`
    写入 `evidence_state_t`
  - `relative_orientation + orientation_confidence` 第一版仍由视觉分支主导

当前关于 evidence strength 的新增收口：

- 单步阶段先计算：
  - `raw_visual_evidence_strength`
  - `raw_audio_evidence_strength`
- 它们表示“仅基于当前步证据”的瞬时模态证据强度
- 它们作为当前步 token 输入第一轮时序模态校正阶段
- 第一轮时序模态校正阶段会结合历史差分与当前结构证据，对其进行校正，得到：
  - `visual_evidence_strength`
  - `audio_evidence_strength`
- `raw_*_evidence_strength` 与 `*_evidence_strength` 都是运行时中间量，不作为离线数据集主监督 target
- 校正后的 `*_evidence_strength` 用于 belief 层和 `track_confidence` 的运行时更新，并写回历史状态/记忆
- 第一版不要求第一轮显式产出大量新的模态 meta 量，显式校正后的证据强度是其主要附加输出之一

后续强调的扩展方向：

- 视觉分割结果与音频结构分解结果，可进一步形成 segmentation-derived evidence
- 例如：
  - visual segmentation summary
  - audio segmentation summary
  - segmentation overlap / alignment evidence
- 这类证据尤其适合进入第一轮时序模态校正阶段
- 但第一版先不冻结其具体字段形式

当前关于视觉结构差分的新增收口：

- 第一版不优先使用 raw segmentation image 差分作为主要时序证据
- 视觉侧更优先采用：
  - keypoints 在图像上的位移变化

当前 synthetic 视觉恢复线的新增约束：

- synthetic 线允许使用与真实数据线不同的 semantic label 试作规则
- 该规则当前只用于 synthetic calibration / audit
- 不回写 canonical segmentation GT
- 当前已落地一个 `msaa-aware semantic label` 试作版：
  - semantic render 可启用 `MSAA`
  - 样本同时导出 strict / trial 两套分割结果
  - `strict` / `trial` 的差异只允许来自逐像素调色板解码策略
  - 不允许通过连通域合并、形态学后处理或其他空间 relabel 规则修补 target 污染
  - synthetic 默认导出只保留活动标签版本
  - `strict / trial` 对比产物仅在显式审计开关下导出
- synthetic 观察机 / 敌机位姿当前已从固定姿态切换为受约束随机采样：
  - 保持高于地面
  - 低于 flight ceiling
  - 不越出 arena
  - 不进入 box obstacles
- synthetic 样本导出除了按 `sample_xxxxxx/` 组织外，还提供扁平目录：
  - `rgb/`
  - `seg_color/`
  - `seg_color_strict/`
  - `seg_color_trial/`
  - `metadata/`
- 当前 `msaa_trial` 是试作版：
  - 其目标是把受 target `MSAA` 覆盖影响的像素纳入 target class
  - 当前接受的实现方向是：
    - semantic palette render
    - target-biased palette decode
  - 它用于 synthetic 审计，不是 canonical GT 规则
  - keypoint confidence / visibility mask
- 当前 synthetic 采样校准的较优基线已更新为：
  - `synthetic_visual_prototype_v4b_msaa_trial`
  - 其特征是：
    - `None` 比例已压低到极低
    - `px200_plus` 显著减少
    - `px5_to99` 中间桶开始形成可用覆盖
- 这样可以以更低维、更结构化的方式表达目标在图像平面中的运动趋势
- 只要关键点整体变化大致符合同一趋势，就足以构成很强的时序校正证据
- 语义分割结果仍然有价值，但第一版更适合作为：
  - 目标/背景可见性判断
  - `raw_visual_evidence_strength` 的辅助依据
  - 后续扩展的结构证据来源

### 3.4 Hierarchical Temporal Belief Transformer

当前共识：

- 时序层仍然以 Transformer 为主干
- 但采用分层、分阶段的处理语义，而不是单一黑箱序列编码
- 同一个分层时序模块内部包含两轮时序校正：
  - 第一轮：多模态时序校正
  - 第二轮：运动状态一致性时序校正
- 中间显式产出 `coarse_state_t`
- 最终显式产出 `belief_state_t`

时序层职责：

- 处理短时失视
- 在时间轴上校正单步模态冲突
- 维持运动连续性
- 区分真实机动与观测噪声
- 形成可供 Part 3 使用的稳定状态表达

当前优先接受的第一版结构是：

1. `Temporal Modality Calibration Stage`
   - 输入：当前 `evidence_state_t` + `evidence_t` 与历史对应序列
   - 输出：`coarse_state_t`
2. `Temporal Belief Update Stage`
   - 输入：历史 belief 序列 + 当前 `coarse_state_t`
   - 输出：`belief_state_t`

其中需要特别强调：

- 第一版时序层在实现上仍然采用单个 Transformer 主干
- 但该主干内部按阶段组织，而不是两个完全独立的大 Transformer
- 可以理解为一个 `Hierarchical Temporal Belief Transformer`
- 它在同一 backbone 内保留“模态时序校正”和“belief 时序更新”两种处理语义

也就是说，当前主线不再是“纯 Transformer 直接回归当前状态”，而是：

```text
single-step evidence
-> temporal modality calibration
-> coarse_state_t
-> temporal belief update
-> belief_state_t
-> policy_view_t
```

其中：

- 第一轮时序校正的目标是：利用历史 evidence 对当前模态冲突进行时间域校正
- 第二轮时序校正的目标是：利用历史 belief 对当前状态进行运动一致性校正
- `coarse_state_t` 是第一轮时序校正后的中间粗状态
- `belief_state_t` 是第二轮时序校正后的最终内部 belief
- `policy_view_t` 是最终给 RL 的消费视图

第二轮时序校正的当前新增共识：

- `coarse_state_t` 与 `belief_state_t` 第一版完全同形
- `coarse_state_t -> belief_state_t` 的更新不只依赖当前粗状态本身
- 还应显式利用状态差分与派生运动差分作为重要校正依据
- 第二轮 token 的状态块当前冻结为：
  - `relative_position`
  - `relative_orientation`
  - `position_confidence`
  - `orientation_confidence`
  - `track_confidence`
- 第二轮 token 的差分块当前冻结为：
  - `delta_position`
  - `delta_orientation`
  - `linear_velocity`
  - `angular_velocity`
- 其中：
  - `delta_orientation` 表示当前姿态相对上一时刻姿态的相对旋转
  - 内部姿态表示统一采用 `rotation_6d`
  - `angular_velocity` 表示由相邻姿态相对旋转通过 `Log(ΔR) / dt` 得到的角速度向量
- 第一版这些差分量不作为独立 token 类型，而是并入每个时间步的 belief-update token
- 也就是说，第二轮第一版采用：
  - 每个时间步保留当前状态块
  - 每个时间步同时保留相对前一时刻的局部状态差分块
- 这样与第一轮“每步都带局部结构差分”的设计原则保持一致
- 在 Part 2 完成详细设计并评估计算量时，若需要进一步压缩复杂度，可统一回退为：
  - 历史步骤只保留状态块
  - 仅在最后一步显式提供 delta 块
  - 或进一步移除 `angular_velocity`，仅保留一阶旋转差分 `delta_orientation`

第一轮 `Temporal Modality Calibration Stage` 的 token 语义当前已冻结为：

1. `state_token_t`
   - 语义：当前时刻单步层已经做过一致性校正后的粗状态骨架
   - 字段：
     - `relative_position: Vec3`
     - `relative_orientation: rotation_6d`
     - `position_confidence: Scalar`
     - `orientation_confidence: Scalar`

2. `visual_token_t`
   - 语义：当前时刻视觉侧的结构化证据与质量信息
   - 字段：
     - `visual_embedding`
     - `kp_summary_t`
     - `kp_delta_t`
     - `kp_visibility_mask_t`
     - `kp_confidence_t`
     - `raw_visual_evidence_strength: Scalar`

3. `audio_token_t`
   - 语义：当前时刻音频侧的结构化证据与质量信息
   - 字段：
     - `audio_embedding`
     - `binaural_energy_t`
     - `binaural_cue_vector_t`
     - `delta_binaural_cue_t`
     - `raw_audio_evidence_strength: Scalar`

第一轮 token 设计的当前原则：

- 每个时间步都保留原始结构摘要
- 每个时间步都保留相对前一时刻的局部差分
- 序列首帧差分默认置零
- `visibility` 与 `confidence` 分开保留，不合并
- `binaural_energy_t` 与 `binaural_cue_vector_t` 都保留为显式结构摘要，避免音频证据完全塌缩为单个 embedding
- 当前按“每个时间步都带局部 delta”冻结
- 在 Part 2 完成详细设计并评估计算量时，若需要进一步压缩复杂度，可统一回退为：
  - 仅在最后一步显式提供 delta
  - 历史步骤只保留原始结构摘要
- 时间相关输入当前冻结为：
  - `dt_to_prev`
  - `time_from_now`
  - `has_prev_step`
- 其中：
  - `dt_to_prev` 用于表达局部相邻时间差
  - `time_from_now` 用于表达该 token 距当前时刻的远近
  - `has_prev_step` 用于区分“真实零差分”和“窗口首帧无前驱”

时序窗口与时间编码的当前冻结结果：

- 第一轮与第二轮先统一使用相同窗口：
  - `T1 = T2`
- 窗口不假设严格固定步长，而采用：
  - `window_seconds`
  - `max_steps`
  两层约束共同决定
- 实际输入为最近 `window_seconds` 时间范围内、最多 `max_steps` 个时间步
- 第一版默认结构长度先取：
  - `max_steps = 32`
- `window_seconds` 作为超参数保留
- “飞机完成一圈飞行的最短时间”可作为 `window_seconds` 调参时的经验上界参考
- 当前时间编码以相对时间语义为主，而不是依赖绝对位置编码

补充约束：

- 当信息不足以支撑可靠状态估计时，Part 2 不输出随机状态
- 也不引入离散的 “unknown token”
- 当前策略是：
  - 输出基于历史 belief 延续或保守更新得到的状态
  - 同时通过 confidence / uncertainty 表达“当前知道得不够”
- 也就是说，信息不足时输出的是退化的保守 belief，而不是随机数值状态

## 4. 当前仍未钉死、但必须尽快收口的主契约

这两份设计稿方向正确，但如果要继续推进到可实现方案，当前最需要补的是下面这些主契约。

### 4.1 最终输出状态的字段定义

目前文档里已经明确了“belief state”的概念，但还没有完全冻结：

- 具体输出哪些状态量
- 各状态量属于哪种坐标系
- 哪些量是主监督目标
- 哪些量只是辅助量

这一项必须收口。

当前建议优先把输出拆成三层：

1. 几何主状态
   - 相对位置
   - 相对朝向
2. 不确定性与证据状态
   - visibility
   - confidence / uncertainty
   - source dominance
3. 派生运动特征
   - 由相邻时刻 belief 差分得到的速度/趋势量
   - 不作为第一版 `belief_state_t` 主预测目标

### 4.2 坐标系与表示方式

当前文档里默认在谈“位置/姿态”，但没有完全写清：

- 世界坐标还是自机坐标
- 绝对姿态还是相对姿态
- 旋转表示是欧拉角、矩阵、四元数还是 6D rotation representation

如果这一点不先定，后面的：

- 监督标签
- loss
- temporal delta
- Part 3 接口

都会漂。

当前建议：

- Part 2 对外主表示优先使用“自机参考系下的敌机相对状态”
- 内部视觉分支可继续使用适合 PnP 的姿态表示
- 对外最终输出应选一个更适合训练和时序建模的统一表示

当前已新增的收口共识：

- 视觉分支的原始几何输出可以位于相机参考系
- 但 `evidence_state_t`、`coarse_state_t` 与 `belief_state_t` 的标准坐标系统一采用“自机参考系下的敌机相对状态”
- 因此，相机系到自机系的主数值变换属于 Part 2 内部的 canonicalization 过程
- 不把这一主变换拖到 `policy_view_t` 或后置 adapter 层再做
- 这样时序层中的 `prediction / mismatch / correction` 才能在统一参考系下工作

当前新增的重要实现前约束：

- 当前 Part 1 实现中，飞机对象空间与相机局部空间的“前向”语义不必强行统一
  - 飞机对象空间当前冻结为：
    - `+Z` 前向
    - `+X` 左向
    - `+Y` 上向
  - 相机局部系继续遵循图形相机习惯，前向等价于 `-Z` 朝屏幕内
- Part 2 实现前必须先完成飞机对象空间与资产、左右机翼标签、运行时映射的一致化
- 当前冻结的目标语义为：
  - 飞机对象空间采用右手系
  - 前向为 `+Z`
  - 左向为 `+X`
  - 上向为 `+Y`
- 只有在这一语义统一完成后，才应正式实现：
  - 相机外参
  - 相对位姿 canonicalization
  - 关键点 / 分割 / 几何恢复的最终坐标接口

因此，Part 2 后续所有与姿态、位姿、外参、相对状态相关的设计，统一以以下目标语义为准：

- 世界空间：
  - `+Y` 向上
  - 水平面为 `XZ`
- 飞机机体空间：
  - `forward = +Z`
  - `left = +X`
  - `up = +Y`
- 相机局部空间：
  - `forward = -Z`
  - `right = +X`
  - `up = +Y`
- 图像空间：
  - 原点在左上角
  - `x` 向右
  - `y` 向下

当前姿态表示收口为：

- `relative_position` 第一版采用 3D Cartesian representation
- `relative_orientation` 第一版内部统一采用 `rotation_6d`
- 需要时再通过解码器还原为 rotation matrix / quaternion / 其他外部需要的形式
- 文档中应区分：
  - `6DoF pose`
    - 表示位置 + 定向自由度
  - `rotation_6d`
    - 表示内部神经网络使用的旋转参数化

### 4.3 单步校正层到底输出“候选”还是“单值粗估计”

当前文档中的单步粗估计/中间粗状态仍然偏向单值表示。

但从状态估计角度，还需要尽快决定：

- 单步层是否只输出一个粗估计
- 还是输出小规模候选集 + 权重

这会直接影响：

- 时序层复杂度
- 不确定性表达方式
- Part 3 的消费方式

当前建议先保守：

- 单步层输出单值粗估计
- 同时输出置信度/不确定性
- 不在第一版把整个系统做成 mixture-of-hypotheses

### 4.4 不确定性是如何表达的

“带置信度”已经是共识，但还需要定义格式。

第一版建议不要过度复杂化，优先选择可训练、可解释的表达：

- 每个主状态块的 confidence
- 视觉/音频证据质量
- 当前 source dominance
- 上次可靠视觉观测距今的时间

而不是第一版就上完整概率滤波分布参数化。

补充共识：

- `modality_dominance` 第一版优先使用连续值表示，而不是离散类别
- 例如可用 `[0, 1]` 或 `[-1, 1]` 范围来表达当前证据主导性
- 这样更利于训练、分析和时序平滑

### 4.5 时序层窗口和记忆语义

当前只写了“最近若干步”，但对实现来说需要进一步钉死：

- 近历史窗口长度
- 是否需要显式 memory token
- 是否只建模 fixed-length context
- 是否加入时间间隔编码

第一版建议：

- 先做固定长度窗口
- 明确包含 delta 特征
- 不急着引入额外复杂长记忆机制

补充共识：

- `time_since_last_visual_lock` 更适合作为 side-channel 输入，而不是模型输出
- 它的主要用途是帮助更新逻辑判断“当前应更信历史预测还是更信当前观测”
- 不建议第一版把它当作需要显式监督的结果量

### 4.6 速度项的语义

当前已经进一步收口：第一版不把速度项放进 `belief_state_t` 主契约。

建议：

- 位置与朝向仍然是主监督对象
- 速度、角运动等导数量优先由相邻时刻 belief 差分显式派生
- 第一版不把“直接速度回归”作为主路径
- 如后续验证有必要，可再把相关运动量引入 auxiliary head 或 adapter 附加特征

### 4.7 训练监督与数据来源

当前最重要的工程优势在于：

- Part 1 已有 authoritative environment
- authoritative recording 已冻结
- replay 与 dataset 导出链已可工作

因此 Part 2 的监督源不应模糊：

- 主监督目标来自 authoritative world state
- 多模态输入来自 recording / reconstruction / dataset

需要进一步明确的是：

- 训练样本粒度
- 输入窗口定义
- 标签时间对齐方式
- visual/audio 对齐策略

当前新增的监督设计共识：

- `evidence_state_t`、`coarse_state_t`、`belief_state_t` 都接受显式监督
- 分阶段监督强度按 coarse-to-fine 递增：
  - `λ_evidence < λ_coarse < λ_belief`
- 也就是说：
  - 越早期、越粗糙的估计，允许更高误差
  - 越后期、越接近最终输出的估计，监督越强
- `position_confidence` 与 `orientation_confidence` 在各阶段都做辅助监督
- confidence 监督的总权重应额外乘上独立的 `λ_conf`
- 第一版建议将误差映射到 `[0, 1]` 区间作为 confidence target
- 具体可采用单调递减映射，例如：
  - `target_conf = exp(-α * error)`
- 第一版建议对 confidence 使用连续回归监督，而不是离散分类监督
- 当前进一步补充的音频单步监督共识：
  - 单步音频分支不再只提供运行时粗几何输出
  - 还需要显式监督：
    - `doa_unit_vector_body`
    - `log_distance_scalar`
    - `doa_conf`
    - `dist_conf`
  - 监督源直接来自 authoritative `gt_relative_position`
  - 第一版对称参考视觉侧 confidence 设计：
    - `gt_doa_unit_vector_body = normalize(gt_relative_position)`
    - `gt_log_distance_scalar = log(||gt_relative_position|| + eps)`
    - `target_doa_conf`
    - `target_dist_conf`
      由对应误差单调映射在线构造
  - 当前正式冻结的 confidence target 公式为：
    - `e_doa = arccos(clamp(dot(pred_doa_unit, gt_doa_unit), -1, 1))`
    - `e_dist = |pred_log_distance - gt_log_distance|`
    - `target_doa_conf = exp(-ln2 * (e_doa / doa_half_angle)^2)`
    - `target_dist_conf = exp(-ln2 * (e_dist / dist_half_error)^2)`
  - 当前参数语义：
    - `doa_half_angle`
      - 单位为弧度
      - 表示方向误差达到该量级时，`doa_conf = 0.5`
    - `dist_half_error`
      - 定义在 `log-distance` 空间
      - 表示距离误差达到该量级时，`dist_conf = 0.5`
  - 当前分层语义明确为：
    - `doa_conf / dist_conf`
      - 只表示音频粗几何头自身置信度
    - `pos_fusion_conf`
      - 表示视觉/音频位置融合后的最终单步状态置信度
    - 两者不混用，也不共享 target
  - 当前第一版继续保留：
    - `audio_position_confidence = sqrt(doa_conf * dist_conf)`
  - 当前判断：
    - 该定义仍符合“任一音频几何子头不可靠，就应显著压低音频位置权重”的融合语义
    - 在单步音频几何监督补齐后，它不再是鼓励塌缩的无监督门控
  - 当前冻结的单步融合顺序为：
    1. 视觉先给出：
       - `pos_vis`
       - `pos_vis_conf`
    2. 音频先给出：
       - `doa`
       - `doa_conf`
       - `dist`
       - `dist_conf`
    3. 再将音频极坐标解释为位置语义
    4. 然后先完成位置级融合：
       - `pos_fusion`
       - `pos_fusion_conf`
    5. 最后再进入整个单步 evidence / state 的总损失
  - 当前额外记录的风险项：
    - 单步视觉分支当前输出的 `pos_vis` 仍主要是隐式学习得到的 body-frame 结果
    - 后续应在进入融合层前补上：
      - 显式相机几何恢复
      - `camera -> aircraft_body`
        的坐标变换
    - 再将显式 body-frame 视觉位置送入融合层

当前关于 confidence target 的具体草案：

- `e_pos`
  - 采用位置欧氏距离误差
- `e_ori`
  - 内部姿态表示采用 6D rotation representation
  - 但姿态误差仍使用几何正确的旋转角误差
- `position_confidence` / `orientation_confidence` 的 target 采用动态尺度版本：

```text
target_pos_conf = exp(-ln2 * (e_pos / pos_scale_t)^2)
target_ori_conf = exp(-ln2 * (e_ori / ori_scale_t)^2)
```

其中：

- `linear_velocity = Δp / dt`
- `angular_velocity = Log(ΔR) / dt`
- 第一版建议：

```text
pos_scale_t = max(pos_floor, dt * k_v * ||linear_velocity_t||)
ori_scale_t = max(ori_floor, dt * k_r * ||angular_velocity_t||)
```

当前第一版初始建议：

- `k_v = 1.0`
- `k_r = 1.0`
- `pos_floor`
  - 采用相对典型单步位移的经验下限
  - 第一版经验取值为 `0.3`
- `ori_floor`
  - 采用相对典型单步角变化的经验下限
  - 第一版经验取值为 `0.1`

说明：

- `pos_floor` 与 `ori_floor` 当前都作为经验超参数记录，而非理论常数
- 当前共识是：
  - 位置变化在固定翼场景中难以长期接近零，因此位置误差应保持较强敏感度
  - 姿态变化在巡航等状态下可长期较小，因此姿态误差应给予更宽容下限

当前关于“原始信息不足”的最新收口：

- `track_confidence` 仍保留，其语义仍是：
  - 系统当前是否仍然对敌机保持有效跟踪
- 但它不再作为离线规则 target，也不再要求精确监督值
- 第一版采用方案 A：
  - `track_confidence` 视为最终 belief 层与 `policy_view_t` 中的重要运行时强弱信号
  - 不构造 `target_track_conf`
  - 不要求单步层或第一轮时序层显式输出可监督的 `track_confidence`
- 当前字段约束改为：
  - `track_confidence` 不进入 `evidence_state_t`
  - `track_confidence` 不进入 `coarse_state_t`
  - `track_confidence` 保留在 `belief_state_t`
  - `track_confidence` 保留在 `policy_view_t`
- 运行时更新仍可参考“衰减 + 刷新”的思想，但这属于模型内部中间逻辑，不再写成离线标签契约

当前关于 `visual_evidence_strength` / `audio_evidence_strength` 的最新收口：

- 这两个量都不再作为离线规则 target
- 它们是运行时中间量，服务于：
  - 第一轮时序模态校正
  - belief 层内部更新
  - `track_confidence` 的运行时强弱更新

`visual_evidence_strength_t` 当前只保留两项运行时分量：

- `v_kp`
  - 关键点可见性 / 关键点质量对应的运行时量
- `v_geo`
  - 由重投影误差映射得到的运行时几何量
  - 当前建议形式：

```text
v_geo = exp(-ln2 * (e_reproj / e_half)^2)
```

第一版运行时组合为：

```text
visual_evidence_strength_t = 0.5 * v_kp + 0.5 * v_geo
```

说明：

- `v_visible` 已从该组合中移除
- `v_geo` 不再作为离线标签导出，而是留给单步视觉模块与运行时 PnP/重投影链路计算
- `kp_delta` 等时序结构证据仍保留在第一轮 token 中，由时序模态模块直接处理

当前进一步冻结的运行时计算链：

1. 单步视觉模块先输出：
   - segmentation prediction
   - keypoint prediction
   - 单步 PnP / 姿态恢复所需几何中间量
   - reprojection error `e_reproj`
2. 然后在单步视觉模块内部构造：
   - `v_kp`
     - 由关键点可见比例、关键点质量或其轻量聚合量得到
   - `v_geo`
     - 由重投影误差映射得到：

```text
v_geo = exp(-ln2 * (e_reproj / e_half)^2)
```

3. 再得到：

```text
raw_visual_evidence_strength
= 0.5 * v_kp + 0.5 * v_geo
```

4. 第一轮时序模态校正阶段读取：
   - `visual_embedding`
   - `raw_visual_evidence_strength`
   - `kp_delta` 等结构化时序证据
5. 第一轮输出校正后的：
   - `visual_evidence_strength`
6. `visual_evidence_strength` 不直接作为监督目标，而用于：
   - belief 层内部更新
   - `track_confidence` 的运行时强弱更新
   - 写回历史记忆

当前约束：

- `v_kp / v_geo / raw_visual_evidence_strength / visual_evidence_strength` 都是运行时中间量
- 它们不进入离线 dataset 主 schema
- `e_half` 是运行时配置超参数，不是预计算统计量
- `e_reproj` 当前正式定义为：
  - 单相机 PnP 解出的姿态将 canonical 3D keypoints 重投影回图像后，
    与预测 2D keypoints 的可见性加权平均像素误差
- 运行时 `v_geo` 定义为：

```text
v_geo = exp(-ln2 * (e_reproj / e_half)^2)
```

- `ADD / ADD-S` 不参与运行时 `v_geo` 定义
- `ADD / ADD-S` 只保留给后续离线 pose 评估与分析

当前关于视觉分割类别的新增收口：

- 第一版视觉分割类别采用：
  - `fighter1`
  - `fighter2`
  - `background`
- 这比简单二分类更适合双机狗斗场景
- 也更有利于：
  - 目标身份区分
  - 遮挡/交叠情况下的可见性判断
  - `keypoint_visibility` 的像素一致性判定

当前关于关键点集合的新增收口：

- 第一版关键点集合采用固定集合，不在训练中动态变化
- 第一版不采用 3D bounding box corners
- 关键点选取工具参考 `PVNet` 风格思路：
  - 优先选择位于物体表面的点
  - 同时尽量最大化点间距/覆盖机体几何范围
- 工具应支持：
  - 从飞机 mesh 表面采样候选点集
  - 先加入 `object_center`
  - 再在表面候选点上执行 farthest point sampling (FPS)
  - 自动选点后人工微调
- `k` 的语义明确为：
  - 仅表示表面 keypoint 数量
  - `object_center` 额外单独保留
- 第一版默认建议：
  - `FPS 8 + center`
- 备选增强配置：
  - `FPS 12 + center`
- 说明：
  - `FPS 8 + center` 是结合几何恢复、时序关键点差分和 token 维度后的保守工程起点
  - 它不是 PnP 的理论最小需求
  - 若后续验证发现遮挡鲁棒性或姿态稳定性不足，可升至 `FPS 12 + center`
- 应输出固定 keypoint schema 文件，至少记录：
  - `schema_id`
  - `model_id`
  - `selection_method`
  - `surface_keypoint_count`
  - `has_object_center`
  - `points_3d_object`
  - `point_labels`
  - 可选的人工微调记录
- 当前 canonical v1 已落地为：
  - [fighter_surface_fps8_plus_center_v1.json](/run/media/ayano/SharedProjects/general_projects/dfb/config/dfb_state_estimation/keypoints/fighter_surface_fps8_plus_center_v1.json)
- 这份 schema 当前绑定：
  - `model_id = fighter_plane_v1`
  - `coordinate_convention_id = dfb_aircraft_body_rhs_forward_pos_z_v1`
  - 默认 `8` 个表面点加 `object_center`
- 当前 v1 点集以现有 `fighter_plane.gltf` 的 node translation 与 mesh bounds 为锚点固化
- 后续若引入专门的 surface FPS 选点工具，应保持 schema id 语义稳定，避免训练与数据导出引用漂移

`audio_evidence_strength_t` 的当前运行时分解建议为：

- 当前 active binaural cue schema 已落地为：
  - [binaural_cue_schema_v1.json](/run/media/ayano/SharedProjects/general_projects/dfb/config/dfb_state_estimation/audio/binaural_cue_schema_v1.json)
- 当前 active runtime evidence 配置已落地为：
  - [binaural_runtime_evidence_v1.json](/run/media/ayano/SharedProjects/general_projects/dfb/config/dfb_state_estimation/audio/binaural_runtime_evidence_v1.json)
- 当前预计算结构特征已经切换为：
  - `binaural_energy_t`
  - `binaural_cue_vector_t`
- `a_energy / a_cue` 不作为离线标签导出
- 第一版运行时定义为平滑 `[0, 1]` 量：

```text
a_energy = 1 - exp(- E_sum / energy_scale)
s_gcc = 1 - exp(- gcc_peak_value / gcc_scale)
s_coh = clamp(interaural_coherence, 0, 1)
s_reverb = 1 - exp(- reverb_ratio_proxy / reverb_scale)
s_directness = 1 - exp(- directness_proxy / directness_scale)
s_direct = 0.5 * s_reverb + 0.5 * s_directness
a_cue = (s_gcc * s_coh * s_direct)^(1/3)
audio_evidence_strength_t = 0.5 * a_energy + 0.5 * a_cue
```

- 音频时序一致性不作为该量的离线 target，而由 `delta_binaural_cue_t` 和时序模块处理

当前进一步冻结的运行时计算链：

1. 单步音频模块读取：
   - `binaural_energy_t`
   - `binaural_cue_vector_t`
   - audio backbone 与 cue 分支输出
2. 在单步音频模块内部构造：

```text
raw_audio_evidence_strength = 0.5 * a_energy + 0.5 * a_cue
```

3. 第一轮时序模态校正阶段读取：
   - `audio_embedding`
   - `raw_audio_evidence_strength`
   - `delta_binaural_cue_t` 等时序结构证据
4. 第一轮输出校正后的：
   - `audio_evidence_strength`
5. `audio_evidence_strength` 用于：
   - belief 层内部更新
   - `track_confidence` 的运行时强弱更新
   - 写回历史记忆

当前约束：

- `a_energy / a_cue / raw_audio_evidence_strength / audio_evidence_strength` 都是运行时中间量
- 它们不进入离线 dataset 主 schema
- `energy_scale / gcc_scale / reverb_scale / directness_scale` 是运行时配置超参数，不是预计算标签

当前关于 `track_confidence` 与 evidence strength 的运行时关系进一步冻结为：

- `track_confidence` 只存在于：
  - `belief_state_t`
  - `policy_view_t`
- 它不要求单步层或第一轮时序层显式产出监督值
- 运行时可消费：
  - `visual_evidence_strength`
  - `audio_evidence_strength`
  - belief 历史稳定性
  - 短时失证据衰减逻辑
- 但这条链不再要求对应的离线 `target_track_conf`

当前关于 binaural cue 的新增收口：

- `binaural_energy_t` 第一版固定为：
  - `E_L = mean(x_L^2)`
  - `E_R = mean(x_R^2)`
  - `E_sum = E_L + E_R`
  - `E_diff_norm = (E_L - E_R) / (E_sum + ε)`
- `binaural_cue_vector_t` 第一版固定为：
  - `gcc_peak_lag`
  - `gcc_peak_value`
  - `ild`
  - `ipd_low`
  - `ipd_mid`
  - `interaural_coherence`
  - `reverb_ratio_proxy`
  - `directness_proxy`
- `delta_binaural_cue_t` 不作为单步原始持久化字段
- 它在进入时序 token 组装前显式计算

当前额外约束：

- 规则 target 主要用于监督，不要求模型显式复刻规则分解本身
- 模型输入仍以 embedding、结构化差分和证据强度为主
- 规则分解项可由数据准备阶段或辅助标注逻辑计算

当前新增的第一版运行时权重默认值：

- 视觉证据量：
  - `visual_evidence_strength_t = 0.5 * v_kp + 0.5 * v_geo`
- 音频证据量：
  - `audio_evidence_strength_t = 0.5 * a_energy + 0.5 * a_cue`
- `track_confidence` 不再依赖离线 `target_track_conf`，仅保留为运行时高层信号

当前关于 `Part2Sample` 数据分组契约的新增收口：

为兼顾训练加载效率、规则 target 预计算和后续模块化扩展，当前建议将 Part 2 训练样本按目录/分组组织。

当前新增的数据时间语义约束：

- 需要明确区分两类 step：
  - `simulation step`
    - 游戏内部每次帧更新 / 仿真更新对应的细粒度 step
  - `model step`
    - Part 2 / Part 3 模型真正读到的观测 step
    - 是从 `simulation step` 序列上做出的不保证均匀的采样
    - 且只保留该时刻本应可获得的信息
- 因此：
  - 底层物理存储优先保留 `simulation step` 级基础数据
  - `model step` 序列由数据读取层构造
  - 训练时允许对 `model step` 采样间隔加入随机扰动，以模拟实时运行中的不稳定读取节奏

当前新增的数据存储与读取主契约：

- 物理存储：
  - 采用 `chunked npz + meta.json`
  - chunk 内保存 `simulation step` 级基础数据与预计算标签
- 读取层：
  - 同时提供两种视图：
    - `StepDataset -> StepSample`
    - `WindowDataset -> WindowSample`
  - `WindowSample` 以某个 `model step` 为终点，提供其历史窗口
  - `WindowSample` 由 `simulation step` 数据经过不均匀采样后构造
- 机器可读契约：
  - 现在就需要补充独立 schema 文件
  - 至少应包含：
    - `storage_schema`
    - `view_schema`
    - 字段分组
    - dtype / shape
    - 坐标系语义
    - 时间语义

当前新增的 `meta.json` / `schema.json` 分工收口：

当前 canonical 文件路径已经冻结为：

- `config/dfb_state_estimation/schema.json`
- `config/dfb_state_estimation/meta.template.json`

约定：

- repo 内的 `config/dfb_state_estimation/schema.json` 是 canonical 契约
- 数据集实例目录中仍应写出一份实际使用的 `schema.json`
- `config/dfb_state_estimation/meta.template.json` 是实例级 `meta.json` 的模板，不直接等价于某一份真实数据集元信息

### `meta.json`

语义：

- 描述一份具体数据集实例
- 面向“这份数据是什么、来自哪里、如何索引”

当前建议字段：

- `dataset_id`
- `dataset_version`
- `source_episodes`
- `chunk_index`
  - chunk 文件列表
  - 每个 chunk 的 step 范围
  - 每个 chunk 的时间范围
- `camera_ids`
- `visual_resolution`
- `audio_layout`
  - 例如 `7.1`
- `keypoint_schema_id`
- `coordinate_convention_id`
- `total_simulation_steps`
- `timestamp_range`
- 可选：
  - `train/val/test` split 信息
  - 预计算统计量摘要

当前约束：

- `meta.json` 负责描述具体数据实例
- 不重复定义字段 dtype / shape / 视图契约
- 当前 id/path 对照 manifest 已落地为：
  - [asset_ids.json](/run/media/ayano/SharedProjects/general_projects/dfb/config/dfb_state_estimation/ids/asset_ids.json)
- 它统一声明：
  - `coordinate_convention_id`
  - `dataset_schema_id`
  - `keypoint_schema_id`
  - `audio_cue_schema_id`
  以及各自的 canonical 文件路径

### `schema.json`

语义：

- 描述字段级机器可读契约
- 面向“字段应该是什么、如何被解释”

当前建议包含两层：

1. `storage_schema`
   - 描述磁盘物理存储字段
   - 对应：
     - `core/`
     - `vision_labels/`
     - `audio_features/`
     - `rule_targets/`
   - 每个字段至少记录：
     - `name`
     - `dtype`
     - `shape`
     - `group`
     - `semantic`
     - `coordinate_frame`
     - `time_level`
       - `simulation_step`
       - `episode_level`
       - 其他需要的层级
     - `precomputed`
     - `runtime_only`

2. `view_schema`
   - 描述读取层暴露出的视图契约
   - 当前至少包含：
     - `StepSample`
     - `WindowSample`
   - 每个视图字段至少记录：
     - `name`
     - `source_group`
     - `source_field`
     - `windowed`
     - `current_step_only`
     - `derived_at_load_time`
     - `semantic`
     - `coordinate_frame`
     - `time_semantic`

当前约束：

- `schema.json` 负责定义通用格式契约
- 不负责记录具体 chunk 文件名、episode 列表或实例统计
- 数据集目录中应放一份实际使用的 `schema.json`，保证数据资产自描述

当前新增的 `view_schema` 字段级收口：

### `StepSample`

语义：

- 单个 `model step` 的完整单步观测与监督包
- 面向：
  - 单步视觉模块
  - 单步音频模块
  - 单步 evidence extraction
  - 单步规则 target 训练与调试

当前建议字段：

1. 索引与时间
   - `sample_id`
   - `episode_id`
   - `simulation_step_index`
   - `model_step_index`
   - `timestamp`
   - `dt_to_prev`
   - `time_from_now`
     - 对 `StepSample` 固定为 `0`
   - `has_prev_step`

2. 当前观测输入
   - `front_camera_image`
   - `rear_camera_image`
   - `audio_window_binaural`

3. 当前 ego / 相机上下文
   - `ego_position_world`
   - `ego_orientation_world`
   - `ego_linear_velocity_world`
   - `ego_angular_velocity_body`
   - `camera_extrinsics[camera_id]`
   - `observed_role`

4. 当前主监督标签
   - `gt_relative_position`
   - `gt_relative_orientation`
   - `gt_linear_velocity`
   - `gt_angular_velocity`

5. 当前视觉标签
   - `segmentation_mask[camera_id]`
   - `keypoints_2d[camera_id]`
   - `keypoint_visibility[camera_id]`
   - 可选：
     - `keypoint_confidence_target[camera_id]`

6. 当前音频预计算特征
   - `binaural_energy_t`
   - `binaural_cue_vector_t`

7. 当前规则 target
   - `target_pos_conf`
   - `target_ori_conf`

当前约束：

- `StepSample` 只包含当前时刻 `t` 的字段
- 不包含历史窗口堆叠
- 不包含依赖模型输出的 runtime-only 几何验证量

### `WindowSample`

语义：

- 以某个 `model step` 为终点的时序窗口样本
- 面向：
  - 第一轮 `Temporal Modality Calibration Stage`
  - 第二轮 `Temporal Belief Update Stage`
  - 整体 Part 2 时序训练

当前建议字段：

1. 窗口索引与有效性
   - `episode_id`
   - `window_end_model_step_index`
   - `window_end_simulation_step_index`
   - `window_length_steps`
   - `window_time_span`
   - `window_valid_mask`

2. 窗口时间字段
   - `timestamps[T]`
   - `dt_to_prev[T]`
   - `time_from_now[T]`
   - `has_prev_step[T]`

3. 窗口观测输入序列
   - `front_camera_image[T]`
   - `rear_camera_image[T]`
   - `audio_window_binaural[T]`

4. 窗口 ego / 相机上下文序列
   - `ego_position_world[T]`
   - `ego_orientation_world[T]`
   - `ego_linear_velocity_world[T]`
   - `ego_angular_velocity_body[T]`
   - `camera_extrinsics[camera_id, T]`

5. 窗口主监督标签序列
   - `gt_relative_position[T]`
   - `gt_relative_orientation[T]`
   - `gt_linear_velocity[T]`
   - `gt_angular_velocity[T]`

6. 窗口视觉标签序列
   - `segmentation_mask[camera_id, T]`
   - `keypoints_2d[camera_id, T]`
   - `keypoint_visibility[camera_id, T]`

7. 窗口音频预计算特征序列
   - `binaural_energy_t[T]`
   - `binaural_cue_vector_t[T]`

8. 窗口规则 target 序列
   - `target_pos_conf[T]`
   - `target_ori_conf[T]`

当前约束：

- `WindowSample` 是 `StepSample` 的窗口化视图，而不是另一套独立 schema
- 窗口中的 step 是 `model step` 序列，而不是简单连续的 `simulation step` 序列
- 窗口内每一步只包含该步本应可见的信息
- 第一版允许：
  - 整窗辅助监督
  - 最后一步主监督

### 1. `core/`

语义：

- 主输入
- 主监督
- 时序窗口基础信息

建议字段：

- `timestamps`
- `dt_to_prev`
- `time_from_now`
- `has_prev_step`
- 双相机图像窗口
- 双耳/agent-hearing 音频窗口
- `ego_state`
- `camera_extrinsics`
- `observed_role`
- `gt_relative_position`
- `gt_relative_orientation`
- `gt_linear_velocity`
- `gt_angular_velocity`

其中当前进一步冻结为：

`ego_state`

- `ego_position_world: Vec3`
- `ego_orientation_world: rotation_6d`
- `ego_linear_velocity_world: Vec3`
- `ego_angular_velocity_body: Vec3`

`camera_extrinsics`

- `camera_position_body: Vec3`
- `camera_orientation_body: rotation_6d`

说明：

- 这些字段可作为训练标签和 GT 构造支撑信息使用
- 但不意味着它们可被飞行员或策略直接观测
- 它们属于环境可用的 supervisory information，而不是 policy-observable state
- 每个相机各自维护一套 `camera_extrinsics`
- 在双相机场景下，应按相机 ID 或前/后相机分别存储与加载对应外参

### 2. `vision_labels/`

语义：

- 视觉分支训练与规则构造所需标签

建议字段：

- 双相机语义分割标签
  - 类别：
    - `fighter1`
    - `fighter2`
    - `background`
- 固定关键点 2D 标签
- 关键点可见性标签
- 可选关键点置信目标/辅助标签

当前实现状态补充：

- `dfb_tool_dataset pack` 已经按 `vision_labels/` 物理分组写出 `chunked npz`
- 当前这一组处于“部分真实化”状态：
  - `segmentation_mask_*`
    - 已由 `extract` 阶段的 GPU semantic capture pass 真实导出
    - `pack` 直接消费真实 segmentation artifact
    - 不再是零张量占位
  - `keypoints_2d_*`
    - 已由 `label` 阶段基于权威飞机位姿、固定 keypoint schema、相机位姿与成像模型真实投影导出
  - `keypoint_visibility_*`
    - 已由 `label` 阶段基于真实投影结果与真实 segmentation artifact 导出
    - 第一版判定条件为：
      - 点位于相机前方
      - 投影落入图像边界
      - 投影像素命中对应飞机语义分割类别
      - 并且该点在目标机体自身 mesh 光栅化得到的深度缓冲上处于前表面可见层
- 这项替换必须发生在单步视觉模块正式训练之前

真实视觉标签的权威来源当前已冻结为：

- `segmentation_mask_front / segmentation_mask_rear`
  - 来源于 authoritative recording 对应 step 的权威世界状态
  - 在对应相机视角下生成的语义标签图
  - 语义类别固定为：
    - `fighter1`
    - `fighter2`
    - `background`
- `keypoints_2d_front / keypoints_2d_rear`
  - 来源于：
    - 权威飞机位姿
    - 固定 keypoint schema
    - 对应相机外参
    - 对应相机成像模型
  - 通过逐 step、逐相机的 3D keypoint 投影得到
- `keypoint_visibility_front / keypoint_visibility_rear`
  - 来源于投影结果与视角判定
  - 第一版至少结合：
    - 点是否位于相机前方
    - 投影是否落入图像边界
    - 点是否落在对应飞机分割区域内
    - 点深度是否与目标机体自身 mesh 光栅化得到的前表面深度一致
  - 更细的自遮挡/互遮挡判定可作为后续增强，但第一版不得继续用零张量占位

补充说明：

- 这意味着 `vision_labels` 的真实来源是：
  - 权威世界状态
  - 固定 keypoint / camera schema
  - 标签生成过程
- 而不是单步视觉模型当前预测结果
- 单步视觉模型当前预测结果只应用于运行时几何验证量，例如：
  - `pnp_valid`
  - `reprojection_error`

### 3. `audio_features/`

语义：

- 音频结构摘要与可复用预计算特征

建议字段：

- `binaural_energy_t`
- `binaural_cue_vector_t`
- 可选其他轻量声场结构摘要

说明：

- 这些量可以在统一 dataset tool 的 `label / target derivation` 阶段预计算后存储
- 以减少训练时重复计算成本

### 4. `rule_targets/`

语义：

- 规则生成的监督目标

建议字段：

- `target_pos_conf`
- `target_ori_conf`

说明：

- 当前离线主契约只保留可稳定定义的 confidence target
- `v_kp / v_geo / a_energy / a_cue` 以及 `*_evidence_strength` 已降级为运行时中间量，不再写入数据集主 schema

### 5. `runtime_only/`

语义：

- 依赖模型中间输出、无法静态预计算的运行时量

当前明确包含：

- `pnp_valid`
- `reprojection_error`

说明：

- 这两项依赖单步视觉模块的当前中间预测结果
- 不适合作为静态标签直接存入数据集

当前边界共识：

- Part 1 侧应收口为一个统一的 dataset 导出工具，而不是长期维护 `modalities` / `datapacker` 两个并列顶层工具
- 该统一工具内部应按三层分工：
  - `reconstruct / extract`
  - `label / target derivation`
  - `pack / export`
- 统一工具负责导出：
  - 确定性的监督标签
  - 结构化预计算特征
  - 规则 target/规则分解项
- Part 2 训练/推理运行时负责计算：
  - 依赖模型输出的在线几何验证量

当前实现状态：

- `dfb_tool_dataset extract|reconstruct`
  - 已接管旧 `modalities` CLI 的重建逻辑
- `dfb_tool_dataset label`
  - 当前可导出：
    - GT relative state
    - `binaural_energy_t`
    - `binaural_cue_vector_t`
    - `target_pos_conf / target_ori_conf`
- dense segmentation labels、projected keypoints 与 keypoint visibility
  - 已并入统一 `extract + label + pack` 链路
  - 当前剩余增强点主要是更复杂的遮挡判定、运行时重投影几何量与 evidence strength 的实现

## 5. 当前建议的第一版收口方向

为了尽快从“概念设计”进入“可实现设计”，当前建议把 Part 2 第一版收在下面这个范围内。

### 5.1 第一版模型结构

- `Single-Step Evidence Extraction`
- `Hierarchical Temporal Belief Transformer`
- `policy_view_t adapter`

保持不变。

其中 `Hierarchical Temporal Belief Transformer` 当前按下列语义实现：

- 输入：
  - 当前 `evidence_state_t`
  - 当前 `evidence_t`
  - 历史对应序列
  - 历史 belief 序列
  - 可选 side-channel，如 `time_since_last_visual_lock`
- 中间：
  - `Temporal Modality Calibration Stage`
  - `coarse_state_t`
  - `Temporal Belief Update Stage`
- 输出：
  - 最终 belief `belief_state_t`
  - uncertainty / confidence / source reliability 等 meta 量

实现强调：

- 第一版采用单个 fixed-length causal Transformer 主干
- 但内部按分层阶段组织
- 当前不计划在第一版改用更复杂的 state-space model、memory bank 或 mixture tracking

当前新增的第一版结构超参数草案：

#### 单步模块

- 视觉模块：
  - 轻量 PVNet 风格
  - 输出：
    - segmentation head
    - keypoint head
    - visual embedding head
    - `raw_visual_evidence_strength` head
  - 第一版建议：
    - `visual_embedding_dim = 128`

- 音频模块：
  - 轻量多通道音频 encoder
  - 输出：
    - audio embedding head
    - `raw_audio_evidence_strength` head
  - 第一版建议：
    - `audio_embedding_dim = 64`

- `Single-Step Evidence Extraction`：
  - 第一版采用轻量融合层
  - 以 MLP / projection block 为主
  - 不在这一层引入 attention

#### 第一轮：`Temporal Modality Calibration Stage`

- 输入 token：
  - `state_token_t`
  - `visual_token_t`
  - `audio_token_t`
- 第一版建议：
  - `num_layers_stage1 = 2`
  - `hidden_dim_stage1 = 256`
  - `num_heads_stage1 = 8`
  - `ffn_dim_stage1 = 512`
- 第一版输出 head：
  - `coarse_state_head`
  - `visual_evidence_strength_head`
  - `audio_evidence_strength_head`

#### 第二轮：`Temporal Belief Update Stage`

- 输入 token：
  - `belief_update_token_t`
- 第一版建议：
  - `num_layers_stage2 = 3`
  - `hidden_dim_stage2 = 256`
  - `num_heads_stage2 = 8`
  - `ffn_dim_stage2 = 512`
- 第一版输出 head：
  - `belief_state_head`

#### token projection

- 第一版统一采用：
  - 各类 token 先各自 projection 到统一 hidden dim
  - 再加 token type embedding
  - 再加时间相关输入
- 当前第一版建议统一 hidden dim：
  - `hidden_dim = 256`
- 两轮当前建议：
  - hidden dim 相同
  - 不共享参数

#### `policy_view_t adapter`

- 第一版不以可学习 MLP 为主
- 第一版优先采用：
  - 显式字段挑选
  - 显式符号计算
  - 轻量确定性映射
- 也就是说：
  - `policy_view_t` 第一版主要由 `belief_state_t` 通过规则化 adapter 构造
  - 不承担复杂推理职责
- 若后续 Part 3 验证需要更复杂消费视图，再考虑把 adapter 升级为可训练模块

### 5.2 第一版不做的事

- 不做多候选 mixture tracking
- 不做完整显式滤波器替代
- 不做端到端同时输出过多高阶战术量
- 不在第一版把 Part 2 和 Part 3 联训
- 不在第一版实现 `k>1` hypotheses 的 matching / merge / split / pruning

### 5.3 第一版 hypothesis 结构

当前在抽象上允许未来扩展为：

```text
belief_set_t = { (belief_state_t^i, w_t^i) }_{i=1..k}
```

其中：

- `belief_state_t^i` 是第 `i` 个 belief hypothesis
- `w_t^i` 是其 softmax 权重

但第一版实现明确固定为：

- `k = 1`
- `w_1 = 1`

也就是说：

- 当前只做单 hypothesis belief estimation
- 先不引入 hypothesis management 复杂度
- 但接口设计允许以后平滑扩展到 `k>1`

### 5.4 Part 2 内部输出与 Part 3 消费输出分层

当前已明确不再把“内部 belief 表达”和“RL 最终消费视图”混为一谈。

第一版建议采用两层输出：

1. `belief_state_t`
   - Part 2 内部标准输出
   - 用于时序更新、训练监督、分析与诊断
2. `policy_view_t`
   - 给 Part 3 / RL 的最终消费视图
   - 由 `belief_state_t` 经过轻量 adapter 变换得到

当前约束：

- `belief_state_t` 保留必要的 confidence/meta
- `policy_view_t` 第一版不携带分块 confidence，但保留 `track_confidence`
- `policy_view_t` 不承担主坐标系统一职责，只做消费侧轻量映射和派生特征构造

### 5.5 `belief_state_t` 第一版字段冻结结果

当前第一版冻结为：

- `relative_position: Vec3`
- `relative_orientation: rotation_6d`
- `position_confidence: Scalar`
- `orientation_confidence: Scalar`
- `track_confidence: Scalar`
- `linear_velocity: Vec3`
- `angular_velocity: Vec3`

当前明确不进入 `belief_state_t` 的项：

- `throttle_hint`
- `time_since_last_visual_lock`
- `correction_gate`
- `cross_modal_consistency`
- `modality_conflict_score`
- `overall_uncertainty`
- `modality_dominance`
- `source_reliability`

设计说明：

- 位置与姿态是第一版主状态
- 线速度、角运动等导数量优先由相邻时刻 belief 差分显式派生
- `throttle_hint` 暂不进入第一版主契约，若后续验证确有必要，可作为 auxiliary output 或 adapter 附加特征再引入

### 5.6 `policy_view_t` 第一版字段冻结结果

当前第一版 `policy_view_t` 保留：

- `relative_position`
- `relative_orientation`
- `linear_velocity`
- `angular_velocity`
- `track_confidence`

当前理由：

- 相对位置与姿态仍是 RL 的主消费状态
- 一阶派生量在机动判断、趋势感知和时间连续性上有直接价值，因此第一版保留
- 过多 meta/confidence 字段进入最终策略观测，仍然容易带来冗余而非有效行动线索
- 因此第一版将大部分 confidence/meta 保留在 `belief_state_t`，不直接暴露给 `policy_view_t`
- `track_confidence` 例外，因为它可作为 RL 的显式模式切换信号

补充说明：

- 当前 `belief_state_t -> policy_view_t` 的 adapter 第一版应保持轻量
- 其主要职责是：
  - 从内部 belief 提取 RL 直接消费的低维状态
  - 显式构造必要的派生特征
- 第一版不让 adapter 承担主坐标变换职责，也不让其承担主要的不确定性推理职责
- 但在远期扩展上，adapter 可以演化为连接“对手行为建模 / 意图建模”模块的接口层
- 届时它可能负责：
  - 将单 hypothesis belief 扩展为带行为先验的姿态/状态分布
  - 或将 richer belief / distribution 压缩成 RL 实际采信的单一观测视图
- 这一方向当前仅作为远期扩展保留，不影响第一版契约冻结

### 5.7 时序模块输出语义修正

当前进一步明确：

- 第一轮与第二轮时序模块都使用窗口 `{t-L+1, ..., t}` 作为上下文输入
- 但它们的有效解码目标都只有最后一步 `t`

也就是说：

- 时序主干负责“整窗编码”
- heads 只负责“当前步解码”

因此：

- `coarse_state_t`
  - 指当前时刻 `t` 的粗状态
  - 不再把整段窗口内每一步都视为有效输出
- `belief_state_t`
  - 指当前时刻 `t` 的最终 belief
  - 不再把整段窗口内每一步都视为有效输出
- `policy_view_t`
  - 指当前时刻 `t` 的 RL 消费视图
  - 不携带窗口维

窗口在这里的职责是：

- 为当前时刻提供历史上下文
- 而不是要求模型产出一条新的状态序列

这一语义更接近“prefix-to-current”的 GPT 风格推断：

- 历史窗口是上下文
- 当前时刻是唯一有效预测目标

### 5.8 关于 `causal mask` 的当前结论

当前已记录但暂不阻塞实现的点：

- 第一轮和第二轮理论上都可以进一步加上 `causal mask`
- 形成更严格的 GPT-style prefix-to-current 结构

但当前优先级更高的修正是：

- 从“sequence-to-sequence 解码整段窗口”
- 收口到“整窗编码，只解码最后一步”

原因是：

- 当前窗口本身只包含过去到当前 `{t-L+1, ..., t}`
- 主要问题不在“看见未来”
- 而在“错误地把前面 `L-1` 个时间步也当作有效输出”

所以当前顺序是：

1. 先修正两轮时序模块的输出语义
2. 再根据需要评估是否补上 `causal mask`
### 5.7 第一版重点输出

第一版重点不是丰富高阶战术量，而是稳定地产出：

- 统一参考系下的敌机相对位置
- 统一参考系下的敌机相对姿态
- 可供内部更新与分析使用的 uncertainty/meta

这已经足够支撑：

- 状态估计研究
- Part 3 输入设计
- 后续再扩展更细的运动与战术语义

## 6. 与 Part 3 的接口约束

Part 2 不应假设 Part 3 直接消费原始 observation。

当前建议接口语义是：

- Part 3 的主敌机观测来自 Part 2 belief state
- Part 3 仍可保留：
  - 自机状态
  - 自身传感器原始摘要
  - Part 2 的 uncertainty/meta 输出

也就是说，Part 2 应输出：

- belief 主状态
- uncertainty/meta 状态

而不是只输出一个“黑箱 embedding”。

## 7. 当前文档管理建议

建议采用分层管理，而不是把 Part 2 全塞回 `project_context`。

推荐方式：

- `docs/project_context_zh.md`
  - 只保留三部分项目的高层边界
- `docs/part2_context_zh.md`
  - 维护 Part 2 的长期上下文与已收口共识
- `docs/plans/`
  - 放阶段性设计/实现计划

## 8. 下一步讨论顺序建议

如果继续把 Part 2 设计推到“能开工”，我建议严格按这个顺序讨论：

1. 最终输出状态字段
2. 坐标系与旋转表示
3. 单步层输出格式
4. temporal 层输入/输出契约
5. 训练标签与数据样本定义
6. loss 与评估指标
7. 实现切分与代码结构

其中第 1 到第 4 步必须先钉死，否则实现会反复返工。

## 9. 当前从“框架设计”进入“细节冻结”的起点

在 [temporal_belief_module_spec.md](/run/media/ayano/SharedProjects/general_projects/dfb/tmp/temporal_belief_module_spec.md) 的基础上，Part 2 的整体框架设计已经基本完成。

后续讨论不再围绕“是否采用时序 belief update 结构”展开，而是进入细节冻结阶段。

当前应优先冻结的细节顺序是：

1. `belief_state_t` 的最终字段与分块
2. `evidence_state_t`、`evidence_t` 与 `coarse_state_t` 的最终字段与分块
3. `correction_gate`、`mismatch` 等中间量是否进入监督
4. 坐标系、旋转表示、速度语义
5. 数据样本定义、loss 与评估

这部分冻结后，Part 2 就可以进入实现切分与工程落地。
