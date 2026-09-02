# Part 2 Canonical Config Assets

本目录只放 Part 2 / Part 3 共用的 canonical 配置资产。

允许放入：

- `schema.json`
- `meta.template.json`
- keypoint schema
- 双耳音频 cue schema / 历史音频配置
- schema id / convention id 相关配置
- canonical asset id manifest
- train config templates

不允许放入：

- dataset 实例输出
- 训练运行结果
- 临时分析文件

子目录约定：

- `keypoints/`
  - 关键点 schema
- `audio/`
  - 双耳 cue schema、runtime evidence 配置与历史音频配置
- `ids/`
  - schema id / convention id 对照文件
- `train/`
  - Part 2 正式训练配置模板
