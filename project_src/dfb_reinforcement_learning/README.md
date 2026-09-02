# DFB Reinforcement Learning Source Root

本目录承载 Part 3 正式子项目 `dfb_reinforcement_learning` 的源码根。

当前正式版本记录位于：

- `VERSION`

版本策略：

- `dfb_reinforcement_learning` 的版本独立于 `dfb_game`
- 后续仅在 Part 3 本身有实际代码或契约变更时更新
Current implemented training entrypoints:

- `dfb_reinforcement_learning.train.train_bc`
- `dfb_reinforcement_learning.train.train_ppo`
- `dfb_reinforcement_learning.train.train_distill_bc`

Current implemented scene tooling:

- `dfb_reinforcement_learning.tools.generate_tactical_scenes`
