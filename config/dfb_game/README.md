# DFB Game Runtime Config

本目录存放 `dfb_game` 的规范配置资产。

当前已迁入：

- `game.ron`
- `input.ron`
- `scenes/`
- `aircraft_specs/`
- `aircraft_derived/`

约定：

- 运行时默认从 `config/dfb_game/` 读取 `dfb_game` 配置
- `aircraft_derived/` 仍是派生配置占位目录
- 用户直接维护的飞机规格文件位于 `aircraft_specs/`
