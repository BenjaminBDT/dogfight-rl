本目录预留给“派生仿真配置”。

规则：

- 用户应直接修改 `config/dfb_game/aircraft_specs/`
- 仿真运行时消费的是由直接配置推导出的派生配置
- 本目录中的文件不应手工维护

当前阶段：

- 派生配置已经在代码中按 `AircraftSpecConfig -> AircraftConfig` 生成
- 但尚未增加显式导出文件步骤
- 后续若增加导出工具或构建步骤，本目录将存放对应产物
