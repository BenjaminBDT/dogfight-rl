## 一、核心原则

本文件只定义**协作规范与工程约束**，不包含具体项目实现细节。

所有与项目目标、架构、语义、设计相关的内容，统一参考：

> `docs/project_context_zh.md`

在进行任何开发前，应先阅读并理解该文档。

---

## 二、工作流程规范

### 1. 修改前

* 必须先确认：

  * 当前修改是否符合 project_context 中的整体方向
  * 是否会破坏已有语义或数据流

* 对于较大改动：

  * 先写计划（design / plan）
  * 再拆分为有限的小步骤实现

---

### 2. 修改原则

优先级如下：

1. 语义一致性
2. 系统稳定性
3. 可回滚性
4. 功能实现

严禁：

* 临时 patch 堆叠
* 在不明确语义的情况下修改核心逻辑
* 引入隐式行为

---

## 三、Git 与版本管理规范

### 1. 提交规范

* 使用清晰的英文动词开头：

  * `Add ...`
  * `Fix ...`
  * `Refactor ...`
  * `Remove ...`

示例：

* `Add environment step API`
* `Fix replay desync`
* `Refactor aircraft config pipeline`

---

### 2. 提交策略（非常重要）

> 每完成一个“逻辑完整的小改动”，必须提交一次。

目的：

* 保证可回滚
* 避免大规模不可控修改
* 支持快速迭代

---

### 3. 版本号规范

采用三段式版本号：

```
MAJOR.MINOR.PATCH
```

规则：

* PATCH：bug 修复、较小的改动、较小的功能添加
* MINOR：破坏性修改，较大的更新
* MAJOR：极具变革性的修改与更新，往往代表与上个MAJOR版本几乎不兼容

其中 MINOR 和 MAJOR 的更新由用户手动指定，PATCH 在每次完成“逻辑完整的小改动”后增加。

---

## 四、代码规范

### 1. 基本规则

* 使用 Rust 标准风格
* 使用 `cargo fmt`
* 使用 `cargo clippy --all-targets --all-features -D warnings`

---

### 2. 命名规范

* 函数 / 模块：`snake_case`
* 类型 / 枚举：`PascalCase`
* 常量：`SCREAMING_SNAKE_CASE`

---

### 3. 结构原则

* 单模块单职责
* 避免跨层耦合：

  * simulation
  * networking
  * recording
  * training

---

## 五、测试规范

* 所有重要逻辑应具备测试

类型：

* 单元测试：靠近实现（`#[cfg(test)]`）
* 集成测试：`tests/`

提交前必须执行：

```bash
cargo test
cargo clippy
```

---

## 六、数据与接口约束

* 不随意修改：

  * observation 语义
  * action 语义
  * recording 格式

* 若必须修改：

  * 必须同步更新 project_context
  * 明确兼容策略

---

## 七、设计原则

### 1. 优先语义清晰

不太合适的做法：

* 堆叠 patch 修 bug
* 局部修补

期望的做法：

* 寻找根因
* 用统一的语义重构有缺陷的设计

---

### 2. 系统视角

始终从完整链路思考：

```
simulation → recording → reconstruction → dataset → model → policy
```

任何修改都必须考虑其对整条链的影响。

---

### 3. 权威原则

* 能由 server 决定的，必须由 server 决定
* client 不得引入权威语义

---

### 4. 数据优先

* 优先记录“语义数据”而不是原始数据
* 避免不可复现的数据路径

---

## 八、协作规范

* 需求不明确时必须提问
* 禁止在目标不清晰时直接实现
* 优先小步迭代

---

## 九、最终原则

> 如果一个修改可能破坏训练流程或数据一致性，那么它大概率是错误的修改。
