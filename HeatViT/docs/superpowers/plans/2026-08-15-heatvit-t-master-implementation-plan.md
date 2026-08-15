# HeatViT-T FPGA 推理实现总计划

> 本文整合 `docs/superpowers/plans` 中的路线图及五个阶段计划，作为唯一的执行入口。
> 原分阶段文档暂予保留，便于追溯原始任务细节；后续实施应以本文的阶段顺序、公共约束和验收门为准。

## 目标与范围

在现有 `xc7k325tfbg900-3` Vivado 工程中，以 SystemVerilog 实现 HeatViT-T / DeiT-T 的完整定点推理，并通过 Vivado XSim 对一张完整 `224 x 224` 图像进行逐位端到端验证。

模型固定为：`224 x 224 x 3` 输入、`16 x 16` patch、197 个初始 Token、嵌入维度 192、3 个 head（每 head 64 维）、12 个 Transformer block、FFN 维度 768、1000 类。Selector 位于 block 4、7、10 之前；CLS 永久保留，最多额外生成一个 Package Token。

本计划只覆盖 XSim 仿真验证；不覆盖上板、MIG/AXI 集成、时序收敛、功耗、FPS 或 ImageNet 准确率验收。

## 执行前置条件

1. 设置 `HEATVIT_VIVADO_BIN`，指向 Vivado 2023.2 的 `bin` 目录；设置 `HEATVIT_PYTHON`，指向 Python 3.12–3.14。
2. 确认 `xvlog.bat` 与指定 Python 可用，并锁定 `numpy==2.5.2`。
3. Python 黄金模型在量化边界后只能使用显式整数 dtype，不得创建 float/complex Tensor。
4. 未提供 Git 仓库时不得执行 `git init`；任务检查点写入 `build/task-checkpoints.log`。

## 全局工程约束

- 普通激活和权重使用 signed int8；常规 GEMM 使用 signed int8 × signed int8 → signed int32；Attention × V 使用 UQ0.8 × signed int8 → signed int32；最终 logit 为 signed int32。
- 缩放统一采用 signed binary scale exponent。右移、除法均为 nearest、ties-away-from-zero；所有窄化必须饱和，禁止隐式截断。
- LayerNorm、GELU、Softmax、PLAN Sigmoid、除法和平方根均必须为可综合的纯整数 RTL；不得调用浮点或 Divider IP。
- 外部存储为 64-bit ready/valid 行为接口；区域及命令地址 8-byte 对齐，尾拍使用 `mem_w_strb[7:0]`。
- 设计不实例化手工生成的 Vivado IP；DSP48E1 与 BRAM 仅允许由 RTL 推断。
- Testbench 自检成功时唯一输出 `TEST_PASS <top>`，失败时 `$fatal` 且退出码非零；不使用 UVM。
- 全尺寸端到端测试中，每个 Selector 都必须同时发生普通 Token 的保留与剪除。

## 共享契约

`rtl/include/heatvit_pkg.sv` 是唯一公共类型和数值契约来源，必须锁定：

- 320-bit `heatvit_desc_t` 描述符及 `$bits(...) == 320` 检查。
- 固定 opcode：`OP_NOP` 至 `OP_FINISH`，以及 post-op、error（1–7）和 warning（0–2）编码。
- `round_shift_away_s128`、`scale_to_exp_s128`、`sat_s8`、`sat_s32` 等统一定点函数。
- `FLAG_SRC0_UNSIGNED` 仅用于 Attention UQ0.8 × signed int8 V；`FLAG_DYNAMIC_N/K` 用于动态维度。

## 阶段顺序与验收门

| 阶段 | 交付物 | 进入下一阶段的条件 |
| --- | --- | --- |
| 1. 定点基础 | 公共 package、整数黄金模型、非线性与基础 RTL 单元 | Python 与数值 RTL 单元测试全通过 |
| 2. 存储与 GEMM | 64-bit 存储通路、Tile Buffer、8×8 GEMM | 普通、转置、尾块、回压 GEMM 逐位通过 |
| 3. Transformer | Patch Embed、MHSA、FFN、Pre-LN block | N=197 和非 8 倍数 Token 数完整 block 通过 |
| 4. Selector | 分类、head 融合、稳定压缩、Package | 全保留、全剪除、混合剪除和两种回退通过 |
| 5. 集成 | 描述符调度、12 block、3 selector、最终 logit | 全尺寸端到端检查点逐位通过 |

后续阶段只能消费前一阶段已经稳定的接口和数值定义；任何接口或数值定义变动，必须同时更新本计划和受影响回归。

## 阶段 1：定点基础

### 目标

建立可重复执行的 XSim/Python 测试环境，以及后续模块共用的逐位确定性定点、非线性、除法、平方根、Softmax 与 LayerNorm 单元。

### 任务清单

1. 建立配置、编译脚本和公共 package。
2. 实现 Python 定点基准与 `.mem` 向量文件协议。
3. 实现重定标与残差单元。
4. 实现恢复除法和整数平方根。
5. 实现 GELU 与 PLAN Sigmoid。
6. 实现 Attention 和 Selector 两种 Softmax。
7. 实现纯 RTL LayerNorm。
8. 建立阶段 1 回归门。

### 验收

`scripts/run_python_tests.ps1` 和各数值模块 XSim 自检通过；舍入、饱和、边界值和随机固定种子向量均与整数黄金模型逐位一致。

## 阶段 2：存储与统一 GEMM

### 目标

实现 64-bit 行为存储、地址守卫、存储主机、FIFO、可推断 RAM、8×8 MAC Bank、Tile Buffer 和统一 GEMM 调度。

### 任务清单

1. 实现行为存储和地址守卫。
2. 实现存储主机、FIFO 和可推断 RAM。
3. 实现 8×8 MAC Bank。
4. 实现 Python GEMM 基准和矩阵布局测试。
5. 实现 Tile Buffer 和普通 GEMM 调度。
6. 增加 RHS 转置、head 模式和全链路回压覆盖。

### 验收

普通 GEMM、转置 RHS、非整 Tile 尾块、动态维度和随机回压均逐位匹配 Python 基准；所有 ready/valid 负载在 stall 时稳定，地址保持区域内且 8-byte 对齐。

## 阶段 3：Transformer 数据通路

### 目标

实现 Tensor arena、布局/向量执行器、Patch Embedding、三 head MHSA、FFN 与残差，并组合为完整 Pre-LN Transformer block。

### 任务清单

1. 实现 Tensor arena、布局黄金模型和描述符打包。
2. 实现 Layout/Vector Engine 和单描述符 Executor。
3. 实现并验证 Patch Embedding。
4. 实现并验证三 head MHSA。
5. 实现并验证 FFN 与残差。
6. 建立完整 Pre-LN Transformer block 回归。

### 验收

对 N=197 以及非 8 倍数 N 的输入，Patch、QKV、Attention、投影、FFN 和每个残差检查点都逐位匹配黄金模型。

## 阶段 4：动态 Token Selector

### 目标

实现 Selector 的整数基准、候选归约、Local/Global 拼接、head 融合、稳定压缩及单 Package Token 生成，并完成跨阶段状态验证。

### 任务清单

1. 实现 Selector 整数黄金模型。
2. 实现 Reduction 与 Local/Global Concat。
3. 实现 head 权重融合。
4. 实现稳定压缩与单 Package Finalize。
5. 集成完整 Token Selector 描述符序列。
6. 建立 Selector 回归与跨阶段 Package 测试。

### 验收

覆盖全保留、全剪除、混合剪除、无剪除候选和分母为零回退。CLS 始终保留，普通 Token 顺序稳定；Package 至多一个且状态可跨三个 Selector 正确传递。

## 阶段 5：调度与端到端集成

### 目标

固定 198 条描述符，完成 Descriptor ROM、动态 Scheduler、顶层封装、完整黄金模型、内存映像、端到端 XSim、错误/警告覆盖及最终交付文档。

### 任务清单

1. 编译并验证固定 198 条描述符。
2. 实现 Descriptor ROM 和动态 Scheduler。
3. 集成 `heatvit_top`、兼容封装和 Vivado 工程源文件。
4. 实现完整黄金模型和确定性混合剪枝参数生成。
5. 固化端到端内存映像和 manifest 契约。
6. 运行完整尺寸无回压端到端 XSim。
7. 验证随机回压、全部错误码和 warning 锁存。
8. 完成全回归、无 IP 审计和仿真交付文档。

### 端到端契约

- 内存区域基址固定为 input `0x00000000`、weights `0x01000000`、scratch `0x02000000`、output `0x03000000`。
- 所有 `.mem` 每行一个 64-bit 小端 beat，末拍零填充；manifest 记录原始 byte 数、哈希、scale、offset 与 descriptor 索引。
- 18 个检查点、1000 个 logit、输出 scale、三个 Selector 的输入/输出 Token 数和 Package 状态均逐位比较。
- `WATCHDOG_CYCLES = 4 * (gemm_work + memory_beats + nonlinear_work) + 10_000_000`。
- 无回压与随机回压必须得到完全相同的检查点、Selector 状态和 logit；仅周期数允许不同。

### 验收

`scripts/audit_no_ip.ps1` 与 `scripts/run_regression.ps1 -Suite all` 均返回 0。错误码 1–7、warning 0–2 完整覆盖；Vivado Part 保持 `xc7k325tfbg900-3`，顶层为 `heatvit`。

## 最终交付清单

- [ ] 五个阶段回归均通过。
- [ ] `scripts/run_regression.ps1 -Suite all` 返回 0。
- [ ] `build/reports/e2e_summary.json` 记录三个 Selector 的输入/输出 Token 数，且三次均发生普通 Token 剪枝。
- [ ] `build/reports/ip_audit.txt` 包含 `NO_MANUAL_VIVADO_IP_REQUIRED`。
- [ ] `docs/verification/simulation-guide.md`、`memory-and-weight-format.md` 与 `e2e-results.md` 完整记录可复现方法与结果。
- [ ] 最终结论仅表述“仿真逐位通过”，明确不等同于 ImageNet 准确率、时序、功耗、FPS 或上板通过。

## 原始文档索引

本文汇总自以下文件（保留用于任务级实现细节追溯）：

- [总体路线图](2026-08-15-heatvit-t-implementation-roadmap.md)
- [阶段 1：定点基础](2026-08-15-heatvit-t-phase-1-foundation.md)
- [阶段 2：存储与 GEMM](2026-08-15-heatvit-t-phase-2-memory-gemm.md)
- [阶段 3：Transformer](2026-08-15-heatvit-t-phase-3-transformer.md)
- [阶段 4：Token Selector](2026-08-15-heatvit-t-phase-4-selector.md)
- [阶段 5：端到端集成](2026-08-15-heatvit-t-phase-5-integration.md)
