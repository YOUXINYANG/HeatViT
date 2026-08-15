# HeatViT-T FPGA Inference Implementation Roadmap

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 `xc7k325tfbg900-3` Vivado 工程中，以 SystemVerilog 实现 HeatViT-T 完整定点推理，并用 Vivado XSim 对一张完整 224×224 图像进行逐位端到端验证。

**Architecture:** 实现分为五个可独立验收的阶段：工程与定点基础、统一 GEMM 与存储通路、Transformer 数据通路、动态 Token Selector、描述符调度与端到端集成。所有阶段共享 320-bit 描述符、signed binary scale exponent、行为级 64-bit 外部存储模型和 Python 整数黄金模型。

**Tech Stack:** SystemVerilog 2012、Vivado/XSim 2023.2、PowerShell 7 或 Windows PowerShell 5.1、Python 3.12–3.14、NumPy 2.5.2、`.mem` 测试向量。

## Global Constraints

- 固定模型为 HeatViT-T / DeiT-T：输入 `224 x 224 x 3`、Patch `16 x 16`、197 初始 Token、D=192、3 Head、Head Dim=64、12 Block、FFN=768、1000 类。
- Selector 固定放在 Block 4、7、10 之前；CLS 永久保留，输出至多包含一个 Package Token。
- 普通激活和权重为 signed int8；常规 GEMM 为 signed int8×signed int8→int32；Attention×V 为 unsigned UQ0.8×signed int8→int32；最终 Logit 为 signed int32。
- 所有缩放、最近舍入中点远离零、饱和、非线性常量、阈值和回退行为必须与批准规格逐位一致。
- LayerNorm、GELU、Softmax、PLAN Sigmoid、除法和平方根均使用可综合 RTL，不调用浮点或 Divider IP。
- 外部存储接口为 64-bit ready/valid 行为接口；所有区域和命令地址 8-byte 对齐，尾拍使用 `mem_w_strb[7:0]`。
- 只做 Vivado XSim 仿真验证；不做上板、MIG/AXI 集成、时序收敛、功耗、FPS 或 ImageNet 准确率验收。
- 当前不需要手动生成任何 Vivado IP；DSP48E1 和 BRAM 只允许从 RTL 模板推断。
- 测试采用自检式 SystemVerilog，不引入 UVM；Python 黄金模型量化后只执行整数运算。
- 完整端到端测试必须让三个 Selector 均同时保留和剪除至少一个普通 Token，并比较所有规定检查点。
- 规范来源：`docs/superpowers/specs/2026-08-15-heatvit-t-fpga-inference-design.md`。

---

## 执行前置门槛

当前机器探测结果为：`vivado`、`xvlog`、`xelab`、`xsim`、`python` 和 `py` 均不在 `PATH`。开始实施前必须配置：

```powershell
$env:HEATVIT_VIVADO_BIN = 'C:\Xilinx\Vivado\2023.2\bin'
$env:HEATVIT_PYTHON = 'C:\Path\To\Python312\python.exe'
Test-Path "$env:HEATVIT_VIVADO_BIN\xvlog.bat"
Test-Path $env:HEATVIT_PYTHON
& $env:HEATVIT_PYTHON -c "import sys; assert (3,12) <= sys.version_info[:2] <= (3,14)"
```

两条 `Test-Path` 都必须返回 `True` 且 Python 版本断言退出 0。路径示例不是安装要求；执行者应填入本机真实路径。若 Vivado 或 Python 尚未安装，停止实施并请用户提供或安装，不能用其他仿真器悄然替代 XSim。阶段 1 Task 1 随后安装并锁定 NumPy 2.5.2；NumPy 只允许以显式 integer dtype 加速黄金模型，不得产生浮点 Tensor。

版本依据：[NumPy 官方发布列表](https://numpy.org/news/) 与 [NumPy 2.5.0 官方兼容说明](https://numpy.org/doc/stable/release/2.5.0-notes.html)；2.5 系列支持 Python 3.12–3.14，本计划锁定当前补丁版 2.5.2。

当前目录没有 `.git`。实施任务不得自行执行 `git init`。每个任务调用 `scripts/task_checkpoint.ps1`：若用户之后提供 Git 仓库则创建提交，否则把已通过的测试命令和任务名追加到 `build/task-checkpoints.log`。

## 文件结构总览

```text
config/
  heatvit_t.json                 固定模型与量化配置
rtl/
  include/heatvit_pkg.sv         公共类型、常量、描述符和定点函数
  common/                        数值、FIFO、RAM、除法和平方根单元
  memory/                        外存主接口与 Tile Buffer
  compute/                       MAC Bank、GEMM、张量 Executor、布局和残差
  selector/                      分类、Head 融合、压缩和 Package
  top/                           描述符 ROM、调度器和 heatvit_top
HeatViT.srcs/sources_1/new/
  heatvit.sv                     保持原工程 Top 名的薄封装
verification/heatvit_ref/        纯整数 Python 黄金模型
verification/tests/              Python 单元测试
tools/                           描述符和 `.mem` 向量生成器
sim/common/                      TB 公共包与行为存储
sim/tb/                          自检式 SystemVerilog Testbench
sim/vectors/                     小型受控单元向量
build/                           生成的完整向量、日志和 XSim 产物
scripts/                         预检、XSim、回归和 Vivado 同步脚本
docs/superpowers/plans/          本路线图与分阶段计划
```

## 阶段顺序与验收门

| 顺序 | 计划文件 | 独立交付物 | 进入下一阶段的必要条件 |
| ---: | --- | --- | --- |
| 1 | `2026-08-15-heatvit-t-phase-1-foundation.md` | 可复用定点单元、Python 整数基准、XSim 基础设施 | Python 与全部数值 RTL 单元测试通过 |
| 2 | `2026-08-15-heatvit-t-phase-2-memory-gemm.md` | 64-bit 存储通路、Tile Buffer、3×8×8 GEMM | GEMM 普通/转置/尾块/回压逐位通过 |
| 3 | `2026-08-15-heatvit-t-phase-3-transformer.md` | Patch Embed、MHSA、FFN、完整 Pre-LN Block | N=197 与非 8 倍数 N 的完整 Block 通过 |
| 4 | `2026-08-15-heatvit-t-phase-4-selector.md` | 分类、Head 权重、稳定压缩、单 Package Selector | 全保留、全剪、混合剪与两种回退通过 |
| 5 | `2026-08-15-heatvit-t-phase-5-integration.md` | 描述符调度、12 Block、3 Selector、最终 Logit | 完整尺寸端到端所有检查点逐位通过 |

阶段必须按表中顺序执行。后续阶段可以读取前一阶段稳定接口，但不得在未更新对应计划和回归的情况下改变其端口或数值定义。

## 总体验收清单

- [ ] 阶段 1：定点与非线性基础通过。
- [ ] 阶段 2：存储与统一 GEMM 通过。
- [ ] 阶段 3：Transformer Block 通过。
- [ ] 阶段 4：Token Selector 通过。
- [ ] 阶段 5：端到端回归通过。
- [ ] `scripts/run_regression.ps1 -Suite all` 返回退出码 0。
- [ ] `build/reports/e2e_summary.json` 记录三个 Selector 的输入/输出 Token 数且三次均发生剪枝。
- [ ] `build/reports/ip_audit.txt` 证明工程未实例化需手工生成的 Vivado IP。
- [ ] 最终文档明确“仿真通过”不等于 ImageNet 准确率或板级性能通过。
