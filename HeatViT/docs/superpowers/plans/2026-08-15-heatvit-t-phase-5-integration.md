# HeatViT-T Phase 5: Scheduler and End-to-End Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 集成固定 12 Block、三个动态 Selector、Final LayerNorm 和 1000 类分类头，并以完整 224×224 输入在 XSim 中通过所有逐位检查点。

**Architecture:** 生成器编译恰好 198 条 320-bit 描述符到可综合 ROM；Scheduler 串行提交唯一 Tensor Executor，并在三个 Selector Finalize 后原子更新 Token/Package 状态。`heatvit_top` 锁存四个内存区域，汇总错误/警告；现有 `heatvit.sv` 只作为保持 Vivado 工程 Top 名的端口透传封装。

**Tech Stack:** SystemVerilog 2012、Vivado/XSim 2023.2、Python 3.12–3.14、NumPy 2.5.2、PowerShell、前四阶段全部模块。

## Global Constraints

- 固定完整流：Patch → Block 1..3 → Selector 1 → Block 4..6 → Selector 2 → Block 7..9 → Selector 3 → Block 10..12 → Final LN → Head。
- 描述符数固定为 198、合法 index 为 0..197；Patch 3 条、每 Block 13 条、每 Selector 12 条、Final LN/Head/Finish 各 1 条。
- Scheduler 初始状态固定为 `current_token_count=197`、`current_package_present=0`。
- 只有 OP_SELECTOR_FINALIZE 可以更新 Token/Package 状态；更新值必须在 2..197 且 Token 数不得增加。
- Final LayerNorm 对全部当前 Token 执行；分类 GEMM 只读取 index 0 的 CLS 并写 1000 个 little-endian signed int32 Logit。
- 完整向量使用固定 seed 20260815 和确定性合成权重，不声明任何分类准确率。
- 三个 Selector 各自必须保留至少一个且剪除至少两个普通 Token，确保三次 Token 数都实际下降。
- 所有要求检查点必须逐位比较：Patch、12 个 Block、3 个 Selector、Final LN、1000 Logit 和尺度。
- 致命错误 1..7 必须停止命令、清 busy、不发 done；warning bits 0..2 必须锁存至下一 start 或 reset。
- 完整无回压和伪随机回压两轮均需通过；watchdog 由 manifest 中的工作量估算生成。
- 不做综合资源/Fmax/板级验收，不生成或实例化手工 Vivado IP。

---

## 固定描述符索引

| Index | 内容 |
| ---: | --- |
| 0..2 | Patchify、Patch GEMM、CLS+Position |
| 3..41 | Block 1..3，每个 13 条 |
| 42..53 | Selector 1 |
| 54..92 | Block 4..6 |
| 93..104 | Selector 2 |
| 105..143 | Block 7..9 |
| 144..155 | Selector 3 |
| 156..194 | Block 10..12 |
| 195 | Final LayerNorm |
| 196 | CLS→1000 分类 GEMM，int32 写回 |
| 197 | OP_FINISH |

每个 13 条 Block 的内部顺序固定为 LN1、QKV GEMM、QKV unpack、QKᵀ、Attention Softmax、Attention×V、Head concat、Projection、Residual1、LN2、FC1+GELU、FC2、Residual2。每个 12 条 Selector 顺序使用阶段 4 已锁定列表。

## 文件映射

| 文件 | 单一职责 |
| --- | --- |
| `verification/heatvit_ref/descriptor.py` | 320-bit descriptor schema、合法性和 `.mem` 编码 |
| `verification/heatvit_ref/weights.py` | 确定性合成参数与 Selector 混合剪枝校准 |
| `verification/heatvit_ref/model.py` | 完整 HeatViT-T 整数推理和检查点 |
| `verification/tests/test_schedule.py` | 198 条顺序、flag、地址和动态维度测试 |
| `verification/tests/test_model.py` | 小配置流程与完整 manifest 契约测试 |
| `tools/generate_descriptors.py` | 最终 ROM、memory map 和人类可读 listing |
| `tools/generate_e2e_vectors.py` | 四区域 `.mem`、检查点和 summary 生成 |
| `sim/generated/e2e_tb_config.sv` | JSON manifest 的 XSim 可读常量 package |
| `rtl/generated/heatvit_descriptors.mem` | 默认 198×320-bit ROM 初始化 |
| `rtl/top/heatvit_descriptor_rom.sv` | 单周期同步 ROM 读取 |
| `rtl/top/heatvit_scheduler.sv` | descriptor 提交、Token 状态和控制流 |
| `rtl/top/heatvit_top.sv` | 控制锁存、Executor、内存与状态汇总 |
| `HeatViT.srcs/sources_1/new/heatvit.sv` | 原工程 Top 兼容封装 |
| `sim/tb/tb_scheduler.sv` | descriptor/state/error 单元测试 |
| `sim/tb/tb_heatvit_e2e.sv` | 完整尺寸端到端 Testbench |
| `sim/tb/tb_heatvit_errors.sv` | 七个 fatal code 与三个 warning 测试 |
| `scripts/sync_vivado_project.tcl` | 将 RTL 加入现有 `.xpr` 并设置 Top |
| `scripts/audit_no_ip.ps1` | 源码/XPR IP 实例审计 |
| `docs/verification/simulation-guide.md` | 环境、向量、运行与结果解读 |
| `docs/verification/memory-and-weight-format.md` | 四区域布局、矩阵次序、尺度和替换权重契约 |

## 锁定顶层端口

```systemverilog
module heatvit_top #(
  parameter string DESC_MEM_FILE = "rtl/generated/heatvit_descriptors.mem"
) (
  input  logic                clk,
  input  logic                rst_n,
  input  logic                start,
  input  logic [31:0]         input_base,
  input  logic [31:0]         input_bytes,
  input  logic [31:0]         weight_base,
  input  logic [31:0]         weight_bytes,
  input  logic [31:0]         scratch_base,
  input  logic [31:0]         scratch_bytes,
  input  logic [31:0]         output_base,
  input  logic [31:0]         output_bytes,
  output logic                busy,
  output logic                done,
  output logic                error_valid,
  output logic [7:0]          error_code,
  output logic [7:0]          warning_flags,
  output heatvit_pkg::heatvit_scale_t output_scale_exp,
  output logic                mem_cmd_valid,
  input  logic                mem_cmd_ready,
  output logic                mem_cmd_write,
  output logic [31:0]         mem_cmd_addr,
  output logic [15:0]         mem_cmd_len,
  output logic                mem_w_valid,
  input  logic                mem_w_ready,
  output logic [63:0]         mem_w_data,
  output logic [7:0]          mem_w_strb,
  output logic                mem_w_last,
  input  logic                mem_r_valid,
  output logic                mem_r_ready,
  input  logic [63:0]         mem_r_data,
  input  logic                mem_r_last
);
```

兼容封装 `heatvit` 必须暴露完全相同的非参数端口。完整 Testbench 固定驱动 `input_base=32'h0000_0000`、`weight_base=32'h0100_0000`、`scratch_base=32'h0200_0000`、`output_base=32'h0300_0000`；各 `*_bytes` 使用 manifest 的有效且已向上 8-byte 对齐容量。

### Task 1: 编译并验证固定 198 条描述符

**Files:**
- Modify: `verification/heatvit_ref/descriptor.py`
- Create: `verification/tests/test_schedule.py`
- Create: `tools/generate_descriptors.py`
- Create: `rtl/generated/heatvit_descriptors.mem`
- Create: `build/vectors/e2e/descriptor_listing.csv`
- Create: `build/vectors/e2e/memory_map.json`

**Interfaces:**
- Consumes: `config/heatvit_t.json`、op_sequence helpers、四个 8-byte 对齐 Tensor arenas。
- Produces: 扩展后的 `Descriptor.validate()`、`build_schedule(memory_map) -> list[Descriptor]`、`patch_sequence`、`block_sequence`、`selector_sequence`、198 行每行 80 hex digits 的 descriptor ROM 和带 label 的 listing。

- [ ] **Step 1: 写 schedule 失败测试**

断言 descriptor 总数 198、OP_FINISH 只在 index 197、Selector 起点为 42/93/144、Final LN/Head 为 195/196。每个 QK descriptor 必须 transpose+head mode+dynamic N，每个 Attention Softmax 必须 dynamic N，每个 AV 必须 head mode+flag18+dynamic K；Transformer 动态算子必须 flag3 且 `param0[1:0]=00`，每个 Selector 的前 11 条候选计算 descriptor 必须 flag3 且 `param0[1:0]=01`，第 12 条 Finalize 必须 flag3 且 `param0[1:0]=00`；分类头必须 M=1 且 flag7/15 置位。

- [ ] **Step 2: 运行并确认 descriptor 模块缺失**

Run: `& $env:HEATVIT_PYTHON -m unittest verification.tests.test_schedule -v`

Expected: FAIL，缺失 `verification.heatvit_ref.descriptor`。

- [ ] **Step 3: 实现 schema 验证和固定 schedule builder**

Schema 必须拒绝 unknown opcode、reserved 非零、scale 超出 -32..31、非 Attention×V 的 flag18、非动态算子的 flag3、零维度 GEMM 和任何未对齐 Tensor 偏移。每行 descriptor 按 packed struct 位序编码，`reserved` 位于最低 4 bit，`opcode` 位于最高 8 bit；round-trip 测试逐字段比对。

```python
def build_schedule(memory_map):
    descs = list(patch_sequence(memory_map))
    for block_index in range(1, 13):
        if block_index in (4, 7, 10):
            selector_index = {4: 1, 7: 2, 10: 3}[block_index]
            descs.extend(selector_sequence(selector_index, memory_map))
        descs.extend(block_sequence(block_index, memory_map))
    descs.extend(final_layernorm_sequence(memory_map))
    descs.extend(classifier_sequence(memory_map))
    descs.append(Descriptor.finish())
    if len(descs) != 198:
        raise ValueError(f"descriptor count {len(descs)} != 198")
    return descs
```

`final_layernorm_sequence` 和 `classifier_sequence` 各返回一条 descriptor，`Descriptor.finish()` 返回唯一 OP_FINISH；它们与前三个 sequence helpers 一并在 `tools/generate_descriptors.py` 中定义。

- [ ] **Step 4: 生成 ROM/listing 并运行索引测试**

Run:

```powershell
& $env:HEATVIT_PYTHON tools/generate_descriptors.py --config config/heatvit_t.json --rom rtl/generated/heatvit_descriptors.mem --listing build/vectors/e2e/descriptor_listing.csv --map build/vectors/e2e/memory_map.json
& $env:HEATVIT_PYTHON -m unittest verification.tests.test_schedule -v
```

Expected: 198 行 ROM、198 行数据 listing、全部测试 `ok`；四个区域无重叠，所有 Tensor 偏移为 8-byte 对齐。

- [ ] **Step 5: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'build: generate fixed HeatViT descriptor schedule' -Paths verification/heatvit_ref/descriptor.py,verification/tests/test_schedule.py,tools/generate_descriptors.py,rtl/generated/heatvit_descriptors.mem -TestCommand 'python -m unittest verification.tests.test_schedule -v'
```

### Task 2: 实现 Descriptor ROM 和动态 Scheduler

**Files:**
- Create: `rtl/top/heatvit_descriptor_rom.sv`
- Create: `rtl/top/heatvit_scheduler.sv`
- Create: `sim/tb/tb_scheduler.sv`

**Interfaces:**
- Consumes: `start`、顶层单周期 `abort`、Executor desc ready/done/error/state-update、同步 descriptor ROM。
- Produces: `exec_desc_valid/exec_desc`、busy/done/error、current token/package state、current descriptor index。

- [ ] **Step 1: 写 198 条顺序与状态更新失败测试**

用假 Executor 每次在 1 至 5 周期后 done；在 indices 53/104/155 分别返回 `(N,package)=(150,1),(90,1),(48,1)`。TB 断言下一动态 descriptor 看到更新 N，普通 descriptor 的 state_update_valid 被忽略且触发内部 assertion；index 197 只产生顶层 done 而不提交新内存操作。

- [ ] **Step 2: 运行并确认 Scheduler 缺失**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_scheduler`

Expected: FAIL，缺失 `heatvit_scheduler`。

- [ ] **Step 3: 实现 ROM pipeline 和状态机**

FSM 固定为 `IDLE -> ROM_REQ -> ROM_WAIT -> EXEC_ISSUE -> EXEC_WAIT -> ADVANCE -> COMPLETE/ERROR`。`abort` 在任意非 IDLE 状态立即回到 IDLE、撤销尚未握手的 descriptor valid 且不产生 done。start 锁存后 N=197/package=0/index=0，Patch 最终输出固定写 Activation A。只有 index 53/104/155 的 OP_SELECTOR_FINALIZE done 可接受 state update。若 next N 不在 2..197 或大于 current N，报 error 4。每个 Block 的 Residual2 和每个 Selector Finalize 共 15 条 descriptor 置 flag4、写当前 inactive Activation Buffer；成功后切换 activation select。其他 descriptor 的 flag4 必须为零，Final LN 读取切换后的 current Buffer。

```systemverilog
case (state)
  IDLE: if (start) begin
    current_token_count <= 8'd197;
    current_package_present <= 1'b0;
    desc_index <= 16'd0;
    state <= ROM_REQ;
  end
  ROM_REQ:    state <= ROM_WAIT;
  ROM_WAIT:   state <= (rom_desc.opcode == OP_FINISH) ? COMPLETE : EXEC_ISSUE;
  EXEC_ISSUE: if (exec_desc_ready) state <= EXEC_WAIT;
  EXEC_WAIT:  if (exec_error_valid) state <= ERROR;
              else if (exec_done) state <= ADVANCE;
  ADVANCE: begin desc_index <= desc_index + 1'b1; state <= ROM_REQ; end
  COMPLETE: begin done <= 1'b1; state <= IDLE; end
  ERROR: begin error_valid <= 1'b1; state <= IDLE; end
endcase
if (abort) begin state <= IDLE; exec_desc_valid <= 1'b0; done <= 1'b0; end
```

- [ ] **Step 4: 运行正常、非法状态和 Executor error 测试**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_scheduler`

Expected: `TEST_PASS tb_scheduler`；正常恰好提交 indices 0..196，非法 N 返回 4，Executor error 原码透传，FINISH 产生单周期 done。

- [ ] **Step 5: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'feat: add descriptor ROM and dynamic scheduler' -Paths rtl/top/heatvit_descriptor_rom.sv,rtl/top/heatvit_scheduler.sv,sim/tb/tb_scheduler.sv -TestCommand 'scripts/run_xsim.ps1 -Top tb_scheduler'
```

### Task 3: 集成 heatvit_top、兼容封装和 Vivado 工程源文件

**Files:**
- Create: `rtl/top/heatvit_top.sv`
- Modify: `HeatViT.srcs/sources_1/new/heatvit.sv`
- Create: `sim/tb/tb_heatvit_top_smoke.sv`
- Create: `scripts/sync_vivado_project.tcl`

**Interfaces:**
- Consumes: 批准规格第 7 节全部顶层端口。
- Produces: `heatvit_top` 和同端口透传的 `heatvit` 工程 Top。

- [ ] **Step 1: 写 reset/start/端口失败测试**

同步 `rst_n=0` 两拍后检查 busy/done/error/warnings 全零；合法 start 必须锁存 base/bytes，即使 TB 下一拍改变输入也不影响 memory trace；busy 时 start 预期 error 7、停止新命令、若有已接受 Burst 则排空后 busy 清零，且 done 不产生。

- [ ] **Step 2: 运行并确认 top 缺失或空端口**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_heatvit_top_smoke`

Expected: FAIL，现有空 `heatvit` 无法连接规定端口或缺失 `heatvit_top`。

- [ ] **Step 3: 实现顶层锁存与错误/警告规则**

`heatvit_top` 实例化一个 Scheduler 和一个 Tensor Executor。合法 start 清 error/warning；Executor warning pulse OR 进 8-bit 锁存寄存器低三位；任何 error 同时向 Scheduler/Executor 发 abort 并置 `abort_pending`，Executor 排空已握手 Burst期间不得产生新 command，收到 `exec_abort_done` 后清 `abort_pending`。顶层 `busy = scheduler_busy | abort_pending`，因此错误后的 busy 只持续到协议安全点；成功 FINISH 产生一拍 done。分类 descriptor 完成时把 `desc.dst_scale_exp` 锁存到 signed `output_scale_exp[5:0]`。

```systemverilog
always_ff @(posedge clk) begin
  if (!rst_n) begin
    warning_flags <= 8'd0;
    error_valid <= 1'b0;
    error_code <= ERR_NONE;
    done <= 1'b0;
    scheduler_abort <= 1'b0;
    executor_abort <= 1'b0;
    abort_pending <= 1'b0;
  end else begin
    done <= 1'b0;
    scheduler_abort <= 1'b0;
    executor_abort <= 1'b0;
    if (exec_abort_done) abort_pending <= 1'b0;
    if (start && !busy) begin
      warning_flags <= 8'd0;
      error_valid <= 1'b0;
      error_code <= ERR_NONE;
      input_base_reg <= input_base;
      input_bytes_reg <= input_bytes;
      weight_base_reg <= weight_base;
      weight_bytes_reg <= weight_bytes;
      scratch_base_reg <= scratch_base;
      scratch_bytes_reg <= scratch_bytes;
      output_base_reg <= output_base;
      output_bytes_reg <= output_bytes;
    end else begin
      warning_flags[2:0] <= warning_flags[2:0] | exec_warning_pulse;
    end
    if (start && busy) begin
      error_valid <= 1'b1;
      error_code <= ERR_BUSY_START;
      scheduler_abort <= 1'b1;
      executor_abort <= 1'b1;
      abort_pending <= 1'b1;
    end
    if (exec_error_valid) begin
      error_valid <= 1'b1;
      error_code <= exec_error_code;
      scheduler_abort <= 1'b1;
      executor_abort <= 1'b1;
      abort_pending <= 1'b1;
    end
    if (scheduler_error_valid) begin
      error_valid <= 1'b1;
      error_code <= scheduler_error_code;
      scheduler_abort <= 1'b1;
      executor_abort <= 1'b1;
      abort_pending <= 1'b1;
    end
  end
end
assign busy = scheduler_busy | abort_pending;
```

- [ ] **Step 4: 把现有 heatvit.sv 改成薄封装**

文件只保留 timescale、完整顶层端口和一例 `heatvit_top u_top` 的一一命名连接，不放控制或计算逻辑。Tcl 脚本打开 `HeatViT.xpr`，分别对 `rtl/include/*.sv`、`rtl/common/*.sv`、`rtl/memory/*.sv`、`rtl/compute/*.sv`、`rtl/selector/*.sv`、`rtl/top/*.sv` 执行 `glob -nocomplain` 后用 `add_files -norecurse` 添加，并加入 ROM `.mem`；设置 sources_1 top 为 `heatvit`，更新 compile order，保存并关闭。不得创建 IP catalog 对象。

```systemverilog
heatvit_top u_top (
  .clk, .rst_n, .start,
  .input_base, .input_bytes, .weight_base, .weight_bytes,
  .scratch_base, .scratch_bytes, .output_base, .output_bytes,
  .busy, .done, .error_valid, .error_code, .warning_flags,
  .output_scale_exp,
  .mem_cmd_valid, .mem_cmd_ready, .mem_cmd_write, .mem_cmd_addr, .mem_cmd_len,
  .mem_w_valid, .mem_w_ready, .mem_w_data, .mem_w_strb, .mem_w_last,
  .mem_r_valid, .mem_r_ready, .mem_r_data, .mem_r_last
);
```

- [ ] **Step 5: 运行 smoke 并同步工程**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_heatvit_top_smoke
& "$env:HEATVIT_VIVADO_BIN\vivado.bat" -mode batch -source scripts/sync_vivado_project.tcl -tclargs HeatViT.xpr
```

Expected: `TEST_PASS tb_heatvit_top_smoke`；Vivado Tcl 返回 0，工程 Part 仍为 `xc7k325tfbg900-3`、Top 为 `heatvit`。

- [ ] **Step 6: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'feat: integrate HeatViT inference top' -Paths rtl/top/heatvit_top.sv,HeatViT.srcs/sources_1/new/heatvit.sv,sim/tb/tb_heatvit_top_smoke.sv,scripts/sync_vivado_project.tcl,HeatViT.xpr -TestCommand 'scripts/run_xsim.ps1 -Top tb_heatvit_top_smoke'
```

### Task 4: 实现完整黄金模型和确定性混合剪枝参数生成

**Files:**
- Create: `verification/heatvit_ref/weights.py`
- Create: `verification/heatvit_ref/model.py`
- Create: `verification/tests/test_model.py`
- Create: `tools/generate_e2e_vectors.py`

**Interfaces:**
- Consumes: Patch/Transformer/Selector 黄金函数和固定模型配置。
- Produces: immutable `HeatViTParams` 与 `ModelResult` dataclasses、`HeatViTModel.infer(image, params) -> ModelResult`、确定性完整权重、所有检查点和 selector summary。

- [ ] **Step 1: 写流程顺序和 checkpoint 失败测试**

用缩小测试配置 D=12、3 Head、2 Block、一个 Selector 验证调用顺序和 checkpoint keys；正式配置必须有 `patch`、`block_01` 至 `block_12`、`selector_01` 至 `selector_03`、`final_ln`、`logits` 共 18 个强制检查点，并记录每个 scale、shape、SHA-256。

- [ ] **Step 2: 运行并确认 model/weights 缺失**

Run: `& $env:HEATVIT_PYTHON -m unittest verification.tests.test_model -v`

Expected: FAIL，缺失 `verification.heatvit_ref.model` 或 `weights`。

- [ ] **Step 3: 实现逐层模型和参数生成器**

所有普通参数由 `random.Random(20260815)` 生成：主权重范围 `[-8,7]`，普通 Bias 范围 `[-64,63]`，Gamma 以 64 为中心并钳入 int8，Beta 范围 `[-4,4]`。生成顺序按 patch、block index、selector index、final LN、head 固定。合成量化表固定为 input/activation/weight/CLS/position/Beta exponent=-7、Gamma exponent=-6、Q8.16=-16、Attention UQ0.8=-8、Selector Q0.16=-16、Logit accumulator=-14；Bias exponent 始终由对应两输入 exponent 之和导出。每层参数记录 shape、scale 和 byte offset。模型每层只调用已经单元测试过的整数函数；NumPy 数组必须显式为 integer dtype，每次 matmul 后先验证范围再缩窄，禁止 float/complex 和其他推理框架。

```python
def infer(self, image, params):
    self.package_present = False
    x = patch_embedding(image, params.patch)
    checkpoints = {"patch": x.copy()}
    selector_summary = []
    selector_number = 0
    for block_number in range(1, 13):
        if block_number in (4, 7, 10):
            selector_number += 1
            input_count = len(x)
            selected = token_selector(x, self.package_present,
                                      params.selectors[selector_number - 1])
            x = selected.tokens
            self.package_present = selected.package_present
            selector_summary.append({"input_tokens": input_count,
                                     "output_tokens": len(x),
                                     "kept_normal": selected.kept_normal_count,
                                     "pruned_normal": selected.pruned_normal_count,
                                     "package_present": self.package_present})
            checkpoints[f"selector_{selector_number:02d}"] = x.copy()
        x, _ = transformer_block(x, params.blocks[block_number - 1])
        checkpoints[f"block_{block_number:02d}"] = x.copy()
    final_ln = layer_norm(x, params.final_norm)
    logits = gemm([final_ln[0]], params.head.weight, params.head.bias, False)[0]
    checkpoints["final_ln"] = final_ln
    checkpoints["logits"] = logits
    return ModelResult(logits=logits, output_scale_exp=-14,
                       checkpoints=checkpoints, selector_summary=selector_summary)
```

- [ ] **Step 4: 确定性校准三个 Selector 的混合剪枝**

对 stage 1..3，依次尝试 attempt 0..255 的 score 最后一层权重 seed `20260815 + stage*1000 + attempt`；Head weight 分支设为确定性相等权重。计算每候选无 Bias keep-drop logit 差，若所有差相等则试下一 attempt；否则把 keep/drop Bias 差设为负的中位 logit 差，并完整运行 Softmax/fuse/finalize。接受条件为至少 1 个 kept normal、至少 2 个 pruned normal；首个满足者写入 manifest。256 次均失败时生成器返回非零并打印 stage，禁止放宽条件。

- [ ] **Step 5: 运行小模型测试并生成完整黄金结果**

Run:

```powershell
& $env:HEATVIT_PYTHON -m unittest verification.tests.test_model -v
& $env:HEATVIT_PYTHON tools/generate_e2e_vectors.py --seed 20260815 --output build/vectors/e2e
```

Expected: unittest 全部 `ok`；generator 返回 0；summary 显示三个 Selector 均 `kept_normal>=1`、`pruned_normal>=2` 且 N 严格下降。

- [ ] **Step 6: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'test: add complete integer HeatViT golden model' -Paths verification/heatvit_ref/weights.py,verification/heatvit_ref/model.py,verification/tests/test_model.py,tools/generate_e2e_vectors.py -TestCommand 'python -m unittest verification.tests.test_model -v'
```

### Task 5: 固化端到端内存映像和 manifest 契约

**Files:**
- Create: `verification/tests/test_e2e_manifest.py`
- Modify: `tools/generate_e2e_vectors.py`
- Create: `build/vectors/e2e/input.mem`
- Create: `build/vectors/e2e/weights.mem`
- Create: `build/vectors/e2e/scratch_init.mem`
- Create: `build/vectors/e2e/output_init.mem`
- Create: `build/vectors/e2e/checkpoints/`
- Create: `build/vectors/e2e/manifest.json`
- Create: `sim/generated/e2e_tb_config.sv`

**Interfaces:**
- Consumes: 完整 ModelResult、descriptor listing、memory map。
- Produces: 行为存储可直接加载的四区域映像、JSON manifest 和 XSim 可直接 import 的 `e2e_tb_config_pkg`。

- [ ] **Step 1: 写 manifest 失败测试**

测试强制检查 seed、part、198 descriptor hash、四区域 base/bytes、18 个 checkpoint 的 descriptor index/offset/bytes/scale/hash、三个 Selector 的 in/out N/package、expected logits scale、watchdog_cycles。另解析 `sim/generated/e2e_tb_config.sv`，逐项证明其中整数常量与 JSON 相同；删除任一键或改动任一 SV 常量时测试必须明确指出键路径。

- [ ] **Step 2: 运行并确认现有 manifest 不完整**

Run: `& $env:HEATVIT_PYTHON -m unittest verification.tests.test_e2e_manifest -v`

Expected: FAIL，报告第一个缺失字段。

- [ ] **Step 3: 实现内存映像和 watchdog 计算**

四个区域映像和所有 checkpoint `.mem` 每行一个 64-bit 小端 Beat，最低地址 byte 位于该行最低 8 bit；最后 Beat 用零填充，manifest 另记录原始有效 bytes。四个固定 base 分别为 `0x00000000/0x01000000/0x02000000/0x03000000`，生成器必须证明实际容量不跨入下一 base。watchdog 计算为：

```text
gemm_work = sum over descriptors of ceil(M/8) * ceil(N/24) * K
memory_beats = sum over descriptor declared reads and writes of ceil(bytes/8)
nonlinear_work = total LayerNorm elements*4 + Softmax elements*4 + Selector elements*4
watchdog_cycles = 4*(gemm_work + memory_beats + nonlinear_work) + 10_000_000
```

所有动态 M/N/K 使用黄金运行时实际 Token 数代入。生成器写完后重新读取四个 `.mem` 并验证 SHA-256，避免只验证内存中的对象。`e2e_tb_config_pkg` 固定包含四区域 base/bytes、WATCHDOG_CYCLES、OUTPUT_SCALE_EXP、三个 SELECTOR_IN_N/OUT_N/PACKAGE，以及 18 个 CHECKPOINT_DESC_INDEX/OFFSET/BYTES/SCALE 数组；检查点 index 固定为 `2,15,28,41,53,66,79,92,104,117,130,143,155,168,181,194,195,196`。

- [ ] **Step 4: 重生成并运行 manifest 契约测试**

Run:

```powershell
& $env:HEATVIT_PYTHON tools/generate_e2e_vectors.py --seed 20260815 --output build/vectors/e2e
& $env:HEATVIT_PYTHON -m unittest verification.tests.test_e2e_manifest -v
```

Expected: 全部 `ok`；两次生成的全部 hash、Selector counts 和 watchdog_cycles 相同。

- [ ] **Step 5: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'test: lock end-to-end vector manifest' -Paths verification/tests/test_e2e_manifest.py,tools/generate_e2e_vectors.py,sim/generated/e2e_tb_config.sv -TestCommand 'python -m unittest verification.tests.test_e2e_manifest -v'
```

### Task 6: 运行完整尺寸无回压端到端 XSim

**Files:**
- Create: `sim/tb/tb_heatvit_e2e.sv`
- Modify: `sim/common/behavioral_memory.sv`
- Modify: `scripts/run_regression.ps1`

**Interfaces:**
- Consumes: 四区域映像、198 条 ROM、`e2e_tb_config_pkg`、JSON manifest 和完整 heatvit Top。
- Produces: 18 个逐位检查点比较、最终日志和 `TEST_PASS tb_heatvit_e2e`。

- [ ] **Step 1: 写端到端 Testbench 并确认首次失败**

TB 加载四区域映像，驱动批准顶层接口，并用 Testbench 内的层次化 monitor 观察 `dut.u_top.u_scheduler.current_desc_index` 与 Executor done，在规定 index 读取对应 checkpoint 区；不得为此向 RTL 添加仿真专用端口或条件编译逻辑。首次运行应因尚未完成的 checkpoint 比较或连接错误而失败，不能先屏蔽比较。

- [ ] **Step 2: 实现全部检查点与协议断言**

TB import `e2e_tb_config_pkg`，不在 SystemVerilog 中解析 JSON。每个 byte/int32 mismatch 打印 checkpoint、线性 index、期望/实际和当前 descriptor。断言 valid stall payload 稳定、有效控制/数据无 X/Z、所有 memory 命令在区域内、N 在 2..197、package 状态与 package 常量相同。达到 WATCHDOG_CYCLES 时 `$fatal`。

```systemverilog
assert property (@(posedge clk) mem_cmd_valid && !mem_cmd_ready |=>
                 $stable({mem_cmd_write,mem_cmd_addr,mem_cmd_len}));
assert property (@(posedge clk) mem_w_valid && !mem_w_ready |=>
                 $stable({mem_w_data,mem_w_strb,mem_w_last}));
assert property (@(posedge clk) busy |-> !$isunknown({busy,error_valid,mem_cmd_valid,mem_w_valid,mem_r_ready}));
always_ff @(posedge clk) begin
  if (cycle_count >= WATCHDOG_CYCLES) $fatal(1, "e2e watchdog");
end
```

- [ ] **Step 3: 运行无回压完整推理**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_heatvit_e2e -PlusArgs '+VECTOR_DIR=build/vectors/e2e +STALL_MASK=0'
```

Expected: `TEST_PASS tb_heatvit_e2e`；18 个 checkpoint、1000 Logit、output_scale_exp、三个 N/package 状态全部逐位一致，done 恰好一拍，error_valid=0。

- [ ] **Step 4: 将 e2e-no-stall 加入回归但不重复生成向量**

`run_regression.ps1 -Suite e2e` 先验证 manifest hash，只在 `-RegenerateVectors` 显式指定时调用生成器，然后运行 e2e。这样失败重跑不会无意更换预期值。

```powershell
if ($RegenerateVectors) {
  & $env:HEATVIT_PYTHON tools/generate_e2e_vectors.py --seed 20260815 --output build/vectors/e2e
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& $env:HEATVIT_PYTHON -m unittest verification.tests.test_e2e_manifest -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_heatvit_e2e -PlusArgs '+VECTOR_DIR=build/vectors/e2e +STALL_MASK=0'
```

- [ ] **Step 5: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'test: pass full-size HeatViT end-to-end simulation' -Paths sim/tb/tb_heatvit_e2e.sv,sim/common/behavioral_memory.sv,scripts/run_regression.ps1 -TestCommand 'scripts/run_regression.ps1 -Suite e2e'
```

### Task 7: 验证随机回压、全部错误码和警告锁存

**Files:**
- Create: `sim/tb/tb_heatvit_errors.sv`
- Modify: `sim/tb/tb_heatvit_e2e.sv`
- Modify: `scripts/run_regression.ps1`

**Interfaces:**
- Consumes: 可替换 descriptor ROM 参数、memory error injection 和 warning test vectors。
- Produces: error code 1..7、warning bit 0..2 以及 e2e backpressure 回归。

- [ ] **Step 1: 写错误矩阵失败测试**

用七个最小 ROM/注入案例分别触发：unknown opcode、非法维度、未对齐/越界地址、非法 Token update、read last 长度错误、Softmax zero-sum 内部注入、busy start。每例断言新命令停止、busy=0、error_valid=1、code 精确、done=0。

- [ ] **Step 2: 写警告锁存失败测试**

分别触发 Head 零分母、Package 零分母、LayerNorm 负方差；警告发生后继续成功操作仍保持，下一合法 start 清零，reset 也清零。warning 不得阻止最终 done。

- [ ] **Step 3: 实现 ROM/memory 仿真注入参数并运行**

注入逻辑只存在于 TB 和 behavioral memory；任何 `FORCE_*` plusarg 不得进入 `rtl/`。Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_heatvit_errors
```

Expected: `TEST_PASS tb_heatvit_errors`，七个错误和三个警告各命中一次。

- [ ] **Step 4: 运行完整随机回压推理**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_heatvit_e2e -PlusArgs '+VECTOR_DIR=build/vectors/e2e +STALL_MASK=3'
```

Expected: 与无回压运行相同的所有 checkpoint/hash/count/logits，cycle count 可以不同但小于 watchdog。

- [ ] **Step 5: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'test: cover HeatViT errors warnings and backpressure' -Paths sim/tb/tb_heatvit_errors.sv,sim/tb/tb_heatvit_e2e.sv,scripts/run_regression.ps1 -TestCommand 'scripts/run_xsim.ps1 -Top tb_heatvit_errors'
```

### Task 8: 完成全回归、无 IP 审计和仿真交付文档

**Files:**
- Create: `scripts/audit_no_ip.ps1`
- Create: `docs/verification/simulation-guide.md`
- Create: `docs/verification/memory-and-weight-format.md`
- Create: `docs/verification/e2e-results.md`
- Modify: `scripts/run_regression.ps1`

**Interfaces:**
- Consumes: 全部阶段测试、现有 `HeatViT.xpr` 和最终日志。
- Produces: `run_regression.ps1 -Suite all`、`build/reports/ip_audit.txt`、可复现仿真说明和结果摘要。

- [ ] **Step 1: 写 IP 审计脚本并确认禁止模式为零**

脚本递归收集 `rtl` 下所有 `.sv` 并连同 `HeatViT.xpr` 搜索 `xpm_`、`blk_mem_gen`、`floating_point`、`div_gen`、`axi_`、`IPSources`；另用 `Get-ChildItem -Recurse -Filter *.xci` 检查文件名。任何命中返回 1，否则写 `NO_MANUAL_VIVADO_IP_REQUIRED` 到报告。

- [ ] **Step 2: 完成仿真指南和结果边界**

仿真指南给出环境变量、向量生成、单套件/全套件命令、日志位置、常见失败定位和预计长仿真说明。内存/权重文档逐 Tensor 列出 shape、行优先/per-head 次序、byte offset、scale exponent、Bias 规则、64-bit `.mem` 小端格式和替换约束：新权重只有在 shape/layout/scale 与 descriptor 一致时可直接替换，否则必须重生成 descriptor 与黄金检查点。结果文档记录实际三个 Token count、warning/error、checkpoint hash、Logit hash 和 cycle count，并明确：合成权重无分类意义、未验证 ImageNet 准确率、时序、功耗、FPS 或上板功能。

```text
simulation-guide.md: prerequisites -> vector generation -> unit suites -> e2e -> failure triage
memory-and-weight-format.md: region map -> tensor table -> scale table -> .mem encoding -> replacement rules
e2e-results.md: tool versions -> selector counts -> checkpoint hashes -> logits hash -> cycle count -> exclusions
```

- [ ] **Step 3: 运行最终全回归**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/audit_no_ip.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_regression.ps1 -Suite all
```

Expected: 两条命令均退出 0；foundation、gemm、transformer、selector、scheduler、errors、e2e no-stall 和 e2e backpressure 全部通过。

- [ ] **Step 4: 检查最终交付清单**

Run:

```powershell
Get-ChildItem build/reports
Select-String -LiteralPath build/reports/ip_audit.txt -Pattern '^NO_MANUAL_VIVADO_IP_REQUIRED$'
Select-String -LiteralPath docs/verification/e2e-results.md -Pattern 'ImageNet|时序|上板'
```

Expected: 报告目录包含 regression summary、e2e summary、IP audit；两个 Select-String 均成功。

- [ ] **Step 5: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'docs: finalize simulated HeatViT inference delivery' -Paths scripts/audit_no_ip.ps1,scripts/run_regression.ps1,docs/verification/simulation-guide.md,docs/verification/memory-and-weight-format.md,docs/verification/e2e-results.md -TestCommand 'scripts/run_regression.ps1 -Suite all'
```

## 阶段 5 与项目完成条件

- `scripts/run_regression.ps1 -Suite all` 退出码为 0。
- 198 条 descriptor 顺序、三个动态 state update 和所有地址检查通过。
- 无回压与随机回压完整 224×224 推理的 18 个检查点和 1000 Logit 完全一致。
- error 1..7、warning 0..2 全覆盖且协议符合规格。
- Vivado 工程 Part 为 `xc7k325tfbg900-3`、Top 为 `heatvit`，所有设计逻辑为可综合 SystemVerilog。
- `ip_audit.txt` 明确当前不需要用户手动生成任何 Vivado IP。
- 交付结论严格限定为“仿真逐位通过”，不扩展为准确率或板级性能声明。
