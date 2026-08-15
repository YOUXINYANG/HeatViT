# HeatViT-T Phase 3: Transformer Datapath Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在单一 GEMM 引擎上完成 Patch Embedding、CLS/位置编码、MHSA、FFN 和完整 Pre-LN Transformer Block，并对 HeatViT-T 实际维度逐位验证。

**Architecture:** 高层算子不复制计算核，而是向 `heatvit_tensor_executor` 提交 320-bit 描述符。Executor 解码 GEMM、LayerNorm、Residual、Softmax 和布局转换，持有唯一 GEMM、唯一共享 divider 及必要矢量/布局引擎；组件 Testbench 读取短描述符序列来验证 Patch、MHSA、FFN 和 Block。

**Tech Stack:** SystemVerilog 2012、Vivado XSim 2023.2、Python 3.12–3.14、NumPy 2.5.2、阶段 1 数值单元、阶段 2 GEMM/行为存储。

## Global Constraints

- 固定维度为 D=192、3 Head、Head Dim=64、FFN=768；动态 N 范围为 2 至 197。
- Block 固定采用 `Y=X+MSA(LN(X))`、`Z=Y+FFN(LN(Y))` 的 Pre-LN 顺序。
- Q/K/V 使用一个 `[192][576]` 行优先权重，列顺序为 Q、K、V；布局引擎输出 `[kind][head][token][64]`。
- Attention Score 存为 little-endian signed int32；其 scale exponent 必须包含 `1/sqrt(64)=1/8` 的 `-3` 指数调整。
- Attention 概率存为 UQ0.8 byte；Attention×V 必须置 descriptor flag 18，按 unsigned×signed 计算。
- Context 临时布局为 `[head][token][64]`；Head concat 后恢复 `[token][192]`。
- 所有中间 Tensor 位于 Scratch 区，并由 8-byte 对齐 allocator 分配；不得用固定数组容纳完整 N×N×3 Score。
- 每个组件测试必须同时比较结果、写入范围、输出尺度和 memory command trace。
- 不实例化第二个 GEMM 或第二个 divider；不使用任何 Vivado IP。

---

## 文件映射

| 文件 | 单一职责 |
| --- | --- |
| `verification/heatvit_ref/memory.py` | 8-byte 对齐 Tensor arena 和小端读写 |
| `verification/heatvit_ref/layout.py` | NHWC Patchify、CLS/位置、QKV unpack、Head concat |
| `verification/heatvit_ref/descriptor.py` | 320-bit descriptor 字段与 packed bit-order pack/unpack |
| `verification/heatvit_ref/transformer.py` | 整数 Patch/MHSA/FFN/Block 黄金运算 |
| `verification/heatvit_ref/op_sequence.py` | 生成组件级 320-bit descriptor 序列 |
| `verification/tests/test_layout.py` | 布局与 arena 单元测试 |
| `verification/tests/test_transformer.py` | Transformer 黄金模型测试 |
| `rtl/compute/heatvit_layout_engine.sv` | Patchify、Add Pos、QKV unpack 和 Head concat |
| `rtl/compute/heatvit_vector_engine.sv` | Memory-streaming requant、Residual、GELU 和 LayerNorm 适配 |
| `rtl/compute/heatvit_tensor_executor.sv` | 单描述符解码、子单元仲裁和错误传播 |
| `tools/generate_transformer_vectors.py` | 组件权重、输入、descriptor 和检查点生成 |
| `sim/tb/tb_patch_embedding.sv` | 完整 196 Patch 测试 |
| `sim/tb/tb_mhsa.sv` | 三 Head Attention 测试 |
| `sim/tb/tb_ffn.sv` | 192→768→192 测试 |
| `sim/tb/tb_transformer_block.sv` | 完整 Pre-LN Block 测试 |

## 锁定 Tensor Executor 接口

```systemverilog
module heatvit_tensor_executor (
  input  logic                 clk,
  input  logic                 rst_n,
  input  logic                 abort,
  input  logic                 desc_valid,
  output logic                 desc_ready,
  input  heatvit_pkg::heatvit_desc_t desc,
  input  logic [7:0]           current_token_count,
  input  logic                 current_package_present,
  input  logic [31:0]          input_base,
  input  logic [31:0]          input_bytes,
  input  logic [31:0]          weight_base,
  input  logic [31:0]          weight_bytes,
  input  logic [31:0]          scratch_base,
  input  logic [31:0]          scratch_bytes,
  input  logic [31:0]          output_base,
  input  logic [31:0]          output_bytes,
  output logic                 busy,
  output logic                 done,
  output logic                 error_valid,
  output logic [7:0]           error_code,
  output logic                 abort_done,
  output logic [2:0]           warning_pulse,
  output logic                 state_update_valid,
  output logic [7:0]           next_token_count,
  output logic                 next_package_present,
  output logic                 mem_cmd_valid,
  input  logic                 mem_cmd_ready,
  output logic                 mem_cmd_write,
  output logic [31:0]          mem_cmd_addr,
  output logic [15:0]          mem_cmd_len,
  output logic                 mem_w_valid,
  input  logic                 mem_w_ready,
  output logic [63:0]          mem_w_data,
  output logic [7:0]           mem_w_strb,
  output logic                 mem_w_last,
  input  logic                 mem_r_valid,
  output logic                 mem_r_ready,
  input  logic [63:0]          mem_r_data,
  input  logic                 mem_r_last
);
```

Executor 每次只接受一个 descriptor；`done` 与 `error_valid` 互斥且各一拍。`abort` 立即禁止新的 child/memory request，并让 memory master 排空已握手 Burst；排空后 Executor 回到 IDLE 并向顶层脉冲 `abort_done`，整个取消过程不产生 done/error。flag 3 置位时依据 `param0[1:0]` 把 descriptor.m 覆盖为 current N 或 N−1，`2'b10/11` 报 error 2；flags 19/20 分别把 descriptor.n/k 覆盖为 current N。未置对应 flag 的维度不得动态改写。阶段 3 的 opcode 不改变 Token 状态，因此 `state_update_valid=0`；阶段 4 的原子 Selector Finalize 才允许更新 `next_token_count/next_package_present`。

阶段 3 opcode 的字段语义固定为：

| Opcode | 有效维度 | 地址角色 |
| --- | --- | --- |
| OP_PATCHIFY | `m=196,n=768` | src0=Input NHWC，dst=Scratch patch matrix |
| OP_COPY_ADD_POS | `m=197,n=192` | src0=patch embed，src1=position Weight，aux=CLS Weight，dst=Activation A |
| OP_GEMM | `m/n/k` 为矩阵维度 | src0=A，src1=B，bias=可选 Bias，aux=可选 Residual，dst=C |
| OP_LAYERNORM | `m=N,n=192` | src0=activation，src1=Gamma Weight，aux=Beta Weight，dst=normalized |
| OP_RESIDUAL | `m=N,n=192` | src0=branch，aux=residual，dst=sum |
| OP_QKV_UNPACK | `m=N,n=576,heads=3` | src0=fused QKV，dst=`[kind][head][token][64]` |
| OP_HEAD_CONCAT | `m=N,n=192,heads=3` | src0=`[head][token][64]`，dst=`[token][192]` |
| OP_ATTN_SOFTMAX | `m=N,n=N,heads=3` | src0=`[head][N][N]` int32 Score，dst=UQ0.8 probability |

未列出的地址字段必须为零；所需 Weight 地址必须选 Weight region，所有临时值必须选 Scratch region。

### Task 1: 实现 Tensor arena、布局黄金模型和描述符打包

**Files:**
- Create: `verification/heatvit_ref/memory.py`
- Create: `verification/heatvit_ref/layout.py`
- Create: `verification/heatvit_ref/descriptor.py`
- Create: `verification/heatvit_ref/op_sequence.py`
- Create: `verification/tests/test_layout.py`

**Interfaces:**
- Consumes: Phase 1 fixed types、320-bit descriptor 字段定义。
- Produces: immutable `Descriptor` dataclass（`pack() -> int`、`unpack(word) -> Descriptor`、`finish() -> Descriptor`）、`TensorArena.allocate(name, byte_count) -> int`、四个布局函数和 80-hex-digit formatter。

- [ ] **Step 1: 写布局失败测试**

用 4×4×3 图像和 patch=2 验证第一个 Patch 展平顺序为像素 `(0,0),(0,1),(1,0),(1,1)`，每像素 R/G/B 连续；验证 QKV 输入 `[token][Q192,K192,V192]` 被转换为 `[kind][head][token][64]`；验证 concat 是精确逆变换。Arena 连续分配 1、8、9 bytes 时偏移必须是 0、8、16。

- [ ] **Step 2: 运行并确认模块导入失败**

Run: `& $env:HEATVIT_PYTHON -m unittest verification.tests.test_layout -v`

Expected: FAIL，缺失 `verification.heatvit_ref.memory` 或 `layout`。

- [ ] **Step 3: 实现布局和 descriptor 小端打包**

关键索引必须使用：

```python
image_index = ((row * width) + col) * 3 + channel
patch_index = (((patch_row * patch) + in_row) * patch + in_col) * 3 + channel
qkv_index = token * 576 + kind * 192 + head * 64 + lane
head_major_index = ((kind * 3 + head) * tokens + token) * 64 + lane
concat_index = token * 192 + head * 64 + lane
```

Descriptor 按 `heatvit_desc_t` packed 位序显式移位：`reserved` 占最低 4 bit、`opcode` 占最高 8 bit；`.mem` 每行用恰好 80 个高位在左的十六进制字符表示一个 320-bit word。测试解包后逐字段比对，禁止依赖 Python struct 的本机对齐或字节序。

- [ ] **Step 4: 运行布局测试和 descriptor round-trip**

Run: `& $env:HEATVIT_PYTHON -m unittest verification.tests.test_layout -v`

Expected: 全部 `ok`，包括 N=197、D=192 的 QKV round-trip 和所有分配偏移 `% 8 == 0`。

- [ ] **Step 5: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'test: define transformer tensor layouts' -Paths verification/heatvit_ref/memory.py,verification/heatvit_ref/layout.py,verification/heatvit_ref/descriptor.py,verification/heatvit_ref/op_sequence.py,verification/tests/test_layout.py -TestCommand 'python -m unittest verification.tests.test_layout -v'
```

### Task 2: 实现 Layout/Vector Engine 和单描述符 Executor

**Files:**
- Create: `rtl/compute/heatvit_layout_engine.sv`
- Create: `rtl/compute/heatvit_vector_engine.sv`
- Create: `rtl/compute/heatvit_tensor_executor.sv`
- Create: `sim/tb/tb_tensor_executor.sv`

**Interfaces:**
- Consumes: 锁定 Executor 接口、阶段 1 数值客户端、阶段 2 GEMM 和 memory master。
- Produces: OP_PATCHIFY、OP_COPY_ADD_POS、OP_GEMM、OP_LAYERNORM、OP_RESIDUAL、OP_QKV_UNPACK、OP_HEAD_CONCAT、OP_ATTN_SOFTMAX 的执行能力。

- [ ] **Step 1: 写 opcode、动态 M 和错误传播失败测试**

逐一提交上述八种 opcode 的最小合法 descriptor；提交 opcode `8'hff` 预期 error 1；提交 reserved 非零、N=0、Head 模式 heads=2 或 LayerNorm input scale=+1 预期 error 2；flag 3、`param0[1:0]=00` 下 descriptor.m=99、current_token_count=13 时有效 M 必须为 13，`param0[1:0]=01` 时为 12，`10/11` 必须报 error 2。另用 flags 19/20 把 descriptor.n/k 从 99 覆盖为 13，并从 memory trace 证明两个地址循环均使用覆盖值。分别在 descriptor 接受前、memory command handshake 后拉高 abort，断言无新命令且 Executor 最终回到 desc_ready。

- [ ] **Step 2: 运行并确认 Executor 缺失**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_tensor_executor`

Expected: FAIL，缺失 `heatvit_tensor_executor`。

- [ ] **Step 3: 实现解码、地址解析和唯一资源实例**

Executor 内只实例化一个 `heatvit_gemm_engine`、一个 `heatvit_mem_master`、一个 `heatvit_udiv` 和一个 `heatvit_div_arbiter`。Softmax 接 client 0、LayerNorm 接 client 1、client 2 暂时拉低请求。descriptor 接受后先验证 opcode/flags/维度/地址，再进入 `DISPATCH -> WAIT_CHILD -> COMPLETE`；任何 child error 立即转 `ERROR` 且停止新命令。

```systemverilog
case (desc_reg.opcode)
  OP_GEMM:          child_sel <= CHILD_GEMM;
  OP_PATCHIFY,
  OP_COPY_ADD_POS,
  OP_QKV_UNPACK,
  OP_HEAD_CONCAT:   child_sel <= CHILD_LAYOUT;
  OP_LAYERNORM,
  OP_RESIDUAL,
  OP_ATTN_SOFTMAX:  child_sel <= CHILD_VECTOR;
  default: begin error_code <= ERR_OPCODE; state <= ERROR; end
endcase
if (desc_reg.flags[FLAG_DYNAMIC_M]) begin
  case (desc_reg.param0[1:0])
    DYN_M_CURRENT:    m_eff <= {8'd0, current_token_count};
    DYN_M_CANDIDATES: m_eff <= {8'd0, current_token_count - 1'b1};
    default: begin error_code <= ERR_DIMENSION; state <= ERROR; end
  endcase
end
if (desc_reg.flags[FLAG_DYNAMIC_N]) n_eff <= current_token_count;
if (desc_reg.flags[FLAG_DYNAMIC_K]) k_eff <= current_token_count;
```

- [ ] **Step 4: 实现四种布局/矢量内存循环**

`OP_PATCHIFY` 使用批准的 NHWC/Patch 光栅索引；`OP_COPY_ADD_POS` 写 CLS 后对 196 个 Patch 逐元素重定标并加位置编码；`OP_QKV_UNPACK` 和 `OP_HEAD_CONCAT` 使用 Task 1 索引。Vector engine 把内存 byte 或 little-endian int32 明确拆包，所有写回通过 `mem_w_strb`。`OP_ATTN_SOFTMAX` 读取 int32 Score 后依据 `src0_scale_exp` 重定标到目标指数 -16 的 signed Q8.16，并在送入 Softmax 前饱和到 24-bit。

```systemverilog
patch_src = input_base + (((image_row * 224) + image_col) * 3) + channel;
patch_dst = scratch_base + desc.dst_offset +
            ((((patch_row * 16) + in_row) * 16 + in_col) * 3 + channel);
qkv_src = src0_addr + ((token * 576) + (kind * 192) + (head * 64) + lane);
qkv_dst = dst_addr + ((((kind * 3) + head) * m_eff + token) * 64 + lane);
concat_dst = dst_addr + ((token * 192) + (head * 64) + lane);
```

- [ ] **Step 5: 运行 opcode 与随机回压测试**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_tensor_executor -PlusArgs '+STALL_MASK=3'`

Expected: `TEST_PASS tb_tensor_executor`；八个 opcode 均执行，非法 descriptor 在任何 memory command 前失败。

- [ ] **Step 6: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'feat: add tensor executor and layout operations' -Paths rtl/compute/heatvit_layout_engine.sv,rtl/compute/heatvit_vector_engine.sv,rtl/compute/heatvit_tensor_executor.sv,sim/tb/tb_tensor_executor.sv -TestCommand 'scripts/run_xsim.ps1 -Top tb_tensor_executor'
```

### Task 3: 实现并验证 Patch Embedding

**Files:**
- Create: `verification/heatvit_ref/transformer.py`
- Create: `verification/tests/test_transformer.py`
- Create: `tools/generate_transformer_vectors.py`
- Create: `sim/tb/tb_patch_embedding.sv`

**Interfaces:**
- Consumes: 224×224×3 NHWC int8、`[768][192]` Patch 权重、Bias、CLS、`[197][192]` 位置编码。
- Produces: immutable `PatchParams` dataclass、`patch_embedding(image, params) -> list[list[int]]`、Scratch 中 `[197][192]` int8 激活和 scale exponent。

- [ ] **Step 1: 写 Patch 黄金测试**

用 16×16 的单 Patch 输入验证 patchify 和 GEMM；再用完整 224×224 输入验证 196 个 Patch 的首、末和中间索引。CLS 位置必须只使用独立 CLS 向量加 position row 0，Patch i 必须加 position row `i+1`。

- [ ] **Step 2: 运行并确认 transformer 函数缺失**

Run: `& $env:HEATVIT_PYTHON -m unittest verification.tests.test_transformer.TransformerTest.test_patch_embedding -v`

Expected: FAIL，缺失 `patch_embedding`。

- [ ] **Step 3: 实现黄金函数和三描述符序列**

序列固定为：

```text
OP_PATCHIFY      input NHWC -> scratch.patch_matrix [196][768]
OP_GEMM          patch_matrix * patch_weight + patch_bias -> scratch.patch_embed [196][192]
OP_COPY_ADD_POS  cls + patch_embed + position -> scratch.activation_a [197][192]
```

每一步在 Python 和 descriptor 中记录输入/输出 scale exponent；加法前调用共同 requant。

- [ ] **Step 4: 生成完整尺寸向量并运行 XSim**

Run:

```powershell
& $env:HEATVIT_PYTHON tools/generate_transformer_vectors.py --case patch --seed 20260815 --output build/vectors/patch
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_patch_embedding -PlusArgs '+VECTOR_DIR=build/vectors/patch'
```

Expected: `TEST_PASS tb_patch_embedding`；比较 `[197][192]` 全部 37824 bytes、scale 和三条 descriptor 的 memory trace。

- [ ] **Step 5: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'feat: implement full-size patch embedding' -Paths verification/heatvit_ref/transformer.py,verification/tests/test_transformer.py,tools/generate_transformer_vectors.py,sim/tb/tb_patch_embedding.sv -TestCommand 'scripts/run_xsim.ps1 -Top tb_patch_embedding'
```

### Task 4: 实现并验证三 Head MHSA

**Files:**
- Modify: `verification/heatvit_ref/transformer.py`
- Modify: `verification/tests/test_transformer.py`
- Modify: `tools/generate_transformer_vectors.py`
- Create: `sim/tb/tb_mhsa.sv`

**Interfaces:**
- Consumes: `[N][192]` int8、LN gamma/beta、QKV/Projection 权重与 Bias。
- Produces: immutable `MhsaParams` dataclass、`mhsa(x, params) -> (output, checkpoints)`；N 由 descriptor flag 3 使用 current token count。

- [ ] **Step 1: 写 N=9 的 MHSA 失败测试**

用三个 Head 不同的对角/反对角合成权重，使 Q、K、V 和三个 Score 明确不同。检查 Score shape `[3][9][9]`、每行概率 shape、Context `[3][9][64]`、concat `[9][192]` 和投影输出。至少一个 Attention 概率必须等于 `128`，用于捕获 unsigned flag 遗漏。

- [ ] **Step 2: 运行并确认 MHSA 黄金函数缺失**

Run: `& $env:HEATVIT_PYTHON -m unittest verification.tests.test_transformer.TransformerTest.test_mhsa -v`

Expected: FAIL，缺失 `mhsa` 或检查点键。

- [ ] **Step 3: 实现固定 MHSA 描述符序列**

```text
OP_LAYERNORM       X -> LN1
OP_GEMM            LN1 * Wqkv + Bqkv -> fused_qkv
OP_QKV_UNPACK      fused_qkv -> [Q/K/V][head][N][64]
OP_GEMM transpose  Q * K^T -> score_int32, dst_scale=q_scale+k_scale-3
OP_ATTN_SOFTMAX    score_int32 -> probability_uq0_8
OP_GEMM flag18     probability_uq0_8 * V -> context
OP_HEAD_CONCAT     context -> concat
OP_GEMM            concat * Wproj + Bproj -> msa
```

Softmax 必须逐 Head、逐行调用，row_len=N；三个 Head 之间不得共享 row max 或 denominator。

- [ ] **Step 4: 运行 N=9 与 N=197 测试**

Run:

```powershell
& $env:HEATVIT_PYTHON tools/generate_transformer_vectors.py --case mhsa --tokens 9 --seed 20260815 --output build/vectors/mhsa9
& $env:HEATVIT_PYTHON tools/generate_transformer_vectors.py --case mhsa --tokens 197 --seed 20260815 --output build/vectors/mhsa197
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_mhsa -PlusArgs '+VECTOR_DIR=build/vectors/mhsa9'
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_mhsa -PlusArgs '+VECTOR_DIR=build/vectors/mhsa197'
```

Expected: 两轮 `TEST_PASS tb_mhsa`，QKV、Score、Probability、Context、Concat 和 MSA 检查点均逐位一致。

- [ ] **Step 5: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'feat: implement three-head fixed-point MHSA' -Paths verification/heatvit_ref/transformer.py,verification/tests/test_transformer.py,tools/generate_transformer_vectors.py,sim/tb/tb_mhsa.sv -TestCommand 'scripts/run_xsim.ps1 -Top tb_mhsa'
```

### Task 5: 实现并验证 FFN 与残差

**Files:**
- Modify: `verification/heatvit_ref/transformer.py`
- Modify: `verification/tests/test_transformer.py`
- Modify: `tools/generate_transformer_vectors.py`
- Create: `sim/tb/tb_ffn.sv`

**Interfaces:**
- Consumes: `[N][192]` residual 输入、LN gamma/beta、`[192][768]`/`[768][192]` 两层权重和 Bias。
- Produces: immutable `FfnParams` dataclass、`ffn(y, params) -> (z, checkpoints)`。

- [ ] **Step 1: 写非 8 倍数 N=13 的失败测试**

检查 LN2、第一层 int32 累加、GELU int8 写回、第二层输出、残差尺度对齐和最终 int8。权重包含正负极值但由 generator 约束累加不超 int32；尾部第 14 至 16 行的 dst 哨兵不得改变。

- [ ] **Step 2: 运行并确认 FFN 函数或 descriptor 序列缺失**

Run: `& $env:HEATVIT_PYTHON -m unittest verification.tests.test_transformer.TransformerTest.test_ffn -v`

Expected: FAIL，缺失 `ffn`。

- [ ] **Step 3: 实现 FFN 序列**

```text
OP_LAYERNORM  Y -> LN2
OP_GEMM       LN2 * W1 + B1, post_op=GELU -> hidden [N][768]
OP_GEMM       hidden * W2 + B2 -> ffn_out [N][192]
OP_RESIDUAL   Y + ffn_out -> Z [N][192]
```

第一层 N Tile 尾块与 768 列整块并存，第二层复用同一 GEMM；禁止新增 FFN 专用乘法阵列。

- [ ] **Step 4: 运行 N=13 与 N=197 测试**

Run:

```powershell
& $env:HEATVIT_PYTHON tools/generate_transformer_vectors.py --case ffn --tokens 13 --seed 20260815 --output build/vectors/ffn13
& $env:HEATVIT_PYTHON tools/generate_transformer_vectors.py --case ffn --tokens 197 --seed 20260815 --output build/vectors/ffn197
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_ffn -PlusArgs '+VECTOR_DIR=build/vectors/ffn13'
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_ffn -PlusArgs '+VECTOR_DIR=build/vectors/ffn197'
```

Expected: 两轮 `TEST_PASS tb_ffn`，四个检查点逐位一致且 dst 尾部哨兵未改写。

- [ ] **Step 5: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'feat: implement fixed-point transformer FFN' -Paths verification/heatvit_ref/transformer.py,verification/tests/test_transformer.py,tools/generate_transformer_vectors.py,sim/tb/tb_ffn.sv -TestCommand 'scripts/run_xsim.ps1 -Top tb_ffn'
```

### Task 6: 实现完整 Pre-LN Transformer Block 回归

**Files:**
- Modify: `verification/heatvit_ref/transformer.py`
- Modify: `verification/tests/test_transformer.py`
- Modify: `tools/generate_transformer_vectors.py`
- Create: `sim/tb/tb_transformer_block.sv`
- Modify: `scripts/run_regression.ps1`

**Interfaces:**
- Consumes: Patch 输出或前一 Block `[N][192]`、一个 Block 的全部权重和尺度。
- Produces: immutable `BlockParams` dataclass、`transformer_block(x, params) -> (z, checkpoints)` 和 `run_regression.ps1 -Suite transformer`。

- [ ] **Step 1: 写完整顺序失败测试**

Testbench 读取 descriptor 序列并逐项提交 Executor，强制断言顺序为 LN1、QKV、QK、Softmax、AV、Projection、Residual1、LN2、FC1/GELU、FC2、Residual2。若交换任意两个 Residual/LN 操作，最终结果必须与黄金不匹配。

- [ ] **Step 2: 运行并确认完整 Block 检查点缺失**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_transformer_block -PlusArgs '+VECTOR_DIR=build/vectors/block197'`

Expected: FAIL，向量目录或 `block_output.mem` 尚不存在。

- [ ] **Step 3: 生成 N=197 和 N=13 的完整 Block 向量**

Run:

```powershell
& $env:HEATVIT_PYTHON tools/generate_transformer_vectors.py --case block --tokens 197 --seed 20260815 --output build/vectors/block197
& $env:HEATVIT_PYTHON tools/generate_transformer_vectors.py --case block --tokens 13 --seed 20260816 --output build/vectors/block13
```

Expected: 每个目录包含 input、weights、descriptors、LN1、MSA、Y、LN2、hidden、FFN、Z 和 manifest，所有 SHA-256 校验通过。

- [ ] **Step 4: 运行两个完整 Block 与回压回归**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_transformer_block -PlusArgs '+VECTOR_DIR=build/vectors/block197 +STALL_MASK=0'
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_transformer_block -PlusArgs '+VECTOR_DIR=build/vectors/block13 +STALL_MASK=3'
```

Expected: 两轮 `TEST_PASS tb_transformer_block`，每个检查点及最终 Z 全部逐位一致。

- [ ] **Step 5: 加入阶段回归并执行**

`-Suite transformer` 依次运行 `foundation`、`gemm`、Tensor Executor、Patch、MHSA9、FFN13、Block197、Block13；长测试打印每个 descriptor 的 index、opcode、起止周期和 Token 数。

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_regression.ps1 -Suite transformer`

Expected: 退出码 0，无 watchdog、越界、未知值或协议断言。

- [ ] **Step 6: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'test: gate complete transformer block regression' -Paths verification/heatvit_ref/transformer.py,verification/tests/test_transformer.py,tools/generate_transformer_vectors.py,sim/tb/tb_transformer_block.sv,scripts/run_regression.ps1 -TestCommand 'scripts/run_regression.ps1 -Suite transformer'
```

## 阶段 3 完成条件

- 阶段 1、2 回归继续通过。
- 完整 224×224 Patch Embedding 输出 197×192 bytes 逐位匹配。
- N=9/197 MHSA、N=13/197 FFN 和 N=13/197 完整 Block 逐位匹配。
- `Attention×V` 对 UQ0.8 的 128 编码按正数处理。
- Executor 中只有一个 GEMM、一个 divider，且所有 dynamic M 访问使用 current token count。
