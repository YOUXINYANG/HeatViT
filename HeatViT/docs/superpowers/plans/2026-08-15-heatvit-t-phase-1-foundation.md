# HeatViT-T Phase 1: Fixed-Point Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立可重复执行的 XSim/Python 测试环境，并实现 HeatViT-T 后续模块共用的逐位确定定点、非线性、除法、平方根、Softmax 和 LayerNorm 基础单元。

**Architecture:** SystemVerilog package 固定所有类型、常量、320-bit 描述符和错误码；标量数值单元使用显式位宽与 start/busy/done 接口。Python/NumPy 实现同一整数算法并生成 `.mem`，SystemVerilog Testbench 对边界向量和固定种子向量逐项比较。

**Tech Stack:** SystemVerilog 2012、Vivado XSim 2023.2、PowerShell、Python 3.12–3.14、NumPy 2.5.2。

## Global Constraints

- 目标 Part 必须保持 `xc7k325tfbg900-3`，仿真器必须为 Vivado XSim 2023.2。
- 普通权重/激活为 signed int8，GEMM 累加为 signed int32，`scale_exp` 为 signed 6-bit。
- 右移采用 round-to-nearest、ties-away-from-zero；所有窄化必须饱和，禁止隐式截断。
- 一般整数除法由 floor quotient/remainder 生成最近舍入商；当 `2*remainder >= denominator` 时商的绝对值加一，signed 结果随后恢复符号。
- GELU、Softmax、PLAN Sigmoid、LayerNorm、除法和平方根只能使用可综合整数 RTL。
- Attention 概率为 UQ0.8；Selector 概率/Head Weight 为 17-bit Q0.16；主要非线性内部值为 signed Q8.16。
- Python 黄金模型在量化边界以后不得创建 float/complex dtype；NumPy 只使用显式 `int8/int32/int64/uint8/uint32/uint64`，禁止 PyTorch 或其他数值框架。
- 自检式 Testbench 必须在成功时打印唯一 `TEST_PASS <top>`，失败必须 `$fatal` 并返回非零退出码。
- 当前不需要任何手工 Vivado IP。

---

## 文件映射

| 文件 | 单一职责 |
| --- | --- |
| `config/heatvit_t.json` | 固定模型、Q 格式、常量和测试种子 |
| `rtl/include/heatvit_pkg.sv` | 类型、枚举、描述符、定点纯函数 |
| `rtl/common/heatvit_requant.sv` | 48-bit 输入到 int8/int32 的显式重定标 |
| `rtl/common/heatvit_residual.sv` | 两尺度残差对齐、求和与 int8 写回 |
| `rtl/common/heatvit_udiv.sv` | 64-bit 无符号恢复除法 |
| `rtl/common/heatvit_div_arbiter.sv` | Softmax、LayerNorm、Package 共用除法器的三客户端仲裁 |
| `rtl/common/heatvit_isqrt.sv` | 无符号整数平方根 |
| `rtl/common/heatvit_gelu.sv` | 论文 GELU 定点近似 |
| `rtl/common/heatvit_plan_sigmoid.sv` | PLAN 分段 Sigmoid |
| `rtl/common/heatvit_softmax_core.sv` | 行缓存、最大值、指数和归一化 |
| `rtl/common/heatvit_softmax_attention.sv` | `delta2=0.5`、UQ0.8 输出封装 |
| `rtl/common/heatvit_softmax_selector.sv` | `delta2=1.0`、Q0.16 输出封装 |
| `rtl/common/heatvit_layernorm.sv` | D=192 两遍定点 LayerNorm |
| `verification/heatvit_ref/fixed.py` | Python 位宽、舍入、饱和和打包基准 |
| `verification/heatvit_ref/nonlinear.py` | Python GELU、PLAN、Softmax、LayerNorm 基准 |
| `verification/requirements.txt` | 锁定黄金模型加速依赖 NumPy 2.5.2 |
| `verification/tests/` | Python 单元测试 |
| `tools/generate_unit_vectors.py` | 生成确定性 `.mem` 和 JSON manifest |
| `sim/common/tb_pkg.sv` | TB 断言、文件和 ready/valid 辅助任务 |
| `sim/tb/tb_*.sv` | 各数值单元自检 Testbench |
| `scripts/run_xsim.ps1` | 按 package-first 顺序编译、展开和运行一个 TB |
| `scripts/run_python_tests.ps1` | 使用指定 Python 运行 unittest |
| `scripts/task_checkpoint.ps1` | Git 存在时提交，否则写本地检查点日志 |

## 锁定公共接口

`rtl/include/heatvit_pkg.sv` 必须定义：

```systemverilog
typedef logic signed [7:0]  heatvit_s8_t;
typedef logic signed [23:0] heatvit_q8_16_t;
typedef logic signed [31:0] heatvit_s32_t;
typedef logic signed [47:0] heatvit_s48_t;
typedef logic signed [127:0] heatvit_s128_t;
typedef logic signed [5:0]  heatvit_scale_t;
typedef logic        [16:0] heatvit_uq0_16_t;

typedef struct packed {
  logic [7:0] opcode;
  logic [23:0] flags;
  logic [15:0] m;
  logic [15:0] n;
  logic [15:0] k;
  logic [3:0] heads;
  logic [31:0] src0_offset;
  logic [31:0] src1_offset;
  logic [31:0] bias_offset;
  logic [31:0] aux_offset;
  logic [31:0] dst_offset;
  heatvit_scale_t src0_scale_exp;
  heatvit_scale_t src1_scale_exp;
  heatvit_scale_t aux_scale_exp;
  heatvit_scale_t dst_scale_exp;
  logic [15:0] next_index;
  logic [15:0] param0;
  logic [15:0] param1;
  logic [3:0] reserved;
} heatvit_desc_t;
```

Opcode 编码固定为：

```systemverilog
typedef enum logic [7:0] {
  OP_NOP                 = 8'd0,
  OP_PATCHIFY            = 8'd1,
  OP_COPY_ADD_POS        = 8'd2,
  OP_GEMM                = 8'd3,
  OP_LAYERNORM           = 8'd4,
  OP_RESIDUAL            = 8'd5,
  OP_QKV_UNPACK          = 8'd6,
  OP_HEAD_CONCAT         = 8'd7,
  OP_ATTN_SOFTMAX        = 8'd8,
  OP_SELECTOR_SOFTMAX    = 8'd9,
  OP_REDUCE_MEAN         = 8'd10,
  OP_CONCAT_LOCAL_GLOBAL = 8'd11,
  OP_HEAD_FUSE           = 8'd12,
  OP_SELECTOR_FINALIZE   = 8'd13,
  OP_FINISH              = 8'd14
} heatvit_opcode_e;
```

Post-op、flag、错误和警告编码固定为：

```systemverilog
typedef enum logic [2:0] {
  POST_NONE             = 3'd0,
  POST_REQUANT          = 3'd1,
  POST_GELU             = 3'd2,
  POST_ATTN_SOFTMAX     = 3'd3,
  POST_SELECTOR_SOFTMAX = 3'd4,
  POST_PLAN             = 3'd5,
  POST_LAYERNORM        = 3'd6
} heatvit_postop_e;

localparam int FLAG_RHS_TRANSPOSE = 0;
localparam int FLAG_BIAS_ENABLE = 1;
localparam int FLAG_AUX_ENABLE = 2;
localparam int FLAG_DYNAMIC_M = 3;
localparam int FLAG_SWAP_ACTIVATION = 4;
localparam int FLAG_HEAD_MODE = 5;
localparam int FLAG_HEAD_CONCAT = 6;
localparam int FLAG_OUTPUT_INT32 = 7;
localparam int FLAG_SRC0_INPUT = 11;
localparam int FLAG_SRC1_SCRATCH = 12;
localparam int FLAG_BIAS_SCRATCH = 13;
localparam int FLAG_AUX_WEIGHT = 14;
localparam int FLAG_DST_OUTPUT = 15;
localparam int FLAG_TOKEN_TAIL = 16;
localparam int FLAG_CHANNEL_TAIL = 17;
localparam int FLAG_SRC0_UNSIGNED = 18;
localparam int FLAG_DYNAMIC_N = 19;
localparam int FLAG_DYNAMIC_K = 20;

localparam logic [1:0] DYN_M_CURRENT = 2'b00;
localparam logic [1:0] DYN_M_CANDIDATES = 2'b01;
localparam logic [1:0] REDUCE_AXIS_CANDIDATES = 2'b00;
localparam logic [1:0] REDUCE_AXIS_HEAD_LANES = 2'b01;

typedef enum logic [7:0] {
  ERR_NONE = 8'd0, ERR_OPCODE = 8'd1, ERR_DIMENSION = 8'd2,
  ERR_ADDRESS = 8'd3, ERR_TOKEN_COUNT = 8'd4, ERR_MEMORY_PROTOCOL = 8'd5,
  ERR_SOFTMAX_ZERO_SUM = 8'd6, ERR_BUSY_START = 8'd7
} heatvit_error_e;

localparam int WARN_HEAD_DEN_ZERO = 0;
localparam int WARN_PACKAGE_DEN_ZERO = 1;
localparam int WARN_LN_NEGATIVE_VARIANCE = 2;
```

并用 `$bits(heatvit_desc_t) == 320` 进行 elaboration-time 检查。数值函数名称固定为：

```systemverilog
function automatic logic signed [127:0] round_shift_away_s128(
  input logic signed [127:0] value,
  input logic [6:0] shift
);
function automatic logic signed [127:0] scale_to_exp_s128(
  input logic signed [127:0] value,
  input heatvit_scale_t src_exp,
  input heatvit_scale_t dst_exp
);
function automatic logic signed [7:0] sat_s8(input logic signed [127:0] value);
function automatic logic signed [31:0] sat_s32(input logic signed [127:0] value);
```

### Task 1: 建立配置、编译脚本和公共 package

**Files:**
- Create: `config/heatvit_t.json`
- Create: `verification/requirements.txt`
- Create: `rtl/include/heatvit_pkg.sv`
- Create: `sim/common/tb_pkg.sv`
- Create: `sim/tb/tb_pkg_smoke.sv`
- Create: `scripts/run_xsim.ps1`
- Create: `scripts/run_python_tests.ps1`
- Create: `scripts/task_checkpoint.ps1`

**Interfaces:**
- Consumes: `$env:HEATVIT_VIVADO_BIN`、`$env:HEATVIT_PYTHON`。
- Produces: 上述锁定 typedef/function、`run_xsim.ps1 -Top <name> [-PlusArgs <string>]`、`run_python_tests.ps1 -Pattern <file>`。

- [ ] **Step 1: 写 package 编译契约测试**

```systemverilog
module tb_pkg_smoke;
  import heatvit_pkg::*;
  heatvit_desc_t desc;
  initial begin
    if ($bits(desc) != 320) $fatal(1, "descriptor width=%0d", $bits(desc));
    if (GELU_A_Q16 != -18927) $fatal(1, "GELU_A_Q16");
    if (LN_EPS_Q32 != 48'd4295) $fatal(1, "LN_EPS_Q32");
    $display("TEST_PASS tb_pkg_smoke");
    $finish;
  end
endmodule
```

- [ ] **Step 2: 直接编译并确认测试因 package 缺失而失败**

Run:

```powershell
& "$env:HEATVIT_VIVADO_BIN\xvlog.bat" -sv sim/tb/tb_pkg_smoke.sv
```

Expected: 非零退出码，日志包含无法找到 `heatvit_pkg`。

- [ ] **Step 3: 实现配置与 package**

`config/heatvit_t.json` 固定写入以下键值：

```json
{
  "seed": 20260815,
  "image": [224, 224, 3],
  "patch": 16,
  "tokens": 197,
  "embed_dim": 192,
  "heads": 3,
  "head_dim": 64,
  "blocks": 12,
  "ffn_dim": 768,
  "classes": 1000,
  "selector_before_blocks": [4, 7, 10],
  "gemm_tile": {"heads": 3, "input": 8, "output": 8},
  "gelu_q16": {"a": -18927, "b": -115933, "delta": 32768, "inv_sqrt2": 46341},
  "exp_q16": {"ln2": 45426, "quad": 23495, "offset": 88670, "constant": 22544},
  "softmax_delta_q16": {"attention": 32768, "selector": 65536},
  "layernorm_epsilon_q32": 4295,
  "synthetic_scale_exp": {
    "input": -7,
    "activation": -7,
    "weight": -7,
    "cls_position_beta": -7,
    "gamma": -6,
    "q8_16": -16,
    "attention_uq0_8": -8,
    "selector_q0_16": -16,
    "logit": -14
  }
}
```

在 package 中实现锁定 struct、常量、错误码 1 至 7、警告位 0 至 2、Post-op 枚举，以及正负对称的舍入函数。flags bit 18 固定命名为 `FLAG_SRC0_UNSIGNED`，只允许 UQ0.8 Attention×signed int8 V 使用；bits 19/20 分别命名为 `FLAG_DYNAMIC_N/K`，bits 23:21 保留为零。对负数先取 129-bit magnitude，加入 `1 << (shift-1)` 后右移，再恢复符号；`shift=0` 直接返回输入。

`verification/requirements.txt` 的完整内容固定为：

```text
numpy==2.5.2
```

- [ ] **Step 4: 实现可重复运行脚本**

`run_xsim.ps1` 必须接收 `-Top` 和可选字符串 `-PlusArgs`，按 `rtl/include`、其他 `rtl`、`HeatViT.srcs/sources_1/new`、`sim/common`、存在时的 `sim/generated`、目标 TB 的顺序收集绝对 `.sv` 路径，为每个 Top 建立独立 `build/xsim/<top>` 日志目录，但从仓库根目录调用工具以保持 `.mem` 相对路径稳定，并依次执行：

```powershell
& "$VivadoBin\xvlog.bat" -sv @Sources
& "$VivadoBin\xelab.bat" $Top -s "${Top}_snapshot" -timescale 1ns/1ps
$RunArgs = @("${Top}_snapshot", '-runall', '-onerror', 'quit', '-onfinish', 'quit')
foreach ($Arg in ($PlusArgs -split ' ')) {
  if ($Arg) {
    $RunArgs += '-testplusarg'
    $RunArgs += $Arg.TrimStart('+')
  }
}
& "$VivadoBin\xsim.bat" @RunArgs
```

Plusarg 转换遵循 [AMD XSim 2023.2 `-testplusarg` 选项](https://docs.amd.com/r/2023.2-English/ug900-vivado-logic-simulation/xsim-Executable-Options)。任一 `$LASTEXITCODE` 非零立即退出。`task_checkpoint.ps1` 接收 `-Message`、`-Paths`、`-TestCommand`；有 `.git` 时执行 `git add -- <paths>` 和 `git commit -m <message>`，否则创建 `build` 并向 `build/task-checkpoints.log` 追加三个字段。

脚本启动时必须验证 `xvlog.bat/xelab.bat/xsim.bat` 均存在；若 `$env:XILINX_VIVADO` 未定义，则把它设置为 `Split-Path $env:HEATVIT_VIVADO_BIN -Parent`。`run_python_tests.ps1` 同样先验证解释器路径与 NumPy 版本，再从仓库根目录调用 `-m unittest discover -s verification/tests -p <Pattern>`。

- [ ] **Step 5: 安装并核对锁定的 Python 依赖**

Run:

```powershell
& $env:HEATVIT_PYTHON -m pip install -r verification/requirements.txt
& $env:HEATVIT_PYTHON -c "import numpy as np; assert np.__version__ == '2.5.2'"
```

Expected: 两条命令退出码均为 0。依赖下载需要在执行阶段按环境权限流程取得网络批准。

- [ ] **Step 6: 运行 package smoke test**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_pkg_smoke
```

Expected: 退出码 0，stdout 唯一成功标记为 `TEST_PASS tb_pkg_smoke`。

- [ ] **Step 7: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'build: establish HeatViT simulation foundation' -Paths config/heatvit_t.json,verification/requirements.txt,rtl/include/heatvit_pkg.sv,sim/common/tb_pkg.sv,sim/tb/tb_pkg_smoke.sv,scripts/run_xsim.ps1,scripts/run_python_tests.ps1,scripts/task_checkpoint.ps1 -TestCommand 'scripts/run_xsim.ps1 -Top tb_pkg_smoke'
```

### Task 2: 实现 Python 定点基准和向量文件协议

**Files:**
- Create: `verification/heatvit_ref/__init__.py`
- Create: `verification/heatvit_ref/fixed.py`
- Create: `verification/tests/__init__.py`
- Create: `verification/tests/test_fixed.py`
- Create: `tools/generate_unit_vectors.py`
- Create: `sim/vectors/fixed/manifest.json`
- Create: `sim/vectors/fixed/requant.mem`

**Interfaces:**
- Consumes: `config/heatvit_t.json`。
- Produces: `round_shift_away(value: int, shift: int) -> int`、`sat_signed(value: int, bits: int) -> int`、`requant(value, src_exp, dst_exp, bits) -> int`、小端 `.mem`/manifest 格式。

- [ ] **Step 1: 写失败的 Python 边界测试**

```python
import unittest
from verification.heatvit_ref.fixed import round_shift_away, sat_signed, requant

class FixedTest(unittest.TestCase):
    def test_ties_away_from_zero(self):
        self.assertEqual(round_shift_away(1, 1), 1)
        self.assertEqual(round_shift_away(-1, 1), -1)
        self.assertEqual(round_shift_away(3, 1), 2)
        self.assertEqual(round_shift_away(-3, 1), -2)

    def test_saturation(self):
        self.assertEqual(sat_signed(128, 8), 127)
        self.assertEqual(sat_signed(-129, 8), -128)

    def test_scale_conversion(self):
        self.assertEqual(requant(255, -8, -7, 8), 127)
        self.assertEqual(requant(-255, -8, -7, 8), -128)
```

- [ ] **Step 2: 运行测试并确认导入失败**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_python_tests.ps1 -Pattern test_fixed.py
```

Expected: FAIL，错误为 `No module named 'verification.heatvit_ref.fixed'`。

- [ ] **Step 3: 实现纯整数函数**

标量核心实现必须等价于：

```python
def round_shift_away(value: int, shift: int) -> int:
    if shift < 0:
        return value << (-shift)
    if shift == 0:
        return value
    magnitude = abs(value)
    rounded = (magnitude + (1 << (shift - 1))) >> shift
    return -rounded if value < 0 else rounded

def sat_signed(value: int, bits: int) -> int:
    lower = -(1 << (bits - 1))
    upper = (1 << (bits - 1)) - 1
    return min(upper, max(lower, value))

def requant(value: int, src_exp: int, dst_exp: int, bits: int) -> int:
    shift = dst_exp - src_exp
    shifted = round_shift_away(value, shift) if shift >= 0 else value << (-shift)
    return sat_signed(shifted, bits)
```

拒绝 Python `float` 输入；`.mem` 每行保存一个无前缀十六进制字，manifest 保存 seed、位宽、记录数和 SHA-256。

- [ ] **Step 4: 生成并验证确定性向量**

Run:

```powershell
& $env:HEATVIT_PYTHON tools/generate_unit_vectors.py --suite fixed --seed 20260815 --output sim/vectors/fixed
& $env:HEATVIT_PYTHON -m unittest verification.tests.test_fixed -v
```

Expected: 两次运行生成相同 SHA-256，三个测试均 `ok`。

- [ ] **Step 5: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'test: add integer fixed-point reference' -Paths verification/heatvit_ref/__init__.py,verification/heatvit_ref/fixed.py,verification/tests/__init__.py,verification/tests/test_fixed.py,tools/generate_unit_vectors.py,sim/vectors/fixed -TestCommand 'python -m unittest verification.tests.test_fixed -v'
```

### Task 3: 实现重定标和残差单元

**Files:**
- Create: `rtl/common/heatvit_requant.sv`
- Create: `rtl/common/heatvit_residual.sv`
- Create: `sim/tb/tb_requant_residual.sv`
- Modify: `tools/generate_unit_vectors.py`

**Interfaces:**
- Consumes: `round_shift_away_s128`、`sat_s8`、fixed suite vectors。
- Produces: 组合 `heatvit_requant` 和一拍 ready/valid `heatvit_residual`。

- [ ] **Step 1: 写失败的 RTL 测试**

测试依次驱动 `(value,src_exp,dst_exp)` 为 `(1,0,1)`、`(-1,0,1)`、`(3,0,1)`、`(-3,0,1)`、`(1024,0,0)`，预期 int8 为 `1,-1,2,-2,127`。残差驱动 `main=64@-7`、`aux=64@-8`、`out=-7`，预期 `96`；再驱动正负饱和案例。

```systemverilog
check_requant(48'sd1,   6'sd0, 6'sd1,   8'sd1);
check_requant(-48'sd1,  6'sd0, 6'sd1,  -8'sd1);
check_requant(48'sd3,   6'sd0, 6'sd1,   8'sd2);
check_requant(-48'sd3,  6'sd0, 6'sd1,  -8'sd2);
check_residual(8'sd64, -6'sd7, 8'sd64, -6'sd8, -6'sd7, 8'sd96);
```

- [ ] **Step 2: 编译并确认模块缺失**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_requant_residual`

Expected: FAIL，缺失 `heatvit_requant` 或 `heatvit_residual`。

- [ ] **Step 3: 实现显式扩位、舍入和饱和**

`heatvit_requant` 端口固定为 `in_value[47:0]`、`src_scale_exp[5:0]`、`dst_scale_exp[5:0]`、`out_value[7:0]`、`saturated`，内部先符号扩展到 signed 128-bit。`heatvit_residual` 使用 `s_valid/s_ready` 和 `m_valid/m_ready`，把两个 int8 扩成 signed 128-bit 并对齐到较细尺度，在 128-bit 求和，再重定标到目标尺度；完整 exponent 差值不得在对齐前截断。stall 时所有 m 端口保持稳定。

```systemverilog
common_exp = (main_scale_exp < aux_scale_exp) ? main_scale_exp : aux_scale_exp;
main_wide = $signed(main_value) <<< (main_scale_exp - common_exp);
aux_wide  = $signed(aux_value)  <<< (aux_scale_exp  - common_exp);
sum_wide  = main_wide + aux_wide;
scaled     = scale_to_exp_s128(sum_wide, common_exp, out_scale_exp);
next_value = sat_s8(scaled);
```

- [ ] **Step 4: 运行边界与随机向量测试**

Run:

```powershell
& $env:HEATVIT_PYTHON tools/generate_unit_vectors.py --suite requant --seed 20260815 --output sim/vectors/requant
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_requant_residual
```

Expected: `TEST_PASS tb_requant_residual`，至少比较 1024 个固定种子案例并覆盖正负中点及两端饱和。

- [ ] **Step 5: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'feat: add exact requant and residual units' -Paths rtl/common/heatvit_requant.sv,rtl/common/heatvit_residual.sv,sim/tb/tb_requant_residual.sv,tools/generate_unit_vectors.py -TestCommand 'scripts/run_xsim.ps1 -Top tb_requant_residual'
```

### Task 4: 实现恢复除法和整数平方根

**Files:**
- Create: `rtl/common/heatvit_udiv.sv`
- Create: `rtl/common/heatvit_div_arbiter.sv`
- Create: `rtl/common/heatvit_isqrt.sv`
- Create: `sim/tb/tb_udiv_isqrt.sv`
- Modify: `verification/heatvit_ref/fixed.py`
- Modify: `verification/tests/test_fixed.py`
- Modify: `tools/generate_unit_vectors.py`

**Interfaces:**
- Consumes: unsigned integers，调用方保证 start 只在 busy=0 时有效；三个 divider 客户端使用 `req_valid/req_ready/num/den` 与 `rsp_valid/quot/rem/div_zero`。
- Produces: `heatvit_udiv #(NUM_W=64,DEN_W=64,QUOT_W=64)`、`heatvit_isqrt #(RAD_W=48)` 的 start/busy/done 接口，以及固定优先级 0→1→2 的 `heatvit_div_arbiter`。

- [ ] **Step 1: 写除法/平方根失败测试**

Python 断言 `udiv(10,3)==(3,1)`、`isqrt(0)==(0,0)`、`isqrt(15)==(3,6)`、`isqrt(16)==(4,0)`。RTL 对同一组案例逐周期等待 `done`，并断言除零时 `divide_by_zero=1`、其他结果不更新为商。三个 arbiter 客户端同周期发请求时必须依次收到 0、1、2 三个响应，且响应只返回原请求客户端。

- [ ] **Step 2: 运行并确认函数或模块不存在**

Run:

```powershell
& $env:HEATVIT_PYTHON -m unittest verification.tests.test_fixed -v
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_udiv_isqrt
```

Expected: 两项均 FAIL，分别报告缺失函数和模块。

- [ ] **Step 3: 实现逐位恢复算法**

除法每周期处理一个商位，保持 `(remainder << 1) | numerator_msb`，大于等于 denominator 时相减并置商位。平方根每周期处理 radicand 的两位，试除数为 `(root << 2) | 1`。复位清零 busy/done；done 只脉冲一拍；start while busy 由单元 assertion 报错。仲裁器在接受请求时锁存 grant，直至 divider done 才向对应客户端发一个 `rsp_valid`，期间不切换 grant。

```systemverilog
shifted_rem = {remainder[62:0], numerator_shift[63]};
numerator_shift <= {numerator_shift[62:0], 1'b0};
if (shifted_rem >= denominator_reg) begin
  remainder <= shifted_rem - denominator_reg;
  quotient_shift <= {quotient_shift[62:0], 1'b1};
end else begin
  remainder <= shifted_rem;
  quotient_shift <= {quotient_shift[62:0], 1'b0};
end
```

- [ ] **Step 4: 运行穷举小位宽、64-bit 除法和 48-bit 平方根随机测试**

Run:

```powershell
& $env:HEATVIT_PYTHON tools/generate_unit_vectors.py --suite divsqrt --seed 20260815 --output sim/vectors/divsqrt
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_udiv_isqrt
```

Expected: `TEST_PASS tb_udiv_isqrt`；8-bit 输入穷举、1024 个 64-bit 除法、1024 个 48-bit 平方根案例以及除零均通过；平方根始终满足 `root^2 <= radicand < (root+1)^2`。

- [ ] **Step 5: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'feat: add restoring divider and integer square root' -Paths rtl/common/heatvit_udiv.sv,rtl/common/heatvit_div_arbiter.sv,rtl/common/heatvit_isqrt.sv,sim/tb/tb_udiv_isqrt.sv,verification/heatvit_ref/fixed.py,verification/tests/test_fixed.py,tools/generate_unit_vectors.py -TestCommand 'scripts/run_xsim.ps1 -Top tb_udiv_isqrt'
```

### Task 5: 实现 GELU 和 PLAN Sigmoid

**Files:**
- Create: `rtl/common/heatvit_gelu.sv`
- Create: `rtl/common/heatvit_plan_sigmoid.sv`
- Create: `verification/heatvit_ref/nonlinear.py`
- Create: `verification/tests/test_nonlinear.py`
- Create: `sim/tb/tb_gelu_plan.sv`
- Modify: `tools/generate_unit_vectors.py`

**Interfaces:**
- Consumes: signed 24-bit Q8.16 scalar。
- Produces: start/busy/done GELU Q8.16 和 PLAN unsigned Q0.16。

- [ ] **Step 1: 写公式边界测试**

PLAN 必测输入编码 `-327680,-155648,-65536,0,65536,155648,327680`，并检查 `sigmoid(-x)=65536-sigmoid(x)`。GELU 必测 `x=0` 输出 0、`x=±32768` 及 Q8.16 最小/最大安全输入，预期由 Python 整数函数给出。

- [ ] **Step 2: 运行并确认失败**

Run: `& $env:HEATVIT_PYTHON -m unittest verification.tests.test_nonlinear -v`

Expected: FAIL，缺失 `verification.heatvit_ref.nonlinear`。

- [ ] **Step 3: 按批准系数实现整数流水**

GELU 固定使用 `a=-18927`、`b=-115933`、`delta1=32768`、`inv_sqrt2=46341`，并严格按下列整数顺序执行：

```text
u_q16       = round_shift_away(x_q16 * 46341, 16)
clip_q16    = min(abs(u_q16), 115933)
t_q16       = clip_q16 - 115933
t2_q16      = round_shift_away(t_q16 * t_q16, 16)
poly_q16    = round_shift_away(-18927 * t2_q16, 16) + 65536
erf_mag_q16 = round_shift_away(32768 * poly_q16, 16)
l_erf_q16   = sign(u_q16) * erf_mag_q16
y_q16       = round_shift_away(x_q16 * (65536 + l_erf_q16), 17)
```

`sign(0)=0`，最终饱和到 signed 24-bit。PLAN 四段精确使用移位 `/4`、`/8`、`/32` 和常数 `1/2`、`5/8`、`27/32`，负输入最后执行 `65536-y_abs`。

- [ ] **Step 4: 生成向量并运行 Python/RTL 对照**

Run:

```powershell
& $env:HEATVIT_PYTHON tools/generate_unit_vectors.py --suite nonlinear --seed 20260815 --output sim/vectors/nonlinear
& $env:HEATVIT_PYTHON -m unittest verification.tests.test_nonlinear -v
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_gelu_plan
```

Expected: unittest 全部 `ok`，XSim 输出 `TEST_PASS tb_gelu_plan`；每个分段阈值的 `-1/0/+1 LSB` 均被覆盖。

- [ ] **Step 5: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'feat: add fixed-point GELU and PLAN sigmoid' -Paths rtl/common/heatvit_gelu.sv,rtl/common/heatvit_plan_sigmoid.sv,verification/heatvit_ref/nonlinear.py,verification/tests/test_nonlinear.py,sim/tb/tb_gelu_plan.sv,tools/generate_unit_vectors.py -TestCommand 'scripts/run_xsim.ps1 -Top tb_gelu_plan'
```

### Task 6: 实现两种 Softmax

**Files:**
- Create: `rtl/common/heatvit_softmax_core.sv`
- Create: `rtl/common/heatvit_softmax_attention.sv`
- Create: `rtl/common/heatvit_softmax_selector.sv`
- Create: `sim/tb/tb_softmax.sv`
- Modify: `verification/heatvit_ref/nonlinear.py`
- Modify: `verification/tests/test_nonlinear.py`
- Modify: `tools/generate_unit_vectors.py`

**Interfaces:**
- Consumes: `start`、`row_len[7:0]`、Q8.16 `s_data` ready/valid 行流，以及外部共享 divider 的 client request/response 端口。
- Produces: Attention UQ0.8 或 Selector Q0.16 的 `m_data/m_last` ready/valid 流、`done`、`error_zero_sum`；模块内部不得实例化第二个 divider。

- [ ] **Step 1: 写行级失败测试**

覆盖长度 1、2、3、197；全相等值、一个明显最大值、Q8.16 负极值和输出回压。Selector 长度 2 且输入相等时两个输出必须都是 `32768`；Attention `delta2=0.5` 的单元素行输出必须是 UQ0.8 的 `128`。

- [ ] **Step 2: 运行并确认 Softmax 模块缺失**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_softmax`

Expected: FAIL，日志报告缺失 `heatvit_softmax_attention` 或 `heatvit_softmax_selector`。

- [ ] **Step 3: 实现三遍行处理**

第一遍缓存最多 197 个 Q8.16 元素并求最大值；第二遍计算：

```text
x_tilde = x - row_max
z = floor((-x_tilde) / 45426)
p = x_tilde + z * 45426
exp_q16 = round_q16(23495 * square_q16(p + 88670)) + 22544
scaled_exp = exp_q16 >> z
```

累加整数行和 `S` 后只向共享 divider client 发一次 `(1<<32)/S` 请求，用 quotient/remainder 生成 33-bit `recip_q32`。第三遍逐元素严格执行 `ratio_q16=round(E_i*recip_q32/2^16)`、`scaled_q16=round(ratio_q16*delta2_q16/2^16)`。Selector wrapper 保持 17-bit Q0.16，Attention wrapper再最近舍入右移 8 bit到 UQ0.8。standalone TB 把该 client 端口连接到一个 64-bit `heatvit_udiv`。stall 时输出保持稳定，`row_len=0` 在命令入口直接 `$fatal`，内部行和为零置 `error_zero_sum`。

- [ ] **Step 4: 运行确定性与回压回归**

Run:

```powershell
& $env:HEATVIT_PYTHON tools/generate_unit_vectors.py --suite softmax --seed 20260815 --output sim/vectors/softmax
& $env:HEATVIT_PYTHON -m unittest verification.tests.test_nonlinear -v
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_softmax
```

Expected: `TEST_PASS tb_softmax`，逐项匹配至少 256 行，其中包含长度 197 和随机 backpressure。

- [ ] **Step 5: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'feat: add attention and selector softmax' -Paths rtl/common/heatvit_softmax_core.sv,rtl/common/heatvit_softmax_attention.sv,rtl/common/heatvit_softmax_selector.sv,sim/tb/tb_softmax.sv,verification/heatvit_ref/nonlinear.py,verification/tests/test_nonlinear.py,tools/generate_unit_vectors.py -TestCommand 'scripts/run_xsim.ps1 -Top tb_softmax'
```

### Task 7: 实现纯 RTL LayerNorm

**Files:**
- Create: `rtl/common/heatvit_layernorm.sv`
- Create: `sim/tb/tb_layernorm.sv`
- Modify: `verification/heatvit_ref/nonlinear.py`
- Modify: `verification/tests/test_nonlinear.py`
- Modify: `tools/generate_unit_vectors.py`

**Interfaces:**
- Consumes: 配置握手 `cfg_valid/cfg_ready`（四个 scale exponent），随后恰好 192 组 `x/gamma/beta` int8 流，以及外部共享 divider 的 client request/response 端口。
- Produces: 恰好 192 个 int8 输出流、`done`、`warn_negative_variance`；模块内部实例化一个 `heatvit_isqrt`，但不得实例化 divider。

- [ ] **Step 1: 写零方差和非对称向量测试**

零向量配 `gamma=64`、`beta=0` 必须输出全零；常量非零向量必须产生零归一化项；递增/递减/正负混合三组输入由 Python 给出逐元素预期。额外构造一个因定点舍入导致 `E[x^2]-mean^2=-1` 的内部案例，预期方差钳零且警告置位。

- [ ] **Step 2: 运行并确认模块缺失**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_layernorm`

Expected: FAIL，缺失 `heatvit_layernorm`。

- [ ] **Step 3: 实现两遍 FSM**

状态顺序固定为 `IDLE -> LOAD_ACCUM -> MEAN -> VARIANCE -> SQRT -> RECIP -> NORMALIZE -> DRAIN -> DONE`。只接受 input scale `[-32,0]`。LOAD 阶段按批准公式构造/累加带 32 个小数位的 `sum_x_q32` 与 `sum_square_q32`；mean 和 E[x²] 都从 64-bit divider quotient/remainder 按统一规则舍入；`mean_square_q32=round(mean_q32^2/2^32)`；负方差钳零；加入 `4295`；48-bit isqrt 产生 floor `std_q16`；divider 只计算一次 `inv_std_q32=round(2^48/std_q16)`；NORMALIZE 用 `round((x_q32-mean_q32)*inv_std_q32/2^48)` 得到并饱和为 signed 24-bit Q8.16，再执行 gamma/beta、舍入和 int8 饱和。

```text
x_q32            = x_int << (input_scale_exp + 32)
square_q32       = round(x_int*x_int * 2^(2*input_scale_exp + 32))
mean_q32         = round(sum_x_q32 / 192)
e2_q32           = round(sum_square_q32 / 192)
variance_q32     = max(0, e2_q32 - round(mean_q32*mean_q32 / 2^32))
std_q16          = floor_sqrt(variance_q32 + 4295)
inv_std_q32      = round(2^48 / std_q16)
normalized_q16   = sat_q8_16(round((x_q32-mean_q32)*inv_std_q32 / 2^48))
```

- [ ] **Step 4: 运行完整通道和回压测试**

Run:

```powershell
& $env:HEATVIT_PYTHON tools/generate_unit_vectors.py --suite layernorm --seed 20260815 --output sim/vectors/layernorm
& $env:HEATVIT_PYTHON -m unittest verification.tests.test_nonlinear -v
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_layernorm
```

Expected: `TEST_PASS tb_layernorm`，至少 64 个完整 192 通道 Token 逐位匹配，包含连续 50% 随机输出 stall。

- [ ] **Step 5: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'feat: add pure RTL layer normalization' -Paths rtl/common/heatvit_layernorm.sv,sim/tb/tb_layernorm.sv,verification/heatvit_ref/nonlinear.py,verification/tests/test_nonlinear.py,tools/generate_unit_vectors.py -TestCommand 'scripts/run_xsim.ps1 -Top tb_layernorm'
```

### Task 8: 建立阶段 1 回归门

**Files:**
- Create: `scripts/run_regression.ps1`
- Create: `verification/tests/test_config_contract.py`
- Create: `docs/verification/fixed-point-contract.md`

**Interfaces:**
- Consumes: 阶段 1 的全部 Python/RTL 测试。
- Produces: `run_regression.ps1 -Suite foundation` 和非零失败传播。

- [ ] **Step 1: 写配置一致性失败测试**

`test_config_contract.py` 读取 JSON 与 `heatvit_pkg.sv`，用正则提取所有批准常量并逐项断言；还断言 descriptor 字段宽度之和为 320。先故意只列出测试，再运行以确认缺失 `docs/verification/fixed-point-contract.md` 的文档检查失败。

- [ ] **Step 2: 实现回归清单和数值契约文档**

`run_regression.ps1 -Suite foundation` 按顺序运行：Python fixed/nonlinear/config、`tb_pkg_smoke`、`tb_requant_residual`、`tb_udiv_isqrt`、`tb_gelu_plan`、`tb_softmax`、`tb_layernorm`。任何子命令非零立即返回同一非零码。文档逐项列出 Q 格式、所有常量、舍入伪代码、饱和范围和异常行为。

```powershell
$FoundationTops = @('tb_pkg_smoke','tb_requant_residual','tb_udiv_isqrt','tb_gelu_plan','tb_softmax','tb_layernorm')
& $env:HEATVIT_PYTHON -m unittest verification.tests.test_fixed verification.tests.test_nonlinear verification.tests.test_config_contract -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
foreach ($Top in $FoundationTops) {
  & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top $Top
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
```

- [ ] **Step 3: 运行阶段回归两次以检查确定性**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_regression.ps1 -Suite foundation
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_regression.ps1 -Suite foundation
```

Expected: 两次退出码均为 0，所有 Testbench 输出对应 `TEST_PASS`，两次生成 manifest 的 SHA-256 相同。

- [ ] **Step 4: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'test: gate fixed-point foundation regression' -Paths scripts/run_regression.ps1,verification/tests/test_config_contract.py,docs/verification/fixed-point-contract.md -TestCommand 'scripts/run_regression.ps1 -Suite foundation'
```

## 阶段 1 完成条件

- `scripts/run_regression.ps1 -Suite foundation` 退出码为 0。
- 所有 Python 与 RTL 结果逐位一致，无 `X/Z` 泄漏。
- stall 期间 ready/valid 输出保持稳定。
- descriptor 宽度、常量和配置文件通过自动一致性检查。
- 未实例化任何 Xilinx IP 原语或生成文件。
