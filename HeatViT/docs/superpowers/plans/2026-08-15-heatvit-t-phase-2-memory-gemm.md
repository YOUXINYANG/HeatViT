# HeatViT-T Phase 2: Memory and Unified GEMM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现带严格边界检查的 64-bit 行为存储通路、可推断片上缓冲和 `TH=3, TI=8, TO=8` 的统一 int8 GEMM 引擎。

**Architecture:** 一个 GEMM 命令独占存储客户端，依次搬入 A Tile、最多三个 B Tile 和 Bias，再让三个 8×8 Bank 跨 K 周期累加，最后按描述符写回 int8 或 int32。普通模式下 Bank 对应三个相邻 N Tile；Head 模式下 Bank 0/1/2 固定对应 Attention Head 0/1/2。

**Tech Stack:** SystemVerilog 2012、Vivado XSim 2023.2、Python 3.12–3.14、NumPy 2.5.2、64-bit ready/valid 行为存储。

## Global Constraints

- 必须复用阶段 1 的 `heatvit_pkg.sv`、舍入、饱和和测试脚本，不复制数值逻辑。
- GEMM 固定解释为行优先 `A[M][K] * B[K][N]`，Bias 按 N 连续；右矩阵转置由 descriptor flag 0 控制。
- 每个 Bank 每个 K 周期计算 8 个 M 行与 8 个 N 列的外积，共 64 个 int8 乘法；三个 Bank 共 192 个乘法。
- 常规 int8×int8 乘积显式扩展后累加到 signed int32；flag 18 置位时 src0 为 unsigned UQ0.8、src1 仍为 signed int8，乘积按 unsigned×signed 扩展到 signed 17-bit。K 尾部不读取或累加无效元素。
- M/N 尾块必须用 valid mask 屏蔽计算与写回；不得依赖填充数据恰好为零。
- 外存数据为 64-bit 小端；命令地址 8-byte 对齐；最后写 Beat 使用 `mem_w_strb[7:0]`。
- 存储模型必须支持确定性伪随机 read、write、command backpressure 和协议错误注入。
- 本阶段只要求仿真，不以资源利用率或 Fmax 为验收标准；不得实例化 BMG、AXI 或 MIG。

---

## 文件映射

| 文件 | 单一职责 |
| --- | --- |
| `rtl/common/heatvit_sdp_ram.sv` | 同步读、字节写使能、可推断 BRAM 模板 |
| `rtl/common/heatvit_rv_fifo.sv` | 小深度 ready/valid 解耦 FIFO |
| `rtl/memory/heatvit_addr_guard.sv` | 区域对齐、Burst 范围和 32-bit 溢出检查 |
| `rtl/memory/heatvit_mem_master.sv` | 单 outstanding 读写 Burst 协议引擎 |
| `rtl/memory/heatvit_tile_buffer.sv` | A/B/Bias/输出双缓冲封装 |
| `rtl/compute/heatvit_mac_bank.sv` | 单个 8×8 外积累加 Bank |
| `rtl/compute/heatvit_gemm_engine.sv` | 描述符驱动 Tile 循环、三 Bank 调度和写回 |
| `sim/common/behavioral_memory.sv` | `.mem` 初始化、存储、回压和协议断言 |
| `verification/heatvit_ref/gemm.py` | 纯整数矩阵乘法和写回基准 |
| `verification/tests/test_gemm.py` | 黄金 GEMM 单元测试 |
| `tools/generate_gemm_vectors.py` | 小矩阵和完整尺寸 Tile 向量生成器 |
| `sim/tb/tb_memory_path.sv` | 地址与 Burst 协议测试 |
| `sim/tb/tb_mac_bank.sv` | Bank 逐周期测试 |
| `sim/tb/tb_gemm_engine.sv` | GEMM 模式、尾块和回压测试 |

## 锁定存储客户端接口

`heatvit_mem_master` 的外侧端口与设计规格完全一致：

```systemverilog
output logic        mem_cmd_valid;
input  logic        mem_cmd_ready;
output logic        mem_cmd_write;
output logic [31:0] mem_cmd_addr;
output logic [15:0] mem_cmd_len;
output logic        mem_w_valid;
input  logic        mem_w_ready;
output logic [63:0] mem_w_data;
output logic [7:0]  mem_w_strb;
output logic        mem_w_last;
input  logic        mem_r_valid;
output logic        mem_r_ready;
input  logic [63:0] mem_r_data;
input  logic        mem_r_last;
```

内侧请求固定为单 outstanding：`req_valid/req_ready`、`req_write`、`req_addr[31:0]`、`req_bytes[31:0]`；写流和读流均为 64-bit ready/valid/last/strb。`req_bytes` 可非 8 倍数，主接口换算为向上取整 Beat 数。

### Task 1: 实现行为存储和地址守卫

**Files:**
- Create: `rtl/memory/heatvit_addr_guard.sv`
- Create: `sim/common/behavioral_memory.sv`
- Create: `sim/tb/tb_memory_path.sv`

**Interfaces:**
- Consumes: 四个锁存 base/bytes、命令地址和 Beat 数。
- Produces: `addr_ok`、`addr_error_code`，以及锁定外存协议的仿真模型。

- [ ] **Step 1: 写失败的边界与协议测试**

测试区域设为 base `0x1000`、bytes `0x100`。必须接受 `[0x1000,1 beat]` 和 `[0x10f8,1 beat]`，拒绝未对齐 `0x1001`、零 Beat、`[0x10f8,2 beats]` 及 `0xfffffff8,2 beats` 溢出。行为存储必须在 `mem_w_last` 早到/晚到、读长度不匹配和越界时 `$fatal`。

- [ ] **Step 2: 运行并确认 DUT 缺失**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_memory_path`

Expected: FAIL，缺失 `heatvit_addr_guard`。

- [ ] **Step 3: 实现无回绕范围判断和小端存储模型**

范围判断使用 33-bit：

```systemverilog
first_ext = {1'b0, cmd_addr};
size_ext  = {14'd0, cmd_len, 3'b000};
last_ext  = first_ext + size_ext - 1'b1;
addr_ok   = (cmd_len != 0) && (cmd_addr[2:0] == 0) &&
            !last_ext[32] && (first_ext >= {1'b0, region_base}) &&
            (last_ext < ({1'b0, region_base} + {1'b0, region_bytes}));
```

行为存储用 byte 数组，`data[8*i +: 8]` 对应 `memory[address+i]`；只在 `strb[i]` 为 1 时写入。LFSR seed 固定由 TB 参数传入，stall 决策不能调用非确定性的 `$urandom`。

- [ ] **Step 4: 运行无回压和 50% 回压两轮测试**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_memory_path -PlusArgs '+STALL_MASK=0'
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_memory_path -PlusArgs '+STALL_MASK=3'
```

Expected: 两轮均输出 `TEST_PASS tb_memory_path`。

- [ ] **Step 5: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'test: define external memory protocol and bounds' -Paths rtl/memory/heatvit_addr_guard.sv,sim/common/behavioral_memory.sv,sim/tb/tb_memory_path.sv -TestCommand 'scripts/run_xsim.ps1 -Top tb_memory_path'
```

### Task 2: 实现存储主机、FIFO 和可推断 RAM

**Files:**
- Create: `rtl/common/heatvit_sdp_ram.sv`
- Create: `rtl/common/heatvit_rv_fifo.sv`
- Create: `rtl/memory/heatvit_mem_master.sv`
- Create: `sim/tb/tb_mem_master.sv`

**Interfaces:**
- Consumes: 锁定内侧请求与外侧内存接口。
- Produces: 一次只接受一个请求的 read/write 搬运器；`done`、`protocol_error` 或 `abort_done` 单周期脉冲。

- [ ] **Step 1: 写请求稳定性和尾拍失败测试**

依次发送 1、7、8、9、31 bytes 的读写；检查命令 Beat 数为 `1,1,1,2,4`，尾拍 strobe 分别为 `01,7f,ff,01,7f`。command ready、write ready、read valid 分别 stall 3 至 11 周期，TB 断言 stalled valid payload 不变。另在 command handshake 前和 Burst 中途各拉高一次 `abort`：前者不得发命令，后者必须合法排空已接受 Burst、屏蔽本地读输出并产生 `abort_done`，且不得启动下一请求。

- [ ] **Step 2: 运行并确认模块缺失**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_mem_master`

Expected: FAIL，缺失 `heatvit_mem_master`。

- [ ] **Step 3: 实现单请求 FSM 与存储模板**

FSM 固定为 `IDLE -> COMMAND -> DATA -> COMPLETE`，并含 `DRAIN_ABORT`。写命令只在 command handshake 后接受写数据；读命令只在 command handshake 后拉高 `mem_r_ready`。abort 在 COMMAND handshake 前取消 valid 并直接 abort_done；handshake 后进入 DRAIN_ABORT：读 Burst 继续拉高 ready 并丢弃至 last，写 Burst 对剩余 Beat 主动发送 `data=0,strb=0` 且在规定 Beat 产生 last，因 strobe 全零不得修改存储。两种路径都不再向本地 client 交付数据或请求新数据。最后 Beat 的正常 strobe 由 `req_bytes[2:0]` 生成，余数为零时必须是 `8'hff`。单请求最大为 `65535*8=524280` bytes；更大 Tensor 必须由调用方拆成多个请求，超限请求直接返回 protocol_error 且不发命令。`heatvit_sdp_ram` 使用单 always_ff、同步读、逐 byte 写使能；`heatvit_rv_fifo` 深度参数只允许 2 的幂并跟踪读写指针与计数。

```systemverilog
case (state)
  IDLE:     if (req_valid && req_ready) state <= COMMAND;
  COMMAND:  if (mem_cmd_valid && mem_cmd_ready) state <= DATA;
  DATA:     if (beat_handshake && expected_last && observed_last) state <= COMPLETE;
  COMPLETE: begin done <= 1'b1; state <= IDLE; end
  DRAIN_ABORT: if (external_last_handshake) begin abort_done <= 1'b1; state <= IDLE; end
  default:  state <= IDLE;
endcase
last_strb = (req_bytes[2:0] == 3'd0) ? 8'hff : ((9'h001 << req_bytes[2:0]) - 1'b1);
```

- [ ] **Step 4: 运行协议与 RAM read-after-write 测试**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_mem_master`

Expected: `TEST_PASS tb_mem_master`，覆盖全部五种长度、三条 stall 路径、两种 abort 时点和 early/late last 错误注入。

- [ ] **Step 5: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'feat: add burst memory master and inferred buffers' -Paths rtl/common/heatvit_sdp_ram.sv,rtl/common/heatvit_rv_fifo.sv,rtl/memory/heatvit_mem_master.sv,sim/tb/tb_mem_master.sv -TestCommand 'scripts/run_xsim.ps1 -Top tb_mem_master'
```

### Task 3: 实现 8×8 MAC Bank

**Files:**
- Create: `rtl/compute/heatvit_mac_bank.sv`
- Create: `sim/tb/tb_mac_bank.sv`
- Create: `tools/generate_gemm_vectors.py`

**Interfaces:**
- Consumes: 每个 K 周期 8 个 A byte、8 个 B signed int8、`a_unsigned`、8-bit row mask、8-bit column mask、`clear_accum`、`accum_valid`。
- Produces: 64 个 signed int32 accumulator 和 `accum_done`。

- [ ] **Step 1: 写外积和 mask 失败测试**

第一周期 A 为 `[1,2,3,4,5,6,7,8]`、B 为 `[1,-1,2,-2,3,-3,4,-4]`；第二周期 A 全 1、B 全 2。检查每个 `(row,col)` 等于两次乘积之和。再令 row mask=`8'h0f`、column mask=`8'h03`，断言其余 56 个 accumulator 保持零。

- [ ] **Step 2: 运行并确认 Bank 缺失**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_mac_bank`

Expected: FAIL，缺失 `heatvit_mac_bank`。

- [ ] **Step 3: 实现 64 路显式 signed MAC**

对每个有效 `(r,c)` 执行：

```systemverilog
logic signed [15:0] product;
product = $signed(a_lane[r]) * $signed(b_lane[c]);
accum[r][c] <= $signed(accum[r][c]) + {{16{product[15]}}, product};
```

`a_unsigned=0` 时使用上面的 signed×signed 乘法；`a_unsigned=1` 时先把 A 零扩展到 9-bit、B 符号扩展到 9-bit，再生成 signed 18-bit 乘积。`clear_accum` 的优先级高于 `accum_valid`；无效 row/column 永远不改变 accumulator；复位与 clear 后全部 64 项为零。B 必须始终按 signed 解释。

- [ ] **Step 4: 运行边界和 K=768 累加测试**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_mac_bank`

Expected: `TEST_PASS tb_mac_bank`；包括 `-128*-128`、`127*127`、unsigned `128 * -128` 和 K=768 最大安全累加，逐项匹配 Python。

- [ ] **Step 5: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'feat: add 8 by 8 signed MAC bank' -Paths rtl/compute/heatvit_mac_bank.sv,sim/tb/tb_mac_bank.sv,tools/generate_gemm_vectors.py -TestCommand 'scripts/run_xsim.ps1 -Top tb_mac_bank'
```

### Task 4: 实现 Python GEMM 基准和矩阵布局测试

**Files:**
- Create: `verification/heatvit_ref/gemm.py`
- Create: `verification/tests/test_gemm.py`
- Modify: `tools/generate_gemm_vectors.py`
- Create: `sim/vectors/gemm/manifest.json`

**Interfaces:**
- Consumes: `fixed.requant`、行优先 int8 A/B、int32 Bias 和 descriptor 标志。
- Produces: `gemm(a, b, bias, transpose_b) -> list[list[int]]`、`gemm_writeback(accum, src_exp, dst_exp, output_bits) -> list[list[int]]` 和打包后的内存映像。

- [ ] **Step 1: 写布局和转置失败测试**

```python
def test_row_major_and_transpose(self):
    a = [[1, 2, 3], [-1, 0, 1]]
    b = [[1, 2], [3, 4], [5, 6]]
    self.assertEqual(gemm(a, b, [7, -7], False), [[29, 21], [11, -3]])
    bt = [[1, 3, 5], [2, 4, 6]]
    self.assertEqual(gemm(a, bt, [7, -7], True), [[29, 21], [11, -3]])
```

- [ ] **Step 2: 运行并确认导入失败**

Run: `& $env:HEATVIT_PYTHON -m unittest verification.tests.test_gemm -v`

Expected: FAIL，缺失 `verification.heatvit_ref.gemm`。

- [ ] **Step 3: 实现精确循环与内存打包**

标量参考固定循环顺序为 M、N、K；完整尺寸路径把 A/B 显式转换为 NumPy `int64` 后调用矩阵乘法，并在加入 Bias 前断言结果处于 int32 范围。每个乘积保持 signed 16-bit 数值语义，Bias 在 K 归约后加入。`gemm_writeback` 根据 flag 7 选择小端 int32 或依据尺度调用 int8 requant；转置只改变 B 视图/地址公式，不改数值布局。测试必须证明标量路径和 NumPy 路径逐项相等。

```python
import numpy as np

def gemm(a, b, bias, transpose_b):
    a64 = np.asarray(a, dtype=np.int64)
    b64 = np.asarray(b, dtype=np.int64)
    rhs = b64.T if transpose_b else b64
    accum = a64 @ rhs
    if bias is not None:
        accum = accum + np.asarray(bias, dtype=np.int64)[None, :]
    if np.any(accum < -(1 << 31)) or np.any(accum > (1 << 31) - 1):
        raise OverflowError("GEMM accumulator exceeds int32")
    return accum.astype(np.int32).tolist()
```

- [ ] **Step 4: 运行 Python 测试并生成六组矩阵**

生成维度 `(1,1,1)`、`(7,9,5)`、`(8,24,8)`、`(9,25,17)`、`(197,192,192)`、`(8,8,64)`，后两组分别覆盖普通模式和 Head 模式。

Run:

```powershell
& $env:HEATVIT_PYTHON -m unittest verification.tests.test_gemm -v
& $env:HEATVIT_PYTHON tools/generate_gemm_vectors.py --seed 20260815 --output sim/vectors/gemm
```

Expected: unittest `ok`，manifest 中六组 SHA-256 均存在且重跑不变。

- [ ] **Step 5: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'test: add row-major integer GEMM reference' -Paths verification/heatvit_ref/gemm.py,verification/tests/test_gemm.py,tools/generate_gemm_vectors.py,sim/vectors/gemm -TestCommand 'python -m unittest verification.tests.test_gemm -v'
```

### Task 5: 实现 Tile Buffer 和普通 GEMM 调度

**Files:**
- Create: `rtl/memory/heatvit_tile_buffer.sv`
- Create: `rtl/compute/heatvit_gemm_engine.sv`
- Create: `sim/tb/tb_gemm_engine.sv`

**Interfaces:**
- Consumes: `cmd_valid/cmd_ready`、完整 `heatvit_desc_t`、四个 region base/bytes、锁定存储客户端接口。
- Produces: `busy`、单周期 `done`、`error_valid/error_code`，以及 descriptor 指定的 dst Tensor。

- [ ] **Step 1: 写普通模式失败测试**

从行为存储加载 `(M,N,K)=(7,9,5)` 和 `(9,25,17)`，分别测试无 Bias/int8 写回与 Bias/int32 写回。TB 在 done 后逐 byte 比较 dst 区，并断言所有 padding byte 保持初始值 `8'ha5`。

- [ ] **Step 2: 运行并确认 GEMM engine 缺失**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_gemm_engine -PlusArgs '+CASE=ordinary'`

Expected: FAIL，缺失 `heatvit_gemm_engine`。

- [ ] **Step 3: 实现普通三 Bank Tile 循环**

外层顺序固定为 `m_tile += 8`、`n_group += 24`、`k += 1`。每个 K 周期向三个 Bank 广播 8 个 A 行值，分别提供 B 的 `[n_group+0..7]`、`[+8..15]`、`[+16..23]`。加载下一 Tile 与当前计算允许 ping-pong，但第一版只需保证协议正确；不得引入第二套计算路径。Bias 加入 accumulator 后再按 descriptor 尺度重定标。

```text
for m0 in range(0, M, 8):
  for n0 in range(0, N, 24):
    clear all three banks
    for k0 in range(0, K):
      A_lanes = A[m0:m0+8, k0]
      bank0_B = B[k0, n0+0:n0+8]
      bank1_B = B[k0, n0+8:n0+16]
      bank2_B = B[k0, n0+16:n0+24]
      accumulate outer products under row/column masks
    add Bias, requantize, and write only valid rows/columns
```

- [ ] **Step 4: 运行普通模式和尾块测试**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_gemm_engine -PlusArgs '+CASE=ordinary'
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_gemm_engine -PlusArgs '+CASE=tail'
```

Expected: 两轮 `TEST_PASS tb_gemm_engine`；行为存储地址 trace 不包含任何超出 A/B/Bias/dst 声明范围的访问。

- [ ] **Step 5: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'feat: add descriptor-driven tiled GEMM' -Paths rtl/memory/heatvit_tile_buffer.sv,rtl/compute/heatvit_gemm_engine.sv,sim/tb/tb_gemm_engine.sv -TestCommand 'scripts/run_xsim.ps1 -Top tb_gemm_engine -PlusArgs +CASE=ordinary'
```

### Task 6: 增加转置、Head 模式和全链路回压

**Files:**
- Modify: `rtl/compute/heatvit_gemm_engine.sv`
- Modify: `sim/tb/tb_gemm_engine.sv`
- Modify: `verification/tests/test_gemm.py`
- Modify: `scripts/run_regression.ps1`

**Interfaces:**
- Consumes: descriptor flag 0、flag 5、flag 18 和 `heads=3`。
- Produces: 转置 B 地址生成、Bank-to-Head 固定映射、unsigned-src0 模式、`run_regression.ps1 -Suite gemm`。

- [ ] **Step 1: 写转置与 Head 模式失败测试**

转置案例使用 `(M,N,K)=(8,8,64)` 且存储 B 为 `[N][K]`。Head 案例为三个互不相同的 8×8×64 矩阵，Bank 0/1/2 预期输出设置不同哨兵值，确保交换 Bank 会失败。unsigned-src0 案例必须含 A=`8'h80` 和 B=`-128`，预期乘积为 `-16384` 而不是 `+16384`。非法 `heads=2` 或在非 Attention×V opcode 上置 flag 18，必须在发出首个 memory command 前返回 error code 2。

- [ ] **Step 2: 运行并确认现有 engine 不支持这些标志**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_gemm_engine -PlusArgs '+CASE=transpose'
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_gemm_engine -PlusArgs '+CASE=head'
```

Expected: 至少一轮逐位比较失败；非法配置不得被误判为成功。

- [ ] **Step 3: 实现转置地址和 Head 隔离**

普通 B 地址为 `base + (k_index*N + n_index)`；转置 B 地址为 `base + (n_index*K + k_index)`。Head 模式把每个 Bank 的 B 基址增加 `head * K * N_per_head`，禁止跨 Head 拼接 accumulator。flag 18 原样送到全部 MAC Bank，并由 opcode 合法性检查限制为 Attention×V。所有 address multiply-add 使用 64-bit 中间值，再由 address guard 检查 32-bit 范围。

```systemverilog
normal_b_offset    = (64'(k_index) * n_eff) + n_index;
transpose_b_offset = (64'(n_index) * k_eff) + k_index;
head_b_offset      = (64'(head_index) * k_eff * n_per_head) +
                     (rhs_transpose ? transpose_b_offset : normal_b_offset);
a_unsigned = desc.flags[FLAG_SRC0_UNSIGNED];
```

- [ ] **Step 4: 运行随机回压和阶段回归**

`-Suite gemm` 必须包含 memory path、mem master、MAC Bank、GEMM ordinary/tail/transpose/head，并以 `STALL_MASK=3` 重跑 GEMM。

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_regression.ps1 -Suite gemm
```

Expected: 退出码 0；所有案例逐位一致；三 Bank 的 `mac_active_cycles` 均大于零。

- [ ] **Step 5: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'feat: complete transpose and head GEMM modes' -Paths rtl/compute/heatvit_gemm_engine.sv,sim/tb/tb_gemm_engine.sv,verification/tests/test_gemm.py,scripts/run_regression.ps1 -TestCommand 'scripts/run_regression.ps1 -Suite gemm'
```

## 阶段 2 完成条件

- `scripts/run_regression.ps1 -Suite foundation` 仍通过。
- `scripts/run_regression.ps1 -Suite gemm` 退出码为 0。
- 普通、转置、Bias、int8/int32 写回、三 Head、M/N/K 尾块和随机回压逐位匹配。
- 所有外存访问通过 RTL 与行为存储两层边界检查。
- RTL 中不存在 `xpm_*`、`blk_mem_gen`、`axis_*`、`floating_point` 或 `div_gen` 实例。
