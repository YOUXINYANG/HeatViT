# HeatViT-T Phase 4: Dynamic Token Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 HeatViT 三 Head Token 分类、Head 权重融合、稳定 Token 压缩和单 Package Token，并验证所有剪枝与回退路径。

**Architecture:** Selector 仍复用唯一 GEMM 完成所有 MLP；专用流式单元只做 reduction、local/global concat、Head fuse 和最终压缩/Package。最终状态转换由一个原子 `OP_SELECTOR_FINALIZE` 完成，只有它能更新 Token 数和 Package-present 状态，避免动态状态跨描述符半更新。

**Tech Stack:** SystemVerilog 2012、Vivado XSim 2023.2、Python 3.12–3.14、NumPy 2.5.2 整数基准、阶段 1 至 3 的 Executor/GEMM/Softmax/divider。

## Global Constraints

- CLS 位于索引 0，永久旁路且不参与分类、全局均值或 Package。
- 候选数 `C=current_token_count-1`；若 `current_package_present=1`，输入 Package 固定是最后一个候选。
- 三个 Head 各自使用独立 64→32 local MLP 和 64→32→16→2 score MLP 权重。
- 所有 per-head 矩阵按 `[head][K][N]` 连续行优先存放；Head 0、1、2 的权重区域互不重叠。
- 二分类列 0 为 Drop、列 1 为 Keep；只融合 Keep 概率，`keep_score >= 32768` 时保留。
- Head 权重分支固定为每候选三维 Head 均值 → 3→3→3 → PLAN Sigmoid。
- local/global 特征布局固定为 `[head][candidate][32]`，concat 后为 `[head][candidate][64]`，前 32 为 local、后 32 为该 Head 的 global。
- incoming Package 即使分类结果为 Keep，也必须进入 Package 累加而不得作为普通 Token 输出。
- 输出顺序固定为 CLS、稳定顺序的 kept normal tokens、可选且至多一个 Package。
- 所有除法采用最近舍入、中点远离零；Head 分母零和 Package 分母零必须执行规定回退并置警告。
- Selector 完成后的合法 Token 数为 2 至 197；任何写回不得超过输入矩阵容量。
- 不增加 GEMM 或 divider 实例，不使用 Vivado IP。

---

## 文件映射

| 文件 | 单一职责 |
| --- | --- |
| `verification/heatvit_ref/selector.py` | 完整整数 Selector 与所有中间检查点 |
| `verification/tests/test_selector.py` | 阈值、融合、压缩和 Package 单元测试 |
| `tools/generate_selector_vectors.py` | 合成 Selector 参数、输入和 `.mem` |
| `rtl/selector/heatvit_reduce_mean.sv` | Token/Channel 轴 signed 最近舍入均值 |
| `rtl/selector/heatvit_feature_concat.sv` | local/global 广播与拼接 |
| `rtl/selector/heatvit_head_fuse.sv` | 三 Head 加权 Keep Score 与零分母回退 |
| `rtl/selector/heatvit_token_compactor.sv` | CLS 复制和普通 Token 稳定压缩 |
| `rtl/selector/heatvit_token_packager.sv` | 被剪 Token/输入 Package 的累加、除法与回退 |
| `rtl/selector/heatvit_selector_finalize.sv` | 原子协调 compactor、packager 和状态更新 |
| `sim/tb/tb_selector_features.sv` | Reduction/concat 测试 |
| `sim/tb/tb_head_fuse.sv` | Head fuse 测试 |
| `sim/tb/tb_selector_finalize.sv` | 压缩/Package 边界测试 |
| `sim/tb/tb_token_selector.sv` | 完整 Selector 描述符序列测试 |

## 锁定 Selector Tensor 布局

```text
input tokens       [N][192] signed int8
candidate tokens   [C][3][64] signed int8
local features     [3][C][32] signed int8
global features    [3][32] signed int8
local_global       [3][C][64] signed int8
head keep scores   [3][C] Q0.16
head statistics    [C][3] signed int8
head weights       [C][3] Q0.16
fused keep scores  [C] Q0.16
output tokens      [N_next][192] signed int8
```

上述所有 Q0.16 Tensor 在 Scratch 中每元素占 4 bytes，bits 16:0 有效、bits 31:17 为零；普通 int8 Tensor 每元素占 1 byte。

阶段 4 专用 opcode 的字段语义固定为：

| Opcode | 有效维度 | 地址角色 |
| --- | --- | --- |
| OP_SELECTOR_SOFTMAX | `m=C,n=2,heads=3` | src0=head logits int8，dst=head Keep Score Q0.16 |
| OP_REDUCE_MEAN | `m=C`，axis=`param0[3:2]` | src0=local feature 或 candidate Head，dst=global feature 或 stats |
| OP_CONCAT_LOCAL_GLOBAL | `m=C,n=64,heads=3` | src0=local，src1=global Scratch，dst=concat |
| OP_HEAD_FUSE | `m=C,n=3` | src0=head Keep Score，src1=Head Weight Scratch，dst=fused Q0.16 |
| OP_SELECTOR_FINALIZE | `m=N,n=192` | src0=input tokens，src1=fused Score Scratch，dst=next Activation |

Selector MLP 与 Head-weight MLP 继续使用 OP_GEMM；未列出的地址字段必须为零。

### Task 1: 实现 Selector 整数黄金模型

**Files:**
- Create: `verification/heatvit_ref/selector.py`
- Create: `verification/tests/test_selector.py`
- Create: `tools/generate_selector_vectors.py`

**Interfaces:**
- Consumes: `fixed.py`、`nonlinear.py`、`gemm.py` 和 `[N][192]` 输入。
- Produces: immutable `SelectorParams`、`FinalizeResult`、`SelectorResult` dataclasses，`token_selector(tokens, package_present, params) -> SelectorResult`，以及同文件私有 helpers `per_head_local_mlp`、`mean_over_candidates`、`concat_local_global`、`per_head_score_mlp`、`mean_over_head_lanes`、`head_weight_mlp`、`fuse_head_scores`、`finalize_tokens`；结果含 output、scores、counts、warning flags 和全部中间 Tensor。

- [ ] **Step 1: 写决策与 Package 失败测试**

```python
def test_threshold_is_inclusive(self):
    result = finalize_tokens(
        cls=[1, 2], normal=[[3, 4], [5, 6]], incoming_package=None,
        scores=[32768, 32767])
    self.assertEqual(result.tokens, [[1, 2], [3, 4], [5, 6]])
    self.assertTrue(result.package_present)

def test_incoming_package_never_becomes_normal(self):
    result = finalize_tokens(
        cls=[1], normal=[[2]], incoming_package=[9],
        scores=[65536, 65536])
    self.assertEqual(result.tokens, [[1], [2], [9]])
    self.assertTrue(result.package_present)
```

第一例第二个 normal 被剪但形成 Package，因此输出为 CLS、第一 normal、Package；第二例 Package 即使 score=1 仍位于末尾且数量为一。

- [ ] **Step 2: 运行并确认导入失败**

Run: `& $env:HEATVIT_PYTHON -m unittest verification.tests.test_selector -v`

Expected: FAIL，缺失 `verification.heatvit_ref.selector`。

- [ ] **Step 3: 实现完整纯整数流程**

`FinalizeResult` 固定字段为 `tokens`、`package_present`、`kept_normal_count`、`pruned_normal_count`、`warnings`。`SelectorResult` 固定字段为上述状态字段加 `token_count`、`local`、`global_features`、`head_scores`、`head_stats`、`head_weights`、`fused_scores`。Head fuse numerator 为 `sum(weight[h]*score[h])`，denominator 为 `sum(weight[h])`；Package 每通道 numerator 为 `sum(score[t]*feature[t][d])`。两个除法都用 quotient/remainder 进行统一舍入。

```python
def token_selector(tokens, package_present, params):
    candidates = tokens[1:]
    local = per_head_local_mlp(candidates, params.local)
    global_features = mean_over_candidates(local)
    local_global = concat_local_global(local, global_features)
    head_scores = per_head_score_mlp(local_global, params.score)
    head_stats = mean_over_head_lanes(candidates)
    head_weights = head_weight_mlp(head_stats, params.head_weight)
    fused_scores, warn_head = fuse_head_scores(head_scores, head_weights)
    finalized = finalize_tokens(tokens[0], candidates, package_present, fused_scores)
    return SelectorResult(tokens=finalized.tokens,
                          token_count=len(finalized.tokens),
                          package_present=finalized.package_present,
                          kept_normal_count=finalized.kept_normal_count,
                          pruned_normal_count=finalized.pruned_normal_count,
                          local=local, global_features=global_features,
                          head_scores=head_scores, head_stats=head_stats,
                          head_weights=head_weights, fused_scores=fused_scores,
                          warnings=warn_head | finalized.warnings)
```

- [ ] **Step 4: 运行六种确定性边界案例**

必须覆盖：全保留无 Package、全剪无 Package、混合剪、阈值相等、已有 Package 再剪、Head/Package 两种零分母。Run:

```powershell
& $env:HEATVIT_PYTHON -m unittest verification.tests.test_selector -v
& $env:HEATVIT_PYTHON tools/generate_selector_vectors.py --suite unit --seed 20260815 --output sim/vectors/selector
```

Expected: unittest 全部 `ok`，manifest 列出六个案例及预期 warning bits。

- [ ] **Step 5: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'test: add integer token selector reference' -Paths verification/heatvit_ref/selector.py,verification/tests/test_selector.py,tools/generate_selector_vectors.py -TestCommand 'python -m unittest verification.tests.test_selector -v'
```

### Task 2: 实现 Reduction 和 Local/Global Concat

**Files:**
- Create: `rtl/selector/heatvit_reduce_mean.sv`
- Create: `rtl/selector/heatvit_feature_concat.sv`
- Create: `sim/tb/tb_selector_features.sv`
- Modify: `rtl/compute/heatvit_tensor_executor.sv`

**Interfaces:**
- Consumes: OP_REDUCE_MEAN 的 `param0[1:0]=DYN_M_CANDIDATES`，以及 `param0[3:2]`（00=候选轴、01=64 通道轴）和 OP_CONCAT_LOCAL_GLOBAL。
- Produces: 锁定布局的 global features、head statistics 和 local_global；divider client 2 用于 signed mean。

- [ ] **Step 1: 写负数均值和广播失败测试**

候选轴输入 `[-2,-1,0,1,2]` 预期均值 0；`[-2,-1]` 预期 ties-away 结果 -2；通道轴用 64 个值构造余数恰好等于分母一半。Concat 使用 local 哨兵 `0x11` 和三个不同 global Head 哨兵，检查每个 candidate 后 32 项只来自同一 Head。

- [ ] **Step 2: 运行并确认模块缺失**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_selector_features`

Expected: FAIL，缺失 `heatvit_reduce_mean` 或 `heatvit_feature_concat`。

- [ ] **Step 3: 实现两种索引模式和 signed 除法舍入**

候选轴按 `[head][candidate][channel]` 累加 C 项；通道轴按 `[candidate][head][lane]` 累加 64 项。对 signed sum 取 magnitude 后请求 client 2 divider，以 quotient/remainder 舍入再恢复符号。Concat 逐 Head/候选写 64 bytes，前后半区不能交错。

```text
candidate-axis: dst[head,channel] = round(sum_c src[head,c,channel] / C)
lane-axis:      dst[c,head]       = round(sum_lane src[c,head,lane] / 64)
concat:         dst[head,c,0:32]  = local[head,c,0:32]
                dst[head,c,32:64] = global[head,0:32]
```

- [ ] **Step 4: 运行 C=1、5、196 和回压测试**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_selector_features -PlusArgs '+STALL_MASK=3'`

Expected: `TEST_PASS tb_selector_features`，三种 C 均逐 byte 匹配且无越界。

- [ ] **Step 5: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'feat: add selector reductions and feature concat' -Paths rtl/selector/heatvit_reduce_mean.sv,rtl/selector/heatvit_feature_concat.sv,sim/tb/tb_selector_features.sv,rtl/compute/heatvit_tensor_executor.sv -TestCommand 'scripts/run_xsim.ps1 -Top tb_selector_features'
```

### Task 3: 实现 Head 权重融合

**Files:**
- Create: `rtl/selector/heatvit_head_fuse.sv`
- Create: `sim/tb/tb_head_fuse.sv`
- Modify: `rtl/compute/heatvit_tensor_executor.sv`

**Interfaces:**
- Consumes: 每候选三个 Q0.16 Keep Score 与三个 Q0.16 Head Weight、divider client 2。
- Produces: 每候选一个 Q0.16 fused score、`warn_head_den_zero`。

- [ ] **Step 1: 写正常、零分母和阈值失败测试**

正常案例 scores=`[0,32768,65536]`、weights=`[65536,65536,0]`，预期 fused=`16384`。全零 weights 时预期三个 score 的等权最近舍入平均并置 warning。构造 fused 分别为 32767、32768，供 finalize 判定边界。

- [ ] **Step 2: 运行并确认模块缺失**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_head_fuse`

Expected: FAIL，缺失 `heatvit_head_fuse`。

- [ ] **Step 3: 实现 36-bit numerator 和回退**

三个 `17×17` 乘积扩为至少 34-bit unsigned 后在 36-bit 累加；denominator 至少 19-bit。非零分母通过 client 2 divider 求最近舍入商并饱和到 0..65536。分母为零时 numerator 改为三个 score 之和、denominator 固定 3，并发 warning pulse。

```systemverilog
weighted_num = (score0 * weight0) + (score1 * weight1) + (score2 * weight2);
weight_den   = weight0 + weight1 + weight2;
if (weight_den == 0) begin
  div_num <= score0 + score1 + score2;
  div_den <= 64'd3;
  warn_head_den_zero <= 1'b1;
end else begin
  div_num <= weighted_num;
  div_den <= weight_den;
end
```

- [ ] **Step 4: 运行 1024 个随机三元组**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_head_fuse`

Expected: `TEST_PASS tb_head_fuse`，正常与零分母均逐项匹配 Python。

- [ ] **Step 5: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'feat: add weighted three-head score fusion' -Paths rtl/selector/heatvit_head_fuse.sv,sim/tb/tb_head_fuse.sv,rtl/compute/heatvit_tensor_executor.sv -TestCommand 'scripts/run_xsim.ps1 -Top tb_head_fuse'
```

### Task 4: 实现稳定压缩和单 Package Finalize

**Files:**
- Create: `rtl/selector/heatvit_token_compactor.sv`
- Create: `rtl/selector/heatvit_token_packager.sv`
- Create: `rtl/selector/heatvit_selector_finalize.sv`
- Create: `sim/tb/tb_selector_finalize.sv`
- Modify: `rtl/compute/heatvit_tensor_executor.sv`

**Interfaces:**
- Consumes: input `[N][192]`、fused score `[C]`、`current_package_present`、divider client 2。
- Produces: output `[N_next][192]`、state update、`warn_package_den_zero`。

- [ ] **Step 1: 写五种输出顺序失败测试**

使用每个 Token 全通道填充不同 index 的哨兵，覆盖：全保留、全剪、交替剪、已有 Package 且全保留 normal、已有 Package 且混合剪。逐 Token 检查顺序、`next_token_count`、`next_package_present`；输入区和输出区设置不同 guard bytes 并检查不越界。

- [ ] **Step 2: 运行并确认 finalize 模块缺失**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_selector_finalize`

Expected: FAIL，缺失 `heatvit_selector_finalize`。

- [ ] **Step 3: 实现单遍分类与双用途累加**

先复制 CLS。对普通候选，score≥32768 时按原序写下一个 kept slot，否则把 192 个 `feature*score` 加入 48-bit signed numerator 并累计 denominator/count。若输入 Package 存在，最后一个候选无条件只进入该累加。扫描结束后：参与数为零则不追加 Package；denominator 非零则逐通道最近舍入除法；denominator 为零则用未加权 feature sum/count 回退并置 warning。最后原子产生：

```systemverilog
next_token_count      = 8'(1 + kept_normal_count + package_will_exist);
next_package_present  = package_will_exist;
state_update_valid    = 1'b1;
```

- [ ] **Step 4: 运行零权重、负特征和 backpressure 测试**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_selector_finalize -PlusArgs '+STALL_MASK=3'`

Expected: `TEST_PASS tb_selector_finalize`；包含 score 全零、signed numerator 为负和商为半数中点的通道。

- [ ] **Step 5: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'feat: add stable compaction and single package token' -Paths rtl/selector/heatvit_token_compactor.sv,rtl/selector/heatvit_token_packager.sv,rtl/selector/heatvit_selector_finalize.sv,sim/tb/tb_selector_finalize.sv,rtl/compute/heatvit_tensor_executor.sv -TestCommand 'scripts/run_xsim.ps1 -Top tb_selector_finalize'
```

### Task 5: 集成完整 Token Selector 描述符序列

**Files:**
- Modify: `verification/heatvit_ref/selector.py`
- Modify: `tools/generate_selector_vectors.py`
- Modify: `rtl/compute/heatvit_tensor_executor.sv`
- Create: `sim/tb/tb_token_selector.sv`

**Interfaces:**
- Consumes: 一个 Selector 的完整 MLP 权重、N/package state 和输入激活。
- Produces: 下一激活、next state、所有 classifier 检查点和 warning flags。

- [ ] **Step 1: 写序列与独立 Head 权重失败测试**

三个 Head 的 local/score 权重使用不同哨兵模式；若误共享任一权重，`head_scores.mem` 必须不匹配。Descriptor 序列固定为：

```text
OP_GEMM head-mode  64->32 + GELU             local
OP_REDUCE_MEAN     candidate axis             global
OP_CONCAT_LOCAL_GLOBAL                        local_global
OP_GEMM head-mode  64->32 + GELU             score_h1
OP_GEMM head-mode  32->16 + GELU             score_h2
OP_GEMM head-mode  16->2                     logits
OP_SELECTOR_SOFTMAX                            head_keep_scores
OP_REDUCE_MEAN     64-channel axis            head_statistics
OP_GEMM            3->3 + GELU                head_weight_hidden
OP_GEMM            3->3 + PLAN                head_weights
OP_HEAD_FUSE                                  fused_scores
OP_SELECTOR_FINALIZE                           output/state
```

- [ ] **Step 2: 运行并确认 Executor 尚未支持全部 Selector opcode**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_token_selector -PlusArgs '+VECTOR_DIR=build/vectors/selector_mixed'`

Expected: FAIL，向量缺失或 Executor 对某个 Selector opcode 返回 error 1。

- [ ] **Step 3: 扩展 Executor 且保持唯一资源**

加入 OP_SELECTOR_SOFTMAX、OP_REDUCE_MEAN、OP_CONCAT_LOCAL_GLOBAL、OP_HEAD_FUSE、OP_SELECTOR_FINALIZE 解码。Softmax 使用 divider client 0，LN 使用 client 1，当前激活的 reduction/head-fuse/finalize 通过 mux 使用 client 2；同一时刻只有一个 client-2 子单元获准请求。只有 Finalize 可拉高 state_update_valid。

```systemverilog
case (desc_reg.opcode)
  OP_GEMM:                child_sel <= CHILD_GEMM;
  OP_PATCHIFY,
  OP_COPY_ADD_POS,
  OP_QKV_UNPACK,
  OP_HEAD_CONCAT:         child_sel <= CHILD_LAYOUT;
  OP_LAYERNORM,
  OP_RESIDUAL,
  OP_ATTN_SOFTMAX:        child_sel <= CHILD_VECTOR;
  OP_SELECTOR_SOFTMAX:    child_sel <= CHILD_SELECTOR_SOFTMAX;
  OP_REDUCE_MEAN:         child_sel <= CHILD_REDUCE;
  OP_CONCAT_LOCAL_GLOBAL: child_sel <= CHILD_CONCAT;
  OP_HEAD_FUSE:           child_sel <= CHILD_HEAD_FUSE;
  OP_SELECTOR_FINALIZE:   child_sel <= CHILD_SELECTOR_FINALIZE;
  default: begin error_code <= ERR_OPCODE; state <= ERROR; end
endcase
client2_req_valid = reduce_req_valid | head_fuse_req_valid | finalize_req_valid;
```

- [ ] **Step 4: 生成并运行完整混合剪枝案例**

Run:

```powershell
& $env:HEATVIT_PYTHON tools/generate_selector_vectors.py --suite full --case mixed --tokens 197 --seed 20260815 --output build/vectors/selector_mixed
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top tb_token_selector -PlusArgs '+VECTOR_DIR=build/vectors/selector_mixed +STALL_MASK=3'
```

Expected: `TEST_PASS tb_token_selector`；local/global、三层 score、softmax、stats、weights、fused、output 和 state 全部逐位一致，且至少两个 normal 被剪、至少一个保留。

- [ ] **Step 5: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'feat: integrate complete HeatViT token selector' -Paths verification/heatvit_ref/selector.py,tools/generate_selector_vectors.py,rtl/compute/heatvit_tensor_executor.sv,sim/tb/tb_token_selector.sv -TestCommand 'scripts/run_xsim.ps1 -Top tb_token_selector'
```

### Task 6: 建立 Selector 阶段回归与跨阶段 Package 测试

**Files:**
- Modify: `scripts/run_regression.ps1`
- Modify: `verification/tests/test_selector.py`
- Create: `docs/verification/token-state-contract.md`

**Interfaces:**
- Consumes: 所有 Selector 单元/集成测试。
- Produces: `run_regression.ps1 -Suite selector` 和动态 Token 状态契约。

- [ ] **Step 1: 写连续三个 Finalize 的失败测试**

第 1 次从无 Package 输入产生 Package；第 2、3 次把上一输出作为下一输入，并同时再剪普通 Token。每次输出只能有一个末尾 Package，Token 数必须非增，CLS 和 kept normal 的相对顺序保持。

- [ ] **Step 2: 补充状态契约文档和回归套件**

文档明确 N、C、normal count、Package index、无/有 Package 的状态转移表，以及所有 warning 的清除/锁存责任。`-Suite selector` 运行 Python、features、head fuse、finalize 六案例、full mixed 和连续三阶段测试，并先执行 transformer 回归。

```powershell
$SelectorTops = @('tb_selector_features','tb_head_fuse','tb_selector_finalize','tb_token_selector')
& powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_regression.ps1 -Suite transformer
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
foreach ($Top in $SelectorTops) {
  & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top $Top
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
```

- [ ] **Step 3: 执行 Selector 阶段回归**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_regression.ps1 -Suite selector`

Expected: 退出码 0；三次连续输出各自恰好零或一个 Package，第二次以后不得出现两个 Package。

- [ ] **Step 4: Commit/checkpoint**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/task_checkpoint.ps1 -Message 'test: gate dynamic token selector regression' -Paths scripts/run_regression.ps1,verification/tests/test_selector.py,docs/verification/token-state-contract.md -TestCommand 'scripts/run_regression.ps1 -Suite selector'
```

## 阶段 4 完成条件

- 阶段 1 至 3 回归继续通过。
- 全保留、全剪、混合剪、阈值相等、已有 Package、两种零分母全部通过。
- 完整 N=197 Selector 的全部中间 Tensor 和输出逐位匹配。
- 三次连续 Finalize 始终保持 CLS、稳定 normal 顺序和至多一个 Package。
- Head denominator 和 Package denominator 警告只在对应回退发生时置位。
