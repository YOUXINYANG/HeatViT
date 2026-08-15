# HeatViT-T 纯 FPGA SystemVerilog 推理设计规格

## 1. 目标

在现有 Vivado 工程中实现可综合的 SystemVerilog RTL，复现 HeatViT-T 的完整单图推理数据通路。设计目标器件保持为工程实际配置的 `xc7k325tfbg900-3`，但本阶段只以 Vivado XSim 仿真作为验收手段。

本规格覆盖：

- 8-bit 输入图像到 1000 类 Logit 的完整推理。
- Patch Embedding、位置编码和 CLS Token。
- 12 个 DeiT-T Transformer Block。
- Block 4、7、10 前的三个 HeatViT Token Selector。
- 动态 Token 压缩和单个 Package Token。
- 完全位于 FPGA RTL 中的 LayerNorm。
- 8-bit 权重/激活、定点非线性近似和统一 GEMM 引擎。
- 整数黄金模型、模块级仿真和完整尺寸端到端逐位回归。

## 2. 非目标

- 不实现模型训练、微调、蒸馏或量化感知训练。
- 不复现 ImageNet Top-1 准确率，也不声称随机测试权重具有分类意义。
- 不集成板级 DDR、MIG、PCIe、AXI、摄像头、显示、串口或主机软件。
- 不执行上板、功耗测试、时序收敛或实测吞吐验证。
- 不支持 HeatViT-S、HeatViT-B 或 LV-ViT 变体。
- 不实现 JPEG/PNG 解码、图像缩放和浮点均值方差预处理。

## 3. 依据和明确偏差

主要依据：

- 本项目中的 `HeatViT：Hardware-Efficient Adaptive Token Pruning for Vision Transformers.pdf`。
- <https://arxiv.org/abs/2211.08110>
- PLAN Sigmoid 分段式：<https://www.mdpi.com/1424-8220/20/11/3168>
- XC7K325T 资源表：<https://docs.amd.com/api/khub/documents/2LByHkO~nSZXcei2D55fTg/content>

论文没有公开完整 RTL、逐层 Q 格式、Token Selector 权重、Head 权重分支隐藏维度或端到端检查点。本设计据此采用以下明确约定：

- 论文把 LayerNorm 放在 ZCU102 ARM 上；本设计使用定点 RTL LayerNorm。
- 训练期 Gumbel-Softmax 不进入推理 RTL；推理使用确定性 Softmax 和 `0.5` 阈值。
- Selector Head 权重分支默认采用 `3 -> 3 -> 3`，尺寸仍由描述符表达。
- Token Package 采用论文公式 (10) 的加权平均；分母为零时采用算术平均回退。
- Attention Softmax 使用论文的 `delta2=0.5`；Selector 二分类 Softmax 使用 `delta2=1.0`，从而保留 `0.5` 判决含义。
- 使用确定性合成权重和输入完成数值验证，不使用训练 checkpoint。
- 使用通用行为级外部存储接口，不复现 ZCU102 的 CPU/DDR 子系统。

## 4. 固定模型配置

| 项目 | 值 |
| --- | --- |
| Batch | 1 |
| 输入 | 预处理并量化后的 `224 x 224 x 3` signed int8 |
| Patch | `16 x 16` |
| Patch 数 | 196 |
| 初始 Token 数 | 197（196 Patch + 1 CLS） |
| Embedding 维度 | 192 |
| Attention Head | 3 |
| 每个 Head 维度 | 64 |
| Transformer Block | 12 |
| FFN 隐藏维度 | 768 |
| 分类数 | 1000 |
| Selector 位置 | Block 4、7、10 之前 |

不实现 distillation token 或 distillation head。

### 4.1 数据与权重布局

- 输入图像采用 NHWC 行优先布局；同一像素的 R、G、B 三个 signed int8 按地址递增排列。
- Patch Token 按图像从上到下、从左到右的光栅顺序编号。每个 Patch 内按行、列、通道顺序展平，形成长度 768 的向量。
- Token 序列索引 0 固定为 CLS；位置编码索引 1 至 196 对应上述 Patch 光栅顺序。
- GEMM 统一解释为 `A[M][K] * B[K][N]`，A、B 和结果均为行优先布局；Bias 按 N 维连续存放。
- Q/K/V 权重存成一个 `[192][576]` 行优先矩阵，输出列顺序固定为 Q、K、V，各占连续 192 列。
- Patch Embedding 权重由卷积形式转换为 `[768][192]`；K 维顺序必须与 Patch 展平顺序一致。
- CLS、位置编码、Gamma、Beta、普通权重和 Bias 各自在生成器给出的 8-byte 对齐偏移处存放。
- 64-bit 存储数据采用小端字节序：最低地址字节位于 `mem_*_data[7:0]`。最终 Logit 以 class 0 至 class 999 顺序存放，每项为小端 signed int32。

## 5. 模型数据流

```text
224 x 224 x 3 int8 image
        |
Patch partition: 196 x 768
        |
Patch embedding GEMM: 768 -> 192
        |
prepend CLS + add position embedding
        |
Transformer Block 1..3
        |
Token Selector 1
        |
Transformer Block 4..6
        |
Token Selector 2
        |
Transformer Block 7..9
        |
Token Selector 3
        |
Transformer Block 10..12
        |
Final LayerNorm
        |
CLS classification GEMM: 192 -> 1000
        |
1000 signed int32 logits + scale exponent
```

每个 Transformer Block 使用 Pre-LN：

```text
Y = X + MSA(LayerNorm(X))
Z = Y + FFN(LayerNorm(Y))
```

MSA 顺序为 Q/K/V 投影、三个 Head 的 `Q*K^T`、缩放、Softmax、`Attention*V`、Head 拼接和输出投影。`sqrt(64)=8`，Attention 缩放可用精确的三位右移结合尺度指数完成。

## 6. 顶层架构

`heatvit_top` 包含：

- 顶层控制和状态寄存器。
- 操作描述符 ROM 与动态调度器。
- 地址生成和通用外部存储主接口。
- 输入、权重和输出 Tile Ping-Pong Buffer。
- 三 Bank 统一 GEMM 引擎。
- 重定标、残差、非线性、LayerNorm、倒数和整数平方根单元。
- Token Classifier、Token Compactor 和 Token Packager。
- 当前 Token 数寄存器及 Activation Ping-Pong 基地址。

大张量保存在抽象外部存储中，片上缓冲仅保存当前 Tile 和局部归约状态。Q、K、V、Attention Score、FFN 中间激活和残差源均使用描述符指定的 Scratch 区域。

## 7. 顶层接口

控制接口：

- `clk`、在 `posedge clk` 采样的同步低有效复位 `rst_n`。
- 单周期 `start`。
- 32-bit byte address：`input_base`、`weight_base`、`scratch_base`、`output_base`。
- 32-bit 区域容量：`input_bytes`、`weight_bytes`、`scratch_bytes`、`output_bytes`。
- `busy`、单周期成功脉冲 `done`。
- `error_valid`、`error_code[7:0]`。
- 锁存到下一次复位或启动的 `warning_flags[7:0]`。
- 最终 Logit 的 signed `output_scale_exp[5:0]`。

`start` 仅允许在 `busy=0` 时拉高。合法启动沿锁存全部基地址和容量、清除旧错误与警告并置 `busy`；`busy=1` 时再次拉高 `start` 属于致命协议错误。`done` 只在成功完成时拉高一个周期。

外部存储接口使用 64-bit 数据宽度：

- 命令通道：`mem_cmd_valid/ready`、`mem_cmd_write`、`mem_cmd_addr[31:0]`、`mem_cmd_len[15:0]`。
- 写通道：`mem_w_valid/ready`、`mem_w_data[63:0]`、`mem_w_strb[7:0]`、`mem_w_last`。
- 读通道：`mem_r_valid/ready`、`mem_r_data[63:0]`、`mem_r_last`。

`mem_cmd_len` 表示 64-bit beat 数且不得为零。命令地址和四个区域基地址必须 8-byte 对齐；向量生成器把每个 Tensor 占用空间向上填充到 8-byte 边界。最后一个写 Beat 只通过 `mem_w_strb` 使能有效字节，读出的填充字节不得参与计算。地址生成器在发出命令前，用锁存的区域容量检查完整 Burst 的首尾地址和 32-bit 加法溢出。Testbench 行为存储模型负责 `.mem` 初始化、命令响应、独立边界复查和可配置 backpressure。

## 8. 描述符调度

描述符为固定 320-bit SystemVerilog packed struct，字段如下；保留位必须写零，非零时视为非法描述符：

| 字段 | 位宽 | 含义 |
| --- | ---: | --- |
| `opcode` | 8 | 算子枚举 |
| `flags` | 24 | 转置、Bias、残差、动态 M、地址区域、写回和 Post-op 控制 |
| `m/n/k` | 各 16 | 静态矩阵维度；动态 M 置位时 `m` 由当前 Token 数覆盖 |
| `heads` | 4 | Head 数，普通 GEMM 为 0，Attention/Selector 为 3 |
| `src0/src1/bias/aux/dst_offset` | 各 32 | 相对于 flags 所选区域基地址的 byte offset |
| `src0/src1/aux/dst_scale_exp` | 各 6 signed | Tensor 尺度；Bias 尺度固定由 GEMM 两输入尺度之和导出 |
| `next_index` | 16 | 成功后的下一描述符索引 |
| `param0/param1` | 各 16 | Opcode 专用的维度、阈值或常量表索引 |
| `reserved` | 4 | 固定为零，使总宽度为 320 bit |

`flags` 的固定编码为：bit 0 右矩阵转置，bit 1 启用 Bias，bit 2 启用 Aux/残差，bit 3 使用动态 M，bit 4 完成后交换 Activation 区，bit 5 Head 模式，bit 6 Head 拼接，bit 7 输出 int32，bits 10:8 为 Post-op 枚举，bit 11 表示 src0 来自 Input 区，bit 12 表示 src1 来自 Scratch 区（否则为 Weight 区），bit 13 表示 Bias 来自 Scratch 区（否则为 Weight 区），bit 14 表示 Aux 来自 Weight 区（否则为 Scratch 区），bit 15 表示 dst 为 Output 区（否则为 Scratch 区），bits 16、17 分别启用 Token 和 Channel 尾块掩码，bit 18 表示 src0 按 unsigned 解释（仅用于 UQ0.8 Attention 与 signed int8 V 相乘），bits 19、20 分别把 N、K 覆盖为当前 Token 数，bits 23:21 保留为零。Post-op 枚举只允许 none、requant、GELU、Attention Softmax、Selector Softmax、PLAN 和 LayerNorm。

固定模型顺序存放在描述符 ROM 中。flag 3 置位时，`param0[1:0]` 固定选择动态 M：`2'b00` 表示当前 Token 数 N，`2'b01` 表示候选数 C=N-1，`2'b10/11` 非法；bits 19/20 置位时分别把 descriptor N/K 覆盖为当前 Token 数。Reduction opcode 另用 `param0[3:2]` 选择候选轴或 64 通道轴。普通 MSA/FFN 使用动态 M；QKᵀ 使用动态 M/N；Attention×V 使用动态 M/K；Selector MLP/Softmax 使用动态候选 M。Selector 完成后更新 Token 数；每个算子完成后按描述符交换 Activation Ping-Pong 基地址。

非法 Opcode、零维度、超出模型上限的维度或越界地址必须在发出存储命令前被拒绝。

## 9. 定点数值契约

### 9.1 普通 Tensor

- 权重和普通激活均为 signed int8，范围 `[-128, 127]`。
- 每个 Tensor 携带 6-bit signed `scale_exp`，范围 `[-32, 31]`。
- 数值解释为 `real = integer * 2^scale_exp`。
- Bias 为 signed int32，尺度等于对应 GEMM 累加尺度。
- 常规 GEMM 为 signed `int8 * int8 -> int32 accumulator`；唯一例外是 Attention×V，其 UQ0.8 左操作数按 unsigned、V 按 signed int8 解释，结果仍累加到 signed int32。
- 最终 1000 个 Logit 保留为 signed int32，并输出共同的 `scale_exp`。

### 9.2 重定标

若累加尺度为 `s_acc`，目标尺度为 `s_out`，则对整数累加值执行 `2^(s_acc-s_out)` 的移位。

- 右移采用最近舍入，正负中点均远离零。
- 一般整数除法也采用最近舍入、中点远离零：底层无符号除法器返回向下取整商和余数，调用方在 `2*remainder >= denominator` 时将商的绝对值加一，再恢复符号。
- 普通 Tensor 的尺度对齐左移在 signed 128-bit 扩展位宽中完成，足以覆盖 48-bit 输入与完整 `scale_exp` 差值；之后再按目标位宽饱和。
- 窄化统一饱和，禁止二进制回绕。
- 禁止依赖 SystemVerilog 隐式 signedness、隐式位宽扩展或隐式截断。

残差相加前先对齐尺度，在 int32 中求和后再重定标为 int8。

### 9.3 概率和中间值

- Attention Softmax 输出为 8-bit unsigned UQ0.8；由于 `delta2=0.5`，最大合法值不超过 0.5。
- Selector Score 和 Head Weight 使用 17-bit unsigned Q0.16，`1.0` 编码为 `65536`。
- 非线性输入和主要中间值使用 signed 24-bit Q8.16。
- 平方、方差、倒数和 Package 累加允许扩展到 32 或 48 bit。
- Q0.16 Selector Score、Head Weight 和 fused score 在外部 Scratch 中各占一个 little-endian unsigned 32-bit word，bits 16:0 有效且 bits 31:17 必须为零；Q8.16 Scratch 值各占一个 little-endian signed 32-bit word，bits 31:24 必须是 bit 23 的符号扩展。

## 10. 统一 GEMM 引擎

默认参数：

- `TH=3`
- `TI=8`
- `TO=8`
- 三个 Bank，每个 Bank 为 `8 x 8` MAC，共 192 条 int8 乘法通道。

Attention 模式下一 Bank 对应一个 Head。普通 GEMM 模式下三个 Bank 处理不同输出 Tile。K 维度跨周期归约。动态 Token 数或矩阵边界不是 8 的倍数时，尾块 Valid Mask 必须阻止无效输入参与累加或写回。

引擎支持：

- 普通和转置右矩阵。
- Bias 加法。
- Head 分组结果保留。
- Head 拼接或普通 Tile 写回。
- 可选重定标和激活派发。

Patch Embedding、Q/K/V、Attention、Projection、FFN、Selector MLP 和分类头全部复用该引擎。

## 11. 非线性和归一化

### 11.1 GELU

使用 HeatViT 公式 (11)、(12)，`delta1=0.5`。Q8.16 常量固定为：

| 常量 | Q8.16 整数 |
| --- | ---: |
| `a=-0.2888` | -18927 |
| `b=-1.769` | -115933 |
| `delta1=0.5` | 32768 |
| `1/sqrt(2)` | 46341 |

所有乘法后按第 9 节规则舍入和饱和。

对 Q8.16 输入 `x`，整数执行顺序固定为：

```text
u       = round_q16(x * inv_sqrt2)
c       = min(abs(u), -b)
t       = c + b
poly    = round_q16(a * round_q16(t*t)) + 1.0
l_erf   = sign(u) * round_q16(delta1 * poly)
gelu    = round((x * (1.0 + l_erf)) / (1<<17))
```

其中 `sign(0)=0`、`-b=115933`；最后结果饱和到 signed 24-bit Q8.16。最后一式的分母 `1<<17` 同时包含 Q16 乘法缩放与 GELU 的 `1/2`。

### 11.2 Softmax 和指数

每行先减去最大值。使用 HeatViT 公式 (13)、(14) 的范围缩减：

```text
x_tilde = x - row_max
z = floor(-x_tilde / ln(2))
p = x_tilde + z*ln(2)
exp(x_tilde) = exp_poly(p) >> z
exp_poly(p) = 0.3585*(p + 1.353)^2 + 0.344
```

Q8.16 常量：

| 常量 | Q8.16 整数 |
| --- | ---: |
| `ln(2)` | 45426 |
| `0.3585` | 23495 |
| `1.353` | 88670 |
| `0.344` | 22544 |
| Attention `delta2=0.5` | 32768 |
| Selector `delta2=1.0` | 65536 |

令 Q8.16 指数整数为 `E_i`、整数行和为 `S`。行和只计算一次 33-bit Q0.32 倒数，逐元素按下式计算：

```text
recip_q32  = round((1<<32) / S)
ratio_q16  = round((E_i * recip_q32) / (1<<16))
scaled_q16 = round((ratio_q16 * delta2_q16) / (1<<16))
```

Selector 直接把 `scaled_q16` 饱和为 17-bit Q0.16；Attention 再将其最近舍入右移 8 bit 并饱和为 UQ0.8。每个 `round` 均使用第 9 节规则。减最大值保证至少一项为正，因此正常 Softmax 行和不能为零；若内部错误导致行和为零，必须报致命错误。

### 11.3 PLAN Sigmoid

输入为 Q8.16，先取绝对值：

```text
abs(x) >= 5.0      : y = 1.0
2.375 <= abs(x)<5  : y = abs(x)/32 + 27/32
1.0 <= abs(x)<2.375: y = abs(x)/8  + 5/8
0 <= abs(x)<1.0    : y = abs(x)/4  + 1/2
x < 0              : y = 1.0 - y
```

阈值 `1.0`、`2.375`、`5.0` 的 Q8.16 编码分别为 `65536`、`155648`、`327680`。输出为 17-bit Q0.16。

### 11.4 LayerNorm

LayerNorm 输入 scale exponent 的合法范围为 `[-32,0]`；超出范围在发出存储命令前报致命错误 2。对每个 Token 的 192 个通道执行两遍处理，所有带 `_q32` 后缀的整数均表示 32 个小数位：

1. `x_q32 = x_int << (input_scale_exp+32)`；缓存 Token，并累加 signed `sum_x_q32`。
2. `square_q32 = round(x_int^2 * 2^(2*input_scale_exp+32))`，累加 unsigned `sum_square_q32`。
3. `mean_q32=round(sum_x_q32/192)`，`e2_q32=round(sum_square_q32/192)`，`mean_square_q32=round(mean_q32^2/(1<<32))`。
4. `variance_q32=e2_q32-mean_square_q32`；若为负则钳位为零并置警告位。
5. 加入 Q16.32 的 `epsilon=10^-6`，整数编码为 `4295`；`std_q16=floor_sqrt(variance_q32+4295)`。
6. 只计算一次 `inv_std_q32=round((1<<48)/std_q16)`，随后 `norm_q16=round(((x_q32-mean_q32)*inv_std_q32)/(1<<48))` 并饱和到 signed 24-bit Q8.16。
7. 第二遍将 Q8.16 `norm_q16` 与 int8 Gamma 相乘、对齐并加 int8 Beta，最后重定标到输出 int8。

共享恢复除法器的 numerator/denominator/quotient 均为 64-bit unsigned；LayerNorm、Softmax、Selector Reduction/Head Fuse 和 Package Token 通过仲裁器复用该单元。整数平方根返回向下取整根和余数，不再额外舍入。

## 12. Token Selector

### 12.1 Token 分类

CLS Token 永久旁路。其他所有输入 Token 都是分类候选；这包括上一 Selector 产生的 Package Token。对候选 Token：

1. 将 192 维拆成三个 64 维 Head 子向量。
2. 每个 Head 独立执行 `Linear(64,32) + GELU`，产生局部特征。
3. 沿 Token 维求局部特征均值，产生 32 维全局特征并广播。
4. 拼接局部和全局特征为 64 维。
5. 每个 Head 独立执行 `Linear(64,32) -> GELU -> Linear(32,16) -> GELU -> Linear(16,2) -> Softmax`；二分类输出列 0 固定为 Drop，列 1 固定为 Keep。
6. 对每个 Token 的三个 64 维 Head 子向量分别求均值，形成三维统计量。
7. Head 权重分支执行 `Linear(3,3) -> GELU -> Linear(3,3) -> PLAN Sigmoid`。
8. 使用 Head 权重对三个二分类 Score 做加权平均。

Head 权重和为零时改用三个 Head 的等权平均，并置 `WARN_HEAD_DEN_ZERO`。

`keep_score >= 0.5` 时保留，低于 `0.5` 时剪除。恰好等于阈值时保留。

### 12.2 Token Packaging

- 保留的普通 Token 维持原相对顺序。
- 上一阶段的 Package Token 必须重新并入本阶段 Package 累加器，不允许 Package Token 数量增长。
- 被剪普通 Token 和输入 Package Token 使用各自 Keep Score 作为权重。
- Package Token 为加权特征和除以权重和。
- 权重和为零时改用参与 Packaging 的 Token 算术平均，并置 `WARN_PACKAGE_DEN_ZERO`。
- 没有任何 Token 进入 Packaging 时不产生 Package Token。
- 输出顺序固定为 `CLS -> kept normal tokens -> optional single package token`。
- 即使所有普通 Token 被剪，输出仍包含 CLS 和一个 Package Token，因此合法 Token 数范围为 `[2,197]`；无剪枝时可为 197。

Selector 结束后更新 `current_token_count`，并将紧凑的稠密 Token 矩阵写入下一 Activation 区域。

## 13. 错误和警告

致命错误码：

| Code | 含义 |
| ---: | --- |
| 1 | 非法描述符 Opcode |
| 2 | 非法维度、Head、flag、保留位或算子专用 scale 配置 |
| 3 | 外部存储地址未对齐或越界 |
| 4 | Token 数超出 `[2,197]` |
| 5 | 外部存储握手/长度协议错误 |
| 6 | 正常 Softmax 行和为零 |
| 7 | `busy=1` 时收到新的 `start` |

致命错误发生时立即禁止新命令并置 `error_valid`，且不产生 `done`。若已有外存 Burst 完成命令握手，读 Burst 必须接收并丢弃至 `last`，写 Burst 必须用零 strobe 补齐剩余 Beat；`busy` 在该协议排空完成后清零。没有已接受 Burst 时 `busy` 在下一拍清零。

警告位：

| Bit | 含义 |
| ---: | --- |
| 0 | Head 权重分母为零，已使用等权平均 |
| 1 | Package 权重分母为零，已使用算术平均 |
| 2 | LayerNorm 负方差已钳位为零 |

警告不停止推理，并一直锁存到下一次复位或启动。

## 14. 黄金模型和测试向量

Python 黄金模型实现与 RTL 相同的整数位宽、signedness、尺度指数、舍入、饱和、近似系数、阈值和回退行为。量化以后不再用浮点运算生成预期结果。

生成器使用固定随机种子并输出：

- 输入图像、权重、Bias、位置编码、CLS 和量化参数 `.mem`。
- 描述符 ROM 初始化内容。
- 各算术单元和计算模块的输入/预期输出。
- Patch Embedding、每个 Block、每个 Selector、Final LayerNorm 和最终 Logit 的检查点。
- 预期 Token 数、Package 是否存在、错误码和警告位。

完整端到端向量的 Selector 参数必须让三个 Selector 都产生至少一个保留普通 Token和至少一个被剪普通 Token，以验证真实动态路径。

## 15. 仿真验证

主仿真器为 Vivado XSim，Testbench 使用自检式 SystemVerilog，不使用 UVM。

### 15.1 算术单元

覆盖：

- int8 最小/最大值、零和符号组合。
- 左右移、半 LSB 舍入、正负中点、饱和和溢出。
- 除法、平方根、GELU 分段边界、Softmax 极值、PLAN 各阈值和 LayerNorm 零方差。
- 固定种子的随机回归。

### 15.2 计算模块

覆盖：

- GEMM 普通/转置、Bias、Head 模式和普通模式。
- M/N/K 不是 8 的倍数时的尾块掩码。
- 可配置存储 backpressure。
- Residual、LayerNorm、MHSA、FFN 和 Patch Embedding。
- Selector 全保留、全剪、混合剪、零 Head 分母、零 Package 分母和输入 Package 合并。

### 15.3 完整 Block

使用完整 HeatViT-T 尺寸验证：`N<=197`、`D=192`、`FFN=768`、三个 64 维 Head。至少包含一个 `N=197` 的无剪枝 Block 和一个非 8 倍数 Token 数的 Block。

### 15.4 完整端到端回归

至少一组完整 `224 x 224 x 3` 输入经过 12 个 Block 和三个 Selector。以下结果必须逐位一致：

- Patch Embedding 输出。
- 每个 Transformer Block 输出。
- 每个 Selector 输出 Token、Token 数和 Package Token。
- Final LayerNorm 输出。
- 1000 个 signed int32 Logit 及其尺度指数。

Testbench 还必须断言：

- 复位释放后没有未知 `X/Z` 传播到有效数据或控制信号。
- Ready/Valid backpressure 时数据和控制保持稳定。
- 不发生 Buffer 或行为存储越界。
- Token 数始终在合法范围。
- Watchdog 限定的最大周期数内结束。

## 16. Vivado IP 策略

当前设计不需要用户手动生成任何 Vivado IP 核。

- 不使用 Floating-Point Operator。
- 不使用 Divider Generator。
- 不使用 Block Memory Generator。
- 不使用 AXI、MIG 或 Clocking Wizard。
- DSP48E1 由乘法和累加代码推断。
- BRAM 由同步数组模板推断。
- 外部存储仅由 Testbench 行为模型提供。

如果范围改为上板验证，MIG、时钟和板级主机接口需要单独设计，并在生成前向用户列出具体 IP 名称、版本、参数和连接方式。

## 17. 验收标准

设计完成的必要条件：

1. 所有设计文件均为可综合 SystemVerilog；仿真专用构造只存在于 Testbench 和行为存储模型。
2. Vivado XSim 能完成编译、展开和所有分层测试。
3. 所有算术、计算模块和完整 Block 测试通过逐位比较。
4. 至少一个完整 HeatViT-T 端到端向量通过所有检查点和最终 Logit 比较。
5. 三个 Selector 在端到端向量中均实际改变 Token 数。
6. 仿真无未解释断言、未知值、越界或超时。
7. 不要求任何手工生成的 Vivado IP。

本阶段不以综合资源报告、时序报告、FPS、功耗或 ImageNet 准确率作为验收条件。
