# HeatViT RTL 框架图（docs/diagrams/）

Mermaid 源码草稿，4 张分层图：模块级联 + 重要输入输出信号（非全部信号）。

渲染方式（任选）：

- GitHub：推送到仓库后 Markdown/网页直接渲染
- [mermaid.live](https://mermaid.live)：粘贴源码即时预览、可导出 PNG/SVG
- VSCode：安装 Markdown Preview Mermaid Support 插件

| 文件 | 内容 | 关键信号面 |
| --- | --- | --- |
| `heatvit_framework_01_top.mmd` | 顶层：top / scheduler / ROM / executor + 外部接口 | start/busy/done/error · 描述符 valid/ready · state_update · mem 三通道 |
| `heatvit_framework_02_executor.mmd` | 执行器内部：opcode 分发 · 7:1 存储 mux · 2:1 外存 mux · 除法仲裁 | child_sel · req_* · div client 0/1/2 |
| `heatvit_framework_03_gemm.mmd` | GEMM 引擎内部 | tile fill/read · bank_accum · GELU/PLAN 后处理 |
| `heatvit_framework_04_selector.mmd` | Selector 链（时间分时经 scratch 级联） | state_update 回传 |

> **注意**：图 4 中模块间**没有直接连线**——每个模块对应一条描述符，经
> `heatvit_mem_master` 读写 scratch 区分时级联；图中的箭头表示数据依赖顺序。

对应权威文档：`docs/heatvit.md` 第七部分 §1.1（纯模块层次图，无信号）。

## draw.io 版（可编辑精修）

与上面 4 张一一对应的 `.drawio` 源文件（按层着色：协议层橙 / 调度绿 / 执行器蓝 /
存储黄 / 守卫红 / 仲裁紫；图 2 的 8 子引擎为可整体拖动的分组容器）：

| draw.io 文件 | 对应 mmd |
| --- | --- |
| `heatvit_framework_01_top.drawio` | `heatvit_framework_01_top.mmd` |
| `heatvit_framework_02_executor.drawio` | `heatvit_framework_02_executor.mmd` |
| `heatvit_framework_03_gemm.drawio` | `heatvit_framework_03_gemm.mmd` |
| `heatvit_framework_04_selector.drawio` | `heatvit_framework_04_selector.mmd` |

用法：[app.diagrams.net](https://app.diagrams.net) → File → Open from → Device 打开
`.drawio`；精修后用 File → Export as → PNG/SVG 导出。改完源文件后若要同步回 mmd，
需人工比对（两版并存，以 draw.io 版为准）。
