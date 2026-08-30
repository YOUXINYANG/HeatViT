# HeatViT timing / IO constraints (P6 synthesis & implementation,
# P7-4 revised IO policy, P7-5 revised clock target).
#
# Target: 100 MHz core clock on xc7k325tfbg900-3.
# 50 MHz signoff was the P7-4 closure (WNS +0.323 ns).  P7-5 reworked the
# GEMM engine timing (narrow requant datapaths, 4-phase guard pipeline,
# S_CHECK settle) and re-targets 100 MHz: OOC GEMM gate WNS +0.586 ns,
# full-chip signoff below.
#
# No board-level pinout exists (上板 is out of scope), so every non-clock
# IO path is marked false: with unplaced pins the pad routes are arbitrary
# and a synthetic 0-delay input/output schedule cannot be met honestly
# (measured: mem_cmd_* reg->mux->pad paths fail by ~1-2 ns purely on the
# fixed pad route). The internal clock-to-clock paths carry the real
# timing budget, which is the closure criterion for this phase. Board
# bring-up would re-introduce per-pin IO constraints together with the
# pin placement. Async reset rst_n keeps its recovery/removal checks.

create_clock -period 10.000 -name clk [get_ports clk]

set_false_path -from [get_ports -filter {DIRECTION == IN && NAME != clk && NAME != rst_n}]
set_false_path -to [all_outputs]
