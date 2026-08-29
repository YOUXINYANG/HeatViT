# HeatViT timing / IO constraints (P6 synthesis & implementation,
# P7-4 revised IO policy).
#
# Target: 50 MHz core clock on xc7k325tfbg900-3.
# 100 MHz attempt (P7-4): post-phys_opt WNS = -7.2 ns with the GEMM
# write-back staging-RAM cone dominating; routing added congestion
# (router deprioritized timing). Fell back to 50 MHz per plan.
#
# No board-level pinout exists (上板 is out of scope), so every non-clock
# IO path is marked false: with unplaced pins the pad routes are arbitrary
# and a synthetic 0-delay input/output schedule cannot be met honestly
# (measured: mem_cmd_* reg->mux->pad paths fail by ~1-2 ns purely on the
# fixed pad route). The internal clock-to-clock paths carry the real
# timing budget, which is the closure criterion for this phase. Board
# bring-up would re-introduce per-pin IO constraints together with the
# pin placement. Async reset rst_n keeps its recovery/removal checks.

create_clock -period 20.000 -name clk [get_ports clk]

set_false_path -from [get_ports -filter {DIRECTION == IN && NAME != clk && NAME != rst_n}]
set_false_path -to [all_outputs]
