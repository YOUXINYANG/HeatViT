# HeatViT timing / IO constraints (P6 synthesis & implementation).
#
# Target: 100 MHz core clock on xc7k325tfbg900-3.
#
# No board-level pinout exists yet (上板 is out of scope), so every
# non-clock port gets a zero-delay IO constraint: the internal
# clock-to-clock paths carry the real timing budget, while IO paths are
# kept honest without pretending a board schedule exists. Async reset
# rst_n is intentionally left for recovery/removal checks.
#
# Note: XDC does not support design-query commands like
# remove_from_collection, so the clock port is excluded from the input
# delay set with a get_ports -filter expression instead.

create_clock -period 10.000 -name clk [get_ports clk]

set_input_delay  -clock [get_clocks clk] 0.000 \
  [get_ports -filter {DIRECTION == IN && NAME != clk}]
set_output_delay -clock [get_clocks clk] 0.000 [all_outputs]
