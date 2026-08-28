# P7-2 OOC synthesis: per-module LUT feedback for the rewritten selector-side
# engines without a full-project run.
#
# Usage:
#   vivado -mode batch -source scripts/p7b_ooc_engines.tcl
set root [file normalize [file dirname [file dirname [info script]]]]
set report_dir [file join $root build/reports]
file mkdir $report_dir

read_verilog -sv [glob -nocomplain [file join $root rtl include *.sv]]
read_verilog -sv [glob -nocomplain [file join $root rtl common *.sv]]
read_verilog -sv [glob -nocomplain [file join $root rtl memory *.sv]]
read_verilog -sv [glob -nocomplain [file join $root rtl selector *.sv]]
read_verilog -sv [glob -nocomplain [file join $root rtl compute *.sv]]

foreach top {heatvit_reduce_mean heatvit_selector_softmax heatvit_head_fuse \
             heatvit_feature_concat heatvit_selector_finalize \
             heatvit_token_packager heatvit_token_compactor} {
  synth_design -top $top -mode out_of_context -part xc7k325tfbg900-3 \
    -flatten_hierarchy rebuilt
  report_utilization -file [file join $report_dir p7b_ooc_${top}_util.txt]
}

puts "P7B_OOC_DONE"
