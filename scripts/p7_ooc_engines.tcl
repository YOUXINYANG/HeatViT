# P7-1 OOC synthesis: fast per-module LUT feedback for the rewritten
# vector/layout engines without a full-project run.
#
# Usage:
#   vivado -mode batch -source scripts/p7_ooc_engines.tcl
set root [file normalize [file dirname [file dirname [info script]]]]
set report_dir [file join $root build/reports]
file mkdir $report_dir

read_verilog -sv [glob -nocomplain [file join $root rtl include *.sv]]
read_verilog -sv [glob -nocomplain [file join $root rtl common *.sv]]
read_verilog -sv [glob -nocomplain [file join $root rtl memory *.sv]]
read_verilog -sv [glob -nocomplain [file join $root rtl selector *.sv]]
read_verilog -sv [glob -nocomplain [file join $root rtl compute *.sv]]

synth_design -top heatvit_vector_engine -mode out_of_context \
  -part xc7k325tfbg900-3 -flatten_hierarchy rebuilt
report_utilization -file [file join $report_dir p7_ooc_vector_util.txt]
report_utilization -hierarchical -file [file join $report_dir p7_ooc_vector_hier.txt]

synth_design -top heatvit_layout_engine -mode out_of_context \
  -part xc7k325tfbg900-3 -flatten_hierarchy rebuilt
report_utilization -file [file join $report_dir p7_ooc_layout_util.txt]
report_utilization -hierarchical -file [file join $report_dir p7_ooc_layout_hier.txt]

puts "P7_OOC_DONE"
