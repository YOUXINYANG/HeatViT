# P7-5 fast timing gate for the GEMM engine at 100 MHz.
#
# Runs out-of-context synthesis through route_design so the result includes
# placement and routing delay rather than relying on a synthesis estimate.
# This is a diagnostic gate; full-chip timing closure still requires the
# normal project implementation flow.

set root [file normalize [file dirname [file dirname [info script]]]]
set report_dir [file join $root build reports]
file mkdir $report_dir

read_verilog -sv [glob -nocomplain [file join $root rtl include *.sv]]
read_verilog -sv [glob -nocomplain [file join $root rtl common *.sv]]
read_verilog -sv [glob -nocomplain [file join $root rtl memory *.sv]]
read_verilog -sv [glob -nocomplain [file join $root rtl compute *.sv]]

synth_design -top heatvit_gemm_engine -mode out_of_context \
  -part xc7k325tfbg900-3 -flatten_hierarchy rebuilt
create_clock -period 10.000 -name clk [get_ports clk]
# OOC timing needs a representative clock-buffer location to model clock
# insertion delay/skew. This does not constrain the full-chip BUFG choice.
set_property HD.CLK_SRC BUFGCTRL_X0Y0 [get_ports clk]

opt_design
place_design
phys_opt_design
route_design

report_utilization -file [file join $report_dir p7c_ooc_gemm_util.txt]
report_timing_summary -delay_type max -max_paths 20 \
  -file [file join $report_dir p7c_ooc_gemm_timing.txt]
report_timing -delay_type max -max_paths 50 -path_type full \
  -file [file join $report_dir p7c_ooc_gemm_paths.txt]
report_route_status -file [file join $report_dir p7c_ooc_gemm_route.txt]

# Vivado emits these auxiliary diagnostics in the launch directory.  The
# canonical copies of the useful results live under build/reports, so avoid
# leaving generated files in the source tree after the timing gate finishes.
foreach auxiliary_file {clockInfo.txt tight_setup_hold_pins.txt} {
  set auxiliary_path [file join [pwd] $auxiliary_file]
  if {[file exists $auxiliary_path]} {
    file delete -force $auxiliary_path
  }
}

set worst_paths [get_timing_paths -delay_type max -max_paths 1]
if {[llength $worst_paths] == 0} {
  puts "P7C_OOC_ERROR no internal timing paths found"
  exit 2
}
set worst_slack [get_property SLACK [lindex $worst_paths 0]]
set route_status [report_route_status -return_string]
puts "P7C_OOC_WNS $worst_slack"
if {[regexp {# of nets with routing errors[. ]*: *0} $route_status]} {
  puts "P7C_OOC_ROUTE_COMPLETE"
} else {
  puts "P7C_OOC_ROUTE_INCOMPLETE"
  exit 3
}
puts "P7C_OOC_DONE"
exit 0
