# P6: post-synthesis opt_design pass for a realistic LUT count.
#
# The raw synthesis LUT count is pessimistic (large mux trees collapse
# during optimization); opt_design gives the number physical
# implementation would start from. Writes
#   build/reports/p6_opt_util.txt
# and exits 0 with an OPT_OK marker.
#
# Usage:
#   vivado -mode batch -source scripts/run_opt_report.tcl -tclargs <project.xpr>

if { $argc < 1 } {
  puts "usage: vivado -mode batch -source scripts/run_opt_report.tcl -tclargs <project.xpr>"
  exit 2
}
set project_path [lindex $argv 0]
set repo_root [file normalize [file dirname [file dirname [info script]]]]

open_project $project_path
open_run synth_1

opt_design

set report_dir [file join $repo_root build/reports]
file mkdir $report_dir
report_utilization -file [file join $report_dir p6_opt_util.txt]

close_project
puts "OPT_OK"
exit 0
