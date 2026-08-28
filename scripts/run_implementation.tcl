# P6 implementation: run impl_1 through route_design (no bitstream) and
# dump utilization / timing / power reports.
#
# On success writes
#   build/reports/p6_impl_util.txt          (flat post-route utilization)
#   build/reports/p6_impl_util_hier.txt     (per-hierarchy utilization)
#   build/reports/p6_timing_summary.txt     (max-delay timing summary)
#   build/reports/p6_timing_min.txt         (min-delay timing summary)
#   build/reports/p6_power.txt              (vectorless power estimate)
# and exits 0 with an IMPL_OK marker (or IMPL_FAILED + exit 1).
#
# Usage:
#   vivado -mode batch -source scripts/run_implementation.tcl -tclargs <project.xpr> [jobs]

if { $argc < 1 } {
  puts "usage: vivado -mode batch -source scripts/run_implementation.tcl -tclargs <project.xpr> \[jobs\]"
  exit 2
}
set project_path [lindex $argv 0]
set jobs 24
if { $argc >= 2 } { set jobs [lindex $argv 1] }
set repo_root [file normalize [file dirname [file dirname [info script]]]]

open_project $project_path

# Guard rails: part and top must stay unchanged.
set current_part [get_property PART [current_project]]
if { $current_part ne "xc7k325tfbg900-3" } {
  puts "ERROR: project PART changed to $current_part (expected xc7k325tfbg900-3)"
  close_project
  exit 1
}
set current_top [get_property top [get_filesets sources_1]]
if { $current_top ne "heatvit" } {
  puts "ERROR: sources_1 top changed to $current_top (expected heatvit)"
  close_project
  exit 1
}

# Launch implementation (reset only a half-finished previous run).
set run_status [get_property STATUS [get_runs impl_1]]
if { $run_status ne "Not started" && $run_status ne "route_design Complete!" } {
  reset_run impl_1
}
launch_runs impl_1 -to_step route_design -jobs $jobs
wait_on_run impl_1 -timeout 86400

set status [get_property STATUS [get_runs impl_1]]
if { $status ne "route_design Complete!" } {
  puts "IMPL_FAILED status=$status"
  close_project
  exit 1
}

open_run impl_1
set report_dir [file join $repo_root build/reports]
file mkdir $report_dir
report_utilization -file [file join $report_dir p6_impl_util.txt]
report_utilization -hierarchical -file [file join $report_dir p6_impl_util_hier.txt]
report_timing_summary -delay_type max -max_paths 20 -file [file join $report_dir p6_timing_summary.txt]
report_timing_summary -delay_type min -max_paths 10 -file [file join $report_dir p6_timing_min.txt]
report_power -file [file join $report_dir p6_power.txt]

close_project
puts "IMPL_OK status=$status"
exit 0
