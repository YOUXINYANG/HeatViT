# P6 synthesis: run synth_1 in-project and dump utilization reports.
#
# Ensures the XDC constraint set is attached, launches synthesis with the
# given job count, waits, and on success writes
#   build/reports/p6_synth_util.txt        (flat utilization)
#   build/reports/p6_synth_util_hier.txt   (per-hierarchy utilization)
# It also prints a black-box scan (must be empty) and exits 0 with a
# SYNTH_OK marker (or SYNTH_FAILED + exit 1).
#
# Usage:
#   vivado -mode batch -source scripts/run_synthesis.tcl -tclargs <project.xpr> [jobs]

if { $argc < 1 } {
  puts "usage: vivado -mode batch -source scripts/run_synthesis.tcl -tclargs <project.xpr> \[jobs\]"
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

# Ensure the constraint set exists and carries the XDC.
if { [get_filesets -quiet constrs_1] eq "" } {
  create_fileset -constrset constrs_1
}
set xdc [file join $repo_root xdc/heatvit.xdc]
if { ![file exists $xdc] } {
  puts "ERROR: missing constraint file $xdc"
  close_project
  exit 1
}
if { [llength [get_files -quiet $xdc]] == 0 } {
  add_files -fileset constrs_1 -norecurse $xdc
}
update_compile_order -fileset sources_1
update_compile_order -fileset constrs_1

# Stage the descriptor ROM .mem into the run dir before elaboration
# ($readmemh resolves relative to the run cwd, see p6_pre_synth.tcl).
set pre_tcl [file join $repo_root scripts/p6_pre_synth.tcl]
if { ![file exists $pre_tcl] } {
  puts "ERROR: missing pre-synth hook $pre_tcl"
  close_project
  exit 1
}
set_property STEPS.SYNTH_DESIGN.TCL.PRE $pre_tcl [get_runs synth_1]

# Launch synthesis (reset only a half-finished previous run).
set run_status [get_property STATUS [get_runs synth_1]]
if { $run_status ne "Not started" && $run_status ne "synth_design Complete!" } {
  reset_run synth_1
}
launch_runs synth_1 -jobs $jobs
wait_on_run synth_1 -timeout 21600

set status [get_property STATUS [get_runs synth_1]]
if { $status ne "synth_design Complete!" } {
  puts "SYNTH_FAILED status=$status"
  close_project
  exit 1
}

open_run synth_1
set report_dir [file join $repo_root build/reports]
file mkdir $report_dir
report_utilization -file [file join $report_dir p6_synth_util.txt]
report_utilization -hierarchical -file [file join $report_dir p6_synth_util_hier.txt]

# Danger scan: black boxes must not exist (no external IP, no unresolved
# modules). Latch inference is caught by the wrapper's log grep instead.
set blackboxes [get_cells -hierarchical -quiet -filter {IS_BLACKBOX == 1}]
puts "SYNTH_BLACKBOX_COUNT [llength $blackboxes]"
if { [llength $blackboxes] > 0 } {
  foreach b $blackboxes { puts "SYNTH_BLACKBOX $b" }
}

close_project
puts "SYNTH_OK status=$status"
exit 0
