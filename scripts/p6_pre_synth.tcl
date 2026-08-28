# P6 pre-synth hook (STEPS.SYNTH_DESIGN.TCL.PRE on synth_1).
#
# heatvit_descriptor_rom loads rtl/generated/heatvit_descriptors.mem via a
# relative $readmemh path, and Vivado synthesis resolves relative paths
# against the synthesis process cwd (= the run directory
# HeatViT.runs/synth_1) — where the file does not exist. This hook stages
# the generated ROM file into the run directory under the same relative
# path before elaboration, so the ROM is actually initialized in the
# synthesized netlist (instead of being silently ignored with
# CRITICAL WARNING [Synth 8-4445]).
#
# The source path is derived from the run's DIRECTORY property (repo root
# is two levels up), so no machine-specific path is hardcoded.

set run_dir [pwd]
if { [get_runs -quiet synth_1] ne "" } {
  set d [get_property DIRECTORY [get_runs synth_1]]
  if { $d ne "" } { set run_dir $d }
}
set repo_root [file normalize [file dirname [file dirname $run_dir]]]
set mem_src [file join $repo_root rtl/generated/heatvit_descriptors.mem]
if { ![file exists $mem_src] } {
  puts "P6_PRE_SYNTH_ERROR: missing $mem_src"
  exit 1
}
set mem_dst [file join $run_dir rtl/generated/heatvit_descriptors.mem]
file mkdir [file dirname $mem_dst]
file copy -force $mem_src $mem_dst
if { ![file exists $mem_dst] } {
  puts "P6_PRE_SYNTH_ERROR: staging failed for $mem_dst"
  exit 1
}
puts "P6_PRE_SYNTH_MEM_STAGED $mem_dst"
