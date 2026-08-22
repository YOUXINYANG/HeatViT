# Sync the existing HeatViT Vivado project with the full RTL source set.
#
# Adds every SystemVerilog source under rtl/ (include, common, memory,
# compute, selector, top), the generated descriptor ROM .mem, and the
# compatibility wrapper; sets the sources_1 top to `heatvit`, updates the
# compile order and saves. No IP catalog objects are created.
#
# Usage:
#   vivado -mode batch -source scripts/sync_vivado_project.tcl -tclargs HeatViT.xpr

if { $argc != 1 } {
  puts "usage: vivado -mode batch -source scripts/sync_vivado_project.tcl -tclargs <project.xpr>"
  exit 2
}
set project_path [lindex $argv 0]
set repo_root [file normalize [file dirname [file dirname [info script]]]]

open_project $project_path

set dirs [list rtl/include rtl/common rtl/memory rtl/compute rtl/selector rtl/top]
foreach dir $dirs {
  set abs_dir [file join $repo_root $dir]
  if { [file isdirectory $abs_dir] } {
    set files [glob -nocomplain -directory $abs_dir -type f -- *.sv]
    if { [llength $files] > 0 } {
      add_files -norecurse $files
    }
  }
}

# Descriptor ROM memory file (non-source data referenced by the ROM).
set rom_mem [file join $repo_root rtl/generated/heatvit_descriptors.mem]
if { [file exists $rom_mem] } {
  add_files -norecurse $rom_mem
}

# Compatibility wrapper keeps the project top name.
add_files -norecurse [file join $repo_root HeatViT.srcs/sources_1/new/heatvit.sv]

set_property top heatvit [current_fileset]
update_compile_order -fileset sources_1

# Guard rails: the project part must stay unchanged.
set current_part [get_property PART [current_project]]
if { $current_part ne "xc7k325tfbg900-3" } {
  puts "ERROR: project PART changed to $current_part (expected xc7k325tfbg900-3)"
  close_project
  exit 1
}

# The in-memory edits are persisted to the .xpr by Vivado on close.
close_project
puts "SYNC_OK part=$current_part top=heatvit"
exit 0
