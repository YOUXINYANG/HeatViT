// Single-cycle synchronous descriptor ROM (Phase 5 Task 2).
//
// Holds the 198 compiled 320-bit descriptors loaded from the generated
// .mem file; the packed struct for ``addr`` appears combinationally so the
// scheduler's ROM_REQ -> ROM_WAIT pipeline sees the requested descriptor in
// the wait state.
module heatvit_descriptor_rom
  import heatvit_pkg::*;
#(
  parameter string DESC_MEM_FILE = "rtl/generated/heatvit_descriptors.mem",
  parameter int    DEPTH         = 198
)(
  input  logic          clk,
  input  logic          rst_n,
  input  logic [15:0]   addr,
  output heatvit_desc_t desc
);

  logic [319:0] mem [0:DEPTH-1];

  initial begin
    $readmemh(DESC_MEM_FILE, mem);
  end

  assign desc = heatvit_desc_t'(mem[addr]);

endmodule
