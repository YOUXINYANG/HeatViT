// Combinational 8-byte-aligned burst range guard.
//
// Checks, in order of the approved contract:
//   * cmd_len is nonzero,
//   * cmd_addr and region_base are 8-byte aligned,
//   * the 33-bit extended last-byte address does not overflow bit 32,
//   * the whole burst lies inside [region_base, region_base + region_bytes).
module heatvit_addr_guard
  import heatvit_pkg::*;
(
  input  logic [31:0] region_base,
  input  logic [31:0] region_bytes,
  input  logic [31:0] cmd_addr,
  input  logic [15:0] cmd_len,
  output logic        addr_ok,
  output logic [7:0]  addr_error_code
);

  logic [32:0] first_ext;
  logic [32:0] size_ext;
  logic [32:0] last_ext;
  logic [32:0] region_end;

  assign first_ext  = {1'b0, cmd_addr};
  assign size_ext   = {14'd0, cmd_len, 3'b000};
  assign last_ext   = first_ext + size_ext - 33'd1;
  assign region_end = {1'b0, region_base} + {1'b0, region_bytes};

  assign addr_ok =
      (cmd_len != 16'd0) &&
      (cmd_addr[2:0] == 3'd0) &&
      (region_base[2:0] == 3'd0) &&
      !last_ext[32] &&
      (first_ext >= {1'b0, region_base}) &&
      (last_ext < region_end);

  assign addr_error_code = addr_ok ? ERR_NONE : ERR_ADDRESS;

endmodule
