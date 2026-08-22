// On-chip tile storage for one GEMM engine: three A tiles, three B tiles and
// the current Bias slice. A/B use inferred SDP RAMs (synchronous 8-byte reads,
// byte-strobed writes); Bias is a small register array with a combinational
// read so writeback can index any column combinationally.
module heatvit_tile_buffer #(
  parameter int A_BYTES   = 6144,
  parameter int B_BYTES   = 6144,
  parameter int BIAS_COLS = 24
)(
  input  logic          clk,
  input  logic          rst_n,
  // Byte-granular fill port. fill_bank: 0-2 = A0..A2, 3-5 = B0..B2,
  // 6 = Bias, 7 = broadcast A (same beat to all three A tiles).
  input  logic [2:0]    fill_bank,
  input  logic          fill_valid,
  input  logic [12:0]   fill_addr,
  input  logic [7:0]    fill_data,
  // 8-byte synchronous reads, byte address must be 8-byte aligned.
  input  logic [12:0]   a_rd_addr [0:2],
  output logic [63:0]   a_rd_data [0:2],
  input  logic [12:0]   b_rd_addr [0:2],
  output logic [63:0]   b_rd_data [0:2],
  // Bias read side: full array exposed for parallel writeback composition.
  output heatvit_pkg::heatvit_s32_t bias_mem_out [0:BIAS_COLS-1]
);

  import heatvit_pkg::*;

  localparam int A_WORDS = A_BYTES / 8;
  localparam int B_WORDS = B_BYTES / 8;
  localparam int RAM_AW  = $clog2(A_WORDS);

  logic [2:0] a_we;
  logic [2:0] b_we;

  heatvit_s32_t bias_mem [0:BIAS_COLS-1];

  assign a_we[0] = rst_n && fill_valid && (fill_bank == 3'd0 || fill_bank == 3'd7);
  assign a_we[1] = rst_n && fill_valid && (fill_bank == 3'd1 || fill_bank == 3'd7);
  assign a_we[2] = rst_n && fill_valid && (fill_bank == 3'd2 || fill_bank == 3'd7);
  assign b_we[0] = rst_n && fill_valid && (fill_bank == 3'd3);
  assign b_we[1] = rst_n && fill_valid && (fill_bank == 3'd4);
  assign b_we[2] = rst_n && fill_valid && (fill_bank == 3'd5);

  genvar g;
  generate
    for (g = 0; g < 3; g++) begin : gen_a
      heatvit_sdp_ram #(.WIDTH(64), .DEPTH(A_WORDS), .AW(RAM_AW)) a_ram (
        .clk   (clk),
        .we    (a_we[g]),
        .waddr (fill_addr[RAM_AW+2:3]),
        .wdata ({8{fill_data}}),
        .wstrb (8'h01 << fill_addr[2:0]),
        .raddr (a_rd_addr[g][RAM_AW+2:3]),
        .rdata (a_rd_data[g])
      );
    end
    for (g = 0; g < 3; g++) begin : gen_b
      heatvit_sdp_ram #(.WIDTH(64), .DEPTH(B_WORDS), .AW(RAM_AW)) b_ram (
        .clk   (clk),
        .we    (b_we[g]),
        .waddr (fill_addr[RAM_AW+2:3]),
        .wdata ({8{fill_data}}),
        .wstrb (8'h01 << fill_addr[2:0]),
        .raddr (b_rd_addr[g][RAM_AW+2:3]),
        .rdata (b_rd_data[g])
      );
    end
  endgenerate

  integer bi;
  initial begin
    for (bi = 0; bi < BIAS_COLS; bi++) bias_mem[bi] = 32'sd0;
  end

  always_ff @(posedge clk) begin
    if (rst_n && fill_valid && fill_bank == 3'd6) begin
      int byte_idx;
      int col;
      int lane;
      byte_idx = fill_addr;
      col      = byte_idx / 4;
      lane     = byte_idx % 4;
      bias_mem[col][8*lane +: 8] <= fill_data;
    end
  end

  assign bias_mem_out = bias_mem;

endmodule
