// Simple dual-port RAM with synchronous read and per-byte write enable.
// A single always_ff keeps the template suitable for BRAM inference.
module heatvit_sdp_ram #(
  parameter int WIDTH = 64,
  parameter int DEPTH = 256,
  parameter int AW    = $clog2(DEPTH)
) (
  input  logic                 clk,
  input  logic                 we,
  input  logic [AW-1:0]        waddr,
  input  logic [WIDTH-1:0]     wdata,
  input  logic [WIDTH/8-1:0]   wstrb,
  input  logic [AW-1:0]        raddr,
  output logic [WIDTH-1:0]     rdata
);

  logic [WIDTH-1:0] mem [DEPTH];

  integer i;
  initial begin
    for (i = 0; i < DEPTH; i++) mem[i] = '0;
  end

  always_ff @(posedge clk) begin
    if (we) begin
      for (int b = 0; b < WIDTH / 8; b++) begin
        if (wstrb[b]) mem[waddr][8*b +: 8] <= wdata[8*b +: 8];
      end
    end
    rdata <= mem[raddr];
  end

endmodule
