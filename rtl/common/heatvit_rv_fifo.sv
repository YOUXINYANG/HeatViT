// Small decoupling ready/valid FIFO. First word falls through. DEPTH must be
// a power of two; write/read pointers and the derived count are tracked with
// one wrap bit each.
module heatvit_rv_fifo #(
  parameter int WIDTH = 64,
  parameter int DEPTH = 4,
  parameter int PTR_W = $clog2(DEPTH)
) (
  input  logic                 clk,
  input  logic                 rst_n,
  input  logic                 s_valid,
  output logic                 s_ready,
  input  logic [WIDTH-1:0]     s_data,
  output logic                 m_valid,
  input  logic                 m_ready,
  output logic [WIDTH-1:0]     m_data
);

  logic [WIDTH-1:0] mem [DEPTH];
  logic [PTR_W:0]   wr_ptr;
  logic [PTR_W:0]   rd_ptr;
  logic [PTR_W:0]   count;

  localparam int DEPTH_FULL = DEPTH;

  integer i;
  initial begin
    if ((DEPTH & (DEPTH - 1)) != 0)
      $fatal(1, "heatvit_rv_fifo: DEPTH must be a power of two");
    for (i = 0; i < DEPTH; i++) mem[i] = '0;
  end

  assign count    = wr_ptr - rd_ptr;
  assign s_ready  = (count != DEPTH_FULL);
  assign m_valid  = (count != '0);
  assign m_data   = mem[rd_ptr[PTR_W-1:0]];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wr_ptr <= '0;
      rd_ptr <= '0;
    end else begin
      if (s_valid && s_ready) begin
        mem[wr_ptr[PTR_W-1:0]] <= s_data;
        wr_ptr <= wr_ptr + 1'b1;
      end
      if (m_valid && m_ready) begin
        rd_ptr <= rd_ptr + 1'b1;
      end
    end
  end

endmodule
