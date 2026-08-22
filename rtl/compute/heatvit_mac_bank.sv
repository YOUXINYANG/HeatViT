// Single 8x8 outer-product accumulation bank.
//
// Each valid cycle computes 64 int8 products (signed x signed, or zero-
// extended unsigned src0 x signed src1), sign-extends each product and adds it
// to the corresponding signed int32 accumulator. Row/column masks gate the
// lanes, clear_accum wins over accum_valid, and accum_done acknowledges one
// cycle after a clear so callers know the bank is ready for a new round.
module heatvit_mac_bank
  import heatvit_pkg::*;
(
  input  logic         clk,
  input  logic         rst_n,
  input  logic         clear_accum,
  input  logic         accum_valid,
  input  logic [7:0]   a_lane [0:7],
  input  heatvit_s8_t  b_lane [0:7],
  input  logic         a_unsigned,
  input  logic [7:0]   row_mask,
  input  logic [7:0]   col_mask,
  output heatvit_s32_t accum [0:7][0:7],
  output logic         accum_done
);

  heatvit_s32_t accum_r [0:7][0:7];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int r = 0; r < 8; r++) begin
        for (int c = 0; c < 8; c++) accum_r[r][c] <= 32'sd0;
      end
      accum_done <= 1'b0;
    end else begin
      accum_done <= 1'b0;
      if (clear_accum) begin
        for (int r = 0; r < 8; r++) begin
          for (int c = 0; c < 8; c++) accum_r[r][c] <= 32'sd0;
        end
        accum_done <= 1'b1;
      end else if (accum_valid) begin
        for (int r = 0; r < 8; r++) begin
          if (row_mask[r]) begin
            for (int c = 0; c < 8; c++) begin
              if (col_mask[c]) begin
                logic signed [15:0] product16;
                logic signed [17:0] product18;
                logic [8:0]        a9;
                logic signed [8:0] b9;
                heatvit_s32_t      addend;
                if (a_unsigned) begin
                  a9         = {1'b0, a_lane[r]};
                  b9         = $signed({b_lane[c][7], b_lane[c]});
                  product18  = $signed(a9) * b9;
                  addend     = {{14{product18[17]}}, product18};
                end else begin
                  product16  = $signed(a_lane[r]) * b_lane[c];
                  addend     = {{16{product16[15]}}, product16};
                end
                accum_r[r][c] <= accum_r[r][c] + addend;
              end
            end
          end
        end
      end
    end
  end

  assign accum = accum_r;

endmodule
