// I-ViT ShiftGELU integer-only GELU (ln2-slope refinement), pipelined.
// Input and output are signed Q8.16; the output is saturated to signed
// 24 bits. Bit-identical results to the retired serial version (the
// division is exact either way), but the 40-bit restoring division now
// runs as a 40-stage pipeline: throughput 1 lane/cycle, latency 41
// cycles. The GEMM post-op feeds a whole tile back-to-back and drains
// the results in order, removing the per-lane 40-cycle serialization
// (e2e cycle impact, see docs/heatvit.md Part 2 section 13.8).
//
// Sequence (bit-exact port of verification/heatvit_ref/nonlinear.gelu):
//   i_p  = x + (x>>1) + (x>>3) + (x>>4)            // 1.702x, (1.1011)b
//   i_p2 = i_p + (i_p>>1) - (i_p>>4)               // * log2(e), (1.0111)b
//   q    = |i_p2| >> 16,  r = |i_p2| & 0xFFFF
//   frac = (r * 11 + 15) >> 4                      // 2^x linear approx,
//                                                  // slope 11/16 ~ ln2
//   e    = i_p2 < 0 ? ((65536 - frac) >> q if q <= 16 else 0)
//                   : (min((65536 + frac) << q, 2^23 - 1) if q <= 7
//                      else 2^23 - 1)              // e^{1.702x} in Q16
//   sig  = round(e * 2^16 / (65536 + e))           // 40/24-bit division
//   y    = round_shift_away(x * sig, 16), saturated to 24 bits
//
// Pipeline: stage 0 latches x and computes num/den from the shift-exp
// core; stages 0..39 consume one numerator bit each (radix-2 restoring,
// MSB first via a left-shifting numerator register); the final stage
// applies the round-half-up carry and the x*sig multiply/round/saturate.
// ``done`` pulses 41 cycles after ``start``; back-to-back starts are the
// intended usage (no start-while-busy error).
module heatvit_gelu
  import heatvit_pkg::*;
(
  input  logic            clk,
  input  logic            rst_n,
  input  logic            start,
  output logic            busy,
  output logic            done,
  input  heatvit_q8_16_t  x_in,
  output heatvit_q8_16_t  y_out
);

  localparam int DIV_NUM_W  = 40;
  localparam int DIV_DEN_W  = 24;
  localparam int DIV_STAGES = 40;

  // ---- shift-exp core (combinational from the live input) --------------
  heatvit_s48_t i_p;
  heatvit_s48_t i_p2;
  heatvit_s48_t abs_i_p2;
  logic [8:0]   q_int;      // |i_p2| >> 16, <= 309
  logic [15:0]  r_q16;      // fractional 16 bits
  logic [19:0]  frac_prod;  // r * 11 + 15
  logic [15:0]  frac;       // (r * 11 + 15) >> 4, <= 45056
  logic [16:0]  i_b;        // 65536 +/- frac
  logic [23:0]  e_pre;      // pre-clamp exponent
  logic [23:0]  e_comb;     // e^{1.702x} in Q16, sat 2^23 - 1

  assign i_p = heatvit_s48_t'(x_in) + (heatvit_s48_t'(x_in) >>> 1)
             + (heatvit_s48_t'(x_in) >>> 3) + (heatvit_s48_t'(x_in) >>> 4);
  assign i_p2 = i_p + (i_p >>> 1) - (i_p >>> 4);
  assign abs_i_p2 = (i_p2 < 0) ? -i_p2 : i_p2;
  assign q_int = abs_i_p2 >> 16;
  assign r_q16 = abs_i_p2[15:0];
  assign frac_prod = (r_q16 * GELU_SLOPE_NUM_Q16) + GELU_SLOPE_ROUND_ADD;
  assign frac = frac_prod >> GELU_SLOPE_SHIFT;
  assign i_b = (i_p2 < 0) ? (17'd65536 - frac) : (17'd65536 + frac);

  always_comb begin
    if (i_p2 < 0)
      e_pre = (q_int <= 9'd16) ? (24'(i_b) >> q_int) : 24'd0;
    else
      e_pre = (q_int <= 9'd7) ? (24'(i_b) << q_int) : 24'd8388607;
  end
  assign e_comb = (e_pre > 24'd8388607) ? 24'd8388607 : e_pre;

  // ---- pipeline registers ----------------------------------------------
  heatvit_q8_16_t        x_r    [0:40];   // input carried to the multiply
  logic [DIV_NUM_W-1:0]  num_r  [0:40];   // left-shifting numerator
  logic [DIV_DEN_W-1:0]  den_r  [0:40];
  logic [DIV_DEN_W-1:0]  rem_r  [0:40];
  logic [DIV_NUM_W-1:0]  quot_r [0:40];
  logic [41:0]           valid_pipe;      // bit k = lane k cycles old

  // Per-stage combinational next values.
  logic [DIV_DEN_W-1:0] shifted_rem [0:39];
  logic                 bit_one     [0:39];
  logic [DIV_DEN_W-1:0] nxt_rem     [0:39];
  logic [DIV_NUM_W-1:0] nxt_quot    [0:39];

  genvar gi;
  generate
    for (gi = 0; gi < DIV_STAGES; gi++) begin : g_div
      assign shifted_rem[gi] = {rem_r[gi][DIV_DEN_W-2:0], num_r[gi][DIV_NUM_W-1]};
      assign bit_one[gi]     = shifted_rem[gi] >= den_r[gi];
      assign nxt_rem[gi]     = bit_one[gi] ? (shifted_rem[gi] - den_r[gi])
                                           : shifted_rem[gi];
      assign nxt_quot[gi]    = {quot_r[gi][DIV_NUM_W-2:0], bit_one[gi]};
    end
  endgenerate

  // Round-half-up sigmoid + final multiply (combinational on stage 40).
  heatvit_s48_t sig_wide;
  heatvit_s48_t prod_y;
  heatvit_s48_t y_scaled;
  heatvit_s48_t y_sat;
  heatvit_q8_16_t y_comb;

  assign sig_wide = heatvit_s48_t'(quot_r[40][16:0])
                  + ((({1'b0, rem_r[40]} << 1) >= {1'b0, den_r[40]})
                         ? 48'sd1 : 48'sd0);
  assign prod_y   = heatvit_s48_t'(x_r[40]) * sig_wide;
  assign y_scaled = round_shift_away_s48(prod_y, 16);
  assign y_sat    = (y_scaled > 48'sd8388607)  ? 48'sd8388607 :
                    (y_scaled < -48'sd8388608) ? -48'sd8388608 : y_scaled;
  assign y_comb   = heatvit_q8_16_t'(y_sat[23:0]);

  // ---- control ----------------------------------------------------------
  assign busy = |valid_pipe[40:0];   // any lane before its completion cycle
  assign done = valid_pipe[41];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      x_r[0]    <= 24'sd0;
      num_r[0]  <= {DIV_NUM_W{1'b0}};
      den_r[0]  <= {DIV_DEN_W{1'b0}};
      rem_r[0]  <= {DIV_DEN_W{1'b0}};
      quot_r[0] <= {DIV_NUM_W{1'b0}};
      valid_pipe <= 42'd0;
      y_out     <= 24'sd0;
      for (int i = 1; i <= 40; i++) begin
        x_r[i]   <= 24'sd0;
        num_r[i] <= {DIV_NUM_W{1'b0}};
        den_r[i] <= {DIV_DEN_W{1'b0}};
        rem_r[i] <= {DIV_DEN_W{1'b0}};
        quot_r[i] <= {DIV_NUM_W{1'b0}};
      end
    end else begin
      // Stage 0 latches the live input; all stages advance every cycle.
      x_r[0]    <= x_in;
      num_r[0]  <= {e_comb, 16'd0};
      den_r[0]  <= 24'd65536 + e_comb;
      rem_r[0]  <= {DIV_DEN_W{1'b0}};
      quot_r[0] <= {DIV_NUM_W{1'b0}};
      valid_pipe <= {valid_pipe[40:0], start};
      for (int i = 0; i < 40; i++) begin
        x_r[i+1]   <= x_r[i];
        num_r[i+1] <= {num_r[i][DIV_NUM_W-2:0], 1'b0};
        den_r[i+1] <= den_r[i];
        rem_r[i+1] <= nxt_rem[i];
        quot_r[i+1] <= nxt_quot[i];
      end
      y_out     <= y_comb;
    end
  end

endmodule
