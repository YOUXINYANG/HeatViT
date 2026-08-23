// I-ViT ShiftGELU integer-only GELU (ln2-slope refinement).
// Input and output are signed Q8.16; the output is saturated to signed
// 24 bits. Replaces the retired erf-polynomial GELU (HeatViT paper
// Eq. 11/12, delta1 = 0.5), which distorted frozen DeiT-T weights by
// ~25% in saturation and collapsed pure-PTQ accuracy to ~1% (see
// docs/heatvit.md Part 2 section 13.7).
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
// The per-lane 40/24-bit division runs on a local 40-cycle restoring
// divider (heatvit_udiv parameterized with QUOT_W = NUM_W so the whole
// 40-bit numerator is consumed; heatvit_udiv only processes the top
// QUOT_W numerator bits, so QUOT_W must match NUM_W for an exact
// quotient): the GEMM post-op unit is sequential (one lane at a time),
// so the latency is hidden and the executor's shared 3-client divider
// (LN/Softmax channel) stays untouched. Latency: 42 cycles per lane.
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
  localparam int DIV_QUOT_W = 40;

  heatvit_q8_16_t x_r;

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

  logic [DIV_NUM_W-1:0]  num;
  logic [DIV_DEN_W-1:0]  den;
  logic                  div_start;
  logic                  div_busy;
  logic                  div_done;
  logic                  div_dbz;
  logic [DIV_QUOT_W-1:0] div_quot;
  logic [DIV_DEN_W-1:0]  div_rem;

  heatvit_s48_t sig_wide;
  heatvit_s48_t prod_y;
  heatvit_s48_t y_scaled;
  heatvit_s48_t y_sat;
  heatvit_q8_16_t y_comb;

  localparam logic S_DIV  = 1'b0;
  localparam logic S_DONE = 1'b1;
  logic state;

  // ---- shift-exp core (combinational) ----------------------------------
  assign i_p = heatvit_s48_t'(x_r) + (heatvit_s48_t'(x_r) >>> 1)
             + (heatvit_s48_t'(x_r) >>> 3) + (heatvit_s48_t'(x_r) >>> 4);
  assign i_p2 = i_p + (i_p >>> 1) - (i_p >>> 4);
  assign abs_i_p2 = (i_p2 < 0) ? -i_p2 : i_p2;
  assign q_int = abs_i_p2 >> 16;
  assign r_q16 = abs_i_p2[15:0];
  assign frac_prod = (r_q16 * GELU_SLOPE_NUM_Q16) + GELU_SLOPE_ROUND_ADD;
  assign frac = frac_prod >> GELU_SLOPE_SHIFT;
  assign i_b = (i_p2 < 0) ? (17'd65536 - frac) : (17'd65536 + frac);

  always_comb begin
    if (i_p2 < 0) begin
      e_pre = (q_int <= 9'd16) ? (24'(i_b) >> q_int) : 24'd0;
    end else begin
      e_pre = (q_int <= 9'd7) ? (24'(i_b) << q_int) : 24'd8388607;
    end
  end
  assign e_comb = (e_pre > 24'd8388607) ? 24'd8388607 : e_pre;

  // ---- sigmoid division -------------------------------------------------
  assign num = {e_comb, 16'd0};           // e << 16
  assign den = 24'd65536 + e_comb;        // >= 65536, never zero

  heatvit_udiv #(
    .NUM_W  (DIV_NUM_W),
    .DEN_W  (DIV_DEN_W),
    .QUOT_W (DIV_QUOT_W)
  ) u_div (
    .clk            (clk),
    .rst_n          (rst_n),
    .start          (div_start),
    .busy           (div_busy),
    .done           (div_done),
    .divide_by_zero (div_dbz),
    .numerator      (num),
    .denominator    (den),
    .quotient       (div_quot),
    .remainder      (div_rem)
  );

  // round-half-up: sig = quot + (2 * rem >= den); quot < 2^17, only its
  // low 17 bits are used.
  assign sig_wide = heatvit_s48_t'(div_quot[16:0])
                  + ((({1'b0, div_rem} << 1) >= {1'b0, den}) ? 48'sd1
                                                             : 48'sd0);
  assign prod_y   = heatvit_s48_t'(x_r) * sig_wide;
  assign y_scaled = round_shift_away_s48(prod_y, 16);
  assign y_sat    = (y_scaled > 48'sd8388607)  ? 48'sd8388607 :
                    (y_scaled < -48'sd8388608) ? -48'sd8388608 : y_scaled;
  assign y_comb   = heatvit_q8_16_t'(y_sat[23:0]);

  // ---- control ----------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      x_r       <= 24'sd0;
      busy      <= 1'b0;
      done      <= 1'b0;
      y_out     <= 24'sd0;
      state     <= S_DIV;
      div_start <= 1'b0;
    end else begin
      done      <= 1'b0;
      div_start <= 1'b0;
      if (start && busy) $error("heatvit_gelu: start asserted while busy");
      case (state)
        S_DIV: begin
          if (start && !busy) begin
            x_r       <= x_in;
            busy      <= 1'b1;
            div_start <= 1'b1;
            state     <= S_DONE;
          end
        end
        S_DONE: begin
          if (div_done) begin
            y_out <= y_comb;
            busy  <= 1'b0;
            done  <= 1'b1;
            state <= S_DIV;
          end
        end
        default: state <= S_DIV;
      endcase
    end
  end

endmodule
