// HeatViT paper GELU fixed-point approximation. Input and output are
// signed Q8.16; the output is saturated to signed 24 bits.
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

  heatvit_q8_16_t x_r;

  heatvit_s48_t prod_u;
  heatvit_s48_t u_q16;
  heatvit_s48_t abs_u;
  heatvit_s48_t clip_q16;
  heatvit_s48_t t_q16;
  heatvit_s48_t t2_q16;
  heatvit_s48_t poly_q16;
  heatvit_s48_t erf_mag_q16;
  heatvit_s48_t sign_u;
  heatvit_s48_t l_erf_q16;
  heatvit_s48_t one_plus_l;
  heatvit_s48_t prod_y;
  heatvit_s48_t y_scaled;
  heatvit_s48_t y_sat;
  heatvit_q8_16_t y_comb;

  assign prod_u      = heatvit_s48_t'(x_r) * heatvit_s48_t'(INV_SQRT2_Q16);
  assign u_q16       = round_shift_away_s48(prod_u, 16);
  assign abs_u       = (u_q16 < 0) ? -u_q16 : u_q16;
  assign clip_q16    = (abs_u > heatvit_s48_t'(-GELU_B_Q16))
                         ? heatvit_s48_t'(-GELU_B_Q16) : abs_u;
  assign t_q16       = clip_q16 + heatvit_s48_t'(GELU_B_Q16);
  assign t2_q16      = round_shift_away_s48(t_q16 * t_q16, 16);
  assign poly_q16    = round_shift_away_s48(
                         heatvit_s48_t'(GELU_A_Q16) * t2_q16, 16
                       ) + 48'sd65536;
  assign erf_mag_q16 = round_shift_away_s48(
                         heatvit_s48_t'(GELU_DELTA_Q16) * poly_q16, 16
                       );
  assign sign_u      = (u_q16 > 0) ? 48'sd1 : ((u_q16 < 0) ? -48'sd1 : 48'sd0);
  assign l_erf_q16   = sign_u * erf_mag_q16;
  assign one_plus_l  = 48'sd65536 + l_erf_q16;
  assign prod_y      = heatvit_s48_t'(x_r) * one_plus_l;
  assign y_scaled    = round_shift_away_s48(prod_y, 17);
  assign y_sat       = (y_scaled > 48'sd8388607)  ? 48'sd8388607 :
                       (y_scaled < -48'sd8388608) ? -48'sd8388608 : y_scaled;
  assign y_comb      = heatvit_q8_16_t'(y_sat[23:0]);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      x_r   <= 24'sd0;
      busy  <= 1'b0;
      done  <= 1'b0;
      y_out <= 24'sd0;
    end else begin
      done <= 1'b0;
      if (start && busy) $error("heatvit_gelu: start asserted while busy");
      if (start && !busy) begin
        x_r  <= x_in;
        busy <= 1'b1;
      end else if (busy) begin
        y_out <= y_comb;
        busy  <= 1'b0;
        done  <= 1'b1;
      end
    end
  end

endmodule
