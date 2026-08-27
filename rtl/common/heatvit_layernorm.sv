// Two-pass fixed-point LayerNorm over exactly D=192 channels. The input
// scale exponent must be in [-32, 0]. Sum/mean/variance use Q32; the
// reciprocal of the floor standard deviation is requested once from the
// external shared divider, and the internal 48-bit isqrt produces the
// standard deviation. Gamma/Beta are aligned, summed, rounded once, and
// saturated to int8 at the output scale exponent.
module heatvit_layernorm
  import heatvit_pkg::*;
(
  input  logic           clk,
  input  logic           rst_n,
  input  logic           cfg_valid,
  output logic           cfg_ready,
  input  heatvit_scale_t cfg_x_scale_exp,
  input  heatvit_scale_t cfg_gamma_scale_exp,
  input  heatvit_scale_t cfg_beta_scale_exp,
  input  heatvit_scale_t cfg_out_scale_exp,
  output logic           busy,
  output logic           done,
  output logic           warn_negative_variance,
  input  logic           in_valid,
  output logic           in_ready,
  input  heatvit_s8_t    in_x,
  input  heatvit_s8_t    in_gamma,
  input  heatvit_s8_t    in_beta,
  output logic           out_valid,
  input  logic           out_ready,
  output heatvit_s8_t    out_data,
  output logic           div_req_valid,
  input  logic           div_req_ready,
  output logic [63:0]    div_num,
  output logic [63:0]    div_den,
  input  logic           div_rsp_valid,
  input  logic [63:0]    div_quot,
  input  logic [63:0]    div_rem,
  input  logic           div_div_zero
);

  localparam logic [3:0] S_IDLE       = 4'd0;
  localparam logic [3:0] S_LOAD_ACCUM = 4'd1;
  localparam logic [3:0] S_MEAN       = 4'd2;
  localparam logic [3:0] S_VARIANCE   = 4'd3;
  localparam logic [3:0] S_SQRT       = 4'd4;
  localparam logic [3:0] S_RECIP      = 4'd5;
  localparam logic [3:0] S_NORMALIZE  = 4'd6;
  localparam logic [3:0] S_DRAIN      = 4'd7;
  localparam logic [3:0] S_DONE       = 4'd8;

  localparam int D = 192;
  localparam int EPS_Q32 = 4295;

  heatvit_scale_t x_scale_r;
  heatvit_scale_t gamma_scale_r;
  heatvit_scale_t beta_scale_r;
  heatvit_scale_t out_scale_r;

  function automatic heatvit_s128_t x_q32_of(input heatvit_s8_t x,
                                             input heatvit_scale_t x_scale);
    return heatvit_s128_t'(x) <<< (int'(x_scale) + 32);
  endfunction

  function automatic heatvit_s128_t square_q32_of(input heatvit_s8_t x,
                                                  input heatvit_scale_t x_scale);
    heatvit_s128_t square;
    int shift;
    square = heatvit_s128_t'(x) * heatvit_s128_t'(x);
    shift = 2 * int'(x_scale) + 32;
    if (shift >= 0) return square <<< shift;
    return round_shift_away_s128(square, 7'(-shift));
  endfunction

  function automatic heatvit_s8_t normalize_channel(
    input heatvit_s8_t    x,
    input heatvit_s8_t    gamma,
    input heatvit_s8_t    beta,
    input heatvit_s48_t   mean_q32,
    input logic [47:0]    inv_std_q32
  );
    heatvit_s128_t x_q32;
    heatvit_s128_t diff;
    heatvit_s128_t prod;
    heatvit_s128_t norm_wide;
    heatvit_s128_t norm_q16;
    heatvit_s128_t product;
    heatvit_s128_t sum_w;
    int common_exp;
    int final_shift;
    begin
      x_q32 = heatvit_s128_t'(x) <<< (int'(x_scale_r) + 32);
      diff = x_q32 - heatvit_s128_t'(mean_q32);
      prod = diff * heatvit_s128_t'(inv_std_q32);
      norm_wide = round_shift_away_s128(prod, 7'd48);
      if (norm_wide > 128'sd8388607) norm_q16 = 128'sd8388607;
      else if (norm_wide < -128'sd8388608) norm_q16 = -128'sd8388608;
      else norm_q16 = norm_wide;

      product = norm_q16 * heatvit_s128_t'(gamma);
      common_exp = ((int'(gamma_scale_r) - 16) < int'(beta_scale_r))
                     ? (int'(gamma_scale_r) - 16) : int'(beta_scale_r);
      sum_w = (product <<< (int'(gamma_scale_r) - 16 - common_exp))
            + (heatvit_s128_t'(beta) <<< (int'(beta_scale_r) - common_exp));
      final_shift = common_exp - int'(out_scale_r);
      if (final_shift >= 0) sum_w = sum_w <<< final_shift;
      else sum_w = round_shift_away_s128(sum_w, 7'(-final_shift));
      return sat_s8(sum_w);
    end
  endfunction

  logic [3:0]      state;
  logic [7:0]      idx;
  logic [7:0]      out_idx;
  heatvit_s128_t   sum_x_r;
  logic [63:0]     sum_square_r;
  heatvit_s48_t    mean_q32_r;
  logic [47:0]     e2_q32_r;
  logic [47:0]     variance_q32_r;
  logic [23:0]     std_q16_r;
  logic [47:0]     inv_std_q32_r;
  logic            mean_phase;
  logic            req_sent;

  heatvit_s8_t x_buf [D];
  heatvit_s8_t gamma_buf [D];
  heatvit_s8_t beta_buf [D];
  heatvit_s8_t out_buf [D];

  logic        isqrt_start;
  logic        isqrt_busy;
  logic        isqrt_done;
  logic [47:0] isqrt_radicand;
  logic [23:0] isqrt_root;
  logic [47:0] isqrt_remainder;

  heatvit_s128_t mean_wide;
  heatvit_s128_t mean_square_w;
  heatvit_s128_t variance_w;
  logic          variance_negative;
  logic [47:0]   variance_clamped;
  heatvit_s128_t sum_x_next_w;
  heatvit_s128_t square_in_w;
  logic [63:0]   sum_x_mag_w;
  logic [63:0]   rounded_quot_w;

  assign mean_wide          = $signed({{80 {mean_q32_r[47]}}, mean_q32_r});
  assign mean_square_w      = round_shift_away_s128(mean_wide * mean_wide, 7'd32);
  assign variance_w         = heatvit_s128_t'(e2_q32_r) - mean_square_w;
  assign variance_negative  = (variance_w < 0);
  assign variance_clamped   = variance_negative ? 48'd0 : variance_w[47:0];
  assign sum_x_next_w       = sum_x_r + x_q32_of(in_x, x_scale_r);
  assign square_in_w        = square_q32_of(in_x, x_scale_r);
  assign sum_x_mag_w        = (sum_x_next_w < 0) ? (128'd0 - sum_x_next_w) : sum_x_next_w;
  assign rounded_quot_w     = div_quot + (((div_rem << 1) >= div_den) ? 64'd1 : 64'd0);

  assign cfg_ready = (state == S_IDLE);
  assign in_ready  = (state == S_LOAD_ACCUM);
  assign out_data  = out_buf[out_idx];

  heatvit_isqrt #(.RAD_W(48)) u_isqrt (
    .clk       (clk),
    .rst_n     (rst_n),
    .start     (isqrt_start),
    .busy      (isqrt_busy),
    .done      (isqrt_done),
    .radicand  (isqrt_radicand),
    .root      (isqrt_root),
    .remainder (isqrt_remainder)
  );

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state            <= S_IDLE;
      busy             <= 1'b0;
      done             <= 1'b0;
      warn_negative_variance <= 1'b0;
      idx              <= 8'd0;
      out_idx          <= 8'd0;
      x_scale_r        <= 6'sd0;
      gamma_scale_r    <= 6'sd0;
      beta_scale_r     <= 6'sd0;
      out_scale_r      <= 6'sd0;
      sum_x_r          <= 128'sd0;
      sum_square_r     <= 64'd0;
      mean_q32_r       <= 48'sd0;
      e2_q32_r         <= 48'd0;
      variance_q32_r   <= 48'd0;
      std_q16_r        <= 24'd0;
      inv_std_q32_r    <= 48'd0;
      mean_phase       <= 1'b0;
      req_sent         <= 1'b0;
      div_req_valid    <= 1'b0;
      div_num          <= 64'd0;
      div_den          <= 64'd0;
      isqrt_start      <= 1'b0;
      isqrt_radicand   <= 48'd0;
      out_valid        <= 1'b0;
      for (int i = 0; i < D; i++) begin
        x_buf[i]     <= 8'sd0;
        gamma_buf[i] <= 8'sd0;
        beta_buf[i]  <= 8'sd0;
        out_buf[i]   <= 8'sd0;
      end
    end else begin
      done <= 1'b0;
      isqrt_start <= 1'b0;
      case (state)
        S_IDLE: begin
          if (cfg_valid && cfg_ready) begin
            if (cfg_x_scale_exp > 6'sd0)
              $fatal(2, "heatvit_layernorm: input scale %0d out of range",
                     cfg_x_scale_exp);
            x_scale_r     <= cfg_x_scale_exp;
            gamma_scale_r <= cfg_gamma_scale_exp;
            beta_scale_r  <= cfg_beta_scale_exp;
            out_scale_r   <= cfg_out_scale_exp;
            warn_negative_variance <= 1'b0;
            idx        <= 8'd0;
            sum_x_r    <= 128'sd0;
            sum_square_r <= 64'd0;
            busy       <= 1'b1;
            state      <= S_LOAD_ACCUM;
          end
        end
        S_LOAD_ACCUM: begin
          if (in_valid && in_ready) begin
            x_buf[idx]     <= in_x;
            gamma_buf[idx] <= in_gamma;
            beta_buf[idx]  <= in_beta;
            sum_x_r        <= sum_x_r + x_q32_of(in_x, x_scale_r);
            sum_square_r   <= sum_square_r + square_in_w[63:0];
            if (idx == 8'd191) begin
              idx           <= 8'd0;
              mean_phase    <= 1'b0;
              div_num       <= sum_x_mag_w[63:0];
              div_den       <= 64'd192;
              div_req_valid <= 1'b1;
              req_sent      <= 1'b0;
              state         <= S_MEAN;
            end else begin
              idx <= idx + 8'd1;
            end
          end
        end
        S_MEAN: begin
          if (!req_sent && div_req_ready) begin
            div_req_valid <= 1'b0;
            req_sent      <= 1'b1;
          end
          if (div_rsp_valid) begin
            if (!mean_phase) begin
              mean_q32_r <= (sum_x_r < 0)
                ? -heatvit_s48_t'(rounded_quot_w[47:0])
                :  heatvit_s48_t'(rounded_quot_w[47:0]);
              mean_phase    <= 1'b1;
              div_num       <= sum_square_r;
              div_den       <= 64'd192;
              div_req_valid <= 1'b1;
              req_sent      <= 1'b0;
            end else begin
              e2_q32_r <= rounded_quot_w[47:0];
              state    <= S_VARIANCE;
            end
          end
        end
        S_VARIANCE: begin
          variance_q32_r <= variance_clamped;
          if (variance_negative) warn_negative_variance <= 1'b1;
          isqrt_radicand <= variance_clamped + 48'd4295;
          isqrt_start    <= 1'b1;
          state          <= S_SQRT;
        end
        S_SQRT: begin
          if (isqrt_done) begin
            std_q16_r     <= isqrt_root;
            div_num       <= 64'h1_0000_0000_0000;
            div_den       <= 64'(isqrt_root);
            div_req_valid <= 1'b1;
            req_sent      <= 1'b0;
            state         <= S_RECIP;
          end
        end
        S_RECIP: begin
          if (!req_sent && div_req_ready) begin
            div_req_valid <= 1'b0;
            req_sent      <= 1'b1;
          end
          if (div_rsp_valid) begin
            inv_std_q32_r <= rounded_quot_w[47:0];
            idx           <= 8'd0;
            state         <= S_NORMALIZE;
          end
        end
        S_NORMALIZE: begin
          out_buf[idx] <= normalize_channel(
            x_buf[idx], gamma_buf[idx], beta_buf[idx],
            mean_q32_r, inv_std_q32_r
          );
          if (idx == 8'd191) begin
            out_idx   <= 8'd0;
            out_valid <= 1'b1;
            state     <= S_DRAIN;
          end else begin
            idx <= idx + 8'd1;
          end
        end
        S_DRAIN: begin
          if (out_valid && out_ready) begin
            if (out_idx == 8'd191) begin
              out_valid <= 1'b0;
              state     <= S_DONE;
            end else begin
              out_idx <= out_idx + 8'd1;
            end
          end
        end
        S_DONE: begin
          busy  <= 1'b0;
          done  <= 1'b1;
          state <= S_IDLE;
        end
        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
