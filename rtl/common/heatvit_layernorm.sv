// Two-pass fixed-point LayerNorm over exactly D=192 channels. The input
// scale exponent must be in [-32, 0]. Sum/mean/variance use Q32; the
// reciprocal of the floor standard deviation is requested once from the
// external shared divider, and the internal 48-bit isqrt produces the
// standard deviation. Gamma/Beta are aligned, summed, rounded once, and
// saturated to int8 at the output scale exponent.
//
// P7-4 timing rewrite: the normalize pass is a 4-stage stallable pipeline
// that streams its outputs directly (the old out_buf[192] byte-addressed
// register array and its per-byte write mux network are gone). The
// arithmetic is bit-exact with the former single-cycle normalize_channel
// function, only narrowed to the minimal proven widths:
//
//   stage0: x_q32 = x << (x_scale+32)  (40-bit signed), diff = x_q32 - mean
//           (49-bit signed); gamma/beta forwarded.
//   stage1: prod = diff * inv_std_q32 (49x48 -> 97-bit signed).
//   stage2: norm_wide = round_shift_away(prod, 48) -> sat to 24-bit Q8.16.
//   stage3: product = norm_q16 * gamma (24x8), sum_w = align + beta,
//           final scale shift, sat_s8 -> out byte.
//
// Width proofs (see docs, P7-4): |x_q32| < 2^39, |diff| < 2^48,
// |prod| < 2^96, aligned sum intermediates < 2^95, final affine value
// < 2^78, so every intermediate fits its container without truncation and
// every rounding matches the 128-bit original exactly.
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
  localparam logic [3:0] S_DONE       = 4'd8;

  localparam int D = 192;
  localparam int EPS_Q32 = 4295;

  // 2^47: nearest-rounding half-LSB for the >>48 product shift.
  localparam logic signed [96:0] ROUND_HALF_48 = 97'sd140737488355328;

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

  // Nearest-ties-away-from-zero right shift on a 96-bit signed value.
  // |value| fits signed 96 by construction (max |value| < 2^94), so the
  // magnitude, the half-LSB add and the shifted result never overflow.
  function automatic logic signed [95:0] round_shift_away_96(
    input logic signed [95:0] value,
    input int shift
  );
    logic signed [95:0] magnitude;
    logic signed [95:0] rounded_mag;
    if (shift == 0) return value;
    magnitude = value[95] ? -value : value;
    rounded_mag = (magnitude + (96'sd1 << (shift - 1))) >>> shift;
    return value[95] ? -rounded_mag : rounded_mag;
  endfunction

  logic [3:0]      state;
  logic [7:0]      idx;
  logic            mean_phase;
  logic            req_sent;
  heatvit_s128_t   sum_x_r;
  logic [63:0]     sum_square_r;
  heatvit_s48_t    mean_q32_r;
  logic [47:0]     e2_q32_r;
  logic [47:0]     variance_q32_r;
  logic [23:0]     std_q16_r;
  logic [47:0]     inv_std_q32_r;

  heatvit_s8_t x_buf [D];
  heatvit_s8_t gamma_buf [D];
  heatvit_s8_t beta_buf [D];

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

  // ------------------------------------------------------------------
  // Normalize pipeline (4 stages, stallable on the output handshake).
  // ------------------------------------------------------------------
  logic             s1_valid_r;
  logic signed [48:0] diff_r;        // stage0 -> stage1
  heatvit_s8_t     gamma1_r;
  heatvit_s8_t     beta1_r;

  logic             s2_valid_r;
  logic signed [96:0] prod_r;        // stage1 -> stage2
  heatvit_s8_t     gamma2_r;
  heatvit_s8_t     beta2_r;

  logic             s3_valid_r;
  heatvit_q8_16_t  norm_q16_r;       // stage2 -> stage3
  heatvit_s8_t     gamma3_r;
  heatvit_s8_t     beta3_r;

  // P7-5: stage 3 split into 3a/3b/3c.
  logic             s4_valid_r;
  logic signed [31:0] product_r;     // stage3a -> stage3b
  logic signed [6:0]  common_exp4_r;
  logic [6:0]         shift_a_r;
  logic [6:0]         shift_b_r;
  heatvit_s8_t        beta4_r;

  logic             s5_valid_r;
  logic signed [95:0] sum_align_r;   // stage3b -> stage3c
  logic signed [7:0]  final_shift_r;

  heatvit_s8_t     out_byte_r;       // stage3c output register
  logic [7:0]      out_count_r;

  assign out_data = out_byte_r;

  // Stage 0 combinational: read x/gamma/beta at idx, form diff.
  heatvit_s48_t    x_q32_w;
  logic signed [48:0] diff_w;
  assign x_q32_w = heatvit_s48_t'(x_buf[idx]) <<< (int'(x_scale_r) + 32);
  assign diff_w  = $signed({x_q32_w[47], x_q32_w})
                 - $signed({mean_q32_r[47], mean_q32_r});

  // Stage 1 combinational: 49x48 signed multiply (inv_std_q32 is positive).
  (* use_dsp = "yes" *)
  logic signed [96:0] prod_w;
  assign prod_w = diff_r * $signed({1'b0, inv_std_q32_r});

  // Stage 2 combinational: round_shift_away(prod, 48) then saturate Q8.16.
  logic signed [96:0] mag_w;
  logic signed [96:0] sum_half_w;
  logic signed [49:0] norm_wide_w;
  heatvit_q8_16_t     norm_q16_w;
  assign mag_w      = prod_r[96] ? -prod_r : prod_r;
  assign sum_half_w = mag_w + ROUND_HALF_48;
  assign norm_wide_w = prod_r[96]
    ? -$signed(50'($unsigned(sum_half_w[96:48])))
    :  $signed(50'($unsigned(sum_half_w[96:48])));
  always_comb begin
    if (norm_wide_w > 50'sd8388607)       norm_q16_w = 24'sd8388607;
    else if (norm_wide_w < -50'sd8388608) norm_q16_w = -24'sd8388608;
    else                                  norm_q16_w = 24'(norm_wide_w);
  end

  // Stage 3 combinational: gamma/beta affine with scale alignment.
  // P7-5: the original single-cycle version ran the DSP product, two
  // 96-bit variable shifts and the final shift/round in one cone
  // (79 levels, CARRY4=62 + DSP).  It is now split into three register
  // stages with identical arithmetic:
  //   3a: product = norm_q16 * gamma (DSP) + align shift amounts
  //   3b: sum = (product << s_a) + (beta << s_b) on 96 bits
  //   3c: final shift/round on 96 bits + sat8
  logic signed [31:0] product_w;
  logic signed [6:0]  common_exp_w;
  logic [6:0]         shift_a_w;
  logic [6:0]         shift_b_w;
  always_comb begin
    product_w = $signed(norm_q16_r) * $signed(gamma3_r);
    common_exp_w = ((int'(gamma_scale_r) - 16) < int'(beta_scale_r))
                     ? (int'(gamma_scale_r) - 16) : int'(beta_scale_r);
    // Shifts in [0..79]: beta_scale - common reaches 79 when beta_scale is
    // +31 and gamma_scale - 16 is -48, so both need 7 bits.
    shift_a_w = 7'(int'(gamma_scale_r) - 16 - common_exp_w);
    shift_b_w = 7'(int'(beta_scale_r) - common_exp_w);
  end

  logic signed [95:0] sum_align_w;
  logic signed [7:0]  final_shift_w;
  always_comb begin
    sum_align_w = ($signed(96'(product_r)) <<< shift_a_r)
                + ($signed(96'($signed(beta4_r))) <<< shift_b_r);
    final_shift_w = 8'(common_exp4_r - int'(out_scale_r));
  end

  logic signed [95:0] final_w;
  heatvit_s8_t out_byte_w;
  always_comb begin
    if (final_shift_r >= 0)
      final_w = sum_align_r <<< final_shift_r;
    else
      final_w = round_shift_away_96(sum_align_r, -final_shift_r);
    out_byte_w = sat_s8(heatvit_s128_t'(final_w));
  end

  // Pipeline control: advance when the output stage is empty or accepted.
  wire out_accept = out_valid && out_ready;
  wire advance    = !out_valid || out_accept;
  wire feeding    = (idx < 8'd192);

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
      s1_valid_r       <= 1'b0;
      diff_r           <= 49'sd0;
      gamma1_r         <= 8'sd0;
      beta1_r          <= 8'sd0;
      s2_valid_r       <= 1'b0;
      prod_r           <= 97'sd0;
      gamma2_r         <= 8'sd0;
      beta2_r          <= 8'sd0;
      s3_valid_r       <= 1'b0;
      norm_q16_r       <= 24'sd0;
      gamma3_r         <= 8'sd0;
      beta3_r          <= 8'sd0;
      s4_valid_r       <= 1'b0;
      product_r        <= 32'sd0;
      common_exp4_r    <= 7'sd0;
      shift_a_r        <= 7'd0;
      shift_b_r        <= 7'd0;
      beta4_r          <= 8'sd0;
      s5_valid_r       <= 1'b0;
      sum_align_r      <= 96'sd0;
      final_shift_r    <= 8'sd0;
      out_valid        <= 1'b0;
      out_byte_r       <= 8'sd0;
      out_count_r      <= 8'd0;
      for (int i = 0; i < D; i++) begin
        x_buf[i]     <= 8'sd0;
        gamma_buf[i] <= 8'sd0;
        beta_buf[i]  <= 8'sd0;
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
            out_valid     <= 1'b0;
            s1_valid_r    <= 1'b0;
            s2_valid_r    <= 1'b0;
            s3_valid_r    <= 1'b0;
            s4_valid_r    <= 1'b0;
            s5_valid_r    <= 1'b0;
            out_count_r   <= 8'd0;
            state         <= S_NORMALIZE;
          end
        end
        S_NORMALIZE: begin
          // Stage 0 loads channels idx=0..191; the pipeline then flushes.
          // out_valid is the stage-3c slot's valid flag; it holds during an
          // output stall and clears on the final accepted beat.
          if (out_accept && out_count_r == 8'd191) begin
            out_valid <= 1'b0;
            state     <= S_DONE;
          end else begin
            if (advance) begin
              if (feeding) idx <= idx + 8'd1;
              s1_valid_r <= feeding;
              diff_r     <= diff_w;
              gamma1_r   <= gamma_buf[idx];
              beta1_r    <= beta_buf[idx];
              s2_valid_r <= s1_valid_r;
              prod_r     <= prod_w;
              gamma2_r   <= gamma1_r;
              beta2_r    <= beta1_r;
              s3_valid_r <= s2_valid_r;
              norm_q16_r <= norm_q16_w;
              gamma3_r   <= gamma2_r;
              beta3_r    <= beta2_r;
              s4_valid_r    <= s3_valid_r;
              product_r     <= product_w;
              common_exp4_r <= common_exp_w;
              shift_a_r     <= shift_a_w;
              shift_b_r     <= shift_b_w;
              beta4_r       <= beta3_r;
              s5_valid_r    <= s4_valid_r;
              sum_align_r   <= sum_align_w;
              final_shift_r <= final_shift_w;
              out_valid  <= s5_valid_r;
              out_byte_r <= out_byte_w;
            end
            if (out_accept) out_count_r <= out_count_r + 8'd1;
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
