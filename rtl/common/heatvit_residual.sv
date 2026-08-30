// Two-stage registered residual adder: main + aux, scale aligned, int8 out.
//
// P7-5: the original single-stage version ran two cascaded 128-bit
// variable shifters + scale_to_exp_s128 in one combinational cone (111
// levels, CARRY4=93 at full-chip).  The same arithmetic is now split into
// two register stages with proven widths:
//   stage 1: common_exp = min(main, aux); each operand shifted left by
//            (its exp - common) in [0..63]; |value| <= 128 -> each term
//            <= 2^70, the 72-bit signed sum is exact.
//   stage 2: diff = out_exp - common in [-63..62].
//            diff >= 0: s = diff[6:0] in [0..62]; round-away on the
//            72-bit magnitude (|sum| <= 2^71 -> 72-bit cone).
//            diff <  0: k = -diff in [1..63]; k >= 8 sign-saturates
//            (|sum| >= 1 -> |sum<<k| >= 256; the 128-bit overflow
//            saturation in the original is the same extreme); k <= 7
//            -> 79-bit cone (71+7 = 78 < 79, exact).
// The valid/ready protocol keeps the original single-output-slot
// semantics; the result latency grows by one cycle.
module heatvit_residual
  import heatvit_pkg::*;
(
  input  logic           clk,
  input  logic           rst_n,

  input  logic           main_valid,
  output logic           main_ready,
  input  heatvit_s8_t    main_value,
  input  heatvit_scale_t main_scale_exp,

  input  logic           aux_valid,
  output logic           aux_ready,
  input  heatvit_s8_t    aux_value,
  input  heatvit_scale_t aux_scale_exp,

  input  heatvit_scale_t out_scale_exp,
  output logic           out_valid,
  input  logic           out_ready,
  output heatvit_s8_t    out_value
);

  // ---- stage 1: scale alignment + sum ----
  logic signed [5:0] common_exp_c;
  logic signed [71:0] sum_wide_c;
  always_comb begin
    common_exp_c = (main_scale_exp < aux_scale_exp) ? main_scale_exp
                                                    : aux_scale_exp;
    sum_wide_c =
      ($signed({{64{main_value[7]}}, main_value}) <<<
         (int'(main_scale_exp) - int'(common_exp_c))) +
      ($signed({{64{aux_value[7]}}, aux_value}) <<<
         (int'(aux_scale_exp) - int'(common_exp_c)));
  end

  logic             st1_valid_q;
  logic signed [71:0] sum_r;
  logic signed [5:0] common_exp_r;

  // ---- stage 2: scale to the output exponent + saturate ----
  int diff_c;
  logic [6:0] s_c;
  logic [72:0] mag_u;
  logic [72:0] rounded;
  logic signed [78:0] sh_c;
  heatvit_s8_t out_value_next;
  always_comb begin
    diff_c = int'(out_scale_exp) - int'(common_exp_r);
    if (diff_c >= 0) begin
      s_c = diff_c[6:0];  // diff in [0..62] -> s == diff
      if (s_c == 7'd0) begin
        out_value_next = (sum_r > 72'sd127)   ? 8'sd127 :
                         (sum_r < -72'sd128)  ? -8'sd128 :
                         heatvit_s8_t'(sum_r[7:0]);
      end else begin
        mag_u = (sum_r < 72'sd0)
          ? (73'h1000000000000000000 - {1'b0, sum_r}) : {1'b0, sum_r};
        rounded = (mag_u + (73'd1 << (s_c - 7'd1))) >> s_c;
        if (sum_r < 72'sd0) begin
          out_value_next = (rounded > 73'd128) ? -8'sd128 :
                           (8'sd0 - heatvit_s8_t'(rounded[7:0]));
        end else begin
          out_value_next = (rounded > 73'd127) ? 8'sd127 :
                           heatvit_s8_t'(rounded[7:0]);
        end
      end
    end else begin
      if (-diff_c >= 8) begin
        out_value_next = (sum_r == 72'sd0) ? 8'sd0 :
                         ((sum_r < 72'sd0) ? -8'sd128 : 8'sd127);
      end else begin
        // k = -diff in [1..7]: exact in 79 bits (71 + 7 = 78 < 79).
        sh_c = $signed({{7{sum_r[71]}}, sum_r}) <<< (-diff_c);
        out_value_next = (sh_c > 79'sd127)   ? 8'sd127 :
                         (sh_c < -79'sd128)  ? -8'sd128 :
                         heatvit_s8_t'(sh_c[7:0]);
      end
    end
  end

  // Handshake: accept a pair when stage 1 is empty; stage 1 advances into
  // the output slot when it is empty or accepted.  Output stays a single
  // slot (main/aux block when the output is occupied and stage 1 is full,
  // exactly like the original single-stage protocol).
  logic             out_valid_q;
  heatvit_s8_t     out_value_q;
  wire accept2 = (!out_valid_q) || out_ready;
  wire fire    = main_valid && aux_valid && (!st1_valid_q || accept2);

  assign main_ready = fire;
  assign aux_ready  = fire;
  assign out_valid  = out_valid_q;
  assign out_value  = out_value_q;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st1_valid_q  <= 1'b0;
      out_valid_q  <= 1'b0;
      sum_r        <= 72'sd0;
      common_exp_r <= 6'sd0;
      out_value_q  <= 8'sd0;
    end else begin
      if (fire) begin
        st1_valid_q  <= 1'b1;
        sum_r        <= sum_wide_c;
        common_exp_r <= common_exp_c;
      end else if (accept2) begin
        st1_valid_q <= 1'b0;
      end
      if (accept2 && st1_valid_q) begin
        out_valid_q <= 1'b1;
        out_value_q <= out_value_next;
      end else if (out_ready) begin
        out_valid_q <= 1'b0;
      end
    end
  end

endmodule
