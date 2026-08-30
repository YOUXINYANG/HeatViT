// Temporary diagnostic: compare the P7-5 narrow requant functions against
// the original 128-bit implementations under real SystemVerilog semantics.
// NOT part of the permanent test suite.
`timescale 1ns/1ps
module tb_requant_diag;
  import heatvit_pkg::*;

  logic signed [5:0] src0_scale;
  logic signed [5:0] src1_scale;
  logic signed [5:0] dst_scale;
  logic bias_en;
  logic [2:0] post_op;

  // ---------------- original implementations (128-bit) ----------------
  function automatic heatvit_s8_t orig_out8(input logic signed [32:0] sum);
    heatvit_s128_t wide;
    int shift;
    heatvit_s128_t scaled;
    wide = $signed({{95{sum[32]}}, sum});
    shift = int'(dst_scale) - int'(src0_scale) - int'(src1_scale);
    if (shift >= 0) scaled = round_shift_away_s128(wide, shift[6:0]);
    else scaled = wide <<< (-shift);
    return sat_s8(scaled);
  endfunction

  function automatic heatvit_s32_t orig_out32(input logic signed [32:0] sum);
    heatvit_s128_t wide;
    heatvit_s128_t scaled;
    wide = $signed({{95{sum[32]}}, sum});
    scaled = scale_to_exp_s128(wide, src0_scale + src1_scale, dst_scale);
    return sat_s32(scaled);
  endfunction

  function automatic heatvit_s8_t orig_gelu8(input logic signed [23:0] value);
    heatvit_s128_t wide;
    heatvit_s128_t scaled;
    wide = $signed({{104{value[23]}}, value});
    scaled = scale_to_exp_s128(wide, -6'sd16, dst_scale);
    return sat_s8(scaled);
  endfunction

  function automatic heatvit_q8_16_t orig_actq16(
    input heatvit_s32_t accum_value,
    input heatvit_s32_t bias_value
  );
    heatvit_s128_t wide;
    heatvit_s128_t scaled;
    wide = $signed({{96{accum_value[31]}}, accum_value}) +
           (bias_en ? $signed({{96{bias_value[31]}}, bias_value}) : 128'sd0);
    scaled = scale_to_exp_s128(wide, src0_scale + src1_scale, -6'sd16);
    if (scaled > 128'sd8388607) return 24'sd8388607;
    if (scaled < -128'sd8388608) return -24'sd8388608;
    return heatvit_q8_16_t'(scaled[23:0]);
  endfunction

  // ---------------- P7-5 narrow implementations ----------------
  function automatic heatvit_s8_t sat_s8_from_s33(input logic signed [32:0] v);
    if (v > 33'sd127)  return 8'sd127;
    if (v < -33'sd128) return -8'sd128;
    return heatvit_s8_t'(v[7:0]);
  endfunction

  function automatic heatvit_s8_t sat_s8_from_s24(input logic signed [23:0] v);
    if (v > 24'sd127)  return 8'sd127;
    if (v < -24'sd128) return -8'sd128;
    return heatvit_s8_t'(v[7:0]);
  endfunction

  function automatic heatvit_s8_t sat_s8_from_s40(input logic signed [39:0] v);
    if (v > 40'sd127)  return 8'sd127;
    if (v < -40'sd128) return -8'sd128;
    return heatvit_s8_t'(v[7:0]);
  endfunction

  function automatic heatvit_s32_t sat_s32_from_s33(input logic signed [32:0] v);
    if (v > 33'sd2147483647)  return 32'sd2147483647;
    if (v < -33'sd2147483648) return -32'sd2147483648;
    return heatvit_s32_t'(v[31:0]);
  endfunction

  function automatic heatvit_s32_t sat_s32_from_s64(input logic signed [63:0] v);
    if (v > 64'sd2147483647)  return 32'sd2147483647;
    if (v < -64'sd2147483648) return -32'sd2147483648;
    return heatvit_s32_t'(v[31:0]);
  endfunction

  function automatic heatvit_q8_16_t clamp_q8_16_from_s33(
    input logic signed [32:0] v
  );
    if (v > 33'sd8388607)  return 24'sd8388607;
    if (v < -33'sd8388608) return -24'sd8388608;
    return heatvit_q8_16_t'(v[23:0]);
  endfunction

  function automatic heatvit_q8_16_t clamp_q8_16_from_s55(
    input logic signed [54:0] v
  );
    if (v > 55'sd8388607)  return 24'sd8388607;
    if (v < -55'sd8388608) return -24'sd8388608;
    return heatvit_q8_16_t'(v[23:0]);
  endfunction

  function automatic heatvit_s8_t new_out8(input logic signed [32:0] sum);
    int shift;
    logic [6:0] s;
    logic [33:0] mag_u;
    logic [33:0] rounded;
    shift = int'(dst_scale) - int'(src0_scale) - int'(src1_scale);
    if (shift >= 0) begin
      s = shift[6:0];
      if (s == 7'd0) begin
        return sat_s8_from_s33(sum);
      end else if (s <= 7'd32) begin
        mag_u = (sum < 33'sd0) ? (34'h200000000 - {1'b0, sum}) : {1'b0, sum};
        rounded = (mag_u + (34'd1 << (s - 7'd1))) >> s;
        if (sum < 33'sd0) begin
          return (rounded > 34'd128) ? -8'sd128 :
                 (8'sd0 - heatvit_s8_t'(rounded[7:0]));
        end else begin
          return (rounded > 34'd127) ? 8'sd127 : heatvit_s8_t'(rounded[7:0]);
        end
      end else if (s == 7'd33) begin
        return (sum == 33'sh100000000) ? -8'sd1 : 8'sd0;
      end else begin
        return 8'sd0;
      end
    end else begin
      if (-shift >= 8) begin
        return (sum == 33'sd0) ? 8'sd0 :
               ((sum < 33'sd0) ? -8'sd128 : 8'sd127);
      end else begin
        return sat_s8_from_s40($signed({{7{sum[32]}}, sum}) <<< (-shift));
      end
    end
  endfunction

  function automatic heatvit_s32_t new_out32(input logic signed [32:0] sum);
    logic signed [5:0] src_exp;
    int diff;
    logic [5:0] d;
    logic [33:0] mag_u;
    logic [33:0] rounded;
    src_exp = src0_scale + src1_scale;
    diff = int'(dst_scale) - int'(src_exp);
    if (diff == 0) begin
      return sat_s32_from_s33(sum);
    end else if (diff > 0) begin
      d = diff[5:0];
      if (d >= 6'd34) begin
        return 32'sd0;
      end else if (d == 6'd33) begin
        return (sum == 33'sh100000000) ? -32'sd1 : 32'sd0;
      end else begin
        mag_u = (sum < 33'sd0) ? (34'h200000000 - {1'b0, sum}) : {1'b0, sum};
        rounded = (mag_u + (34'd1 << (d - 6'd1))) >> d;
        if (sum < 33'sd0) begin
          return -heatvit_s32_t'(rounded[31:0]);
        end else begin
          return (rounded > 34'd2147483647) ? 32'sd2147483647 :
                 heatvit_s32_t'(rounded[31:0]);
        end
      end
    end else begin
      if (-diff >= 31) begin
        return (sum == 33'sd0) ? 32'sd0 :
               ((sum < 33'sd0) ? -32'sd2147483648 : 32'sd2147483647);
      end else begin
        return sat_s32_from_s64($signed({{31{sum[32]}}, sum}) <<< (-diff));
      end
    end
  endfunction

  function automatic heatvit_s8_t new_gelu8(input logic signed [23:0] value);
    int diff;
    logic [5:0] d;
    logic [24:0] mag_u;
    logic [24:0] rounded;
    diff = int'(dst_scale) + 16;
    if (diff == 0) begin
      return sat_s8_from_s24(value);
    end else if (diff > 0) begin
      d = diff[5:0];
      if (d >= 6'd25) begin
        return 8'sd0;
      end else if (d == 6'd24) begin
        return (value == 24'sh800000) ? -8'sd1 : 8'sd0;
      end else begin
        mag_u = (value < 24'sd0) ? (25'h1000000 - {1'b0, value})
                                : {1'b0, value};
        rounded = (mag_u + (25'd1 << (d - 6'd1))) >> d;
        if (value < 24'sd0) begin
          return (rounded > 25'd128) ? -8'sd128 :
                 (8'sd0 - heatvit_s8_t'(rounded[7:0]));
        end else begin
          return (rounded > 25'd127) ? 8'sd127 : heatvit_s8_t'(rounded[7:0]);
        end
      end
    end else begin
      if (-diff >= 8) begin
        return (value == 24'sd0) ? 8'sd0 :
               ((value < 24'sd0) ? -8'sd128 : 8'sd127);
      end else begin
        return sat_s8_from_s40($signed({{16{value[23]}}, value}) <<< (-diff));
      end
    end
  endfunction

  function automatic heatvit_q8_16_t new_actq16(
    input heatvit_s32_t accum_value,
    input heatvit_s32_t bias_value
  );
    logic signed [32:0] wide;
    logic signed [5:0]  src_exp;
    int diff;
    logic [5:0]  d;
    logic [33:0] mag_u;
    logic [33:0] rounded;
    logic signed [33:0] rsigned;
    wide = $signed({accum_value[31], accum_value}) +
           (bias_en ? $signed({bias_value[31], bias_value}) : 33'sd0);
    src_exp = src0_scale + src1_scale;
    diff = -16 - int'(src_exp);
    if (diff == 0) begin
      return clamp_q8_16_from_s33(wide);
    end else if (diff > 0) begin
      d = diff[5:0];
      mag_u = (wide < 33'sd0) ? (34'h200000000 - {1'b0, wide}) : {1'b0, wide};
      rounded = (mag_u + (34'd1 << (d - 6'd1))) >> d;
      rsigned = (wide < 33'sd0) ? -$signed(rounded) : $signed(rounded);
      return clamp_q8_16_from_s33(rsigned);
    end else begin
      if (-diff >= 23) begin
        return (wide == 33'sd0) ? 24'sd0 :
               ((wide < 33'sd0) ? -24'sd8388608 : 24'sd8388607);
      end else begin
        return clamp_q8_16_from_s55(
          $signed({{22{wide[32]}}, wide}) <<< (-diff));
      end
    end
  endfunction

  int errors;
  int trials;
  logic signed [32:0] sum33;
  logic signed [23:0] val24;
  heatvit_s32_t acc, bias;
  int s0, s1, sd;

  initial begin
    errors = 0;
    trials = 0;
    bias_en = 1'b1;
    post_op = 3'd0;

    // Full sweep over all scale triples and targeted values.
    for (s0 = -32; s0 <= 31; s0++) begin
      src0_scale = s0;
      for (s1 = -32; s1 <= 31; s1++) begin
        src1_scale = s1;
        for (sd = -32; sd <= 31; sd++) begin
          dst_scale = sd;
          for (int t = 0; t < 32; t++) begin
            case (t)
              0: sum33 = 0;
              1: sum33 = 1;
              2: sum33 = -1;
              3: sum33 = 127;
              4: sum33 = 128;
              5: sum33 = -128;
              6: sum33 = -129;
              7: sum33 = 33'sh7FFFFFFF;
              8: sum33 = -33'sd2147483648;
              9: sum33 = 33'sh100000000;
              10: sum33 = 33'sh100000001;
              11: sum33 = 33'shFFFFFFFE;
              12: sum33 = 33'sh0FFFFFFF;
              13: sum33 = 33'sh10000000;
              14: sum33 = -33'sd536870912;
              15: sum33 = 33'sd4294967295;
              16: sum33 = 33'sh0FFFFFF8;
              17: sum33 = 33'sh10000008;
              18: sum33 = 33'shFFFFFFF0;
              19: sum33 = 33'sh00010080;
              20: sum33 = 33'shFFFF00FF;
              21: sum33 = 33'sh10000003;
              22: sum33 = 33'sh10000002;
              23: sum33 = 33'sh0000FFFF;
              default: sum33 = 33'($urandom);
            endcase
            val24 = 24'(sum33[23:0]);
            acc = sum33[31:0];
            bias = {sum33[16:0], sum33[31:17]};
            trials = trials + 4;
            if (orig_out8(sum33) !== new_out8(sum33)) begin
              $display("MISMATCH out8 sum=%0d s0=%0d s1=%0d sd=%0d: orig=%0d new=%0d",
                       sum33, s0, s1, sd, orig_out8(sum33), new_out8(sum33));
              errors++;
            end
            if (orig_out32(sum33) !== new_out32(sum33)) begin
              $display("MISMATCH out32 sum=%0d s0=%0d s1=%0d sd=%0d: orig=%0d new=%0d",
                       sum33, s0, s1, sd, orig_out32(sum33), new_out32(sum33));
              errors++;
            end
            if (orig_gelu8(val24) !== new_gelu8(val24)) begin
              $display("MISMATCH gelu8 val=%0d sd=%0d: orig=%0d new=%0d",
                       val24, sd, orig_gelu8(val24), new_gelu8(val24));
              errors++;
            end
            if (orig_actq16(acc, bias) !== new_actq16(acc, bias)) begin
              $display("MISMATCH actq16 acc=%0d bias=%0d s0=%0d s1=%0d: orig=%0d new=%0d",
                       acc, bias, s0, s1, orig_actq16(acc, bias),
                       new_actq16(acc, bias));
              errors++;
            end
            if (errors > 10) begin
              $display("DIAG_FAIL");
              $finish;
            end
          end
        end
      end
    end
    $display("DIAG_DONE trials=%0d errors=%0d", trials, errors);
    if (errors != 0) begin
      $display("DIAG_FAIL");
      $finish;
    end
    $display("DIAG_PASS");
    $finish;
  end
endmodule
