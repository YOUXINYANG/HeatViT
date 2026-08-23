`timescale 1ns / 1ps

module tb_pkg_smoke;
  import heatvit_pkg::*;

  heatvit_desc_t desc;

  initial begin
    if ($bits(desc) != 320) $fatal(1, "descriptor width=%0d", $bits(desc));
    if (GELU_SLOPE_NUM_Q16 != 11) $fatal(1, "GELU_SLOPE_NUM_Q16");
    if (GELU_SLOPE_SHIFT != 4) $fatal(1, "GELU_SLOPE_SHIFT");
    if (GELU_SLOPE_ROUND_ADD != 15) $fatal(1, "GELU_SLOPE_ROUND_ADD");
    if (GELU_EXP_NEG_Q_MAX != 16) $fatal(1, "GELU_EXP_NEG_Q_MAX");
    if (GELU_EXP_POS_Q_MAX != 7) $fatal(1, "GELU_EXP_POS_Q_MAX");
    if (LN_EPS_Q32 != 48'd4295) $fatal(1, "LN_EPS_Q32");
    $display("TEST_PASS tb_pkg_smoke");
    $finish;
  end

endmodule
