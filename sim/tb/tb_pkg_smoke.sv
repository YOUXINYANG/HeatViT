`timescale 1ns / 1ps

module tb_pkg_smoke;
  import heatvit_pkg::*;

  heatvit_desc_t desc;

  initial begin
    if ($bits(desc) != 320) $fatal(1, "descriptor width=%0d", $bits(desc));
    if (GELU_A_Q16 != -18927) $fatal(1, "GELU_A_Q16");
    if (LN_EPS_Q32 != 48'd4295) $fatal(1, "LN_EPS_Q32");
    $display("TEST_PASS tb_pkg_smoke");
    $finish;
  end

endmodule
