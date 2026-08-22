`ifndef TB_PKG_SV
`define TB_PKG_SV

// Shared testbench helpers used by the self-checking HeatViT-T testbenches.
package tb_pkg;

  task automatic tb_check(input logic ok, input string msg);
    if (!ok) begin
      $display("TB_CHECK_FAIL: %s", msg);
      $fatal(1, "tb_check failed: %s", msg);
    end
  endtask

  task automatic tb_fatal(input string msg);
    $fatal(1, "%s", msg);
  endtask

endpackage

`endif
