`timescale 1ns / 1ps

// Task 3: heatvit_top reset / start / busy-start smoke test.
//
// No memory model is attached (mem_cmd_ready stays low) so the first
// descriptor parks inside the executor; the smoke verifies the reset
// state, the legal-start latch of the four region registers (peeked
// hierarchically), and the busy-start error path: error 7, no done, and
// busy clearing after the executor drains its accepted burst.
module tb_heatvit_top_smoke;
  import heatvit_pkg::*;
  import tb_pkg::*;

  logic        clk   = 1'b0;
  logic        rst_n = 1'b0;

  logic        start;
  logic [31:0] input_base;
  logic [31:0] input_bytes;
  logic [31:0] weight_base;
  logic [31:0] weight_bytes;
  logic [31:0] scratch_base;
  logic [31:0] scratch_bytes;
  logic [31:0] output_base;
  logic [31:0] output_bytes;
  logic        busy;
  logic        done;
  logic        error_valid;
  logic [7:0]  error_code;
  logic [7:0]  warning_flags;
  logic [5:0]  output_scale_exp;
  logic        mem_cmd_valid;
  logic        mem_cmd_ready;
  logic        mem_cmd_write;
  logic [31:0] mem_cmd_addr;
  logic [15:0] mem_cmd_len;
  logic        mem_w_valid;
  logic        mem_w_ready;
  logic [63:0] mem_w_data;
  logic [7:0]  mem_w_strb;
  logic        mem_w_last;
  logic        mem_r_valid;
  logic        mem_r_ready;
  logic [63:0] mem_r_data;
  logic        mem_r_last;

  heatvit_top dut (
    .clk             (clk),
    .rst_n           (rst_n),
    .start           (start),
    .input_base      (input_base),
    .input_bytes     (input_bytes),
    .weight_base     (weight_base),
    .weight_bytes    (weight_bytes),
    .scratch_base    (scratch_base),
    .scratch_bytes   (scratch_bytes),
    .output_base     (output_base),
    .output_bytes    (output_bytes),
    .busy            (busy),
    .done            (done),
    .error_valid     (error_valid),
    .error_code      (error_code),
    .warning_flags   (warning_flags),
    .output_scale_exp(output_scale_exp),
    .mem_cmd_valid   (mem_cmd_valid),
    .mem_cmd_ready   (mem_cmd_ready),
    .mem_cmd_write   (mem_cmd_write),
    .mem_cmd_addr    (mem_cmd_addr),
    .mem_cmd_len     (mem_cmd_len),
    .mem_w_valid     (mem_w_valid),
    .mem_w_ready     (mem_w_ready),
    .mem_w_data      (mem_w_data),
    .mem_w_strb      (mem_w_strb),
    .mem_w_last      (mem_w_last),
    .mem_r_valid     (mem_r_valid),
    .mem_r_ready     (mem_r_ready),
    .mem_r_data      (mem_r_data),
    .mem_r_last      (mem_r_last)
  );

  always #5 clk = ~clk;

  task automatic pulse_start();
    start = 1'b1;
    @(posedge clk);
    #1;
    start = 1'b0;
  endtask

  initial begin
    #10000000;
    $display("WATCHDOG: busy=%0b error=%0b", busy, error_valid);
    tb_fatal("tb_heatvit_top_smoke watchdog");
  end

  initial begin
    start         = 1'b0;
    input_base    = 32'h00000000;
    input_bytes   = 32'd0;
    weight_base   = 32'h01000000;
    weight_bytes  = 32'd0;
    scratch_base  = 32'h02000000;
    scratch_bytes = 32'd0;
    output_base   = 32'h03000000;
    output_bytes  = 32'd0;
    mem_cmd_ready = 1'b0;
    mem_w_ready   = 1'b0;
    mem_r_valid   = 1'b0;
    mem_r_data    = 64'd0;
    mem_r_last    = 1'b0;

    // Reset checks: two cycles after sync reset deassert, all status is
    // quiet.
    repeat (2) @(posedge clk);
    if (busy || done || error_valid || warning_flags != 8'd0) begin
      $display("reset state busy=%b done=%b err=%b warn=%h", busy, done,
               error_valid, warning_flags);
      tb_fatal("reset state not quiet");
    end
    repeat (2) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
    #1;
    if (busy || done || error_valid || warning_flags != 8'd0) begin
      $display("post-reset busy=%b done=%b err=%b warn=%h", busy, done,
               error_valid, warning_flags);
      tb_fatal("post-reset state not quiet");
    end

    // Legal start latches the region registers.
    input_base    = 32'h11110000;
    input_bytes   = 32'd150528;
    weight_base   = 32'h22220000;
    weight_bytes  = 32'd5828104;
    scratch_base  = 32'h33330000;
    scratch_bytes = 32'd1543752;
    output_base   = 32'h44440000;
    output_bytes  = 32'd4000;
    pulse_start();
    @(posedge clk);
    #1;
    if (!busy) tb_fatal("legal start must assert busy");
    // Change the inputs next cycle: the latched regs must keep the old
    // values (verified hierarchically).
    input_base  = 32'hDEADBEEF;
    weight_base = 32'hCAFEBABE;
    scratch_base = 32'hBADC0DE0;
    output_base = 32'h0BADF00D;
    @(posedge clk);
    #1;
    if (dut.input_base_r    != 32'h11110000 ||
        dut.weight_base_r   != 32'h22220000 ||
        dut.scratch_base_r  != 32'h33330000 ||
        dut.output_base_r   != 32'h44440000 ||
        dut.input_bytes_r   != 32'd150528 ||
        dut.weight_bytes_r  != 32'd5828104 ||
        dut.scratch_bytes_r != 32'd1543752 ||
        dut.output_bytes_r  != 32'd4000) begin
      $display("latched regs: %h %h %h %h / %0d %0d %0d %0d",
               dut.input_base_r, dut.weight_base_r, dut.scratch_base_r,
               dut.output_base_r, dut.input_bytes_r, dut.weight_bytes_r,
               dut.scratch_bytes_r, dut.output_bytes_r);
      tb_fatal("region latch mismatch");
    end

    // Busy start: error 7, no done, busy clears after the drain.
    @(posedge clk);
    #1;
    if (!dut.u_executor.busy) begin
      // The executor may not have accepted the first descriptor yet; wait
      // until it is busy before the busy-start injection.
      wait (dut.u_executor.busy);
      @(posedge clk);
      #1;
    end
    pulse_start();
    wait (error_valid);
    #1;
    if (error_code != ERR_BUSY_START) begin
      $display("busy-start error_code=%0d expected=7", error_code);
      tb_fatal("busy start must report error 7");
    end
    if (done) tb_fatal("done must not pulse on busy-start error");
    wait (!busy);
    @(posedge clk);
    #1;
    if (done) tb_fatal("done must not pulse after the abort drain");

    $display("TEST_PASS tb_heatvit_top_smoke");
    $finish;
  end

endmodule
