`timescale 1ns / 1ps

// P5 repro: directed divisions from the failing e2e LN2 row 0.
module tb_udiv_p5_repro;
  import tb_pkg::*;

  logic        clk   = 1'b0;
  logic        rst_n = 1'b0;

  logic        u_start;
  logic        u_busy;
  logic        u_done;
  logic        u_div_zero;
  logic [63:0] u_num;
  logic [63:0] u_den;
  logic [63:0] u_quot;
  logic [63:0] u_rem;

  heatvit_udiv #(.NUM_W(64), .DEN_W(64), .QUOT_W(64)) dut_udiv (
    .clk            (clk),
    .rst_n          (rst_n),
    .start          (u_start),
    .busy           (u_busy),
    .done           (u_done),
    .divide_by_zero (u_div_zero),
    .numerator      (u_num),
    .denominator    (u_den),
    .quotient       (u_quot),
    .remainder      (u_rem)
  );

  always #5 clk = ~clk;

  task automatic udiv_run(input logic [63:0] n, input logic [63:0] d);
    u_num = n;
    u_den = d;
    u_start = 1'b1;
    @(posedge clk);
    #1;
    u_start = 1'b0;
    wait (u_done);
  endtask

  int fails = 0;
  task automatic check_case(
    input logic [63:0] n, input logic [63:0] d,
    input logic [63:0] exp_q, input logic [63:0] exp_r,
    input string name
  );
    udiv_run(n, d);
    if (u_quot !== exp_q || u_rem !== exp_r) begin
      $display("FAIL %s: num=%0d den=%0d got q=%0d r=%0d expect q=%0d r=%0d",
               name, n, d, u_quot, u_rem, exp_q, exp_r);
      fails = fails + 1;
    end else begin
      $display("PASS %s: q=%0d r=%0d", name, u_quot, u_rem);
    end
  endtask

  initial begin
    u_start = 1'b0;
    u_num = 64'd0;
    u_den = 64'd0;
    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
    #1;

    // Block-12 LN2 row 0 e2: sum_square / 192.
    check_case(64'h7FC0000000, 64'd192, 64'd2857719125, 64'd64,
               "b12_e2");
    // The quotient the e2e actually captured.
    check_case(64'h7FA8C0000000, 64'd192, 64'd2855621973, 64'd64,
               "b12_e2_implied");
    // Block-11 LN2 row 0 e2 (x_scale=-3 path): raw divider result.
    check_case(64'h6680000000, 64'd192, 64'd2292886186, 64'd128,
               "b11_e2");
    // Mean division.
    check_case(64'h5880000000, 64'd192, 64'd1979711488, 64'd0,
               "b12_mean");
    // inv divisions (correct std 44104, observed std 44080).
    check_case(64'h1_0000_0000_0000, 64'd44104, 64'd6382073660, 64'd10016,
               "b12_inv_correct");
    check_case(64'h1_0000_0000_0000, 64'd44080, 64'd6385548473, 64'd20816,
               "b12_inv_observed");

    if (fails == 0) $display("TEST_PASS tb_udiv_p5_repro");
    else begin
      $display("TEST_FAIL tb_udiv_p5_repro fails=%0d", fails);
      $fatal(1, "divider repro failed");
    end
    $finish;
  end

endmodule
