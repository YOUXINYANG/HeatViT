`timescale 1ns / 1ps

module tb_udiv_isqrt;
  import tb_pkg::*;

  localparam int VEC_SLOTS = 4096;
  localparam string UDIV_PATH = "sim/vectors/divsqrt/udiv.mem";
  localparam string ISQRT_PATH = "sim/vectors/divsqrt/isqrt.mem";

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

  logic        s_start;
  logic        s_busy;
  logic        s_done;
  logic [47:0] s_rad;
  logic [23:0] s_root;
  logic [47:0] s_rem;

  logic [2:0]  a_req_valid;
  logic [2:0]  a_req_ready;
  logic [63:0] a_num [2:0];
  logic [63:0] a_den [2:0];
  logic [2:0]  a_rsp_valid;
  logic [63:0] a_quot [2:0];
  logic [63:0] a_rem [2:0];
  logic [2:0]  a_div_zero;

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

  heatvit_isqrt #(.RAD_W(48)) dut_isqrt (
    .clk       (clk),
    .rst_n     (rst_n),
    .start     (s_start),
    .busy      (s_busy),
    .done      (s_done),
    .radicand  (s_rad),
    .root      (s_root),
    .remainder (s_rem)
  );

  heatvit_div_arbiter #(.NUM_W(64), .DEN_W(64), .QUOT_W(64)) dut_arbiter (
    .clk       (clk),
    .rst_n     (rst_n),
    .req_valid (a_req_valid),
    .req_ready (a_req_ready),
    .num       (a_num),
    .den       (a_den),
    .rsp_valid (a_rsp_valid),
    .quot      (a_quot),
    .rem       (a_rem),
    .div_zero  (a_div_zero)
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

  task automatic isqrt_run(input logic [47:0] rad);
    s_rad = rad;
    s_start = 1'b1;
    @(posedge clk);
    #1;
    s_start = 1'b0;
    wait (s_done);
  endtask

  logic [63:0] prev_q;
  logic [63:0] prev_r;
  int i;
  int n;
  int d;
  logic [63:0] exp_q;
  logic [63:0] exp_r;
  logic [47:0] rad;
  logic [23:0] exp_root;
  logic [47:0] exp_rem;

  logic [255:0] uvec [VEC_SLOTS];
  logic [127:0] svec [VEC_SLOTS];
  int u_count;
  int s_count;
  int udiv_limit = 4096;
  int isqrt_limit = 1024;
  int pair;

  initial begin
    if ($test$plusargs("fast")) begin
      udiv_limit  = 256;
      isqrt_limit = 64;
    end
    if (udiv_limit < 0 || udiv_limit > 4096) udiv_limit = 4096;
    if (isqrt_limit < 0 || isqrt_limit > 1024) isqrt_limit = 1024;

    u_start = 1'b0;
    u_num = 64'd0;
    u_den = 64'd0;
    s_start = 1'b0;
    s_rad = 48'd0;
    for (i = 0; i < 3; i++) begin
      a_req_valid[i] = 1'b0;
      a_num[i] = 64'd0;
      a_den[i] = 64'd0;
    end

    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
    #1;

    // Exhaustive small-width division, including divide-by-zero.
    prev_q = 64'd0;
    prev_r = 64'd0;
    for (pair = 0; pair < udiv_limit; pair++) begin
      n = pair / 64;
      d = pair % 64;
      if (pair % 500 == 0) $display("udiv progress %0d/%0d", pair, udiv_limit);
      udiv_run(64'(n), 64'(d));
      if (d == 0) begin
        if (!u_div_zero) tb_fatal("udiv: missing divide_by_zero");
        if (u_quot !== prev_q || u_rem !== prev_r) tb_fatal("udiv: zero-div updated result");
      end else begin
        if (u_div_zero) tb_fatal("udiv: spurious divide_by_zero");
        if (u_quot != 64'(n / d) || u_rem != 64'(n % d)) begin
          $display("udiv mismatch: %0d/%0d got q=%0d r=%0d", n, d, u_quot, u_rem);
          tb_fatal("udiv exhaustive mismatch");
        end
      end
      prev_q = u_quot;
      prev_r = u_rem;
    end

    // Exhaustive small-width square root with invariant checks.
    for (i = 0; i < isqrt_limit; i++) begin
      if (i % 500 == 0) $display("isqrt progress %0d/%0d", i, isqrt_limit);
      isqrt_run(48'(i));
      if (48'(s_root) * 48'(s_root) > 48'(i)) begin
        $display("isqrt root too large: rad=%0d root=%0d", i, s_root);
        tb_fatal("isqrt invariant root^2 <= radicand");
      end
      if (48'(s_root + 1) * 48'(s_root + 1) <= 48'(i)) begin
        $display("isqrt root too small: rad=%0d root=%0d", i, s_root);
        tb_fatal("isqrt invariant radicand < (root+1)^2");
      end
      if (s_rem != 48'(i) - 48'(s_root) * 48'(s_root)) tb_fatal("isqrt remainder mismatch");
    end

    // Vector-driven 64-bit division cases.
    $readmemh(UDIV_PATH, uvec);
    u_count = uvec[0][15:0];
    if (u_count <= 0) tb_fatal("missing udiv vectors");
    for (i = 1; i <= u_count; i++) begin
      udiv_run(uvec[i][63:0], uvec[i][127:64]);
      if (u_div_zero) tb_fatal("udiv vector: unexpected divide_by_zero");
      if (u_quot != uvec[i][191:128] || u_rem != uvec[i][255:192]) begin
        $display("udiv vector[%0d] mismatch: q=%0d r=%0d", i, u_quot, u_rem);
        tb_fatal("udiv vector mismatch");
      end
    end

    // Vector-driven 48-bit square root cases.
    $readmemh(ISQRT_PATH, svec);
    s_count = svec[0][15:0];
    if (s_count <= 0) tb_fatal("missing isqrt vectors");
    for (i = 1; i <= s_count; i++) begin
      isqrt_run(svec[i][47:0]);
      if (s_root != svec[i][79:48] || s_rem != svec[i][127:80]) begin
        $display("isqrt vector[%0d] mismatch: root=%0d rem=%0d", i, s_root, s_rem);
        tb_fatal("isqrt vector mismatch");
      end
    end

    // Three simultaneous arbiter requests must be served 0, 1, 2 in order.
    for (i = 0; i < 3; i++) a_req_valid[i] = 1'b1;
    a_num[0] = 64'd10;  a_den[0] = 64'd3;
    a_num[1] = 64'd20;  a_den[1] = 64'd6;
    a_num[2] = 64'd100; a_den[2] = 64'd7;
    #1;
    if (!a_req_ready[0]) tb_fatal("arbiter: no immediate grant to client 0");
    @(posedge clk);
    #1;
    a_req_valid[0] = 1'b0;

    wait (a_rsp_valid[0]);
    if ({a_rsp_valid[2], a_rsp_valid[1], a_rsp_valid[0]} != 3'b001)
      tb_fatal("arbiter: first response not one-hot to client 0");
    if (a_quot[0] != 64'd3 || a_rem[0] != 64'd1 || a_div_zero[0]) tb_fatal("arbiter: client 0 wrong");

    wait (a_req_ready[1]);
    @(posedge clk);
    #1;
    a_req_valid[1] = 1'b0;

    wait (a_rsp_valid[1]);
    if ({a_rsp_valid[2], a_rsp_valid[1], a_rsp_valid[0]} != 3'b010)
      tb_fatal("arbiter: second response not one-hot to client 1");
    if (a_quot[1] != 64'd3 || a_rem[1] != 64'd2 || a_div_zero[1]) tb_fatal("arbiter: client 1 wrong");

    wait (a_req_ready[2]);
    @(posedge clk);
    #1;
    a_req_valid[2] = 1'b0;

    wait (a_rsp_valid[2]);
    if ({a_rsp_valid[2], a_rsp_valid[1], a_rsp_valid[0]} != 3'b100)
      tb_fatal("arbiter: third response not one-hot to client 2");
    if (a_quot[2] != 64'd14 || a_rem[2] != 64'd2 || a_div_zero[2]) tb_fatal("arbiter: client 2 wrong");

    $display("TEST_PASS tb_udiv_isqrt");
    $finish;
  end

endmodule
