`timescale 1ns / 1ps

module tb_gelu_plan;
  import heatvit_pkg::*;
  import tb_pkg::*;

  localparam int VEC_SLOTS = 4096;
  localparam string VEC_PATH = "sim/vectors/nonlinear/nonlinear.mem";

  logic           clk   = 1'b0;
  logic           rst_n = 1'b0;

  logic           g_start;
  logic           g_busy;
  logic           g_done;
  heatvit_q8_16_t g_x;
  heatvit_q8_16_t g_y;

  logic            p_start;
  logic            p_busy;
  logic            p_done;
  heatvit_q8_16_t  p_x;
  heatvit_uq0_16_t p_y;

  heatvit_gelu dut_gelu (
    .clk   (clk),
    .rst_n (rst_n),
    .start (g_start),
    .busy  (g_busy),
    .done  (g_done),
    .x_in  (g_x),
    .y_out (g_y)
  );

  heatvit_plan_sigmoid dut_plan (
    .clk   (clk),
    .rst_n (rst_n),
    .start (p_start),
    .busy  (p_busy),
    .done  (p_done),
    .x_in  (p_x),
    .y_out (p_y)
  );

  always #5 clk = ~clk;

  task automatic gelu_run(input heatvit_q8_16_t x);
    g_x = x;
    g_start = 1'b1;
    @(posedge clk);
    #1;
    g_start = 1'b0;
    wait (g_done);
    #1;
  endtask

  task automatic plan_run(input heatvit_q8_16_t x);
    p_x = x;
    p_start = 1'b1;
    @(posedge clk);
    #1;
    p_start = 1'b0;
    wait (p_done);
    #1;
  endtask

  logic [64:0] vec [VEC_SLOTS];
  int vec_count;
  int i;
  heatvit_q8_16_t  x_q16;
  heatvit_q8_16_t  exp_gelu;
  heatvit_uq0_16_t exp_plan;

  initial begin
    g_start = 1'b0;
    g_x     = 24'sd0;
    p_start = 1'b0;
    p_x     = 24'sd0;

    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
    #1;

    // Protocol and zero-input checks before the vector sweep.
    g_x = 24'sd0;
    g_start = 1'b1;
    @(posedge clk);
    #1;
    if (!g_busy) tb_fatal("gelu: busy not asserted after start");
    g_start = 1'b0;
    wait (g_done);
    #1;
    if (g_busy) tb_fatal("gelu: busy not cleared after done");
    if (g_y !== 24'sd0) begin
      $display("gelu(0) mismatch: got=%0d expected=0", g_y);
      tb_fatal("gelu zero-input mismatch");
    end

    p_x = 24'sd0;
    p_start = 1'b1;
    @(posedge clk);
    #1;
    if (!p_busy) tb_fatal("plan: busy not asserted after start");
    p_start = 1'b0;
    wait (p_done);
    #1;
    if (p_busy) tb_fatal("plan: busy not cleared after done");
    if (p_y !== 17'd32768) begin
      $display("plan(0) mismatch: got=%0d expected=32768", p_y);
      tb_fatal("plan zero-input mismatch");
    end

    $readmemh(VEC_PATH, vec);
    vec_count = vec[0][15:0];
    if (vec_count <= 0) begin
      $display("could not read %s", VEC_PATH);
      tb_fatal("missing nonlinear vectors");
    end
    for (i = 1; i <= vec_count; i++) begin
      x_q16    = $signed(vec[i][23:0]);
      exp_gelu = $signed(vec[i][47:24]);
      exp_plan = vec[i][64:48];

      gelu_run(x_q16);
      if (g_y !== exp_gelu) begin
        $display("gelu vector[%0d] mismatch: x=%0d got=%0d expected=%0d",
                 i, x_q16, g_y, exp_gelu);
        tb_fatal("gelu vector mismatch");
      end

      plan_run(x_q16);
      if (p_y !== exp_plan) begin
        $display("plan vector[%0d] mismatch: x=%0d got=%0d expected=%0d",
                 i, x_q16, p_y, exp_plan);
        tb_fatal("plan vector mismatch");
      end
    end

    $display("TEST_PASS tb_gelu_plan");
    $finish;
  end

endmodule
