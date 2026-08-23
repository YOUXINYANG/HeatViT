`timescale 1ns / 1ps

module tb_gelu_pipe;
  import heatvit_pkg::*;
  import tb_pkg::*;

  logic           clk   = 1'b0;
  logic           rst_n = 1'b0;
  logic           start;
  logic           busy;
  logic           done;
  heatvit_q8_16_t x_in;
  heatvit_q8_16_t y_out;

  heatvit_gelu dut (
    .clk   (clk),
    .rst_n (rst_n),
    .start (start),
    .busy  (busy),
    .done  (done),
    .x_in  (x_in),
    .y_out (y_out)
  );

  always #5 clk = ~clk;

  task automatic feed(input heatvit_q8_16_t x);
    x_in  = x;
    start = 1'b1;
    @(posedge clk);
    #1;
    start = 1'b0;
  endtask

  initial begin
    start = 1'b0;
    x_in  = 24'sd0;
    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
    #1;
    $display("tb: reset released");

    // Protocol: single lane.
    feed(24'sd5410434);
    if (!busy) tb_fatal("busy not asserted after start");
    $display("tb: single lane fed, waiting done");
    wait (done);
    #1;
    if (busy) tb_fatal("busy not cleared after done");
    if (y_out !== 24'sd5368495) begin
      $display("single-lane mismatch: got=%0d expected=5368495", y_out);
      tb_fatal("gelu single-lane mismatch");
    end
    $display("tb: single lane ok");

    // Back-to-back: four lanes, one start per cycle; results must arrive
    // in order starting 41 cycles after the first start.
    @(posedge clk); #1;
    feed(24'sd32768);
    feed(24'sd8388607);
    feed(-24'sd8388608);
    feed(24'sd0);
    $display("tb: 4 lanes fed back-to-back");
    // Expected outputs in feed order. With back-to-back feeds ``done`` is
    // a level spanning one cycle per lane and ``y_out`` updates every
    // cycle, so sample lane 0 on the first asserted cycle and lanes 1..3
    // on the following clock edges.
    wait (done);
    #1;
    if (y_out !== 24'sd22817) begin
      $display("lane0 mismatch: got=%0d expected=22817", y_out);
      tb_fatal("gelu pipe lane0");
    end
    $display("tb: lane0 ok");
    for (int i = 1; i < 4; i++) begin
      @(posedge clk);
      #1;
      if (!done) begin
        $display("lane%0d: done deasserted early", i);
        tb_fatal("gelu pipe done level");
      end
      case (i)
        1: if (y_out !== 24'sd8323583) begin
             $display("lane1 mismatch: got=%0d expected=8323583", y_out);
             tb_fatal("gelu pipe lane1");
           end
        2: if (y_out !== 24'sd0) begin
             $display("lane2 mismatch: got=%0d expected=0", y_out);
             tb_fatal("gelu pipe lane2");
           end
        3: if (y_out !== 24'sd0) begin
             $display("lane3 mismatch: got=%0d expected=0", y_out);
             tb_fatal("gelu pipe lane3");
           end
      endcase
      $display("tb: lane%0d ok", i);
    end

    $display("TEST_PASS tb_gelu_pipe");
    $finish;
  end

endmodule
