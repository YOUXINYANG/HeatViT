`timescale 1ns / 1ps

module tb_requant_residual;
  import heatvit_pkg::*;
  import tb_pkg::*;

  localparam int VEC_SLOTS = 4096;
  localparam string VEC_PATH = "sim/vectors/requant/requant.mem";

  logic          clk   = 1'b0;
  logic          rst_n = 1'b0;

  heatvit_s48_t  req_value;
  heatvit_scale_t req_src;
  heatvit_scale_t req_dst;
  heatvit_s8_t   req_out;
  logic          req_sat;

  logic          res_main_valid;
  logic          res_main_ready;
  heatvit_s8_t   res_main_value;
  heatvit_scale_t res_main_scale;
  logic          res_aux_valid;
  logic          res_aux_ready;
  heatvit_s8_t   res_aux_value;
  heatvit_scale_t res_aux_scale;
  heatvit_scale_t res_out_scale;
  logic          res_out_valid;
  logic          res_out_ready;
  heatvit_s8_t   res_out_value;

  heatvit_requant dut_requant (
    .in_value      (req_value),
    .src_scale_exp (req_src),
    .dst_scale_exp (req_dst),
    .out_value     (req_out),
    .saturated     (req_sat)
  );

  heatvit_residual dut_residual (
    .clk            (clk),
    .rst_n          (rst_n),
    .main_valid     (res_main_valid),
    .main_ready     (res_main_ready),
    .main_value     (res_main_value),
    .main_scale_exp (res_main_scale),
    .aux_valid      (res_aux_valid),
    .aux_ready      (res_aux_ready),
    .aux_value      (res_aux_value),
    .aux_scale_exp  (res_aux_scale),
    .out_scale_exp  (res_out_scale),
    .out_valid      (res_out_valid),
    .out_ready      (res_out_ready),
    .out_value      (res_out_value)
  );

  always #5 clk = ~clk;

  task automatic check_requant(
    input heatvit_s48_t value,
    input heatvit_scale_t src,
    input heatvit_scale_t dst,
    input heatvit_s8_t expected
  );
    req_value = value;
    req_src   = src;
    req_dst   = dst;
    #1;
    if (req_out !== expected) begin
      $display("requant mismatch: value=%0d src=%0d dst=%0d got=%0d expected=%0d",
               value, src, dst, req_out, expected);
      tb_fatal("requant boundary mismatch");
    end
  endtask

  task automatic check_residual(
    input heatvit_s8_t main_value,
    input heatvit_scale_t main_scale,
    input heatvit_s8_t aux_value,
    input heatvit_scale_t aux_scale,
    input heatvit_scale_t out_scale,
    input heatvit_s8_t expected
  );
    res_main_valid  = 1'b1;
    res_main_value  = main_value;
    res_main_scale  = main_scale;
    res_aux_valid   = 1'b1;
    res_aux_value   = aux_value;
    res_aux_scale   = aux_scale;
    res_out_scale   = out_scale;
    res_out_ready   = 1'b1;
    @(posedge clk);  // fire: stage 1 latches the pair
    @(posedge clk);  // stage 2 presents the result (P7-5 two-stage pipeline)
    #1;
    if (!res_out_valid) tb_fatal("residual: missing out_valid");
    if (res_out_value !== expected) begin
      $display("residual mismatch: got=%0d expected=%0d", res_out_value, expected);
      tb_fatal("residual boundary mismatch");
    end
    res_main_valid = 1'b0;
    res_aux_valid  = 1'b0;
    @(posedge clk);  // trailing re-presentation (out_ready still asserted)
    @(posedge clk);  // consumed -> out_valid clears
    #1;
  endtask

  logic [67:0] vec [VEC_SLOTS];
  int          vec_count;
  int          i;
  heatvit_s128_t wide;
  heatvit_s128_t scaled;

  initial begin
    req_value = 48'sd0;
    req_src   = 6'sd0;
    req_dst   = 6'sd0;
    res_main_valid  = 1'b0;
    res_main_value  = 8'sd0;
    res_main_scale  = 6'sd0;
    res_aux_valid   = 1'b0;
    res_aux_value   = 8'sd0;
    res_aux_scale   = 6'sd0;
    res_out_scale   = 6'sd0;
    res_out_ready   = 1'b0;

    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    check_requant(48'sd1,    6'sd0, 6'sd1,   8'sd1);
    check_requant(-48'sd1,   6'sd0, 6'sd1,  -8'sd1);
    check_requant(48'sd3,    6'sd0, 6'sd1,   8'sd2);
    check_requant(-48'sd3,   6'sd0, 6'sd1,  -8'sd2);
    check_requant(48'sd1024, 6'sd0, 6'sd0,   8'sd127);
    if (!req_sat) tb_fatal("requant: saturation flag not set for 1024");

    $readmemh(VEC_PATH, vec);
    vec_count = vec[0][15:0];
    if (vec_count <= 0) begin
      $display("could not read %s", VEC_PATH);
      tb_fatal("missing requant vectors");
    end
    for (i = 1; i <= vec_count; i++) begin
      req_value = $signed(vec[i][47:0]);
      req_src   = $signed(vec[i][53:48]);
      req_dst   = $signed(vec[i][59:54]);
      #1;
      if (req_out !== $signed(vec[i][67:60])) begin
        $display("vector[%0d] mismatch: got=%0d expected=%0d",
                 i, req_out, $signed(vec[i][67:60]));
        tb_fatal("requant vector mismatch");
      end
      wide   = $signed({{80 {vec[i][47]}}, vec[i][47:0]});
      scaled = scale_to_exp_s128(wide, req_src, req_dst);
      if (req_sat !== ((scaled > $signed(8'sd127)) || (scaled < $signed(-8'sd128)))) begin
        $display("vector[%0d] saturation flag mismatch", i);
        tb_fatal("requant saturation flag mismatch");
      end
    end

    check_residual(8'sd64, -6'sd7, 8'sd64, -6'sd8, -6'sd7, 8'sd96);
    check_residual(8'sd127, -6'sd7, 8'sd127, -6'sd7, -6'sd7, 8'sd127);
    check_residual(-8'sd128, -6'sd7, -8'sd128, -6'sd7, -6'sd7, -8'sd128);

    // Stall stability: hold out_ready low and verify outputs stay stable.
    res_main_valid = 1'b1;
    res_main_value = 8'sd64;
    res_main_scale = -6'sd7;
    res_aux_valid  = 1'b1;
    res_aux_value  = 8'sd64;
    res_aux_scale  = -6'sd8;
    res_out_scale  = -6'sd7;
    res_out_ready  = 1'b0;
    repeat (3) @(posedge clk);
    #1;
    if (!res_out_valid) tb_fatal("residual: output lost during stall");
    if (res_out_value !== 8'sd96) tb_fatal("residual: output unstable during stall");
    res_out_ready = 1'b1;
    @(posedge clk);
    #1;
    res_main_valid = 1'b0;
    res_aux_valid  = 1'b0;
    @(posedge clk);
    #1;

    $display("TEST_PASS tb_requant_residual");
    $finish;
  end

endmodule
