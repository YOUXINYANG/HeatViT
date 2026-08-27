`timescale 1ns / 1ps

// P5 repro: two LN tokens back-to-back with DIFFERENT input scales and a
// repeated first element, to expose the stale continuous-assign square
// (square_q32_of reads module register x_scale_r as a free variable).
module tb_ln_p5_stale;
  import heatvit_pkg::*;
  import tb_pkg::*;

  logic          clk   = 1'b0;
  logic          rst_n = 1'b0;

  logic           cfg_valid;
  logic           cfg_ready;
  heatvit_scale_t cfg_x_scale;
  heatvit_scale_t cfg_gamma_scale;
  heatvit_scale_t cfg_beta_scale;
  heatvit_scale_t cfg_out_scale;
  logic           busy;
  logic           done;
  logic           warn_negative_variance;

  logic        in_valid;
  logic        in_ready;
  heatvit_s8_t in_x;
  heatvit_s8_t in_gamma;
  heatvit_s8_t in_beta;

  logic        out_valid;
  logic        out_ready;
  heatvit_s8_t out_data;

  logic [2:0]  a_req_valid;
  logic [2:0]  a_req_ready;
  logic [63:0] a_num [2:0];
  logic [63:0] a_den [2:0];
  logic [2:0]  a_rsp_valid;
  logic [63:0] a_quot [2:0];
  logic [63:0] a_rem [2:0];
  logic [2:0]  a_div_zero;

  heatvit_layernorm dut (
    .clk                    (clk),
    .rst_n                  (rst_n),
    .cfg_valid              (cfg_valid),
    .cfg_ready              (cfg_ready),
    .cfg_x_scale_exp        (cfg_x_scale),
    .cfg_gamma_scale_exp    (cfg_gamma_scale),
    .cfg_beta_scale_exp     (cfg_beta_scale),
    .cfg_out_scale_exp      (cfg_out_scale),
    .busy                   (busy),
    .done                   (done),
    .warn_negative_variance (warn_negative_variance),
    .in_valid               (in_valid),
    .in_ready               (in_ready),
    .in_x                   (in_x),
    .in_gamma               (in_gamma),
    .in_beta                (in_beta),
    .out_valid              (out_valid),
    .out_ready              (out_ready),
    .out_data               (out_data),
    .div_req_valid          (a_req_valid[0]),
    .div_req_ready          (a_req_ready[0]),
    .div_num                (a_num[0]),
    .div_den                (a_den[0]),
    .div_rsp_valid          (a_rsp_valid[0]),
    .div_quot               (a_quot[0]),
    .div_rem                (a_rem[0]),
    .div_div_zero           (a_div_zero[0])
  );

  heatvit_div_arbiter #(.NUM_W(64), .DEN_W(64), .QUOT_W(64)) dut_arb (
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

  integer dump_fd;
  integer token_idx = 0;
  initial begin
    dump_fd = $fopen("build/vectors/e2e_p5/img0/diag6_stale.txt", "w");
  end

  // Capture every accumulated square during both tokens.
  always @(posedge clk) begin
    if (rst_n && dut.state == 4'd1 && dut.in_valid && dut.in_ready) begin
      $fwrite(dump_fd, "T%0d %0d %0d %0d %0d\n", token_idx,
              dut.idx, $signed(dut.in_x),
              $signed(dut.square_in_w[63:0]), $signed(dut.sum_square_r));
    end
  end

  heatvit_s8_t row_x [192];
  heatvit_s8_t row_gamma [192];
  heatvit_s8_t row_beta [192];

  task automatic run_token(input heatvit_scale_t x_scale);
    int sent;
    token_idx = token_idx + 1;
    cfg_x_scale = x_scale;
    cfg_gamma_scale = -6'sd6;
    cfg_beta_scale  = -6'sd6;
    cfg_out_scale   = -6'sd4;
    cfg_valid = 1'b1;
    wait (cfg_ready);
    in_x     = row_x[0];
    in_gamma = row_gamma[0];
    in_beta  = row_beta[0];
    in_valid <= 1'b1;
    @(posedge clk);
    #1;
    cfg_valid = 1'b0;
    sent = 0;
    while (sent < 192) begin
      wait (in_ready);
      in_x     = row_x[sent];
      in_gamma = row_gamma[sent];
      in_beta  = row_beta[sent];
      @(posedge clk);
      #1;
      sent++;
    end
    in_valid = 1'b0;
    wait (out_valid);
    out_ready = 1'b1;
    wait (done);
    out_ready = 1'b0;
    @(posedge clk);
    #1;
    if (busy) tb_fatal("busy not cleared after done");
  endtask

  int j;
  initial begin
    cfg_valid = 1'b0;
    in_valid  = 1'b0;
    out_ready = 1'b0;
    in_x = 8'sd0; in_gamma = 8'sd0; in_beta = 8'sd0;
    for (j = 0; j < 192; j++) begin
      row_x[j] = 8'sd0;
      row_gamma[j] = 8'sd64;
      row_beta[j] = 8'sd0;
    end
    // Token 1 at scale -3: last element -1 (so the presented value just
    // before token 2's first handshake is -1).
    row_x[0] = -8'sd1;
    row_x[191] = -8'sd1;
    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
    #1;
    run_token(-6'sd3);
    // Token 2 at scale -2: first TWO elements -1 (same value as token 1's
    // last presented element), third -5.
    row_x[0] = -8'sd1;
    row_x[1] = -8'sd1;
    row_x[2] = -8'sd5;
    row_x[191] = 8'sd0;
    run_token(-6'sd2);

    $display("TEST_PASS tb_ln_p5_stale");
    $finish;
  end

endmodule
