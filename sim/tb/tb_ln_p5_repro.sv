`timescale 1ns / 1ps

// P5 repro: run the captured block-12 LN2 row 0 through the standalone
// heatvit_layernorm unit. Dumps per-element (idx, in_x, square_in_w,
// running sum_square_r) during S_LOAD_ACCUM and the final sums at
// S_VARIANCE, then compares the 192 outputs against golden expectations
// (non-fatal, reports every mismatch).
module tb_ln_p5_repro;
  import heatvit_pkg::*;
  import tb_pkg::*;

  localparam int VEC_SLOTS = 65536;
  localparam string VEC_PATH = "sim/vectors/p5_ln/row0.mem";
  localparam string DUMP_PATH = "build/vectors/e2e_p5/img0/diag4_ln_elem.txt";

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

  // ---- per-element capture ------------------------------------------------
  integer       dump_fd;
  integer       dump_count = 0;
  logic         dumped_final = 1'b0;
  initial begin
    dump_fd = $fopen(DUMP_PATH, "w");
    if (dump_fd == 0) tb_fatal("cannot open dump file");
  end
  always @(posedge clk) begin
    if (rst_n && dut.state == 4'd1 && dut.in_valid && dut.in_ready &&
        dump_count < 192) begin
      $fwrite(dump_fd, "%0d %0d %0d %0d\n", dut.idx, $signed(dut.in_x),
              $signed(dut.square_in_w[63:0]), $signed(dut.sum_square_r));
      dump_count = dump_count + 1;
    end
    if (rst_n && dut.state == 4'd3 && !dumped_final) begin
      $fwrite(dump_fd, "FINAL %0d %0d\n", $signed(dut.sum_x_r),
              $signed(dut.sum_square_r));
      $fwrite(dump_fd, "MEAN %0d E2 %0d MSQ %0d VAR %0d\n",
              $signed(dut.mean_q32_r), $signed(dut.e2_q32_r),
              $signed(dut.mean_square_w), $signed(dut.variance_w));
      dumped_final = 1'b1;
    end
  end

  // ---- single-token driver ------------------------------------------------
  heatvit_s8_t row_x [192];
  heatvit_s8_t row_gamma [192];
  heatvit_s8_t row_beta [192];
  heatvit_s8_t exp_out [192];

  logic [31:0] vec [VEC_SLOTS];
  int j;
  int pos;
  int sent;
  int got;
  int mismatch_count;

  initial begin
    cfg_valid = 1'b0;
    in_valid  = 1'b0;
    out_ready = 1'b0;
    in_x = 8'sd0; in_gamma = 8'sd0; in_beta = 8'sd0;
    mismatch_count = 0;

    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
    #1;

    for (j = 0; j < VEC_SLOTS; j++) vec[j] = 32'h0;
    $readmemh(VEC_PATH, vec);
    if (vec[0][15:0] != 16'd1) begin
      $display("bad vector header: %0d tokens", vec[0][15:0]);
      tb_fatal("missing p5_ln repro vectors");
    end
    pos = 1;
    cfg_x_scale    = $signed(vec[pos][5:0]);
    cfg_gamma_scale = $signed(vec[pos][11:6]);
    cfg_beta_scale  = $signed(vec[pos][17:12]);
    cfg_out_scale   = $signed(vec[pos][23:18]);
    for (j = 0; j < 192; j++) begin
      row_x[j]     = $signed(vec[pos + 1 + j][7:0]);
      row_gamma[j] = $signed(vec[pos + 1 + j][15:8]);
      row_beta[j]  = $signed(vec[pos + 1 + j][23:16]);
      exp_out[j]   = $signed(vec[pos + 1 + j][31:24]);
    end
    $display("scales: x=%0d g=%0d b=%0d out=%0d",
             cfg_x_scale, cfg_gamma_scale, cfg_beta_scale, cfg_out_scale);

    cfg_valid = 1'b1;
    wait (cfg_ready);
    // Vector-engine style presentation: register the first element and
    // assert data-valid at the same edge as the cfg handshake.
    in_x     = row_x[0];
    in_gamma = row_gamma[0];
    in_beta  = row_beta[0];
    in_valid <= 1'b1;
    @(posedge clk);
    #1;
    cfg_valid = 1'b0;
    if (!busy) tb_fatal("busy not asserted after cfg");

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
    got = 0;
    while (got < 192) begin
      if (out_valid && out_ready) begin
        if (out_data !== exp_out[got]) begin
          if (mismatch_count < 32)
            $display("beat[%0d] mismatch: got=%0d expected=%0d",
                     got, $signed(out_data), $signed(exp_out[got]));
          mismatch_count = mismatch_count + 1;
        end
        got++;
      end
      @(posedge clk);
      #1;
    end
    out_ready = 1'b0;
    wait (done);
    #1;
    if (busy) tb_fatal("busy not cleared after done");
    $display("warn_negative_variance=%0d (expected %0d)",
             warn_negative_variance, vec[pos][24]);
    $display("out mismatches: %0d / 192", mismatch_count);
    $display("captured elements: %0d", dump_count);
    if (dump_count != 192) tb_fatal("element capture incomplete");
    $display("TEST_PASS tb_ln_p5_repro");
    $finish;
  end

endmodule
