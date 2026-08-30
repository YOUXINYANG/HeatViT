`timescale 1ns / 1ps

module tb_layernorm;
  import heatvit_pkg::*;
  import tb_pkg::*;

  localparam int VEC_SLOTS = 65536;
  localparam string VEC_PATH = "sim/vectors/layernorm/layernorm.mem";

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

  // 16-bit backpressure LFSR; bit 0 gives a ~50% random output stall.
  logic [15:0] lfsr = 16'hACE1;
  always_ff @(posedge clk) lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};

  heatvit_s8_t row_x [192];
  heatvit_s8_t row_gamma [192];
  heatvit_s8_t row_beta [192];
  heatvit_s8_t exp_out [192];

  task automatic run_token(
    input heatvit_scale_t x_scale,
    input heatvit_scale_t gamma_scale,
    input heatvit_scale_t beta_scale,
    input heatvit_scale_t out_scale,
    input logic            expect_warn,
    input int              stall_cycles
  );
    int sent = 0;
    int got  = 0;

    cfg_x_scale    = x_scale;
    cfg_gamma_scale = gamma_scale;
    cfg_beta_scale  = beta_scale;
    cfg_out_scale   = out_scale;
    cfg_valid = 1'b1;
    wait (cfg_ready);
    @(posedge clk);
    #1;
    cfg_valid = 1'b0;
    if (!busy) tb_fatal("layernorm: busy not asserted after cfg");

    in_valid = 1'b1;
    while (sent < 192) begin
      in_x     = row_x[sent];
      in_gamma = row_gamma[sent];
      in_beta  = row_beta[sent];
      wait (in_ready);
      @(posedge clk);
      #1;
      sent++;
    end
    in_valid = 1'b0;

    wait (out_valid);
    out_ready = 1'b0;
    repeat (stall_cycles) begin
      @(posedge clk);
      #1;
      if (!out_valid) tb_fatal("layernorm: output lost during stall");
      if (out_data !== exp_out[0]) tb_fatal("layernorm: output unstable during stall");
    end
    while (got < 192) begin
      out_ready = lfsr[0];
      if (out_valid && out_ready) begin
        if (out_data !== exp_out[got]) begin
          $display("layernorm beat[%0d] mismatch: got=%0d expected=%0d scales x=%0d g=%0d b=%0d o=%0d",
                   got, out_data, exp_out[got], cfg_x_scale, cfg_gamma_scale,
                   cfg_beta_scale, cfg_out_scale);
          tb_fatal("layernorm output mismatch");
        end
        got++;
      end
      @(posedge clk);
      #1;
    end
    out_ready = 1'b0;
    wait (done);
    #1;
    if (busy) tb_fatal("layernorm: busy not cleared after done");
    if (warn_negative_variance !== expect_warn) begin
      $display("layernorm warn mismatch: got=%0d expected=%0d",
               warn_negative_variance, expect_warn);
      tb_fatal("layernorm warning flag mismatch");
    end
    @(posedge clk);
    #1;
    if (warn_negative_variance !== expect_warn)
      tb_fatal("layernorm: warning flag not latched");
  endtask

  logic [31:0] vec [VEC_SLOTS];
  int token_count;
  int pos;
  int token;
  int j;

  initial begin
    cfg_valid = 1'b0;
    cfg_x_scale = 6'sd0;
    cfg_gamma_scale = 6'sd0;
    cfg_beta_scale = 6'sd0;
    cfg_out_scale = 6'sd0;
    in_valid = 1'b0;
    in_x = 8'sd0;
    in_gamma = 8'sd0;
    in_beta = 8'sd0;
    out_ready = 1'b0;
    for (j = 0; j < 192; j++) begin
      row_x[j] = 8'sd0;
      row_gamma[j] = 8'sd0;
      row_beta[j] = 8'sd0;
      exp_out[j] = 8'sd0;
    end

    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
    #1;

    // Directed zero-vector and stall check before the vector sweep.
    for (j = 0; j < 192; j++) begin
      row_x[j] = 8'sd0;
      row_gamma[j] = 8'sd64;
      row_beta[j] = 8'sd0;
      exp_out[j] = 8'sd0;
    end
    run_token(-6'sd7, -6'sd6, -6'sd7, -6'sd7, 1'b0, 4);

    // Directed negative-variance case (95 x 127, 97 x 90 at scale -23).
    for (j = 0; j < 192; j++) begin
      row_x[j] = (j < 95) ? 8'sd127 : 8'sd90;
      row_gamma[j] = 8'sd64;
      row_beta[j] = 8'sd0;
      exp_out[j] = 8'sd0;
    end
    run_token(-6'sd23, -6'sd6, -6'sd7, -6'sd7, 1'b1, 3);

    $readmemh(VEC_PATH, vec);
    token_count = vec[0][15:0];
    if (token_count <= 0) begin
      $display("could not read %s", VEC_PATH);
      tb_fatal("missing layernorm vectors");
    end
    pos = 1;
    for (token = 0; token < token_count; token++) begin
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
      run_token(cfg_x_scale, cfg_gamma_scale, cfg_beta_scale, cfg_out_scale,
                vec[pos][24], 0);
      pos += 193;
    end
    if (token_count < 64) tb_fatal("layernorm: fewer than 64 tokens checked");

    $display("TEST_PASS tb_layernorm");
    $finish;
  end

endmodule
