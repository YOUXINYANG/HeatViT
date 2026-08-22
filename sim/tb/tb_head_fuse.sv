`timescale 1ns / 1ps

// Task 3: weighted three-head keep-score fusion through the tensor executor.
//
// 1024 deterministic triples (eight batches of C=128) are generated with a
// 32-bit LCG, seeded with the doc's normal case, the fused-boundary cases
// 32767/32768, and forced zero-denominator triples every 64 candidates.
// Each batch runs one OP_HEAD_FUSE descriptor; every fused byte is compared
// against an independent testbench computation (36-bit weighted numerator,
// ties-away division, saturation to 0..65536, zero-den fallback with
// denominator 3). warning_pulse bit 0 must fire exactly for the zero-den
// triples, and an invalid n must report error 2.
module tb_head_fuse;
  import heatvit_pkg::*;
  import tb_pkg::*;

  localparam logic [31:0] SCRATCH_BASE  = 32'h02000000;
  localparam int          SCRATCH_BYTES = 131072;
  localparam int          C = 128;
  localparam int          BATCHES = 8;

  logic        clk   = 1'b0;
  logic        rst_n = 1'b0;

  logic [7:0]  token_count;
  logic        abort;
  logic        desc_valid;
  logic        desc_ready;
  heatvit_desc_t desc_d;
  logic        busy;
  logic        done;
  logic        error_valid;
  logic [7:0]  error_code;
  logic        abort_done;
  logic [2:0]  warning_pulse;

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

  logic        obs_cmd_valid;
  logic        obs_cmd_write;
  logic [31:0] obs_cmd_addr;
  logic [15:0] obs_cmd_len;
  logic        dbg_valid;
  logic        dbg_ready;
  logic [31:0] dbg_addr;
  logic [7:0]  dbg_data;
  logic        dbg_w_valid;
  logic [31:0] dbg_w_addr;
  logic [7:0]  dbg_w_data;

  logic [15:0] stall_mask;
  string       stall_str;
  logic [15:0] parsed_mask;

  heatvit_tensor_executor dut (
    .clk                   (clk),
    .rst_n                 (rst_n),
    .abort                 (abort),
    .desc_valid            (desc_valid),
    .desc_ready            (desc_ready),
    .desc                  (desc_d),
    .current_token_count   (token_count),
    .current_package_present(1'b0),
    .input_base            (32'h00000000),
    .input_bytes           (32'd0),
    .weight_base           (32'h01000000),
    .weight_bytes          (32'd0),
    .scratch_base          (SCRATCH_BASE),
    .scratch_bytes         (SCRATCH_BYTES),
    .output_base           (32'h03000000),
    .output_bytes          (32'd0),
    .busy                  (busy),
    .done                  (done),
    .error_valid           (error_valid),
    .error_code            (error_code),
    .abort_done            (abort_done),
    .warning_pulse         (warning_pulse),
    .state_update_valid    (),
    .next_token_count      (),
    .next_package_present  (),
    .mem_cmd_valid         (mem_cmd_valid),
    .mem_cmd_ready         (mem_cmd_ready),
    .mem_cmd_write         (mem_cmd_write),
    .mem_cmd_addr          (mem_cmd_addr),
    .mem_cmd_len           (mem_cmd_len),
    .mem_w_valid           (mem_w_valid),
    .mem_w_ready           (mem_w_ready),
    .mem_w_data            (mem_w_data),
    .mem_w_strb            (mem_w_strb),
    .mem_w_last            (mem_w_last),
    .mem_r_valid           (mem_r_valid),
    .mem_r_ready           (mem_r_ready),
    .mem_r_data            (mem_r_data),
    .mem_r_last            (mem_r_last)
  );

  behavioral_memory #(
    .SEG_COUNT  (1),
    .SEG0_FILE  (""),
    .SEG0_BASE  (SCRATCH_BASE),
    .SEG0_BYTES (SCRATCH_BYTES),
    .LFSR_INIT  (16'hF00D)
  ) dut_mem (
    .clk          (clk),
    .rst_n        (rst_n),
    .stall_mask   (stall_mask),
    .mem_cmd_valid(mem_cmd_valid),
    .mem_cmd_ready(mem_cmd_ready),
    .mem_cmd_write(mem_cmd_write),
    .mem_cmd_addr (mem_cmd_addr),
    .mem_cmd_len  (mem_cmd_len),
    .mem_w_valid  (mem_w_valid),
    .mem_w_ready  (mem_w_ready),
    .mem_w_data   (mem_w_data),
    .mem_w_strb   (mem_w_strb),
    .mem_w_last   (mem_w_last),
    .mem_r_valid  (mem_r_valid),
    .mem_r_ready  (mem_r_ready),
    .mem_r_data   (mem_r_data),
    .mem_r_last   (mem_r_last),
    .obs_cmd_valid(obs_cmd_valid),
    .obs_cmd_write(obs_cmd_write),
    .obs_cmd_addr (obs_cmd_addr),
    .obs_cmd_len  (obs_cmd_len),
    .dbg_valid    (dbg_valid),
    .dbg_ready    (dbg_ready),
    .dbg_addr     (dbg_addr),
    .dbg_data     (dbg_data),
    .dbg_w_valid  (dbg_w_valid),
    .dbg_w_addr   (dbg_w_addr),
    .dbg_w_data   (dbg_w_data),
    .load_valid   (1'b0),
    .load_ready   (),
    .load_seg     (2'd0),
    .load_bytes   (32'd0),
    .load_file    ("")
  );

  always #5 clk = ~clk;

  logic [7:0] desc_idx;
  int         warn_count;
  int         i;

  // ---------------------------------------------------------------
  // Deterministic triple generation (32-bit LCG).
  // ---------------------------------------------------------------
  logic [31:0] lcg;

  // Independent expected fusion computation.
  function automatic logic [16:0] exp_fuse(
      input logic [16:0] s0, s1, s2, w0, w1, w2);
    logic [35:0] num;
    logic [35:0] den;
    logic [35:0] quot;
    logic [35:0] rem_;
    logic [35:0] wsum;
    wsum = {19'd0, w0} + {19'd0, w1} + {19'd0, w2};
    if (wsum == 36'd0) begin
      num = {19'd0, s0} + {19'd0, s1} + {19'd0, s2};
      den = 36'd3;
    end else begin
      num = {19'd0, s0} * {19'd0, w0} +
            {19'd0, s1} * {19'd0, w1} +
            {19'd0, s2} * {19'd0, w2};
      den = wsum;
    end
    quot = num / den;
    rem_ = num % den;
    if (36'd2 * rem_ >= den) quot = quot + 36'd1;
    if (quot > 36'd65536) quot = 36'd65536;
    return quot[16:0];
  endfunction

  logic [16:0] t_s0 [0:C * BATCHES - 1];
  logic [16:0] t_s1 [0:C * BATCHES - 1];
  logic [16:0] t_s2 [0:C * BATCHES - 1];
  logic [16:0] t_w0 [0:C * BATCHES - 1];
  logic [16:0] t_w1 [0:C * BATCHES - 1];
  logic [16:0] t_w2 [0:C * BATCHES - 1];
  logic [16:0] t_exp [0:C * BATCHES - 1];
  logic        t_zd  [0:C * BATCHES - 1];
  int          exp_zero_den;

  // ---------------------------------------------------------------
  // Tasks.
  // ---------------------------------------------------------------
  task automatic dbg_w(input logic [31:0] addr, input logic [7:0] data);
    dbg_w_valid = 1'b1;
    dbg_w_addr  = addr;
    dbg_w_data  = data;
    @(posedge clk);
    #1;
    dbg_w_valid = 1'b0;
  endtask

  task automatic dbg_rd(input logic [31:0] addr, output logic [7:0] data);
    dbg_valid = 1'b1;
    dbg_addr  = addr;
    @(posedge clk);
    #1;
    data = dbg_data;
    dbg_valid = 1'b0;
  endtask

  task automatic run_desc(input logic [319:0] word);
    desc_d     = word;
    desc_valid = 1'b1;
    wait (desc_ready);
    @(posedge clk);
    #1;
    desc_valid = 1'b0;
    wait (done || error_valid);
    #1;
    if (error_valid) begin
      $display("descriptor error code=%0d", error_code);
      tb_fatal("head fuse descriptor reported an error");
    end
  endtask

  task automatic run_desc_err(input logic [319:0] word,
                              input logic [7:0] want_code);
    desc_d     = word;
    desc_valid = 1'b1;
    wait (desc_ready);
    @(posedge clk);
    #1;
    desc_valid = 1'b0;
    wait (error_valid);
    #1;
    if (error_code != want_code) begin
      $display("error code=%0d expected=%0d", error_code, want_code);
      tb_fatal("head fuse error code mismatch");
    end
  endtask

  task automatic fill_batch(input int batch);
    int c, h;
    for (c = 0; c < C; c++) begin
      for (h = 0; h < 3; h++) begin
        // scores[h][c] = s_h at (h*C + c)*4
        dbg_w(SCRATCH_BASE + (h * C + c) * 4 + 0,
              (h == 0) ? t_s0[batch * C + c][7:0] :
              (h == 1) ? t_s1[batch * C + c][7:0] : t_s2[batch * C + c][7:0]);
        dbg_w(SCRATCH_BASE + (h * C + c) * 4 + 1,
              (h == 0) ? t_s0[batch * C + c][15:8] :
              (h == 1) ? t_s1[batch * C + c][15:8] : t_s2[batch * C + c][15:8]);
        dbg_w(SCRATCH_BASE + (h * C + c) * 4 + 2,
              (h == 0) ? t_s0[batch * C + c][16] :
              (h == 1) ? t_s1[batch * C + c][16] : t_s2[batch * C + c][16]);
        dbg_w(SCRATCH_BASE + (h * C + c) * 4 + 3, 8'h00);
        // weights[c][h] = w_h at (c*3 + h)*4
        dbg_w(SCRATCH_BASE + 3 * C * 4 + (c * 3 + h) * 4 + 0,
              (h == 0) ? t_w0[batch * C + c][7:0] :
              (h == 1) ? t_w1[batch * C + c][7:0] : t_w2[batch * C + c][7:0]);
        dbg_w(SCRATCH_BASE + 3 * C * 4 + (c * 3 + h) * 4 + 1,
              (h == 0) ? t_w0[batch * C + c][15:8] :
              (h == 1) ? t_w1[batch * C + c][15:8] : t_w2[batch * C + c][15:8]);
        dbg_w(SCRATCH_BASE + 3 * C * 4 + (c * 3 + h) * 4 + 2,
              (h == 0) ? t_w0[batch * C + c][16] :
              (h == 1) ? t_w1[batch * C + c][16] : t_w2[batch * C + c][16]);
        dbg_w(SCRATCH_BASE + 3 * C * 4 + (c * 3 + h) * 4 + 3, 8'h00);
      end
    end
  endtask

  task automatic check_batch(input int batch);
    int c;
    logic [16:0] got;
    logic [7:0]  b0, b1, b2;
    for (c = 0; c < C; c++) begin
      dbg_rd(SCRATCH_BASE + 6 * C * 4 + c * 4 + 0, b0);
      dbg_rd(SCRATCH_BASE + 6 * C * 4 + c * 4 + 1, b1);
      dbg_rd(SCRATCH_BASE + 6 * C * 4 + c * 4 + 2, b2);
      got = {1'b0, b2, b1, b0};
      if (got != t_exp[batch * C + c]) begin
        $display("fused[%0d] mismatch: got=%0d want=%0d",
                 batch * C + c, got, t_exp[batch * C + c]);
        tb_fatal("head fuse checkpoint mismatch");
      end
    end
  endtask

  task automatic run_batch(input int batch);
    heatvit_desc_t d;
    d = '0;
    d.opcode      = OP_HEAD_FUSE;
    d.flags       = (1 << FLAG_DYNAMIC_M) | (1 << FLAG_SRC1_SCRATCH);
    d.n           = 16'd3;
    d.heads       = 4'd3;
    d.src0_offset = 32'd0;
    d.src1_offset = 32'd1536;
    d.dst_offset  = 32'd3072;
    d.param0      = DYN_M_CANDIDATES;
    run_desc(d);
    check_batch(batch);
  endtask

  always_ff @(posedge clk) begin
    if (!rst_n) warn_count <= 0;
    else if (warning_pulse[0]) warn_count <= warn_count + 1;
  end

  always_ff @(posedge clk) begin
    if (!rst_n) desc_idx <= 8'd0;
    else if (done) desc_idx <= desc_idx + 8'd1;
  end

  always_ff @(posedge clk) begin
    if (obs_cmd_valid) begin
      if (obs_cmd_addr < SCRATCH_BASE ||
          obs_cmd_addr + {13'd0, obs_cmd_len, 3'b000} >
              SCRATCH_BASE + SCRATCH_BYTES) begin
        $display("trace addr=%h len=%0d outside scratch",
                 obs_cmd_addr, obs_cmd_len);
        tb_fatal("head fuse memory trace outside scratch");
      end
    end
  end

  initial begin
    #10000000000;
    $display("WATCHDOG: desc_idx=%0d busy=%0d", desc_idx, busy);
    tb_fatal("tb_head_fuse watchdog");
  end

  initial begin
    abort      = 1'b0;
    desc_valid = 1'b0;
    desc_d     = '0;
    token_count = 8'd0;
    dbg_valid  = 1'b0;
    dbg_addr   = 32'h00000000;
    dbg_w_valid = 1'b0;
    dbg_w_addr  = 32'h00000000;
    dbg_w_data  = 8'h00;
    lcg         = 32'h20260815;
    exp_zero_den = 0;
    for (i = 0; i < C * BATCHES; i++) begin
      t_zd[i] = 1'b0;
      if (i % 1024 == 0) begin
        t_s0[i] = 17'd0;      t_s1[i] = 17'd32768; t_s2[i] = 17'd65536;
        t_w0[i] = 17'd65536;  t_w1[i] = 17'd65536; t_w2[i] = 17'd0;
      end else if (i % 1024 == 1) begin
        t_s0[i] = 17'd0; t_s1[i] = 17'd0; t_s2[i] = 17'd65536;
        t_w0[i] = 17'd1; t_w1[i] = 17'd1; t_w2[i] = 17'd2;
      end else if (i % 1024 == 2) begin
        t_s0[i] = 17'd0; t_s1[i] = 17'd0; t_s2[i] = 17'd65534;
        t_w0[i] = 17'd1; t_w1[i] = 17'd1; t_w2[i] = 17'd2;
      end else begin
        lcg = (64'(lcg) * 64'd1664525 + 64'd1013904223) &
              64'h00000000FFFFFFFF;
        t_s0[i] = 17'(lcg % 65537);
        lcg = (64'(lcg) * 64'd1664525 + 64'd1013904223) &
              64'h00000000FFFFFFFF;
        t_s1[i] = 17'(lcg % 65537);
        lcg = (64'(lcg) * 64'd1664525 + 64'd1013904223) &
              64'h00000000FFFFFFFF;
        t_s2[i] = 17'(lcg % 65537);
        if ((i % 64) == 63) begin
          t_w0[i] = 17'd0; t_w1[i] = 17'd0; t_w2[i] = 17'd0;
          t_zd[i] = 1'b1;
          exp_zero_den = exp_zero_den + 1;
        end else begin
          lcg = (64'(lcg) * 64'd1664525 + 64'd1013904223) &
                64'h00000000FFFFFFFF;
          t_w0[i] = 17'(lcg % 65537);
          lcg = (64'(lcg) * 64'd1664525 + 64'd1013904223) &
                64'h00000000FFFFFFFF;
          t_w1[i] = 17'(lcg % 65537);
          lcg = (64'(lcg) * 64'd1664525 + 64'd1013904223) &
                64'h00000000FFFFFFFF;
          t_w2[i] = 17'(lcg % 65537);
        end
      end
      t_exp[i] = exp_fuse(t_s0[i], t_s1[i], t_s2[i],
                          t_w0[i], t_w1[i], t_w2[i]);
    end

    if (!$value$plusargs("STALL_MASK=%s", stall_str)) stall_str = "0";
    if ($sscanf(stall_str, "%h", parsed_mask) != 1) parsed_mask = 16'h0000;
    stall_mask = parsed_mask;

    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
    #1;

    token_count = C + 1;
    for (i = 0; i < BATCHES; i++) begin
      fill_batch(i);
      run_batch(i);
    end

    // Invalid n must report error 2.
    begin
      heatvit_desc_t d;
      d = '0;
      d.opcode      = OP_HEAD_FUSE;
      d.flags       = (1 << FLAG_DYNAMIC_M) | (1 << FLAG_SRC1_SCRATCH);
      d.n           = 16'd5;
      d.heads       = 4'd3;
      d.src0_offset = 32'd0;
      d.src1_offset = 32'd1536;
      d.dst_offset  = 32'd3072;
      d.param0      = DYN_M_CANDIDATES;
      run_desc_err(d, ERR_DIMENSION);
    end

    @(posedge clk);
    #1;
    if (desc_idx != 8'd8) begin
      $display("desc_idx=%0d expected=8", desc_idx);
      tb_fatal("expected exactly eight successful descriptor completions");
    end
    if (warn_count != exp_zero_den) begin
      $display("warning pulse count=%0d expected=%0d", warn_count,
               exp_zero_den);
      tb_fatal("head fuse warning pulse count mismatch");
    end

    $display("TEST_PASS tb_head_fuse");
    $finish;
  end

endmodule
