`timescale 1ns / 1ps

// Task 2: selector reductions and local/global feature concat through the
// tensor executor.
//
// Three phases run at C = 1, 5 and 196 candidates (current_token_count =
// C + 1). Each phase fills the local/candidate/global scratch buffers with
// deterministic int8 patterns through the backdoor write port, submits
// OP_REDUCE_MEAN (candidate axis), OP_REDUCE_MEAN (head-lane axis) and
// OP_CONCAT_LOCAL_GLOBAL, and compares every output byte against the same
// formulas re-computed independently here with ties-away rounding.
//
// The candidate-axis column (h=0, j=0) is anchored to c-2 so C=5 covers
// [-2,-1,0,1,2] (mean 0); the lane-axis row (c=0, h=0) is anchored to l-32
// so its sum remainder is exactly half the denominator (ties away). Invalid
// axis and concat-width descriptors must report error 2. The memory trace
// must stay inside the scratch segment.
module tb_selector_features;
  import heatvit_pkg::*;
  import tb_pkg::*;

  localparam logic [31:0] SCRATCH_BASE  = 32'h02000000;
  localparam int          SCRATCH_BYTES = 131072;

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
    .warning_pulse         (),
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

  // ---------------------------------------------------------------
  // Deterministic patterns and independent expected-value computation.
  // ---------------------------------------------------------------
  function automatic logic [7:0] local_byte(input int h, input int c,
                                            input int j);
    int v;
    if (h == 0 && j == 0) v = c - 2;
    else v = ((h * 37 + c * 53 + j * 7 + 13) % 251) - 125;
    return v[7:0];
  endfunction

  function automatic logic [7:0] cand_byte(input int c, input int h,
                                           input int l);
    int v;
    if (c == 0 && h == 0) v = l - 32;
    else v = ((c * 53 + h * 37 + l * 7 + 13) % 251) - 125;
    return v[7:0];
  endfunction

  function automatic logic [7:0] glob_byte(input int h, input int j);
    int v;
    v = 8'hAA + h * 8'h11;
    return v[7:0];
  endfunction

  function automatic logic [7:0] round_mean(input int sum, input int den);
    int mag;
    int quot;
    int rem_;
    int res;
    mag  = (sum < 0) ? -sum : sum;
    quot = mag / den;
    rem_ = mag % den;
    if (2 * rem_ >= den) quot = quot + 1;
    res = (sum < 0) ? -quot : quot;
    if (res > 127) res = 127;
    if (res < -128) res = -128;
    return res[7:0];
  endfunction

  logic [7:0] exp_global [0:95];
  logic [7:0] exp_stats  [0:196 * 3 - 1];
  logic [7:0] exp_concat [0:196 * 64 * 3 - 1];

  int C;
  int local_off, global_off, stats_off, concat_off, cand_off;
  int i;

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
      tb_fatal("selector features descriptor reported an error");
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
      tb_fatal("selector features error code mismatch");
    end
  endtask

  task automatic fill_phase(input int c_count);
    int h, c, j, l;
    for (h = 0; h < 3; h++)
      for (c = 0; c < c_count; c++)
        for (j = 0; j < 32; j++)
          dbg_w(SCRATCH_BASE + local_off + (h * c_count + c) * 32 + j,
                local_byte(h, c, j));
    for (c = 0; c < c_count; c++)
      for (h = 0; h < 3; h++)
        for (l = 0; l < 64; l++)
          dbg_w(SCRATCH_BASE + cand_off + c * 192 + h * 64 + l,
                cand_byte(c, h, l));
    for (h = 0; h < 3; h++)
      for (j = 0; j < 32; j++)
        dbg_w(SCRATCH_BASE + global_off + h * 32 + j, glob_byte(h, j));
  endtask

  task automatic compute_expected(input int c_count);
    int h, c, j, l, s;
    for (h = 0; h < 3; h++)
      for (j = 0; j < 32; j++) begin
        s = 0;
        for (c = 0; c < c_count; c++) s = s + $signed(local_byte(h, c, j));
        exp_global[h * 32 + j] = round_mean(s, c_count);
      end
    for (c = 0; c < c_count; c++)
      for (h = 0; h < 3; h++) begin
        s = 0;
        for (l = 0; l < 64; l++) s = s + $signed(cand_byte(c, h, l));
        exp_stats[c * 3 + h] = round_mean(s, 64);
      end
    for (h = 0; h < 3; h++)
      for (c = 0; c < c_count; c++) begin
        for (j = 0; j < 32; j++)
          exp_concat[(h * c_count + c) * 64 + j] = local_byte(h, c, j);
        // The global half comes from the candidate-axis reduce output.
        for (j = 0; j < 32; j++)
          exp_concat[(h * c_count + c) * 64 + 32 + j] = exp_global[h * 32 + j];
      end
  endtask

  task automatic check_byte(input int addr, input logic [7:0] want,
                            input string name);
    logic [7:0] got;
    dbg_rd(SCRATCH_BASE + addr, got);
    if (got !== want) begin
      $display("%s byte @%0d mismatch: got=%h want=%h", name, addr, got, want);
      tb_fatal("selector features checkpoint mismatch");
    end
  endtask

  task automatic check_phase(input int c_count);
    int h, c, j;
    for (h = 0; h < 3; h++)
      for (j = 0; j < 32; j++)
        check_byte(global_off + h * 32 + j, exp_global[h * 32 + j],
                   "global");
    for (c = 0; c < c_count; c++)
      for (h = 0; h < 3; h++)
        check_byte(stats_off + c * 3 + h, exp_stats[c * 3 + h], "stats");
    for (h = 0; h < 3; h++)
      for (c = 0; c < c_count; c++)
        for (j = 0; j < 64; j++)
          check_byte(concat_off + (h * c_count + c) * 64 + j,
                     exp_concat[(h * c_count + c) * 64 + j], "concat");
  endtask

  task automatic run_phase(input int c_count);
    heatvit_desc_t d;
    token_count = c_count + 1;

    // Candidate-axis reduce: local -> global.
    d = '0;
    d.opcode      = OP_REDUCE_MEAN;
    d.flags       = (1 << FLAG_DYNAMIC_M);
    d.n           = 16'd32;
    d.heads       = 4'd3;
    d.src0_offset = local_off;
    d.dst_offset  = global_off;
    d.param0      = (REDUCE_AXIS_CANDIDATES << 2) | DYN_M_CANDIDATES;
    run_desc(d);

    // Head-lane-axis reduce: candidates -> stats.
    d = '0;
    d.opcode      = OP_REDUCE_MEAN;
    d.flags       = (1 << FLAG_DYNAMIC_M);
    d.n           = 16'd64;
    d.heads       = 4'd3;
    d.src0_offset = cand_off;
    d.dst_offset  = stats_off;
    d.param0      = (REDUCE_AXIS_HEAD_LANES << 2) | DYN_M_CANDIDATES;
    run_desc(d);

    // Concat: local + global -> local_global.
    d = '0;
    d.opcode      = OP_CONCAT_LOCAL_GLOBAL;
    d.flags       = (1 << FLAG_DYNAMIC_M) | (1 << FLAG_SRC1_SCRATCH);
    d.n           = 16'd64;
    d.heads       = 4'd3;
    d.src0_offset = local_off;
    d.src1_offset = global_off;
    d.dst_offset  = concat_off;
    d.param0      = DYN_M_CANDIDATES;
    run_desc(d);

    // Invalid axis (2'b10) must fail with error 2.
    d = '0;
    d.opcode      = OP_REDUCE_MEAN;
    d.flags       = (1 << FLAG_DYNAMIC_M);
    d.n           = 16'd32;
    d.heads       = 4'd3;
    d.src0_offset = local_off;
    d.dst_offset  = global_off;
    d.param0      = (2'd2 << 2) | DYN_M_CANDIDATES;
    run_desc_err(d, ERR_DIMENSION);

    // Concat with n=32 must fail with error 2.
    d = '0;
    d.opcode      = OP_CONCAT_LOCAL_GLOBAL;
    d.flags       = (1 << FLAG_DYNAMIC_M) | (1 << FLAG_SRC1_SCRATCH);
    d.n           = 16'd32;
    d.heads       = 4'd3;
    d.src0_offset = local_off;
    d.src1_offset = global_off;
    d.dst_offset  = concat_off;
    d.param0      = DYN_M_CANDIDATES;
    run_desc_err(d, ERR_DIMENSION);
  endtask

  // ---------------------------------------------------------------
  // Trace check: every command must stay inside the scratch segment.
  // ---------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (obs_cmd_valid) begin
      if (obs_cmd_addr < SCRATCH_BASE ||
          obs_cmd_addr + {13'd0, obs_cmd_len, 3'b000} >
              SCRATCH_BASE + SCRATCH_BYTES) begin
        $display("trace addr=%h len=%0d outside scratch",
                 obs_cmd_addr, obs_cmd_len);
        tb_fatal("selector features memory trace outside scratch");
      end
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n) desc_idx <= 8'd0;
    else if (done) desc_idx <= desc_idx + 8'd1;
  end

  initial begin
    #10000000000;
    $display("WATCHDOG: desc_idx=%0d busy=%0d", desc_idx, busy);
    tb_fatal("tb_selector_features watchdog");
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
    for (i = 0; i < 96; i++) exp_global[i] = 8'h00;
    for (i = 0; i < 196 * 3; i++) exp_stats[i] = 8'h00;
    for (i = 0; i < 196 * 64 * 3; i++) exp_concat[i] = 8'h00;

    if (!$value$plusargs("STALL_MASK=%s", stall_str)) stall_str = "0";
    if ($sscanf(stall_str, "%h", parsed_mask) != 1) parsed_mask = 16'h0000;
    stall_mask = parsed_mask;

    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
    #1;

    for (C = 1; C <= 196; C = (C == 1) ? 5 : (C == 5) ? 196 : 197) begin
      local_off  = 0;
      global_off = 3 * C * 32;
      stats_off  = global_off + 96;
      concat_off = (stats_off + 3 * C + 7) & ~7;
      cand_off   = concat_off + 3 * C * 64;
      $display("phase C=%0d local=%0d global=%0d stats=%0d concat=%0d cand=%0d",
               C, local_off, global_off, stats_off, concat_off, cand_off);
      fill_phase(C);
      compute_expected(C);
      run_phase(C);
      check_phase(C);
    end

    @(posedge clk);
    #1;
    if (desc_idx != 8'd9) begin
      $display("desc_idx=%0d expected=9", desc_idx);
      tb_fatal("expected exactly nine successful descriptor completions");
    end

    $display("TEST_PASS tb_selector_features");
    $finish;
  end

endmodule
