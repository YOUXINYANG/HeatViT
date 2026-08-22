`timescale 1ns / 1ps

// Task 5/6: descriptor-driven tiled GEMM engine against the behavioral memory.
//
// One xsim invocation runs one scenario selected by +CASE. The scenario's
// tensor images are loaded into the input/weight/output segments at runtime,
// the engine is started with the matching descriptor, and after done the
// entire output segment is compared byte-for-byte with the expected image.
// A monitor rejects any observed memory command outside the declared regions.
module tb_gemm_engine;
  import heatvit_pkg::*;
  import tb_pkg::*;
  import gemm_cases_pkg::*;

  // Largest scenario images across all generated cases.
  localparam int MAX_INPUT_BYTES  = 37824;
  localparam int MAX_WEIGHT_BYTES = 36864;
  localparam int MAX_OUTPUT_BYTES = 37824;
  localparam int MAX_OUTPUT_WORDS = MAX_OUTPUT_BYTES / 8;

  logic        clk   = 1'b0;
  logic        rst_n = 1'b0;
  logic [15:0] stall_mask = 16'h0000;

  logic           cmd_valid;
  logic           cmd_ready;
  heatvit_desc_t  desc_d;
  logic           busy;
  logic           done;
  logic           error_valid;
  logic [7:0]     error_code;
  logic [2:0][31:0] mac_active;

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
  logic        load_valid;
  logic        load_ready;
  logic [1:0]  load_seg;
  logic [31:0] load_bytes;
  string       load_file;

  // Case configuration selected at runtime by +CASE.
  logic [15:0] t_m;
  logic [15:0] t_n;
  logic [15:0] t_k;
  logic [3:0]  t_heads;
  logic [23:0] t_flags;
  logic [31:0] t_src0_off;
  logic [31:0] t_src1_off;
  logic [31:0] t_bias_off;
  logic [31:0] t_dst_off;
  logic signed [5:0] t_src0_scale;
  logic signed [5:0] t_src1_scale;
  logic signed [5:0] t_dst_scale;
  logic [31:0] t_input_bytes;
  logic [31:0] t_weight_bytes;
  logic [31:0] t_output_bytes;
  logic [7:0]  t_opcode;
  logic [3:0]  t_reserved;
  logic        err_case;
  logic [7:0]  t_err_code;
  string       t_a_file;
  string       t_w_file;
  string       t_dst_file;
  string       t_exp_file;

  heatvit_gemm_engine dut (
    .clk              (clk),
    .rst_n            (rst_n),
    .cmd_valid        (cmd_valid),
    .cmd_ready        (cmd_ready),
    .desc             (desc_d),
    .input_base       (INPUT_BASE),
    .input_bytes      (t_input_bytes),
    .weight_base      (WEIGHT_BASE),
    .weight_bytes     (t_weight_bytes),
    .scratch_base     (SCRATCH_BASE),
    .scratch_bytes    (32'h00000000),
    .output_base      (OUTPUT_BASE),
    .output_bytes     (t_output_bytes),
    .busy             (busy),
    .done             (done),
    .error_valid      (error_valid),
    .error_code       (error_code),
    .mac_active_cycles(mac_active),
    .mem_cmd_valid    (mem_cmd_valid),
    .mem_cmd_ready    (mem_cmd_ready),
    .mem_cmd_write    (mem_cmd_write),
    .mem_cmd_addr     (mem_cmd_addr),
    .mem_cmd_len      (mem_cmd_len),
    .mem_w_valid      (mem_w_valid),
    .mem_w_ready      (mem_w_ready),
    .mem_w_data       (mem_w_data),
    .mem_w_strb       (mem_w_strb),
    .mem_w_last       (mem_w_last),
    .mem_r_valid      (mem_r_valid),
    .mem_r_ready      (mem_r_ready),
    .mem_r_data       (mem_r_data),
    .mem_r_last       (mem_r_last)
  );

  behavioral_memory #(
    .SEG_COUNT  (3),
    .SEG0_FILE  (""),
    .SEG0_BASE  (INPUT_BASE),
    .SEG0_BYTES (MAX_INPUT_BYTES),
    .SEG1_FILE  (""),
    .SEG1_BASE  (WEIGHT_BASE),
    .SEG1_BYTES (MAX_WEIGHT_BYTES),
    .SEG2_FILE  (""),
    .SEG2_BASE  (OUTPUT_BASE),
    .SEG2_BYTES (MAX_OUTPUT_BYTES),
    .LFSR_INIT  (16'hD00D)
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
    .dbg_w_valid  (1'b0),
    .dbg_w_addr   (32'h00000000),
    .dbg_w_data   (8'h00),
    .load_valid   (load_valid),
    .load_ready   (load_ready),
    .load_seg     (load_seg),
    .load_bytes   (load_bytes),
    .load_file    (load_file)
  );

  always #5 clk = ~clk;

  logic [63:0] exp_words [0:MAX_OUTPUT_WORDS-1];
  int i;
  int b;
  logic [7:0] want_byte;
  logic [63:0] cmd_end;
  logic        saw_cmd;
  string case_str;
  string stall_str;
  logic [15:0] parsed_mask;

  task automatic load_segment(
    input logic [1:0]  seg,
    input logic [31:0] bytes,
    input string       file
  );
    load_seg   = seg;
    load_bytes = bytes;
    load_file  = file;
    load_valid = 1'b1;
    wait (load_ready);
    @(posedge clk);
    #1;
    load_valid = 1'b0;
  endtask

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      saw_cmd <= 1'b0;
    end else if (obs_cmd_valid) begin
      saw_cmd <= 1'b1;
      cmd_end = obs_cmd_addr + {13'd0, obs_cmd_len, 3'b000};
      if (!(((obs_cmd_addr >= INPUT_BASE) &&
             (cmd_end <= INPUT_BASE + t_input_bytes)) ||
            ((obs_cmd_addr >= WEIGHT_BASE) &&
             (cmd_end <= WEIGHT_BASE + t_weight_bytes)) ||
            ((obs_cmd_addr >= OUTPUT_BASE) &&
             (cmd_end <= OUTPUT_BASE + t_output_bytes)))) begin
        $display("trace access outside declared regions: addr=%h len=%0d",
                 obs_cmd_addr, obs_cmd_len);
        tb_fatal("memory trace outside declared regions");
      end
    end
  end

  always_ff @(posedge clk) begin
    if (error_valid) begin
      $display("ENGINE ERROR t=%0t code=%0d state=%0d", $time, error_code, dut.state);
    end
  end

  initial begin
    #100000000;
    $display("WATCHDOG: engine state=%0d m0=%0d n0=%0d ld_kind=%0d ld_bank=%0d ld_idx=%0d",
             dut.state, dut.m0, dut.n0, dut.ld_kind, dut.ld_bank, dut.ld_idx);
    $display("WATCHDOG: mem state=%0d beats=%0d master state=%0d beats=%0d",
             dut_mem.state, dut_mem.beat_count, dut.u_master.state,
             dut.u_master.beat_count);
    $display("WATCHDOG: req_valid=%0d req_ready=%0d r_valid=%0d r_ready=%0d",
             dut.req_valid, dut.req_ready, dut.req_r_valid, dut.req_r_ready);
    tb_fatal("tb_gemm_engine watchdog");
  end

  initial begin
    cmd_valid  = 1'b0;
    desc_d     = '0;
    dbg_valid  = 1'b0;
    dbg_addr   = 32'h00000000;
    load_valid = 1'b0;
    load_seg   = 2'd0;
    load_bytes = 32'h00000000;
    load_file  = "";
    t_opcode   = OP_GEMM;
    t_reserved = 4'd0;
    err_case   = 1'b0;
    t_err_code = 8'd0;

    for (i = 0; i < MAX_OUTPUT_WORDS; i++) exp_words[i] = 64'h0000000000000000;

    if (!$value$plusargs("CASE=%s", case_str)) case_str = "ordinary";
    if (!$value$plusargs("STALL_MASK=%s", stall_str)) stall_str = "0";
    if ($sscanf(stall_str, "%h", parsed_mask) != 1) parsed_mask = 16'h0000;
    stall_mask = parsed_mask;

    case (case_str)
      "mini": begin
        t_m = MINI_M; t_n = MINI_N; t_k = MINI_K; t_heads = MINI_HEADS;
        t_flags = MINI_FLAGS;
        t_src0_off = MINI_SRC0_OFFSET; t_src1_off = MINI_SRC1_OFFSET;
        t_bias_off = MINI_BIAS_OFFSET; t_dst_off = MINI_DST_OFFSET;
        t_src0_scale = MINI_SRC0_SCALE; t_src1_scale = MINI_SRC1_SCALE;
        t_dst_scale = MINI_DST_SCALE;
        t_input_bytes = MINI_INPUT_BYTES; t_weight_bytes = MINI_WEIGHT_BYTES;
        t_output_bytes = MINI_OUTPUT_BYTES;
        t_a_file = MINI_A_FILE; t_w_file = MINI_W_FILE;
        t_dst_file = MINI_DST_FILE; t_exp_file = MINI_EXP_FILE;
      end
      "ordinary": begin
        t_m = ORDINARY_M; t_n = ORDINARY_N; t_k = ORDINARY_K;
        t_heads = ORDINARY_HEADS; t_flags = ORDINARY_FLAGS;
        t_src0_off = ORDINARY_SRC0_OFFSET; t_src1_off = ORDINARY_SRC1_OFFSET;
        t_bias_off = ORDINARY_BIAS_OFFSET; t_dst_off = ORDINARY_DST_OFFSET;
        t_src0_scale = ORDINARY_SRC0_SCALE; t_src1_scale = ORDINARY_SRC1_SCALE;
        t_dst_scale = ORDINARY_DST_SCALE;
        t_input_bytes = ORDINARY_INPUT_BYTES; t_weight_bytes = ORDINARY_WEIGHT_BYTES;
        t_output_bytes = ORDINARY_OUTPUT_BYTES;
        t_a_file = ORDINARY_A_FILE; t_w_file = ORDINARY_W_FILE;
        t_dst_file = ORDINARY_DST_FILE; t_exp_file = ORDINARY_EXP_FILE;
      end
      "full": begin
        t_m = FULL_M; t_n = FULL_N; t_k = FULL_K; t_heads = FULL_HEADS;
        t_flags = FULL_FLAGS;
        t_src0_off = FULL_SRC0_OFFSET; t_src1_off = FULL_SRC1_OFFSET;
        t_bias_off = FULL_BIAS_OFFSET; t_dst_off = FULL_DST_OFFSET;
        t_src0_scale = FULL_SRC0_SCALE; t_src1_scale = FULL_SRC1_SCALE;
        t_dst_scale = FULL_DST_SCALE;
        t_input_bytes = FULL_INPUT_BYTES; t_weight_bytes = FULL_WEIGHT_BYTES;
        t_output_bytes = FULL_OUTPUT_BYTES;
        t_a_file = FULL_A_FILE; t_w_file = FULL_W_FILE;
        t_dst_file = FULL_DST_FILE; t_exp_file = FULL_EXP_FILE;
      end
      "tail": begin
        t_m = TAIL_M; t_n = TAIL_N; t_k = TAIL_K; t_heads = TAIL_HEADS;
        t_flags = TAIL_FLAGS;
        t_src0_off = TAIL_SRC0_OFFSET; t_src1_off = TAIL_SRC1_OFFSET;
        t_bias_off = TAIL_BIAS_OFFSET; t_dst_off = TAIL_DST_OFFSET;
        t_src0_scale = TAIL_SRC0_SCALE; t_src1_scale = TAIL_SRC1_SCALE;
        t_dst_scale = TAIL_DST_SCALE;
        t_input_bytes = TAIL_INPUT_BYTES; t_weight_bytes = TAIL_WEIGHT_BYTES;
        t_output_bytes = TAIL_OUTPUT_BYTES;
        t_a_file = TAIL_A_FILE; t_w_file = TAIL_W_FILE;
        t_dst_file = TAIL_DST_FILE; t_exp_file = TAIL_EXP_FILE;
      end
      "large": begin
        t_m = LARGE_M; t_n = LARGE_N; t_k = LARGE_K; t_heads = LARGE_HEADS;
        t_flags = LARGE_FLAGS;
        t_src0_off = LARGE_SRC0_OFFSET; t_src1_off = LARGE_SRC1_OFFSET;
        t_bias_off = LARGE_BIAS_OFFSET; t_dst_off = LARGE_DST_OFFSET;
        t_src0_scale = LARGE_SRC0_SCALE; t_src1_scale = LARGE_SRC1_SCALE;
        t_dst_scale = LARGE_DST_SCALE;
        t_input_bytes = LARGE_INPUT_BYTES; t_weight_bytes = LARGE_WEIGHT_BYTES;
        t_output_bytes = LARGE_OUTPUT_BYTES;
        t_a_file = LARGE_A_FILE; t_w_file = LARGE_W_FILE;
        t_dst_file = LARGE_DST_FILE; t_exp_file = LARGE_EXP_FILE;
      end
      "head": begin
        t_m = HEAD_M; t_n = HEAD_N; t_k = HEAD_K; t_heads = HEAD_HEADS;
        t_flags = HEAD_FLAGS;
        t_src0_off = HEAD_SRC0_OFFSET; t_src1_off = HEAD_SRC1_OFFSET;
        t_bias_off = HEAD_BIAS_OFFSET; t_dst_off = HEAD_DST_OFFSET;
        t_src0_scale = HEAD_SRC0_SCALE; t_src1_scale = HEAD_SRC1_SCALE;
        t_dst_scale = HEAD_DST_SCALE;
        t_input_bytes = HEAD_INPUT_BYTES; t_weight_bytes = HEAD_WEIGHT_BYTES;
        t_output_bytes = HEAD_OUTPUT_BYTES;
        t_a_file = HEAD_A_FILE; t_w_file = HEAD_W_FILE;
        t_dst_file = HEAD_DST_FILE; t_exp_file = HEAD_EXP_FILE;
      end
      "transpose": begin
        t_m = TRANSPOSE_M; t_n = TRANSPOSE_N; t_k = TRANSPOSE_K;
        t_heads = TRANSPOSE_HEADS; t_flags = TRANSPOSE_FLAGS;
        t_src0_off = TRANSPOSE_SRC0_OFFSET; t_src1_off = TRANSPOSE_SRC1_OFFSET;
        t_bias_off = TRANSPOSE_BIAS_OFFSET; t_dst_off = TRANSPOSE_DST_OFFSET;
        t_src0_scale = TRANSPOSE_SRC0_SCALE; t_src1_scale = TRANSPOSE_SRC1_SCALE;
        t_dst_scale = TRANSPOSE_DST_SCALE;
        t_input_bytes = TRANSPOSE_INPUT_BYTES; t_weight_bytes = TRANSPOSE_WEIGHT_BYTES;
        t_output_bytes = TRANSPOSE_OUTPUT_BYTES;
        t_a_file = TRANSPOSE_A_FILE; t_w_file = TRANSPOSE_W_FILE;
        t_dst_file = TRANSPOSE_DST_FILE; t_exp_file = TRANSPOSE_EXP_FILE;
      end
      "unsigned": begin
        t_m = UNSIGNED_M; t_n = UNSIGNED_N; t_k = UNSIGNED_K;
        t_heads = UNSIGNED_HEADS; t_flags = UNSIGNED_FLAGS;
        t_src0_off = UNSIGNED_SRC0_OFFSET; t_src1_off = UNSIGNED_SRC1_OFFSET;
        t_bias_off = UNSIGNED_BIAS_OFFSET; t_dst_off = UNSIGNED_DST_OFFSET;
        t_src0_scale = UNSIGNED_SRC0_SCALE; t_src1_scale = UNSIGNED_SRC1_SCALE;
        t_dst_scale = UNSIGNED_DST_SCALE;
        t_input_bytes = UNSIGNED_INPUT_BYTES; t_weight_bytes = UNSIGNED_WEIGHT_BYTES;
        t_output_bytes = UNSIGNED_OUTPUT_BYTES;
        t_a_file = UNSIGNED_A_FILE; t_w_file = UNSIGNED_W_FILE;
        t_dst_file = UNSIGNED_DST_FILE; t_exp_file = UNSIGNED_EXP_FILE;
      end
      "err_heads": begin
        t_m = ORDINARY_M; t_n = ORDINARY_N; t_k = ORDINARY_K;
        t_heads = 4'd2; t_flags = ORDINARY_FLAGS | 24'h000020;
        t_src0_off = ORDINARY_SRC0_OFFSET; t_src1_off = ORDINARY_SRC1_OFFSET;
        t_bias_off = ORDINARY_BIAS_OFFSET; t_dst_off = ORDINARY_DST_OFFSET;
        t_src0_scale = ORDINARY_SRC0_SCALE; t_src1_scale = ORDINARY_SRC1_SCALE;
        t_dst_scale = ORDINARY_DST_SCALE;
        t_input_bytes = ORDINARY_INPUT_BYTES; t_weight_bytes = ORDINARY_WEIGHT_BYTES;
        t_output_bytes = ORDINARY_OUTPUT_BYTES;
        t_a_file = ""; t_w_file = ""; t_dst_file = ""; t_exp_file = "";
        err_case = 1'b1; t_err_code = ERR_DIMENSION;
      end
      "err_flag18": begin
        t_m = ORDINARY_M; t_n = ORDINARY_N; t_k = ORDINARY_K;
        t_heads = 4'd0; t_flags = ORDINARY_FLAGS | 24'h040000;
        t_src0_off = ORDINARY_SRC0_OFFSET; t_src1_off = ORDINARY_SRC1_OFFSET;
        t_bias_off = ORDINARY_BIAS_OFFSET; t_dst_off = ORDINARY_DST_OFFSET;
        t_src0_scale = ORDINARY_SRC0_SCALE; t_src1_scale = ORDINARY_SRC1_SCALE;
        t_dst_scale = ORDINARY_DST_SCALE;
        t_input_bytes = ORDINARY_INPUT_BYTES; t_weight_bytes = ORDINARY_WEIGHT_BYTES;
        t_output_bytes = ORDINARY_OUTPUT_BYTES;
        t_a_file = ""; t_w_file = ""; t_dst_file = ""; t_exp_file = "";
        err_case = 1'b1; t_err_code = ERR_DIMENSION;
      end
      "err_reserved": begin
        t_m = ORDINARY_M; t_n = ORDINARY_N; t_k = ORDINARY_K;
        t_heads = ORDINARY_HEADS; t_flags = ORDINARY_FLAGS;
        t_src0_off = ORDINARY_SRC0_OFFSET; t_src1_off = ORDINARY_SRC1_OFFSET;
        t_bias_off = ORDINARY_BIAS_OFFSET; t_dst_off = ORDINARY_DST_OFFSET;
        t_src0_scale = ORDINARY_SRC0_SCALE; t_src1_scale = ORDINARY_SRC1_SCALE;
        t_dst_scale = ORDINARY_DST_SCALE;
        t_input_bytes = ORDINARY_INPUT_BYTES; t_weight_bytes = ORDINARY_WEIGHT_BYTES;
        t_output_bytes = ORDINARY_OUTPUT_BYTES;
        t_a_file = ""; t_w_file = ""; t_dst_file = ""; t_exp_file = "";
        t_reserved = 4'd1;
        err_case = 1'b1; t_err_code = ERR_DIMENSION;
      end
      "err_opcode": begin
        t_m = ORDINARY_M; t_n = ORDINARY_N; t_k = ORDINARY_K;
        t_heads = ORDINARY_HEADS; t_flags = ORDINARY_FLAGS;
        t_src0_off = ORDINARY_SRC0_OFFSET; t_src1_off = ORDINARY_SRC1_OFFSET;
        t_bias_off = ORDINARY_BIAS_OFFSET; t_dst_off = ORDINARY_DST_OFFSET;
        t_src0_scale = ORDINARY_SRC0_SCALE; t_src1_scale = ORDINARY_SRC1_SCALE;
        t_dst_scale = ORDINARY_DST_SCALE;
        t_input_bytes = ORDINARY_INPUT_BYTES; t_weight_bytes = ORDINARY_WEIGHT_BYTES;
        t_output_bytes = ORDINARY_OUTPUT_BYTES;
        t_a_file = ""; t_w_file = ""; t_dst_file = ""; t_exp_file = "";
        t_opcode = 8'hff;
        err_case = 1'b1; t_err_code = ERR_OPCODE;
      end
      default: begin
        $display("unknown CASE=%s", case_str);
        tb_fatal("unknown tb_gemm_engine case");
      end
    endcase

    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
    #1;

    if (!err_case) begin
      load_segment(2'd0, t_input_bytes, t_a_file);
      load_segment(2'd1, t_weight_bytes, t_w_file);
      load_segment(2'd2, t_output_bytes, t_dst_file);
      $readmemh(t_exp_file, exp_words);
    end

    desc_d = '0;
    desc_d.opcode = t_opcode;
    desc_d.flags  = t_flags;
    desc_d.m      = t_m;
    desc_d.n      = t_n;
    desc_d.k      = t_k;
    desc_d.heads  = t_heads;
    desc_d.src0_offset = t_src0_off;
    desc_d.src1_offset = t_src1_off;
    desc_d.bias_offset = t_bias_off;
    desc_d.dst_offset  = t_dst_off;
    desc_d.src0_scale_exp = t_src0_scale;
    desc_d.src1_scale_exp = t_src1_scale;
    desc_d.dst_scale_exp  = t_dst_scale;
    desc_d.reserved       = t_reserved;

    cmd_valid = 1'b1;
    wait (cmd_ready);
    @(posedge clk);
    #1;
    cmd_valid = 1'b0;
    if (!busy) tb_fatal("engine did not assert busy after command");

    if (err_case) begin
      @(posedge clk);
      #1;
      if (!error_valid) tb_fatal("illegal descriptor did not assert error_valid");
      if (error_code != t_err_code) begin
        $display("error code=%0d want=%0d", error_code, t_err_code);
        tb_fatal("illegal descriptor wrong error code");
      end
      if (saw_cmd) tb_fatal("illegal descriptor issued a memory command");
      if (busy) tb_fatal("engine still busy after rejecting descriptor");
      $display("TEST_PASS tb_gemm_engine");
      $finish;
    end

    wait (done);
    #1;
    if (error_valid) begin
      $display("engine error_valid with code=%0d", error_code);
      tb_fatal("gemm engine reported an error");
    end
    if (busy) tb_fatal("engine still busy after done");

    for (b = 0; b < 3; b++) begin
      if (mac_active[b] == 32'd0) begin
        $display("mac bank %0d never accumulated", b);
        tb_fatal("inactive MAC bank");
      end
    end

    for (i = 0; i < t_output_bytes; i++) begin
      dbg_valid = 1'b1;
      dbg_addr  = OUTPUT_BASE + i;
      @(posedge clk);
      #1;
      want_byte = exp_words[i / 8][8 * (i % 8) +: 8];
      if (dbg_data !== want_byte) begin
        $display("dst byte %0d mismatch: got=%h want=%h", i, dbg_data, want_byte);
        tb_fatal("gemm dst image mismatch");
      end
      dbg_valid = 1'b0;
    end

    $display("TEST_PASS tb_gemm_engine");
    $finish;
  end

endmodule
