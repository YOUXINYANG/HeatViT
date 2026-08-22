`timescale 1ns / 1ps

// Task 2: single-descriptor tensor executor with the locked top interface.
//
// The testbench initializes input/weight/scratch through the behavioral
// memory backdoor, submits every Phase-3 opcode with the matching descriptor,
// and checks outputs byte-for-byte. Illegal descriptors must fail before any
// memory command; dynamic M/N/K overrides are proven through the produced
// tensor sizes; abort is exercised before descriptor acceptance and after a
// memory command handshake.
module tb_tensor_executor;
  import heatvit_pkg::*;
  import tb_pkg::*;

  localparam logic [31:0] IN_BASE  = 32'h00000000;
  localparam logic [31:0] WT_BASE  = 32'h01000000;
  localparam logic [31:0] SC_BASE  = 32'h02000000;
  localparam logic [31:0] OUT_BASE = 32'h03000000;
  localparam int IN_BYTES  = 150528;
  localparam int WT_BYTES  = 38576;
  localparam int SC_BYTES  = 245120;
  localparam int OUT_BYTES = 0;

  // Scratch offsets.
  localparam int OFF_PATCH_DST   = 0;       // 150528
  localparam int OFF_ADDPOS_SRC  = 150528;  // 37632
  localparam int OFF_ADDPOS_DST  = 188160;  // 37824
  localparam int OFF_QKV_SRC     = 225984;  // 7488
  localparam int OFF_QKV_DST     = 233472;  // 7488
  localparam int OFF_CONCAT_SRC  = 240960;  // 960
  localparam int OFF_CONCAT_DST  = 241920;  // 960
  localparam int OFF_GEMM_A      = 242880;  // 32
  localparam int OFF_GEMM_B      = 242912;  // unused here
  localparam int OFF_GEMM_DST    = 243088;  // 32
  localparam int OFF_LN_SRC      = 243120;  // 384
  localparam int OFF_LN_DST      = 243504;  // 384
  localparam int OFF_RESID_MAIN  = 243888;  // 384
  localparam int OFF_RESID_AUX   = 244272;  // 384
  localparam int OFF_RESID_DST   = 244656;  // 384
  localparam int OFF_SOFTMAX_SRC = 245040;  // 48
  localparam int OFF_SOFTMAX_DST = 245088;  // 12

  // Weight offsets.
  localparam int WT_POS    = 0;      // 37824
  localparam int WT_CLS    = 37824;  // 192
  localparam int WT_GAMMA  = 38016;  // 192
  localparam int WT_BETA   = 38208;  // 192
  localparam int WT_GEMM_B = 38400;  // 176

  logic        clk   = 1'b0;
  logic        rst_n = 1'b0;
  logic [15:0] stall_mask = 16'h0000;

  logic        abort;
  logic        desc_valid;
  logic        desc_ready;
  heatvit_desc_t desc_d;
  logic [7:0]  token_count;
  logic        package_present;
  logic        busy;
  logic        done;
  logic        error_valid;
  logic [7:0]  error_code;
  logic        abort_done;
  logic [2:0]  warning_pulse;
  logic        state_update_valid;
  logic [7:0]  next_token_count;
  logic        next_package_present;

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

  heatvit_tensor_executor dut (
    .clk                   (clk),
    .rst_n                 (rst_n),
    .abort                 (abort),
    .desc_valid            (desc_valid),
    .desc_ready            (desc_ready),
    .desc                  (desc_d),
    .current_token_count   (token_count),
    .current_package_present(package_present),
    .input_base            (IN_BASE),
    .input_bytes           (IN_BYTES),
    .weight_base           (WT_BASE),
    .weight_bytes          (WT_BYTES),
    .scratch_base          (SC_BASE),
    .scratch_bytes         (SC_BYTES),
    .output_base           (OUT_BASE),
    .output_bytes          (OUT_BYTES),
    .busy                  (busy),
    .done                  (done),
    .error_valid           (error_valid),
    .error_code            (error_code),
    .abort_done            (abort_done),
    .warning_pulse         (warning_pulse),
    .state_update_valid    (state_update_valid),
    .next_token_count      (next_token_count),
    .next_package_present  (next_package_present),
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
    .SEG_COUNT  (3),
    .SEG0_FILE  (""),
    .SEG0_BASE  (IN_BASE),
    .SEG0_BYTES (IN_BYTES),
    .SEG1_FILE  (""),
    .SEG1_BASE  (WT_BASE),
    .SEG1_BYTES (WT_BYTES),
    .SEG2_FILE  (""),
    .SEG2_BASE  (SC_BASE),
    .SEG2_BYTES (SC_BYTES),
    .LFSR_INIT  (16'hCAFE)
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

  logic [63:0] cmd_end;
  int          cmd_count;
  logic        abort_seen;
  string       stall_str;
  string       phase_str;
  logic [15:0] parsed_mask;

  initial begin
    #100000000;
    $display("WATCHDOG: state=%0d child=%0d busy=%0d done=%0d ev=%0d code=%0d phase=%s",
             dut.state, dut.child_sel, busy, done, error_valid, error_code, phase_str);
    $display("WATCHDOG: vec state=%0d op=%0d token=%0d head=%0d idx=%0d rd_len=%0d",
             dut.u_vector.state, dut.u_vector.op_r, dut.u_vector.token,
             dut.u_vector.head, dut.u_vector.idx, dut.u_vector.rd_len);
    $display("WATCHDOG: vec rv=%0d rr=%0d r_ready_r=%0d",
             dut.u_vector.req_r_valid, dut.u_vector.req_r_ready, dut.u_vector.r_ready_r);
    $display("WATCHDOG: master state=%0d beats=%0d mem state=%0d beats=%0d",
             dut.u_master.state, dut.u_master.beat_count,
             dut_mem.state, dut_mem.beat_count);
    tb_fatal("tb_tensor_executor watchdog");
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      cmd_count <= 0;
    end else if (obs_cmd_valid) begin
      cmd_count <= cmd_count + 1;
      if (abort_seen)
        tb_fatal("new memory command observed after abort");
      cmd_end = obs_cmd_addr + {13'd0, obs_cmd_len, 3'b000};
      if (!(((obs_cmd_addr >= IN_BASE) && (cmd_end <= IN_BASE + IN_BYTES)) ||
            ((obs_cmd_addr >= WT_BASE) && (cmd_end <= WT_BASE + WT_BYTES)) ||
            ((obs_cmd_addr >= SC_BASE) && (cmd_end <= SC_BASE + SC_BYTES)))) begin
        $display("trace outside regions: addr=%h len=%0d", obs_cmd_addr, obs_cmd_len);
        tb_fatal("memory trace outside declared regions");
      end
    end
  end

  // Deterministic int8 data functions; the same expressions generate the
  // stimulus and the expected results.
  function automatic logic [7:0] img_val(input int row, input int col, input int ch);
    logic [7:0] v;
    v = 8'(row * 224 + col);
    v = v * 8'd3 + 8'(ch);
    v = v * 8'd37 + 8'd7;
    return v;
  endfunction

  function automatic logic [7:0] pos_val(input int i, input int c);
    return 8'(i * 3 + c) * 8'd11 + 8'd5;
  endfunction

  function automatic logic [7:0] cls_val(input int c);
    return 8'(c) * 8'd13 + 8'd9;
  endfunction

  function automatic logic [7:0] qkv_val(input int token, input int idx);
    return 8'(token * 5 + idx) * 8'd29 + 8'd3;
  endfunction

  function automatic logic [7:0] ctx_val(input int head, input int token, input int lane);
    return 8'(head * 9 + token * 3 + lane) * 8'd17 + 8'd1;
  endfunction

  function automatic heatvit_s8_t sat8(input int v);
    if (v > 127) return 8'sd127;
    if (v < -128) return -8'sd128;
    return heatvit_s8_t'(v);
  endfunction

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

  task automatic run_desc();
    desc_valid = 1'b1;
    wait (desc_ready);
    @(posedge clk);
    #1;
    desc_valid = 1'b0;
  endtask

  task automatic expect_error(input logic [7:0] code, input int cmd_before);
    wait (error_valid);
    #1;
    if (error_code != code) begin
      $display("error code=%0d want=%0d", error_code, code);
      tb_fatal("wrong error code");
    end
    if (cmd_count != cmd_before) tb_fatal("illegal descriptor issued a command");
  endtask

  task automatic expect_done();
    wait (done);
    #1;
    if (error_valid) tb_fatal("unexpected error after done");
    if (busy) tb_fatal("executor still busy after done");
  endtask

  logic [7:0]  got_byte;
  logic [7:0]  want_byte;
  logic [63:0] acc;
  int          i;
  int          j;
  int          c;
  int          kind;
  int          head;
  int          token;
  int          lane;
  int          pr;
  int          pc;
  int          in_row;
  int          in_col;
  int          ch;
  int          expect_loops;
  int          base_cmd_count;

  initial begin
    abort           = 1'b0;
    desc_valid      = 1'b0;
    desc_d          = '0;
    token_count     = 8'd5;
    package_present = 1'b0;
    dbg_valid       = 1'b0;
    dbg_addr        = 32'h00000000;
    dbg_w_valid     = 1'b0;
    dbg_w_addr      = 32'h00000000;
    dbg_w_data      = 8'h00;
    abort_seen      = 1'b0;

    if (!$value$plusargs("STALL_MASK=%s", stall_str)) stall_str = "0";
    if ($sscanf(stall_str, "%h", parsed_mask) != 1) parsed_mask = 16'h0000;
    stall_mask = parsed_mask;

    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
    #1;

    // ------------------------------------------------------------------
    // Illegal descriptors: each must fail before any memory command.
    // ------------------------------------------------------------------
    base_cmd_count = cmd_count;

    desc_d = '0;
    desc_d.opcode = 8'hff;
    run_desc();
    expect_error(ERR_OPCODE, base_cmd_count);

    desc_d = '0;
    desc_d.opcode   = OP_GEMM;
    desc_d.flags    = 24'h000080;
    desc_d.m        = 16'd2;
    desc_d.n        = 16'd2;
    desc_d.k        = 16'd2;
    desc_d.src0_offset = 32'(OFF_GEMM_A);
    desc_d.src1_offset = 32'(WT_GEMM_B);
    desc_d.dst_offset  = 32'(OFF_GEMM_DST);
    desc_d.reserved = 4'd1;
    run_desc();
    expect_error(ERR_DIMENSION, base_cmd_count);

    desc_d = '0;
    desc_d.opcode   = OP_GEMM;
    desc_d.flags    = 24'h000080;
    desc_d.m        = 16'd2;
    desc_d.n        = 16'd0;
    desc_d.k        = 16'd2;
    desc_d.src0_offset = 32'(OFF_GEMM_A);
    desc_d.src1_offset = 32'(WT_GEMM_B);
    desc_d.dst_offset  = 32'(OFF_GEMM_DST);
    run_desc();
    expect_error(ERR_DIMENSION, base_cmd_count);

    desc_d = '0;
    desc_d.opcode   = OP_GEMM;
    desc_d.flags    = 24'h000080 | 24'h000020;
    desc_d.m        = 16'd2;
    desc_d.n        = 16'd8;
    desc_d.k        = 16'd2;
    desc_d.heads    = 4'd2;
    desc_d.src0_offset = 32'(OFF_GEMM_A);
    desc_d.src1_offset = 32'(WT_GEMM_B);
    desc_d.dst_offset  = 32'(OFF_GEMM_DST);
    run_desc();
    expect_error(ERR_DIMENSION, base_cmd_count);

    desc_d = '0;
    desc_d.opcode   = OP_LAYERNORM;
    desc_d.flags    = 24'h000008;
    desc_d.m        = 16'd2;
    desc_d.n        = 16'd192;
    desc_d.src0_offset = 32'(OFF_LN_SRC);
    desc_d.src1_offset = 32'(WT_GAMMA);
    desc_d.aux_offset  = 32'(WT_BETA);
    desc_d.dst_offset  = 32'(OFF_LN_DST);
    desc_d.src0_scale_exp = 6'sd1;
    desc_d.src1_scale_exp = -6'sd6;
    desc_d.aux_scale_exp  = -6'sd7;
    desc_d.dst_scale_exp  = -6'sd7;
    run_desc();
    expect_error(ERR_DIMENSION, base_cmd_count);

    desc_d = '0;
    desc_d.opcode = OP_QKV_UNPACK;
    desc_d.flags  = 24'h000008;
    desc_d.m      = 16'd99;
    desc_d.n      = 16'd576;
    desc_d.heads  = 4'd3;
    desc_d.param0 = 16'd2;
    desc_d.src0_offset = 32'(OFF_QKV_SRC);
    desc_d.dst_offset  = 32'(OFF_QKV_DST);
    run_desc();
    expect_error(ERR_DIMENSION, base_cmd_count);

    // ------------------------------------------------------------------
    // Dynamic M: current N=13 -> m_eff=13 (param0=00) and m_eff=12 (01).
    // ------------------------------------------------------------------
    token_count = 8'd13;

    for (i = 0; i < 13 * 576; i++)
      dbg_w(SC_BASE + OFF_QKV_SRC + i, qkv_val(i / 576, i % 576));

    desc_d = '0;
    desc_d.opcode = OP_QKV_UNPACK;
    desc_d.flags  = 24'h000008;
    desc_d.m      = 16'd99;
    desc_d.n      = 16'd576;
    desc_d.heads  = 4'd3;
    desc_d.param0 = 16'd0;
    desc_d.src0_offset = 32'(OFF_QKV_SRC);
    desc_d.dst_offset  = 32'(OFF_QKV_DST);
    run_desc();
    expect_done();
    for (kind = 0; kind < 3; kind++)
      for (head = 0; head < 3; head++)
        for (token = 0; token < 13; token++)
          for (lane = 0; lane < 64; lane++) begin
            dbg_rd(SC_BASE + OFF_QKV_DST + ((kind * 3 + head) * 13 + token) * 64 + lane,
                   got_byte);
            want_byte = qkv_val(token, kind * 192 + head * 64 + lane);
            if (got_byte !== want_byte) begin
              $display("qkv13 mismatch k=%0d h=%0d t=%0d l=%0d got=%h want=%h",
                       kind, head, token, lane, got_byte, want_byte);
              tb_fatal("QKV dynamic-M output mismatch");
            end
          end

    desc_d.param0 = 16'd1;
    run_desc();
    expect_done();
    // m_eff=12: verify one head's first and last tokens only (layout uses 12).
    dbg_rd(SC_BASE + OFF_QKV_DST + (0 * 3 + 0) * 12 * 64 + 0, got_byte);
    if (got_byte !== qkv_val(0, 0)) tb_fatal("QKV candidate-M first token mismatch");
    dbg_rd(SC_BASE + OFF_QKV_DST + ((2 * 3 + 2) * 12 + 11) * 64 + 63, got_byte);
    if (got_byte !== qkv_val(11, 2 * 192 + 2 * 64 + 63)) begin
      $display("candidate-M last: got=%h want=%h",
               got_byte, qkv_val(11, 2 * 192 + 2 * 64 + 63));
      tb_fatal("QKV candidate-M last token mismatch");
    end

    // ------------------------------------------------------------------
    // PATCHIFY, full 224x224x3 -> 196x768.
    // ------------------------------------------------------------------
    for (pr = 0; pr < 14; pr++)
      for (pc = 0; pc < 14; pc++)
        for (in_row = 0; in_row < 16; in_row++)
          for (in_col = 0; in_col < 16; in_col++)
            for (ch = 0; ch < 3; ch++) begin
              i = pr * 16 + in_row;
              j = pc * 16 + in_col;
              dbg_w(IN_BASE + (i * 224 + j) * 3 + ch, img_val(i, j, ch));
            end

    desc_d = '0;
    desc_d.opcode = OP_PATCHIFY;
    desc_d.flags  = 24'h000800;
    desc_d.m      = 16'd196;
    desc_d.n      = 16'd768;
    desc_d.src0_offset = 32'd0;
    desc_d.dst_offset  = 32'(OFF_PATCH_DST);
    run_desc();
    expect_done();
    for (pr = 0; pr < 14; pr++)
      for (pc = 0; pc < 14; pc++)
        for (i = 0; i < 768; i++) begin
          in_row = i / 48;
          in_col = (i % 48) / 3;
          ch     = (i % 48) % 3;
          dbg_rd(SC_BASE + OFF_PATCH_DST + (pr * 14 + pc) * 768 + i, got_byte);
          want_byte = img_val(pr * 16 + in_row, pc * 16 + in_col, ch);
          if (got_byte !== want_byte) begin
            $display("patchify mismatch pr=%0d pc=%0d i=%0d got=%h want=%h",
                     pr, pc, i, got_byte, want_byte);
            tb_fatal("PATCHIFY output mismatch");
          end
        end

    // ------------------------------------------------------------------
    // COPY_ADD_POS, full 197x192 (equal scales -> plain saturating add).
    // ------------------------------------------------------------------
    for (i = 0; i < 196 * 192; i++)
      dbg_w(SC_BASE + OFF_ADDPOS_SRC + i, qkv_val(i / 192, i % 192));
    for (i = 0; i < 197 * 192; i++)
      dbg_w(WT_BASE + WT_POS + i, pos_val(i / 192, i % 192));
    for (c = 0; c < 192; c++)
      dbg_w(WT_BASE + WT_CLS + c, cls_val(c));

    desc_d = '0;
    desc_d.opcode = OP_COPY_ADD_POS;
    desc_d.flags  = 24'h004000;
    desc_d.m      = 16'd197;
    desc_d.n      = 16'd192;
    desc_d.src0_offset = 32'(OFF_ADDPOS_SRC);
    desc_d.src1_offset = 32'(WT_POS);
    desc_d.aux_offset  = 32'(WT_CLS);
    desc_d.dst_offset  = 32'(OFF_ADDPOS_DST);
    desc_d.src0_scale_exp = -6'sd7;
    desc_d.src1_scale_exp = -6'sd7;
    desc_d.aux_scale_exp  = -6'sd7;
    desc_d.dst_scale_exp  = -6'sd7;
    run_desc();
    expect_done();
    for (c = 0; c < 192; c++) begin
      dbg_rd(SC_BASE + OFF_ADDPOS_DST + c, got_byte);
      if (got_byte !== sat8(int'($signed(cls_val(c))) + int'($signed(pos_val(0, c))))) begin
        $display("addpos cls c=%0d got=%h want=%h cls=%h pos=%h",
                 c, got_byte,
                 sat8(int'($signed(cls_val(c))) + int'($signed(pos_val(0, c)))),
                 cls_val(c), pos_val(0, c));
        tb_fatal("COPY_ADD_POS CLS mismatch");
      end
    end
    for (i = 0; i < 196; i++)
      for (c = 0; c < 192; c++) begin
        dbg_rd(SC_BASE + OFF_ADDPOS_DST + (i + 1) * 192 + c, got_byte);
        want_byte = sat8(int'($signed(qkv_val(i, c))) + int'($signed(pos_val(i + 1, c))));
        if (got_byte !== want_byte) begin
          $display("addpos i=%0d c=%0d got=%h want=%h", i, c, got_byte, want_byte);
          tb_fatal("COPY_ADD_POS patch mismatch");
        end
      end

    // ------------------------------------------------------------------
    // QKV_UNPACK and HEAD_CONCAT with N=5.
    // ------------------------------------------------------------------
    token_count = 8'd5;
    for (i = 0; i < 5 * 576; i++)
      dbg_w(SC_BASE + OFF_QKV_SRC + i, qkv_val(i / 576, i % 576));
    desc_d = '0;
    desc_d.opcode = OP_QKV_UNPACK;
    desc_d.flags  = 24'h000008;
    desc_d.m      = 16'd99;
    desc_d.n      = 16'd576;
    desc_d.heads  = 4'd3;
    desc_d.param0 = 16'd0;
    desc_d.src0_offset = 32'(OFF_QKV_SRC);
    desc_d.dst_offset  = 32'(OFF_QKV_DST);
    run_desc();
    expect_done();
    for (kind = 0; kind < 3; kind++)
      for (head = 0; head < 3; head++)
        for (token = 0; token < 5; token++)
          for (lane = 0; lane < 64; lane++) begin
            dbg_rd(SC_BASE + OFF_QKV_DST + ((kind * 3 + head) * 5 + token) * 64 + lane,
                   got_byte);
            want_byte = qkv_val(token, kind * 192 + head * 64 + lane);
            if (got_byte !== want_byte) tb_fatal("QKV_UNPACK mismatch");
          end

    for (head = 0; head < 3; head++)
      for (token = 0; token < 5; token++)
        for (lane = 0; lane < 64; lane++)
          dbg_w(SC_BASE + OFF_CONCAT_SRC + (head * 5 + token) * 64 + lane,
                ctx_val(head, token, lane));
    desc_d = '0;
    desc_d.opcode = OP_HEAD_CONCAT;
    desc_d.flags  = 24'h000008;
    desc_d.m      = 16'd99;
    desc_d.n      = 16'd192;
    desc_d.heads  = 4'd3;
    desc_d.param0 = 16'd0;
    desc_d.src0_offset = 32'(OFF_CONCAT_SRC);
    desc_d.dst_offset  = 32'(OFF_CONCAT_DST);
    run_desc();
    expect_done();
    for (token = 0; token < 5; token++)
      for (head = 0; head < 3; head++)
        for (lane = 0; lane < 64; lane++) begin
          dbg_rd(SC_BASE + OFF_CONCAT_DST + token * 192 + head * 64 + lane, got_byte);
          want_byte = ctx_val(head, token, lane);
          if (got_byte !== want_byte) tb_fatal("HEAD_CONCAT mismatch");
        end

    // ------------------------------------------------------------------
    // Tiny GEMM (int32 writeback) and dynamic N/K override to 13.
    // ------------------------------------------------------------------
    for (i = 0; i < 4; i++) dbg_w(SC_BASE + OFF_GEMM_A + i, 8'(i + 1));
    for (i = 0; i < 4; i++) dbg_w(WT_BASE + WT_GEMM_B + i, 8'(i + 5));
    desc_d = '0;
    desc_d.opcode = OP_GEMM;
    desc_d.flags  = 24'h000080;
    desc_d.m      = 16'd2;
    desc_d.n      = 16'd2;
    desc_d.k      = 16'd2;
    desc_d.src0_offset = 32'(OFF_GEMM_A);
    desc_d.src1_offset = 32'(WT_GEMM_B);
    desc_d.dst_offset  = 32'(OFF_GEMM_DST);
    desc_d.src0_scale_exp = -6'sd7;
    desc_d.src1_scale_exp = -6'sd7;
    desc_d.dst_scale_exp  = -6'sd14;
    run_desc();
    expect_done();
    // A=[1,2;3,4], B=[5,6;7,8] -> C=[19,22;43,50] as little-endian int32.
    for (i = 0; i < 4; i++) begin
      if (i == 0) acc = 64'd19;
      else if (i == 1) acc = 64'd22;
      else if (i == 2) acc = 64'd43;
      else acc = 64'd50;
      for (j = 0; j < 4; j++) begin
        dbg_rd(SC_BASE + OFF_GEMM_DST + i * 4 + j, got_byte);
        want_byte = acc[8 * j +: 8];
        if (got_byte !== want_byte) begin
          $display("gemm i=%0d j=%0d got=%h want=%h", i, j, got_byte, want_byte);
          tb_fatal("GEMM output mismatch");
        end
      end
    end

    // Dynamic N/K: n=99->13, k=99->13. A[2][13], B[13][13], int32 out.
    token_count = 8'd13;
    for (i = 0; i < 2 * 13; i++)
      dbg_w(SC_BASE + OFF_GEMM_A + i, 8'((i / 13) * 13 + (i % 13) + 1));
    for (i = 0; i < 13 * 13; i++)
      dbg_w(WT_BASE + WT_GEMM_B + i,
            8'(((i / 13) * 7 + (i % 13) * 3) % 200 - 100));
    desc_d = '0;
    desc_d.opcode = OP_GEMM;
    desc_d.flags  = 24'h000080 | 24'h080000 | 24'h100000;
    desc_d.m      = 16'd2;
    desc_d.n      = 16'd99;
    desc_d.k      = 16'd99;
    desc_d.src0_offset = 32'(OFF_GEMM_A);
    desc_d.src1_offset = 32'(WT_GEMM_B);
    desc_d.dst_offset  = 32'(OFF_GEMM_DST);
    desc_d.src0_scale_exp = -6'sd7;
    desc_d.src1_scale_exp = -6'sd7;
    desc_d.dst_scale_exp  = -6'sd14;
    run_desc();
    expect_done();
    for (i = 0; i < 2; i++)
      for (c = 0; c < 13; c++) begin
        acc = 0;
        for (j = 0; j < 13; j++) begin
          int av;
          int bv;
          av = i * 13 + j + 1;
          bv = (j * 7 + c * 3) % 200 - 100;
          acc = acc + av * bv;
        end
        for (j = 0; j < 4; j++) begin
          dbg_rd(SC_BASE + OFF_GEMM_DST + (i * 13 + c) * 4 + j, got_byte);
          want_byte = acc[8 * j +: 8];
          if (got_byte !== want_byte) begin
            $display("gemm override i=%0d c=%0d j=%0d got=%h want=%h",
                     i, c, j, got_byte, want_byte);
            tb_fatal("dynamic N/K GEMM mismatch");
          end
        end
      end

    // ------------------------------------------------------------------
    // RESIDUAL N=2, equal scales -> saturating add.
    // ------------------------------------------------------------------
    phase_str = "residual";
    token_count = 8'd2;
    for (i = 0; i < 2 * 192; i++) begin
      dbg_w(SC_BASE + OFF_RESID_MAIN + i, qkv_val(i / 192, i % 192));
      dbg_w(SC_BASE + OFF_RESID_AUX + i, ctx_val(0, i / 192, i % 192));
    end
    desc_d = '0;
    desc_d.opcode = OP_RESIDUAL;
    desc_d.flags  = 24'h000008;
    desc_d.m      = 16'd99;
    desc_d.n      = 16'd192;
    desc_d.param0 = 16'd0;
    desc_d.src0_offset = 32'(OFF_RESID_MAIN);
    desc_d.aux_offset  = 32'(OFF_RESID_AUX);
    desc_d.dst_offset  = 32'(OFF_RESID_DST);
    desc_d.src0_scale_exp = -6'sd7;
    desc_d.aux_scale_exp  = -6'sd7;
    desc_d.dst_scale_exp  = -6'sd7;
    run_desc();
    expect_done();
    for (i = 0; i < 2 * 192; i++) begin
      dbg_rd(SC_BASE + OFF_RESID_DST + i, got_byte);
      want_byte = sat8(int'($signed(qkv_val(i / 192, i % 192))) +
                       int'($signed(ctx_val(0, i / 192, i % 192))));
      if (got_byte !== want_byte) tb_fatal("RESIDUAL mismatch");
    end

    // ------------------------------------------------------------------
    // LAYERNORM N=2 with zero input, gamma=64, beta=0 -> all zeros.
    // ------------------------------------------------------------------
    phase_str = "layernorm";
    for (i = 0; i < 2 * 192; i++) dbg_w(SC_BASE + OFF_LN_SRC + i, 8'h00);
    for (i = 0; i < 192; i++) begin
      dbg_w(WT_BASE + WT_GAMMA + i, 8'sd64);
      dbg_w(WT_BASE + WT_BETA + i, 8'sd0);
    end
    desc_d = '0;
    desc_d.opcode = OP_LAYERNORM;
    desc_d.flags  = 24'h000008 | 24'h004000;
    desc_d.m      = 16'd99;
    desc_d.n      = 16'd192;
    desc_d.param0 = 16'd0;
    desc_d.src0_offset = 32'(OFF_LN_SRC);
    desc_d.src1_offset = 32'(WT_GAMMA);
    desc_d.aux_offset  = 32'(WT_BETA);
    desc_d.dst_offset  = 32'(OFF_LN_DST);
    desc_d.src0_scale_exp = -6'sd7;
    desc_d.src1_scale_exp = -6'sd6;
    desc_d.aux_scale_exp  = -6'sd7;
    desc_d.dst_scale_exp  = -6'sd7;
    run_desc();
    expect_done();
    for (i = 0; i < 2 * 192; i++) begin
      dbg_rd(SC_BASE + OFF_LN_DST + i, got_byte);
      if (got_byte !== 8'h00) begin
        $display("layernorm i=%0d got=%h", i, got_byte);
        tb_fatal("LAYERNORM zero vector mismatch");
      end
    end

    // ------------------------------------------------------------------
    // ATTN_SOFTMAX N=2, all scores 0 -> every output is 128
    // (P2 fix: attention delta2 = 1.0; was 64 under the old halving 0.5).
    // ------------------------------------------------------------------
    phase_str = "softmax";
    for (i = 0; i < 48; i++) dbg_w(SC_BASE + OFF_SOFTMAX_SRC + i, 8'h00);
    desc_d = '0;
    desc_d.opcode = OP_ATTN_SOFTMAX;
    desc_d.flags  = 24'h000008 | 24'h080000;
    desc_d.m      = 16'd99;
    desc_d.n      = 16'd99;
    desc_d.heads  = 4'd3;
    desc_d.param0 = 16'd0;
    desc_d.src0_offset = 32'(OFF_SOFTMAX_SRC);
    desc_d.dst_offset  = 32'(OFF_SOFTMAX_DST);
    desc_d.src0_scale_exp = -6'sd16;
    run_desc();
    expect_done();
    for (i = 0; i < 12; i++) begin
      dbg_rd(SC_BASE + OFF_SOFTMAX_DST + i, got_byte);
      if (got_byte !== 8'd128) begin
        $display("softmax i=%0d got=%0d", i, got_byte);
        tb_fatal("ATTN_SOFTMAX uniform row mismatch");
      end
    end

    // ------------------------------------------------------------------
    // Abort before descriptor acceptance.
    // ------------------------------------------------------------------
    phase_str = "abort-pre";
    desc_d = '0;
    desc_d.opcode = OP_NOP;
    desc_valid = 1'b1;
    wait (desc_ready);
    abort = 1'b1;
    abort_seen = 1'b1;
    @(posedge clk);
    #1;
    if (!abort_done) tb_fatal("abort before accept did not pulse abort_done");
    if (busy) tb_fatal("executor accepted descriptor during abort");
    desc_valid = 1'b0;
    abort = 1'b0;
    abort_seen = 1'b0;
    @(posedge clk);
    #1;
    if (!desc_ready) tb_fatal("executor not ready after pre-accept abort");

    // ------------------------------------------------------------------
    // Abort after a memory command handshake (full PATCHIFY is slow enough).
    // ------------------------------------------------------------------
    phase_str = "abort-mid";
    base_cmd_count = cmd_count;
    desc_d = '0;
    desc_d.opcode = OP_PATCHIFY;
    desc_d.flags  = 24'h000800;
    desc_d.m      = 16'd196;
    desc_d.n      = 16'd768;
    desc_d.src0_offset = 32'd0;
    desc_d.dst_offset  = 32'(OFF_PATCH_DST);
    run_desc();
    wait (cmd_count > base_cmd_count);
    abort = 1'b1;
    abort_seen = 1'b1;
    wait (abort_done);
    #1;
    abort = 1'b0;
    abort_seen = 1'b0;
    if (busy) tb_fatal("executor still busy after mid-burst abort");
    @(posedge clk);
    #1;
    if (!desc_ready) tb_fatal("executor not ready after mid-burst abort");
    if (state_update_valid) tb_fatal("phase 3 must not update token state");

    $display("TEST_PASS tb_tensor_executor");
    $finish;
  end

endmodule
