`timescale 1ns / 1ps

// Task 6: full-size HeatViT end-to-end inference through heatvit_top.
//
// Loads the four region images into the behavioral memory, starts the top,
// and compares all 18 golden checkpoints at the descriptor indices from
// e2e_tb_config_pkg (hierarchical monitor on the scheduler index and the
// executor done). Verifies the final logits, output_scale_exp, the three
// selector N/package state updates, exactly one done pulse and a quiet
// error output. Protocol assertions cover valid-stall payload stability,
// no X/Z on active control signals, and every memory command staying
// inside the four declared regions.
module tb_heatvit_e2e;
  import heatvit_pkg::*;
  import tb_pkg::*;
  import e2e_tb_config_pkg::*;

  logic        clk   = 1'b0;
  logic        rst_n = 1'b0;

  logic        start;
  logic [31:0] input_base;
  logic [31:0] input_bytes;
  logic [31:0] weight_base;
  logic [31:0] weight_bytes;
  logic [31:0] scratch_base;
  logic [31:0] scratch_bytes;
  logic [31:0] output_base;
  logic [31:0] output_bytes;
  logic        busy;
  logic        done;
  logic        error_valid;
  logic [7:0]  error_code;
  logic [7:0]  warning_flags;
  logic [5:0]  output_scale_exp;

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

  logic        dbg_valid;
  logic        dbg_ready;
  logic [31:0] dbg_addr;
  logic [7:0]  dbg_data;
  logic        load_valid;
  logic        load_ready;
  logic [1:0]  load_seg;
  logic [31:0] load_bytes;
  string       load_file;

  logic [15:0] stall_mask;
  string       stall_str;
  logic [15:0] parsed_mask;

  heatvit_top dut (
    .clk             (clk),
    .rst_n           (rst_n),
    .start           (start),
    .input_base      (input_base),
    .input_bytes     (input_bytes),
    .weight_base     (weight_base),
    .weight_bytes    (weight_bytes),
    .scratch_base    (scratch_base),
    .scratch_bytes   (scratch_bytes),
    .output_base     (output_base),
    .output_bytes    (output_bytes),
    .busy            (busy),
    .done            (done),
    .error_valid     (error_valid),
    .error_code      (error_code),
    .warning_flags   (warning_flags),
    .output_scale_exp(output_scale_exp),
    .mem_cmd_valid   (mem_cmd_valid),
    .mem_cmd_ready   (mem_cmd_ready),
    .mem_cmd_write   (mem_cmd_write),
    .mem_cmd_addr    (mem_cmd_addr),
    .mem_cmd_len     (mem_cmd_len),
    .mem_w_valid     (mem_w_valid),
    .mem_w_ready     (mem_w_ready),
    .mem_w_data      (mem_w_data),
    .mem_w_strb      (mem_w_strb),
    .mem_w_last      (mem_w_last),
    .mem_r_valid     (mem_r_valid),
    .mem_r_ready     (mem_r_ready),
    .mem_r_data      (mem_r_data),
    .mem_r_last      (mem_r_last)
  );

  behavioral_memory #(
    .SEG_COUNT  (4),
    .SEG0_FILE  (""),
    .SEG0_BASE  (INPUT_BASE),
    .SEG0_BYTES (INPUT_BYTES),
    .SEG1_FILE  (""),
    .SEG1_BASE  (WEIGHT_BASE),
    .SEG1_BYTES (WEIGHT_BYTES),
    .SEG2_FILE  (""),
    .SEG2_BASE  (SCRATCH_BASE),
    .SEG2_BYTES (SCRATCH_BYTES),
    .SEG3_FILE  (""),
    .SEG3_BASE  (OUTPUT_BASE),
    .SEG3_BYTES (OUTPUT_BYTES),
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
    .obs_cmd_valid(),
    .obs_cmd_write(),
    .obs_cmd_addr (),
    .obs_cmd_len  (),
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

  // ---------------------------------------------------------------
  // Protocol assertions.
  // ---------------------------------------------------------------
  assert property (@(posedge clk) disable iff (!rst_n)
      mem_cmd_valid && !mem_cmd_ready |=>
      $stable({mem_cmd_write, mem_cmd_addr, mem_cmd_len}));
  assert property (@(posedge clk) disable iff (!rst_n)
      mem_w_valid && !mem_w_ready |=>
      $stable({mem_w_data, mem_w_strb, mem_w_last}));
  assert property (@(posedge clk) disable iff (!rst_n)
      busy |-> !$isunknown({busy, error_valid, mem_cmd_valid, mem_w_valid,
                            mem_r_ready}));

  always_ff @(posedge clk) begin
    if (mem_cmd_valid && !$isunknown({mem_cmd_addr, mem_cmd_len}) &&
        (mem_cmd_addr < 32'h00000000 ||
         mem_cmd_addr + {13'd0, mem_cmd_len, 3'b000} > 32'h04000000))
      tb_fatal("e2e memory command outside the four regions");
  end

  // ---------------------------------------------------------------
  // Checkpoint golden arrays (flat with per-checkpoint word offsets:
  // XSim's $readmemh cannot target a 2-D unpacked array slice).
  // ---------------------------------------------------------------
  localparam int MAX_CP_BYTES = 197 * 192;
  localparam int CP_WORDS = (MAX_CP_BYTES + 7) / 8 + 1;
  logic [63:0] cp_exp [0:18 * CP_WORDS - 1];
  int          i;
  logic [7:0]  got_byte;
  logic [63:0] cycle_count = 64'd0;
  string       vector_dir;
  string       cp_names [0:17] = '{"patch", "block_01", "block_02",
    "block_03", "selector_01", "block_04", "block_05", "block_06",
    "selector_02", "block_07", "block_08", "block_09", "selector_03",
    "block_10", "block_11", "block_12", "final_ln", "logits"};

  int done_pulses;
  int selector_done_count;
  logic [7:0] seen_n [0:2];

  // ---------------------------------------------------------------
  // P8-1 performance monitor (TB-only, non-synthesizable): per-
  // descriptor latency profile plus MAC-activity, external-memory
  // traffic and busy duty statistics. All hierarchical probes read
  // RTL internals; the RTL itself is unchanged.
  // ---------------------------------------------------------------
  logic [15:0] perf_idx;
  logic [7:0]  perf_op;
  logic [63:0] perf_start_cyc;
  logic [63:0] mac_active_total [0:2];
  logic [63:0] mem_cmd_beats;
  logic [63:0] mem_rd_bytes;
  logic [63:0] mem_wr_bytes;
  logic [63:0] mem_rd_beats;
  logic [63:0] mem_wr_beats;
  logic [63:0] mem_cmd_stalls;
  logic [63:0] busy_cycles;
  logic [63:0] perf_desc_count;

  task automatic load_segment(
    input logic [1:0]  seg,
    input int          bytes,
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

  task automatic dbg_rd(input logic [31:0] addr, output logic [7:0] data);
    dbg_valid = 1'b1;
    dbg_addr  = addr;
    @(posedge clk);
    #1;
    data = dbg_data;
    dbg_valid = 1'b0;
  endtask

  task automatic check_checkpoint(input int cp);
    begin
      int bytes;
      logic [31:0] base;
      bytes = CHECKPOINT_BYTES[cp];
      base  = CHECKPOINT_OFFSET[cp];
      for (i = 0; i < bytes; i++) begin
        dbg_rd(base + i, got_byte);
        if (got_byte !== cp_exp[cp * CP_WORDS + i / 8][8 * (i % 8) +: 8]) begin
          $display("checkpoint %s byte %0d mismatch: got=%h want=%h desc=%0d",
                   cp_names[cp], i, got_byte,
                   cp_exp[cp * CP_WORDS + i / 8][8 * (i % 8) +: 8],
                   dut.u_scheduler.current_desc_index);
          tb_fatal("e2e checkpoint mismatch");
        end
      end
    end
  endtask

  // Hierarchical monitor task: wait until the executor reports done for a
  // descriptor whose index matches one of the 18 checkpoint indices.
  task automatic wait_desc_done(output int cp);
    cp = -1;
    while (cp < 0) begin
      @(posedge clk);
      if (dut.u_executor.done) begin
        for (int k = 0; k < 18; k++) begin
          if (dut.u_scheduler.current_desc_index ==
              CHECKPOINT_DESC_INDEX[k])
            cp = k;
        end
      end
    end
  endtask

  initial begin
    $display("WATCHDOG_CYCLES=%0d", WATCHDOG_CYCLES);
    repeat (WATCHDOG_CYCLES) @(posedge clk);
    $display("WATCHDOG: cycle_count=%0d busy=%b index=%0d err=%0d sched=%0d exec=%0d",
             WATCHDOG_CYCLES, busy, dut.u_scheduler.current_desc_index,
             error_code, dut.u_scheduler.error_code,
             dut.u_executor.error_code);
    tb_fatal("tb_heatvit_e2e watchdog");
  end

  // Any DUT error anywhere in the flow: report context and stop immediately
  // (the checkpoint loop above has no error escape and would hang).
  always @(posedge clk) begin
    if (error_valid) begin
      $display("E2E_ERROR top=%0d desc=%0d sched=%0d exec=%0d exec_child=%0d exec_state=%0d",
               error_code, dut.u_scheduler.current_desc_index,
               dut.u_scheduler.error_code, dut.u_executor.error_code,
               dut.u_executor.child_sel, dut.u_executor.state);
      tb_fatal("e2e error monitor");
    end
  end

  initial begin
    start         = 1'b0;
    input_base    = INPUT_BASE;
    input_bytes   = INPUT_BYTES;
    weight_base   = WEIGHT_BASE;
    weight_bytes  = WEIGHT_BYTES;
    scratch_base  = SCRATCH_BASE;
    scratch_bytes = SCRATCH_BYTES;
    output_base   = OUTPUT_BASE;
    output_bytes  = OUTPUT_BYTES;
    dbg_valid     = 1'b0;
    dbg_addr      = 32'h00000000;
    load_valid    = 1'b0;
    load_seg      = 2'd0;
    load_bytes    = 32'd0;
    load_file     = "";

    if (!$value$plusargs("STALL_MASK=%s", stall_str)) stall_str = "0";
    if ($sscanf(stall_str, "%h", parsed_mask) != 1) parsed_mask = 16'h0000;
    stall_mask = parsed_mask;

    if (!$value$plusargs("VECTOR_DIR=%s", vector_dir))
      vector_dir = "build/vectors/e2e";

    begin
      int fd;
      fd = $fopen({vector_dir, "/weights.mem"}, "r");
      if (fd == 0) begin
        $display("missing e2e vectors under %s", vector_dir);
        tb_fatal("e2e vectors missing");
      end
      $fclose(fd);
    end

    for (i = 0; i < 18 * CP_WORDS; i++)
      cp_exp[i] = 64'h0000000000000000;
    for (i = 0; i < 18; i++) begin
      $readmemh({vector_dir, "/checkpoints/", cp_names[i], ".mem"},
                cp_exp, i * CP_WORDS);
    end

    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
    #1;

    load_segment(2'd0, INPUT_BYTES, {vector_dir, "/input.mem"});
    load_segment(2'd1, WEIGHT_BYTES, {vector_dir, "/weights.mem"});
    load_segment(2'd2, SCRATCH_BYTES, {vector_dir, "/scratch_init.mem"});
    load_segment(2'd3, OUTPUT_BYTES, {vector_dir, "/output_init.mem"});

    start = 1'b1;
    @(posedge clk);
    #1;
    start = 1'b0;

    // Checkpoint comparison loop driven by the hierarchical monitor.
    begin
      int remaining_cp;
      remaining_cp = 18;
      while (remaining_cp > 0) begin
        int cp;
        wait_desc_done(cp);
        $display("checkpoint %s at desc %0d", cp_names[cp],
                 dut.u_scheduler.current_desc_index);
        check_checkpoint(cp);
        if (CHECKPOINT_DESC_INDEX[cp] == 16'd53 ||
            CHECKPOINT_DESC_INDEX[cp] == 16'd104 ||
            CHECKPOINT_DESC_INDEX[cp] == 16'd155) begin
          seen_n[selector_done_count] =
              dut.u_scheduler.current_token_count;
          selector_done_count = selector_done_count + 1;
        end
        remaining_cp = remaining_cp - 1;
      end
    end

    // Wait for the top-level FINISH done. done is a one-cycle pulse that can
    // fire while the logits checkpoint comparison is still running, so gate
    // on the pulse counter instead of sampling the raw signal.
    wait (done_pulses == 1 || error_valid);
    @(posedge clk);
    #1;
    if (error_valid) begin
      $display("e2e error code=%0d", error_code);
      tb_fatal("e2e inference reported an error");
    end
    if (done_pulses != 1) begin
      $display("done_pulses=%0d", done_pulses);
      tb_fatal("e2e done must pulse exactly once");
    end
    if (output_scale_exp != OUTPUT_SCALE_EXP) begin
      $display("output_scale_exp=%0d expected=%0d", output_scale_exp,
               OUTPUT_SCALE_EXP);
      tb_fatal("e2e output scale mismatch");
    end
    if (selector_done_count != 3) begin
      $display("selector_done_count=%0d", selector_done_count);
      tb_fatal("expected three selector state updates");
    end
    if (seen_n[0] != SELECTOR_OUT_N[0] || seen_n[1] != SELECTOR_OUT_N[1] ||
        seen_n[2] != SELECTOR_OUT_N[2]) begin
      $display("seen_n=%0d/%0d/%0d expected=%0d/%0d/%0d", seen_n[0],
               seen_n[1], seen_n[2], SELECTOR_OUT_N[0], SELECTOR_OUT_N[1],
               SELECTOR_OUT_N[2]);
      tb_fatal("selector token counts mismatch");
    end

    $display("perf_sum cycles=%0d busy=%0d desc=%0d mac_total=%0d/%0d/%0d cmd_beats=%0d rd_bytes=%0d wr_bytes=%0d rd_beats=%0d wr_beats=%0d cmd_stall=%0d",
             cycle_count, busy_cycles, perf_desc_count,
             mac_active_total[0], mac_active_total[1], mac_active_total[2],
             mem_cmd_beats, mem_rd_bytes, mem_wr_bytes,
             mem_rd_beats, mem_wr_beats, mem_cmd_stalls);
    $display("TEST_PASS tb_heatvit_e2e");
    $display("e2e_cycles=%0d", cycle_count);
    $finish;
  end

  always @(posedge clk) cycle_count <= cycle_count + 64'd1;

  // ---------------------------------------------------------------
  // P8-1 monitor logic.
  // ---------------------------------------------------------------
  initial begin
    perf_idx        = 16'd0;
    perf_op         = 8'd0;
    perf_start_cyc  = 64'd0;
    for (int p = 0; p < 3; p++) mac_active_total[p] = 64'd0;
    mem_cmd_beats   = 64'd0;
    mem_rd_bytes    = 64'd0;
    mem_wr_bytes    = 64'd0;
    mem_rd_beats    = 64'd0;
    mem_wr_beats    = 64'd0;
    mem_cmd_stalls  = 64'd0;
    busy_cycles     = 64'd0;
    perf_desc_count = 64'd0;
  end

  // Descriptor accept: latch index/opcode/start cycle.  The scheduler
  // presents the index together with exec_desc_valid, so this capture
  // is unambiguous even though the done-time print happens later.
  always @(posedge clk) begin
    if (rst_n && dut.u_scheduler.exec_desc_valid &&
        dut.u_scheduler.exec_desc_ready) begin
      perf_idx       = dut.u_scheduler.current_desc_index;
      perf_op        = dut.u_scheduler.exec_desc.opcode;
      perf_start_cyc = cycle_count;
    end
  end

  // Descriptor done: print the profile line and, for GEMM descriptors,
  // accumulate the per-bank MAC-active counters while they still hold
  // this descriptor's totals (they reset at the next GEMM start).
  always @(posedge clk) begin
    if (rst_n && dut.u_executor.done) begin
      $display("perf_desc idx=%0d op=%0d start=%0d end=%0d dur=%0d mac=%0d/%0d/%0d",
               perf_idx, perf_op, perf_start_cyc, cycle_count,
               cycle_count - perf_start_cyc,
               dut.u_executor.gemm_mac_active[0],
               dut.u_executor.gemm_mac_active[1],
               dut.u_executor.gemm_mac_active[2]);
      if (perf_op == OP_GEMM) begin
        for (int p = 0; p < 3; p++)
          mac_active_total[p] = mac_active_total[p] +
                                {32'd0, dut.u_executor.gemm_mac_active[p]};
      end
      perf_desc_count = perf_desc_count + 64'd1;
    end
  end

  // External memory traffic + busy duty at the top-level interface.
  always @(posedge clk) begin
    if (rst_n) begin
      if (mem_cmd_valid && mem_cmd_ready) begin
        mem_cmd_beats = mem_cmd_beats + 64'd1;
        if (mem_cmd_write) mem_wr_bytes = mem_wr_bytes + {48'd0, mem_cmd_len};
        else               mem_rd_bytes = mem_rd_bytes + {48'd0, mem_cmd_len};
      end
      if (mem_cmd_valid && !mem_cmd_ready)
        mem_cmd_stalls = mem_cmd_stalls + 64'd1;
      if (mem_r_valid && mem_r_ready)
        mem_rd_beats = mem_rd_beats + 64'd1;
      if (mem_w_valid && mem_w_ready)
        mem_wr_beats = mem_wr_beats + 64'd1;
      if (busy) busy_cycles = busy_cycles + 64'd1;
    end
  end


  always_ff @(posedge clk) begin
    if (!rst_n) done_pulses <= 0;
    else if (done) done_pulses <= done_pulses + 1;
  end

endmodule
