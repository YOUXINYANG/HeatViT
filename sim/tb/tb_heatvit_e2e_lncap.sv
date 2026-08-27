`timescale 1ns / 1ps

// P5 debug: capture heatvit_layernorm internals during the block-11 and
// block-12 LN2 rows (descriptor indices 178 and 191). On the first two
// S_DRAIN entries per descriptor, dumps u_ln.{x,gamma,beta}_buf,
// mean/inv_std/std and the vector engine's bbuf to text files.
module tb_heatvit_e2e_lncap;
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

  // ---- LN capture monitor -------------------------------------------------
  int cap_count_178 = 0;
  int cap_count_191 = 0;

  // Per-element capture: (desc, idx, in_x, square, running sum) on every
  // S_LOAD_ACCUM handshake near desc 191, plus cfg scales. The file is
  // opened lazily on the first write so vector_dir (parsed from plusargs
  // in the main initial block) is already resolved.
  integer elem_fd = 0;
  int elem_count = 0;
  always @(posedge clk) begin
    if (elem_fd == 0 && vector_dir.len() > 0)
      elem_fd = $fopen({vector_dir, "/diag5_ln_elem.txt"}, "w");
    if (rst_n && dut.u_scheduler.current_desc_index >= 16'd190 &&
        dut.u_scheduler.current_desc_index <= 16'd191) begin
      if (dut.u_executor.u_vector.u_ln.state == 4'd1 &&
          dut.u_executor.u_vector.u_ln.in_valid &&
          dut.u_executor.u_vector.u_ln.in_ready && elem_count < 800) begin
        $fwrite(elem_fd, "D %0d %0d %0d %0d %0d\n",
                dut.u_scheduler.current_desc_index,
                dut.u_executor.u_vector.u_ln.idx,
                $signed(dut.u_executor.u_vector.u_ln.in_x),
                $signed(dut.u_executor.u_vector.u_ln.square_in_w[63:0]),
                $signed(dut.u_executor.u_vector.u_ln.sum_square_r));
        elem_count = elem_count + 1;
      end
      if (dut.u_executor.u_vector.u_ln.state == 4'd0 &&
          dut.u_executor.u_vector.u_ln.cfg_valid &&
          dut.u_executor.u_vector.u_ln.cfg_ready && elem_count < 800) begin
        $fwrite(elem_fd, "C %0d %0d %0d\n",
                dut.u_scheduler.current_desc_index,
                $signed(dut.u_executor.u_vector.u_ln.cfg_x_scale_exp),
                $signed(dut.u_executor.u_vector.u_ln.x_scale_r));
        elem_count = elem_count + 1;
      end
    end
  end

  task automatic dump_ln_row(input int desc_idx, input int row);
    int fd;
    string prefix;
    prefix = $sformatf("%s/diag3_ln_%0d_row%0d", vector_dir, desc_idx, row);
    fd = $fopen({prefix, "_x.txt"}, "w");
    for (int k = 0; k < 192; k++)
      $fwrite(fd, "%0d\n", $signed(dut.u_executor.u_vector.u_ln.x_buf[k]));
    $fclose(fd);
    fd = $fopen({prefix, "_gamma.txt"}, "w");
    for (int k = 0; k < 192; k++)
      $fwrite(fd, "%0d\n", $signed(dut.u_executor.u_vector.u_ln.gamma_buf[k]));
    $fclose(fd);
    fd = $fopen({prefix, "_beta.txt"}, "w");
    for (int k = 0; k < 192; k++)
      $fwrite(fd, "%0d\n", $signed(dut.u_executor.u_vector.u_ln.beta_buf[k]));
    $fclose(fd);
    fd = $fopen({prefix, "_stats.txt"}, "w");
    $fwrite(fd, "mean=%0d e2=%0d msq_c=%0d var_c=%0d rad=%0d std=%0d isq_root=%0d isq_rem=%0d inv=%0d warn=%b\n",
            $signed(dut.u_executor.u_vector.u_ln.mean_q32_r),
            dut.u_executor.u_vector.u_ln.e2_q32_r,
            $signed(dut.u_executor.u_vector.u_ln.mean_square_w),
            $signed(dut.u_executor.u_vector.u_ln.variance_w),
            dut.u_executor.u_vector.u_ln.isqrt_radicand,
            dut.u_executor.u_vector.u_ln.std_q16_r,
            dut.u_executor.u_vector.u_ln.isqrt_root,
            dut.u_executor.u_vector.u_ln.isqrt_remainder,
            dut.u_executor.u_vector.u_ln.inv_std_q32_r,
            dut.u_executor.u_vector.u_ln.warn_negative_variance);
    $fclose(fd);
    $display("CAPTURED ln desc %0d row %0d", desc_idx, row);
  endtask

  always @(posedge clk) begin
    if (rst_n && !error_valid &&
        dut.u_executor.u_vector.u_ln.state == 4'd7) begin  // S_DRAIN
      if (dut.u_scheduler.current_desc_index == 16'd178) begin
        if (cap_count_178 < 2) begin
          dump_ln_row(178, cap_count_178);
          cap_count_178 = cap_count_178 + 1;
        end
      end
      if (dut.u_scheduler.current_desc_index == 16'd191) begin
        if (cap_count_191 < 2) begin
          dump_ln_row(191, cap_count_191);
          cap_count_191 = cap_count_191 + 1;
        end
      end
    end
  end

  // ---- the rest mirrors tb_heatvit_e2e ------------------------------------
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
          $display("checkpoint %s byte %0d mismatch: got=%h want=%h",
                   cp_names[cp], i, got_byte,
                   cp_exp[cp * CP_WORDS + i / 8][8 * (i % 8) +: 8]);
          tb_fatal("e2e checkpoint mismatch");
        end
      end
    end
  endtask

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
    tb_fatal("tb_heatvit_e2e_lncap watchdog");
  end

  always @(posedge clk) begin
    if (error_valid) begin
      $display("E2E_ERROR top=%0d desc=%0d", error_code,
               dut.u_scheduler.current_desc_index);
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
      if (fd == 0) tb_fatal("e2e vectors missing");
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

    begin
      int remaining_cp;
      remaining_cp = 18;
      while (remaining_cp > 0) begin
        int cp;
        wait_desc_done(cp);
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

    wait (done_pulses == 1 || error_valid);
    @(posedge clk);
    #1;
    if (error_valid) tb_fatal("e2e inference reported an error");
    if (done_pulses != 1) tb_fatal("e2e done must pulse exactly once");
    if (output_scale_exp != OUTPUT_SCALE_EXP)
      tb_fatal("e2e output scale mismatch");
    if (selector_done_count != 3)
      tb_fatal("expected three selector state updates");
    if (seen_n[0] != SELECTOR_OUT_N[0] || seen_n[1] != SELECTOR_OUT_N[1] ||
        seen_n[2] != SELECTOR_OUT_N[2])
      tb_fatal("selector token counts mismatch");

    $display("CAP_178=%0d CAP_191=%0d", cap_count_178, cap_count_191);
    $display("e2e_cycles=%0d", cycle_count);
    $finish;
  end

  always @(posedge clk) cycle_count <= cycle_count + 64'd1;

  always_ff @(posedge clk) begin
    if (!rst_n) done_pulses <= 0;
    else if (done) done_pulses <= done_pulses + 1;
  end

endmodule
