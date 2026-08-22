`timescale 1ns / 1ps

// Task 3: full-size patch embedding through the tensor executor.
//
// Runs the three-descriptor sequence PATCHIFY -> GEMM -> COPY_ADD_POS on the
// generated 224x224x3 vectors and compares the [197][192] activation byte for
// byte. The memory trace is checked against the per-descriptor regions from
// sim/generated/patch_tb_config.sv.
module tb_patch_embedding;
  import heatvit_pkg::*;
  import tb_pkg::*;
  import patch_tb_config_pkg::*;

  localparam int ACT_WORDS = ACT_BYTES / 8;

  logic        clk   = 1'b0;
  logic        rst_n = 1'b0;

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
  logic        load_valid;
  logic        load_ready;
  logic [1:0]  load_seg;
  logic [31:0] load_bytes;
  string       load_file;

  heatvit_tensor_executor dut (
    .clk                   (clk),
    .rst_n                 (rst_n),
    .abort                 (abort),
    .desc_valid            (desc_valid),
    .desc_ready            (desc_ready),
    .desc                  (desc_d),
    .current_token_count   (8'd197),
    .current_package_present(1'b0),
    .input_base            (INPUT_BASE),
    .input_bytes           (INPUT_BYTES),
    .weight_base           (WEIGHT_BASE),
    .weight_bytes          (WEIGHT_BYTES),
    .scratch_base          (SCRATCH_BASE),
    .scratch_bytes         (SCRATCH_BYTES),
    .output_base           (OUTPUT_BASE),
    .output_bytes          (OUTPUT_BYTES),
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
    .SEG_COUNT  (3),
    .SEG0_FILE  (""),
    .SEG0_BASE  (INPUT_BASE),
    .SEG0_BYTES (INPUT_BYTES),
    .SEG1_FILE  (""),
    .SEG1_BASE  (WEIGHT_BASE),
    .SEG1_BYTES (WEIGHT_BYTES),
    .SEG2_FILE  (""),
    .SEG2_BASE  (SCRATCH_BASE),
    .SEG2_BYTES (SCRATCH_BYTES),
    .LFSR_INIT  (16'hBEEF)
  ) dut_mem (
    .clk          (clk),
    .rst_n        (rst_n),
    .stall_mask   (16'h0000),
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

  logic [7:0]  desc_idx;
  logic [63:0] cmd_end;

  function automatic logic in_region(
    input logic [31:0] addr,
    input logic [63:0] end_addr,
    input logic [31:0] base,
    input int          bytes
  );
    return (addr >= base) && (end_addr <= {32'd0, base} + bytes);
  endfunction

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      desc_idx <= 8'd0;
    end else begin
      if (done) desc_idx <= desc_idx + 8'd1;
      if (obs_cmd_valid) begin
        cmd_end = obs_cmd_addr + {13'd0, obs_cmd_len, 3'b000};
        case (desc_idx)
          8'd0: begin
            if (!(in_region(obs_cmd_addr, cmd_end, D0_R0_BASE, D0_R0_BYTES) ||
                  in_region(obs_cmd_addr, cmd_end, D0_R1_BASE, D0_R1_BYTES)))
              tb_fatal("desc0 trace outside regions");
          end
          8'd1: begin
            if (!(in_region(obs_cmd_addr, cmd_end, D1_R0_BASE, D1_R0_BYTES) ||
                  in_region(obs_cmd_addr, cmd_end, D1_R1_BASE, D1_R1_BYTES) ||
                  in_region(obs_cmd_addr, cmd_end, D1_R2_BASE, D1_R2_BYTES) ||
                  in_region(obs_cmd_addr, cmd_end, D1_R3_BASE, D1_R3_BYTES)))
              tb_fatal("desc1 trace outside regions");
          end
          8'd2: begin
            if (!(in_region(obs_cmd_addr, cmd_end, D2_R0_BASE, D2_R0_BYTES) ||
                  in_region(obs_cmd_addr, cmd_end, D2_R1_BASE, D2_R1_BYTES) ||
                  in_region(obs_cmd_addr, cmd_end, D2_R2_BASE, D2_R2_BYTES) ||
                  in_region(obs_cmd_addr, cmd_end, D2_R3_BASE, D2_R3_BYTES))) begin
              $display("desc2 trace: addr=%h len=%0d end=%h",
                       obs_cmd_addr, obs_cmd_len, cmd_end);
              tb_fatal("desc2 trace outside regions");
            end
          end
          default: tb_fatal("unexpected descriptor index in trace monitor");
        endcase
      end
    end
  end

  logic [63:0] exp_words [0:ACT_WORDS-1];
  int          i;
  logic [7:0]  got_byte;
  logic [7:0]  want_byte;
  string       vector_dir;
  string       image_file;
  string       weight_file;
  string       expected_file;

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

  task automatic run_desc(input logic [319:0] word);
    desc_d    = word;
    desc_valid = 1'b1;
    wait (desc_ready);
    @(posedge clk);
    #1;
    desc_valid = 1'b0;
    wait (done);
    #1;
    if (error_valid) begin
      $display("descriptor error code=%0d", error_code);
      tb_fatal("patch embedding descriptor reported an error");
    end
  endtask

  task automatic dbg_rd(input logic [31:0] addr, output logic [7:0] data);
    dbg_valid = 1'b1;
    dbg_addr  = addr;
    @(posedge clk);
    #1;
    data = dbg_data;
    dbg_valid = 1'b0;
  endtask

  initial begin
    #500000000;
    $display("WATCHDOG: executor state=%0d busy=%0d done=%0d desc_idx=%0d",
             dut.state, busy, done, desc_idx);
    tb_fatal("tb_patch_embedding watchdog");
  end

  initial begin
    abort      = 1'b0;
    desc_valid = 1'b0;
    desc_d     = '0;
    dbg_valid  = 1'b0;
    dbg_addr   = 32'h00000000;
    load_valid = 1'b0;
    load_seg   = 2'd0;
    load_bytes = 32'd0;
    load_file  = "";
    for (i = 0; i < ACT_WORDS; i++) exp_words[i] = 64'h0000000000000000;

    if (!$value$plusargs("VECTOR_DIR=%s", vector_dir))
      vector_dir = "build/vectors/patch";
    image_file    = {vector_dir, "/image.mem"};
    weight_file   = {vector_dir, "/weight.mem"};
    expected_file = {vector_dir, "/expected.mem"};

    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
    #1;

    load_segment(2'd0, INPUT_BYTES, image_file);
    load_segment(2'd1, WEIGHT_BYTES, weight_file);
    $readmemh(expected_file, exp_words);

    if (OUT_SCALE != -6'sd7) tb_fatal("output scale exponent mismatch");

    run_desc(DESC0);
    run_desc(DESC1);
    run_desc(DESC2);

    @(posedge clk);
    #1;
    if (desc_idx != 8'd3) begin
      $display("desc_idx=%0d expected=3", desc_idx);
      tb_fatal("expected exactly three descriptor completions");
    end

    for (i = 0; i < ACT_BYTES; i++) begin
      dbg_rd(SCRATCH_BASE + ACT_OFFSET + i, got_byte);
      want_byte = exp_words[i / 8][8 * (i % 8) +: 8];
      if (got_byte !== want_byte) begin
        $display("activation byte %0d mismatch: got=%h want=%h", i, got_byte, want_byte);
        tb_fatal("patch embedding activation mismatch");
      end
    end

    $display("TEST_PASS tb_patch_embedding");
    $finish;
  end

endmodule
