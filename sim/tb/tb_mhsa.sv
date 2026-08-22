`timescale 1ns / 1ps

// Task 4: three-head MHSA through the tensor executor.
//
// Runs the eight-descriptor sequence LN -> QKV GEMM -> unpack -> QK^T ->
// Softmax -> Attention*V -> concat -> projection, then compares the QKV,
// Score, Probability, Context, Concat and MSA checkpoints byte for byte.
// The memory trace is checked against the per-descriptor regions from
// sim/generated/mhsa_tb_config.sv.
module tb_mhsa;
  import heatvit_pkg::*;
  import tb_pkg::*;
  import mhsa_tb_config_pkg::*;

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
    .current_token_count   (N[7:0]),
    .current_package_present(1'b0),
    .input_base            (32'h00000000),
    .input_bytes           (32'd0),
    .weight_base           (WEIGHT_BASE),
    .weight_bytes          (WEIGHT_BYTES),
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
    .SEG_COUNT  (2),
    .SEG0_FILE  (""),
    .SEG0_BASE  (WEIGHT_BASE),
    .SEG0_BYTES (WEIGHT_BYTES),
    .SEG1_FILE  (""),
    .SEG1_BASE  (SCRATCH_BASE),
    .SEG1_BYTES (SCRATCH_BYTES),
    .LFSR_INIT  (16'hF00D)
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

  function automatic logic in_desc(
    input int          d,
    input logic [31:0] addr,
    input logic [63:0] end_addr
  );
    case (d)
      0: return in_region(addr, end_addr, D0_R0_BASE, D0_R0_BYTES) ||
                in_region(addr, end_addr, D0_R1_BASE, D0_R1_BYTES) ||
                in_region(addr, end_addr, D0_R2_BASE, D0_R2_BYTES) ||
                in_region(addr, end_addr, D0_R3_BASE, D0_R3_BYTES);
      1: return in_region(addr, end_addr, D1_R0_BASE, D1_R0_BYTES) ||
                in_region(addr, end_addr, D1_R1_BASE, D1_R1_BYTES) ||
                in_region(addr, end_addr, D1_R2_BASE, D1_R2_BYTES) ||
                in_region(addr, end_addr, D1_R3_BASE, D1_R3_BYTES);
      2: return in_region(addr, end_addr, D2_R0_BASE, D2_R0_BYTES) ||
                in_region(addr, end_addr, D2_R1_BASE, D2_R1_BYTES);
      3: return in_region(addr, end_addr, D3_R0_BASE, D3_R0_BYTES) ||
                in_region(addr, end_addr, D3_R1_BASE, D3_R1_BYTES) ||
                in_region(addr, end_addr, D3_R2_BASE, D3_R2_BYTES);
      4: return in_region(addr, end_addr, D4_R0_BASE, D4_R0_BYTES) ||
                in_region(addr, end_addr, D4_R1_BASE, D4_R1_BYTES);
      5: return in_region(addr, end_addr, D5_R0_BASE, D5_R0_BYTES) ||
                in_region(addr, end_addr, D5_R1_BASE, D5_R1_BYTES) ||
                in_region(addr, end_addr, D5_R2_BASE, D5_R2_BYTES);
      6: return in_region(addr, end_addr, D6_R0_BASE, D6_R0_BYTES) ||
                in_region(addr, end_addr, D6_R1_BASE, D6_R1_BYTES);
      7: return in_region(addr, end_addr, D7_R0_BASE, D7_R0_BYTES) ||
                in_region(addr, end_addr, D7_R1_BASE, D7_R1_BYTES) ||
                in_region(addr, end_addr, D7_R2_BASE, D7_R2_BYTES) ||
                in_region(addr, end_addr, D7_R3_BASE, D7_R3_BYTES);
      default: return 1'b0;
    endcase
  endfunction

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      desc_idx <= 8'd0;
    end else begin
      if (done) desc_idx <= desc_idx + 8'd1;
      if (obs_cmd_valid) begin
        cmd_end = obs_cmd_addr + {13'd0, obs_cmd_len, 3'b000};
        if (!in_desc(int'(desc_idx), obs_cmd_addr, cmd_end)) begin
          $display("trace desc=%0d addr=%h len=%0d end=%h",
                   desc_idx, obs_cmd_addr, obs_cmd_len, cmd_end);
          tb_fatal("MHSA memory trace outside declared regions");
        end
      end
    end
  end

  logic [63:0] qkv_exp [0:QKV_BYTES / 8 - 1];
  logic [63:0] score_exp [0:(SCORE_BYTES + 7) / 8 - 1];
  logic [63:0] prob_exp [0:(PROB_BYTES + 7) / 8 - 1];
  logic [63:0] context_exp [0:CONTEXT_BYTES / 8 - 1];
  logic [63:0] concat_exp [0:CONCAT_BYTES / 8 - 1];
  logic [63:0] msa_exp [0:MSA_BYTES / 8 - 1];
  int          i;
  logic [7:0]  got_byte;
  string       vector_dir;
  string       x_file;
  string       weight_file;

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
      tb_fatal("MHSA descriptor reported an error");
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

  task automatic check_region(
    input int          off,
    input int          bytes,
    input string       name
  );
    begin
      logic [7:0] want_byte;
      for (i = 0; i < bytes; i++) begin
        dbg_rd(SCRATCH_BASE + off + i, got_byte);
        case (name)
          "qkv":     want_byte = qkv_exp[i / 8][8 * (i % 8) +: 8];
          "score":   want_byte = score_exp[i / 8][8 * (i % 8) +: 8];
          "prob":    want_byte = prob_exp[i / 8][8 * (i % 8) +: 8];
          "context": want_byte = context_exp[i / 8][8 * (i % 8) +: 8];
          "concat":  want_byte = concat_exp[i / 8][8 * (i % 8) +: 8];
          "msa":     want_byte = msa_exp[i / 8][8 * (i % 8) +: 8];
          default:   want_byte = 8'h00;
        endcase
        if (got_byte !== want_byte) begin
          $display("%s byte %0d mismatch: got=%h want=%h", name, i, got_byte, want_byte);
          tb_fatal("MHSA checkpoint mismatch");
        end
      end
    end
  endtask

  initial begin
    #500000000;
    $display("WATCHDOG: executor state=%0d busy=%0d desc_idx=%0d",
             dut.state, busy, desc_idx);
    tb_fatal("tb_mhsa watchdog");
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
    for (i = 0; i < QKV_BYTES / 8; i++) qkv_exp[i] = 64'h0000000000000000;
    for (i = 0; i < (SCORE_BYTES + 7) / 8; i++) score_exp[i] = 64'h0000000000000000;
    for (i = 0; i < (PROB_BYTES + 7) / 8; i++) prob_exp[i] = 64'h0000000000000000;
    for (i = 0; i < CONTEXT_BYTES / 8; i++) context_exp[i] = 64'h0000000000000000;
    for (i = 0; i < CONCAT_BYTES / 8; i++) concat_exp[i] = 64'h0000000000000000;
    for (i = 0; i < MSA_BYTES / 8; i++) msa_exp[i] = 64'h0000000000000000;

    if (!$value$plusargs("VECTOR_DIR=%s", vector_dir))
      vector_dir = "build/vectors/mhsa9";
    x_file      = {vector_dir, "/x.mem"};
    weight_file = {vector_dir, "/weight.mem"};

    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
    #1;

    load_segment(2'd1, 192 * N, x_file);
    load_segment(2'd0, WEIGHT_BYTES, weight_file);
    $readmemh({vector_dir, "/qkv_expected.mem"}, qkv_exp);
    $readmemh({vector_dir, "/score_expected.mem"}, score_exp);
    $readmemh({vector_dir, "/prob_expected.mem"}, prob_exp);
    $readmemh({vector_dir, "/context_expected.mem"}, context_exp);
    $readmemh({vector_dir, "/concat_expected.mem"}, concat_exp);
    $readmemh({vector_dir, "/msa_expected.mem"}, msa_exp);

    if (OUT_SCALE != -6'sd7) tb_fatal("MHSA output scale mismatch");
    if (SCORE_SCALE != -6'sd17) tb_fatal("MHSA score scale mismatch");
    if (PROB_SCALE != -6'sd8) tb_fatal("MHSA probability scale mismatch");

    run_desc(DESC0);
    run_desc(DESC1);
    run_desc(DESC2);
    run_desc(DESC3);
    run_desc(DESC4);
    run_desc(DESC5);
    run_desc(DESC6);
    run_desc(DESC7);

    @(posedge clk);
    #1;
    if (desc_idx != 8'd8) begin
      $display("desc_idx=%0d expected=8", desc_idx);
      tb_fatal("expected exactly eight descriptor completions");
    end

    check_region(QKV_OFF, QKV_BYTES, "qkv");
    check_region(SCORE_OFF, SCORE_BYTES, "score");
    check_region(PROB_OFF, PROB_BYTES, "prob");
    check_region(CONTEXT_OFF, CONTEXT_BYTES, "context");
    check_region(CONCAT_OFF, CONCAT_BYTES, "concat");
    check_region(MSA_OFF, MSA_BYTES, "msa");

    $display("TEST_PASS tb_mhsa");
    $finish;
  end

endmodule
