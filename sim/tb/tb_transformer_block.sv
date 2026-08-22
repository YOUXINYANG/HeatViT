`timescale 1ns / 1ps

// Task 6: two complete Pre-LN Transformer blocks through the executor.
//
// Block order is fixed: LN1 -> QKV GEMM -> QKV unpack -> QK^T -> Softmax ->
// Attention*V -> Head concat -> Projection -> Residual1 -> LN2 -> FC1+GELU ->
// FC2 -> Residual2, run twice back to back (26 descriptors). The TB asserts
// the opcode order before submitting each descriptor and compares the LN1,
// MSA, Y, LN2, hidden, FFN and Z checkpoints of both blocks byte for byte;
// the dst tail rows beyond N keep their 0xA5 sentinel untouched. Every
// descriptor prints its index, opcode, start/end cycles and token count.
module tb_transformer_block;
  import heatvit_pkg::*;
  import tb_pkg::*;
  import block_tb_config_pkg::*;

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
  logic        dbg_w_valid;
  logic [31:0] dbg_w_addr;
  logic [7:0]  dbg_w_data;
  logic        load_valid;
  logic        load_ready;
  logic [1:0]  load_seg;
  logic [31:0] load_bytes;
  string       load_file;
  logic [15:0] stall_mask = 16'h0000;

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
    .LFSR_INIT  (16'h1357)
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
    .load_valid   (load_valid),
    .load_ready   (load_ready),
    .load_seg     (load_seg),
    .load_bytes   (load_bytes),
    .load_file    (load_file)
  );

  always #5 clk = ~clk;

  logic        [31:0] cycle_count = 32'd0;
  logic        [7:0]  desc_idx;
  logic        [63:0] cmd_end;

  always_ff @(posedge clk) begin
    cycle_count <= cycle_count + 32'd1;
  end

  // Canonical 13-op block order, repeated for the second block.
  localparam logic [7:0] EXPECTED_OP [0:25] = '{
    8'd4, 8'd3, 8'd6, 8'd3, 8'd8, 8'd3, 8'd7, 8'd3, 8'd5, 8'd4, 8'd3, 8'd3, 8'd5,
    8'd4, 8'd3, 8'd6, 8'd3, 8'd8, 8'd3, 8'd7, 8'd3, 8'd5, 8'd4, 8'd3, 8'd3, 8'd5
  };

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
      0:  return in_region(addr, end_addr, D0_R0_BASE, D0_R0_BYTES) ||
                 in_region(addr, end_addr, D0_R1_BASE, D0_R1_BYTES) ||
                 in_region(addr, end_addr, D0_R2_BASE, D0_R2_BYTES) ||
                 in_region(addr, end_addr, D0_R3_BASE, D0_R3_BYTES);
      1:  return in_region(addr, end_addr, D1_R0_BASE, D1_R0_BYTES) ||
                 in_region(addr, end_addr, D1_R1_BASE, D1_R1_BYTES) ||
                 in_region(addr, end_addr, D1_R2_BASE, D1_R2_BYTES) ||
                 in_region(addr, end_addr, D1_R3_BASE, D1_R3_BYTES);
      2:  return in_region(addr, end_addr, D2_R0_BASE, D2_R0_BYTES) ||
                 in_region(addr, end_addr, D2_R1_BASE, D2_R1_BYTES);
      3:  return in_region(addr, end_addr, D3_R0_BASE, D3_R0_BYTES) ||
                 in_region(addr, end_addr, D3_R1_BASE, D3_R1_BYTES) ||
                 in_region(addr, end_addr, D3_R2_BASE, D3_R2_BYTES);
      4:  return in_region(addr, end_addr, D4_R0_BASE, D4_R0_BYTES) ||
                 in_region(addr, end_addr, D4_R1_BASE, D4_R1_BYTES);
      5:  return in_region(addr, end_addr, D5_R0_BASE, D5_R0_BYTES) ||
                 in_region(addr, end_addr, D5_R1_BASE, D5_R1_BYTES) ||
                 in_region(addr, end_addr, D5_R2_BASE, D5_R2_BYTES);
      6:  return in_region(addr, end_addr, D6_R0_BASE, D6_R0_BYTES) ||
                 in_region(addr, end_addr, D6_R1_BASE, D6_R1_BYTES);
      7:  return in_region(addr, end_addr, D7_R0_BASE, D7_R0_BYTES) ||
                 in_region(addr, end_addr, D7_R1_BASE, D7_R1_BYTES) ||
                 in_region(addr, end_addr, D7_R2_BASE, D7_R2_BYTES) ||
                 in_region(addr, end_addr, D7_R3_BASE, D7_R3_BYTES);
      8:  return in_region(addr, end_addr, D8_R0_BASE, D8_R0_BYTES) ||
                 in_region(addr, end_addr, D8_R1_BASE, D8_R1_BYTES) ||
                 in_region(addr, end_addr, D8_R2_BASE, D8_R2_BYTES);
      9:  return in_region(addr, end_addr, D9_R0_BASE, D9_R0_BYTES) ||
                 in_region(addr, end_addr, D9_R1_BASE, D9_R1_BYTES) ||
                 in_region(addr, end_addr, D9_R2_BASE, D9_R2_BYTES) ||
                 in_region(addr, end_addr, D9_R3_BASE, D9_R3_BYTES);
      10: return in_region(addr, end_addr, D10_R0_BASE, D10_R0_BYTES) ||
                 in_region(addr, end_addr, D10_R1_BASE, D10_R1_BYTES) ||
                 in_region(addr, end_addr, D10_R2_BASE, D10_R2_BYTES) ||
                 in_region(addr, end_addr, D10_R3_BASE, D10_R3_BYTES);
      11: return in_region(addr, end_addr, D11_R0_BASE, D11_R0_BYTES) ||
                 in_region(addr, end_addr, D11_R1_BASE, D11_R1_BYTES) ||
                 in_region(addr, end_addr, D11_R2_BASE, D11_R2_BYTES) ||
                 in_region(addr, end_addr, D11_R3_BASE, D11_R3_BYTES);
      12: return in_region(addr, end_addr, D12_R0_BASE, D12_R0_BYTES) ||
                 in_region(addr, end_addr, D12_R1_BASE, D12_R1_BYTES) ||
                 in_region(addr, end_addr, D12_R2_BASE, D12_R2_BYTES);
      13: return in_region(addr, end_addr, D13_R0_BASE, D13_R0_BYTES) ||
                 in_region(addr, end_addr, D13_R1_BASE, D13_R1_BYTES) ||
                 in_region(addr, end_addr, D13_R2_BASE, D13_R2_BYTES) ||
                 in_region(addr, end_addr, D13_R3_BASE, D13_R3_BYTES);
      14: return in_region(addr, end_addr, D14_R0_BASE, D14_R0_BYTES) ||
                 in_region(addr, end_addr, D14_R1_BASE, D14_R1_BYTES) ||
                 in_region(addr, end_addr, D14_R2_BASE, D14_R2_BYTES) ||
                 in_region(addr, end_addr, D14_R3_BASE, D14_R3_BYTES);
      15: return in_region(addr, end_addr, D15_R0_BASE, D15_R0_BYTES) ||
                 in_region(addr, end_addr, D15_R1_BASE, D15_R1_BYTES);
      16: return in_region(addr, end_addr, D16_R0_BASE, D16_R0_BYTES) ||
                 in_region(addr, end_addr, D16_R1_BASE, D16_R1_BYTES) ||
                 in_region(addr, end_addr, D16_R2_BASE, D16_R2_BYTES);
      17: return in_region(addr, end_addr, D17_R0_BASE, D17_R0_BYTES) ||
                 in_region(addr, end_addr, D17_R1_BASE, D17_R1_BYTES);
      18: return in_region(addr, end_addr, D18_R0_BASE, D18_R0_BYTES) ||
                 in_region(addr, end_addr, D18_R1_BASE, D18_R1_BYTES) ||
                 in_region(addr, end_addr, D18_R2_BASE, D18_R2_BYTES);
      19: return in_region(addr, end_addr, D19_R0_BASE, D19_R0_BYTES) ||
                 in_region(addr, end_addr, D19_R1_BASE, D19_R1_BYTES);
      20: return in_region(addr, end_addr, D20_R0_BASE, D20_R0_BYTES) ||
                 in_region(addr, end_addr, D20_R1_BASE, D20_R1_BYTES) ||
                 in_region(addr, end_addr, D20_R2_BASE, D20_R2_BYTES) ||
                 in_region(addr, end_addr, D20_R3_BASE, D20_R3_BYTES);
      21: return in_region(addr, end_addr, D21_R0_BASE, D21_R0_BYTES) ||
                 in_region(addr, end_addr, D21_R1_BASE, D21_R1_BYTES) ||
                 in_region(addr, end_addr, D21_R2_BASE, D21_R2_BYTES);
      22: return in_region(addr, end_addr, D22_R0_BASE, D22_R0_BYTES) ||
                 in_region(addr, end_addr, D22_R1_BASE, D22_R1_BYTES) ||
                 in_region(addr, end_addr, D22_R2_BASE, D22_R2_BYTES) ||
                 in_region(addr, end_addr, D22_R3_BASE, D22_R3_BYTES);
      23: return in_region(addr, end_addr, D23_R0_BASE, D23_R0_BYTES) ||
                 in_region(addr, end_addr, D23_R1_BASE, D23_R1_BYTES) ||
                 in_region(addr, end_addr, D23_R2_BASE, D23_R2_BYTES) ||
                 in_region(addr, end_addr, D23_R3_BASE, D23_R3_BYTES);
      24: return in_region(addr, end_addr, D24_R0_BASE, D24_R0_BYTES) ||
                 in_region(addr, end_addr, D24_R1_BASE, D24_R1_BYTES) ||
                 in_region(addr, end_addr, D24_R2_BASE, D24_R2_BYTES) ||
                 in_region(addr, end_addr, D24_R3_BASE, D24_R3_BYTES);
      25: return in_region(addr, end_addr, D25_R0_BASE, D25_R0_BYTES) ||
                 in_region(addr, end_addr, D25_R1_BASE, D25_R1_BYTES) ||
                 in_region(addr, end_addr, D25_R2_BASE, D25_R2_BYTES);
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
          tb_fatal("transformer block memory trace outside declared regions");
        end
      end
    end
  end

  logic [63:0] b0_ln1_exp [0:LN_BYTES / 8 - 1];
  logic [63:0] b0_msa_exp [0:MSA_BYTES / 8 - 1];
  logic [63:0] b0_y_exp [0:Y_BYTES / 8 - 1];
  logic [63:0] b0_ln2_exp [0:LN_BYTES / 8 - 1];
  logic [63:0] b0_hidden_exp [0:HIDDEN_BYTES / 8 - 1];
  logic [63:0] b0_ffn_out_exp [0:FFN_OUT_BYTES / 8 - 1];
  logic [63:0] b0_z_exp [0:Z_BYTES / 8 - 1];
  logic [63:0] b1_ln1_exp [0:LN_BYTES / 8 - 1];
  logic [63:0] b1_msa_exp [0:MSA_BYTES / 8 - 1];
  logic [63:0] b1_y_exp [0:Y_BYTES / 8 - 1];
  logic [63:0] b1_ln2_exp [0:LN_BYTES / 8 - 1];
  logic [63:0] b1_hidden_exp [0:HIDDEN_BYTES / 8 - 1];
  logic [63:0] b1_ffn_out_exp [0:FFN_OUT_BYTES / 8 - 1];
  logic [63:0] b1_z_exp [0:Z_BYTES / 8 - 1];

  int          i;
  logic [7:0]  got_byte;
  string       vector_dir;
  string       x_file;
  string       weight_file;
  string       stall_str;
  logic [15:0] parsed_mask;
  int          fd;

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

  task automatic run_desc(input int idx, input logic [319:0] word);
    int unsigned start_cycle;
    if (word[319:312] !== EXPECTED_OP[idx]) begin
      $display("desc[%0d] opcode=%h expected=%h",
               idx, word[319:312], EXPECTED_OP[idx]);
      tb_fatal("transformer block descriptor order mismatch");
    end
    desc_d    = word;
    desc_valid = 1'b1;
    start_cycle = cycle_count;
    wait (desc_ready);
    @(posedge clk);
    #1;
    desc_valid = 1'b0;
    wait (done);
    #1;
    if (error_valid) begin
      $display("descriptor error code=%0d", error_code);
      tb_fatal("transformer block descriptor reported an error");
    end
    $display("desc[%0d] opcode=%0d start=%0d end=%0d tokens=%0d",
             idx, word[319:312], start_cycle, cycle_count, N);
  endtask

  task automatic dbg_rd(input logic [31:0] addr, output logic [7:0] data);
    dbg_valid = 1'b1;
    dbg_addr  = addr;
    @(posedge clk);
    #1;
    data = dbg_data;
    dbg_valid = 1'b0;
  endtask

  task automatic dbg_w(input logic [31:0] addr, input logic [7:0] data);
    dbg_w_valid = 1'b1;
    dbg_w_addr  = addr;
    dbg_w_data  = data;
    @(posedge clk);
    #1;
    dbg_w_valid = 1'b0;
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
          "b0_ln1":     want_byte = b0_ln1_exp[i / 8][8 * (i % 8) +: 8];
          "b0_msa":     want_byte = b0_msa_exp[i / 8][8 * (i % 8) +: 8];
          "b0_y":       want_byte = b0_y_exp[i / 8][8 * (i % 8) +: 8];
          "b0_ln2":     want_byte = b0_ln2_exp[i / 8][8 * (i % 8) +: 8];
          "b0_hidden":  want_byte = b0_hidden_exp[i / 8][8 * (i % 8) +: 8];
          "b0_ffn_out": want_byte = b0_ffn_out_exp[i / 8][8 * (i % 8) +: 8];
          "b0_z":       want_byte = b0_z_exp[i / 8][8 * (i % 8) +: 8];
          "b1_ln1":     want_byte = b1_ln1_exp[i / 8][8 * (i % 8) +: 8];
          "b1_msa":     want_byte = b1_msa_exp[i / 8][8 * (i % 8) +: 8];
          "b1_y":       want_byte = b1_y_exp[i / 8][8 * (i % 8) +: 8];
          "b1_ln2":     want_byte = b1_ln2_exp[i / 8][8 * (i % 8) +: 8];
          "b1_hidden":  want_byte = b1_hidden_exp[i / 8][8 * (i % 8) +: 8];
          "b1_ffn_out": want_byte = b1_ffn_out_exp[i / 8][8 * (i % 8) +: 8];
          "b1_z":       want_byte = b1_z_exp[i / 8][8 * (i % 8) +: 8];
          default:      want_byte = 8'h00;
        endcase
        if (got_byte !== want_byte) begin
          $display("%s byte %0d mismatch: got=%h want=%h", name, i, got_byte, want_byte);
          tb_fatal("transformer block checkpoint mismatch");
        end
      end
    end
  endtask

  initial begin
    #1500000000;
    $display("WATCHDOG: executor state=%0d busy=%0d desc_idx=%0d",
             dut.state, busy, desc_idx);
    tb_fatal("tb_transformer_block watchdog");
  end

  initial begin
    abort       = 1'b0;
    desc_valid  = 1'b0;
    desc_d      = '0;
    dbg_valid   = 1'b0;
    dbg_addr    = 32'h00000000;
    dbg_w_valid = 1'b0;
    dbg_w_addr  = 32'h00000000;
    dbg_w_data  = 8'h00;
    load_valid  = 1'b0;
    load_seg    = 2'd0;
    load_bytes  = 32'd0;
    load_file   = "";
    for (i = 0; i < LN_BYTES / 8; i++) begin
      b0_ln1_exp[i] = 64'h0000000000000000;
      b0_ln2_exp[i] = 64'h0000000000000000;
      b1_ln1_exp[i] = 64'h0000000000000000;
      b1_ln2_exp[i] = 64'h0000000000000000;
    end
    for (i = 0; i < MSA_BYTES / 8; i++) begin
      b0_msa_exp[i] = 64'h0000000000000000;
      b1_msa_exp[i] = 64'h0000000000000000;
    end
    for (i = 0; i < Y_BYTES / 8; i++) begin
      b0_y_exp[i] = 64'h0000000000000000;
      b1_y_exp[i] = 64'h0000000000000000;
    end
    for (i = 0; i < HIDDEN_BYTES / 8; i++) begin
      b0_hidden_exp[i] = 64'h0000000000000000;
      b1_hidden_exp[i] = 64'h0000000000000000;
    end
    for (i = 0; i < FFN_OUT_BYTES / 8; i++) begin
      b0_ffn_out_exp[i] = 64'h0000000000000000;
      b1_ffn_out_exp[i] = 64'h0000000000000000;
    end
    for (i = 0; i < Z_BYTES / 8; i++) begin
      b0_z_exp[i] = 64'h0000000000000000;
      b1_z_exp[i] = 64'h0000000000000000;
    end

    if (!$value$plusargs("STALL_MASK=%s", stall_str)) stall_str = "0";
    if ($sscanf(stall_str, "%h", parsed_mask) != 1) parsed_mask = 16'h0000;
    stall_mask = parsed_mask;

    if (!$value$plusargs("VECTOR_DIR=%s", vector_dir))
      vector_dir = "build/vectors/block197";
    x_file      = {vector_dir, "/input.mem"};
    weight_file = {vector_dir, "/weight.mem"};

    fd = $fopen(x_file, "r");
    if (fd == 0) begin
      $display("missing block vector file: %s", x_file);
      tb_fatal("block vectors missing");
    end
    $fclose(fd);
    fd = $fopen(weight_file, "r");
    if (fd == 0) begin
      $display("missing block vector file: %s", weight_file);
      tb_fatal("block vectors missing");
    end
    $fclose(fd);

    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
    #1;

    load_segment(2'd1, 192 * N, x_file);
    load_segment(2'd0, WEIGHT_BYTES, weight_file);
    $readmemh({vector_dir, "/b0_ln1_expected.mem"}, b0_ln1_exp);
    $readmemh({vector_dir, "/b0_msa_expected.mem"}, b0_msa_exp);
    $readmemh({vector_dir, "/b0_y_expected.mem"}, b0_y_exp);
    $readmemh({vector_dir, "/b0_ln2_expected.mem"}, b0_ln2_exp);
    $readmemh({vector_dir, "/b0_hidden_expected.mem"}, b0_hidden_exp);
    $readmemh({vector_dir, "/b0_ffn_out_expected.mem"}, b0_ffn_out_exp);
    $readmemh({vector_dir, "/b0_z_expected.mem"}, b0_z_exp);
    $readmemh({vector_dir, "/b1_ln1_expected.mem"}, b1_ln1_exp);
    $readmemh({vector_dir, "/b1_msa_expected.mem"}, b1_msa_exp);
    $readmemh({vector_dir, "/b1_y_expected.mem"}, b1_y_exp);
    $readmemh({vector_dir, "/b1_ln2_expected.mem"}, b1_ln2_exp);
    $readmemh({vector_dir, "/b1_hidden_expected.mem"}, b1_hidden_exp);
    $readmemh({vector_dir, "/b1_ffn_out_expected.mem"}, b1_ffn_out_exp);
    $readmemh({vector_dir, "/b1_z_expected.mem"}, b1_z_exp);

    if (OUT_SCALE != -6'sd7) tb_fatal("block output scale mismatch");

    // Sentinel-fill the dst tail rows beyond N for both blocks.
    for (i = 0; i < Z0_PAD_BYTES; i++)
      dbg_w(SCRATCH_BASE + Z0_PAD_OFF + i, 8'ha5);
    for (i = 0; i < Z1_PAD_BYTES; i++)
      dbg_w(SCRATCH_BASE + Z1_PAD_OFF + i, 8'ha5);

    run_desc(0, DESC0);
    run_desc(1, DESC1);
    run_desc(2, DESC2);
    run_desc(3, DESC3);
    run_desc(4, DESC4);
    run_desc(5, DESC5);
    run_desc(6, DESC6);
    run_desc(7, DESC7);
    run_desc(8, DESC8);
    run_desc(9, DESC9);
    run_desc(10, DESC10);
    run_desc(11, DESC11);
    run_desc(12, DESC12);
    run_desc(13, DESC13);
    run_desc(14, DESC14);
    run_desc(15, DESC15);
    run_desc(16, DESC16);
    run_desc(17, DESC17);
    run_desc(18, DESC18);
    run_desc(19, DESC19);
    run_desc(20, DESC20);
    run_desc(21, DESC21);
    run_desc(22, DESC22);
    run_desc(23, DESC23);
    run_desc(24, DESC24);
    run_desc(25, DESC25);

    @(posedge clk);
    #1;
    if (desc_idx != 8'd26) begin
      $display("desc_idx=%0d expected=26", desc_idx);
      tb_fatal("expected exactly 26 descriptor completions");
    end

    check_region(LN1_0_OFF, LN_BYTES, "b0_ln1");
    check_region(MSA_0_OFF, MSA_BYTES, "b0_msa");
    check_region(Y_0_OFF, Y_BYTES, "b0_y");
    check_region(LN2_0_OFF, LN_BYTES, "b0_ln2");
    check_region(HIDDEN_0_OFF, HIDDEN_BYTES, "b0_hidden");
    check_region(FFN_OUT_0_OFF, FFN_OUT_BYTES, "b0_ffn_out");
    check_region(Z0_OFF, Z_BYTES, "b0_z");
    check_region(LN1_1_OFF, LN_BYTES, "b1_ln1");
    check_region(MSA_1_OFF, MSA_BYTES, "b1_msa");
    check_region(Y_1_OFF, Y_BYTES, "b1_y");
    check_region(LN2_1_OFF, LN_BYTES, "b1_ln2");
    check_region(HIDDEN_1_OFF, HIDDEN_BYTES, "b1_hidden");
    check_region(FFN_OUT_1_OFF, FFN_OUT_BYTES, "b1_ffn_out");
    check_region(Z1_OFF, Z_BYTES, "b1_z");

    for (i = 0; i < Z0_PAD_BYTES; i++) begin
      dbg_rd(SCRATCH_BASE + Z0_PAD_OFF + i, got_byte);
      if (got_byte !== 8'ha5) begin
        $display("dst tail byte %0d changed: got=%h want=a5", i, got_byte);
        tb_fatal("block 0 dst tail sentinel was overwritten");
      end
    end
    for (i = 0; i < Z1_PAD_BYTES; i++) begin
      dbg_rd(SCRATCH_BASE + Z1_PAD_OFF + i, got_byte);
      if (got_byte !== 8'ha5) begin
        $display("dst tail byte %0d changed: got=%h want=a5", i, got_byte);
        tb_fatal("block 1 dst tail sentinel was overwritten");
      end
    end

    $display("TEST_PASS tb_transformer_block");
    $finish;
  end

endmodule
