`timescale 1ns / 1ps

// Task 5: complete HeatViT Token Selector descriptor sequence through the
// tensor executor.
//
// Runs the fixed 12-descriptor selector sequence (local MLP -> global mean
// -> concat -> score MLP x3 -> selector softmax -> head statistics ->
// head-weight MLP x2 -> head fuse -> finalize) on the full N=197 vector,
// then compares every checkpoint byte for byte: local, global, concat, the
// three score layers, logits, keep scores, stats, head-weight hidden and
// Q0.16 weights, fused scores and the compacted output. The finalize must
// produce exactly one atomic state update with the manifest's
// next_token_count/next_package_present, at least two normal tokens must be
// pruned and at least one kept, and the memory trace must stay inside the
// per-descriptor regions from selector_tb_config.sv.
module tb_token_selector;
  import heatvit_pkg::*;
  import tb_pkg::*;
  import selector_tb_config_pkg::*;

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
  logic        load_valid;
  logic        load_ready;
  logic [1:0]  load_seg;
  logic [31:0] load_bytes;
  string       load_file;

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

  // Invariant: state_update_valid may only assert together with done. The
  // scheduler samples both on the same cycle, so a lone pulse would be lost.
  always @(posedge clk) begin
    if (state_update_valid && !done) begin
      tb_fatal("state_update_valid pulsed without done");
    end
  end

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

  logic [7:0]  desc_idx;
  logic [63:0] cmd_end;
  int          state_pulses;
  int          i;

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
                in_region(addr, end_addr, D1_R1_BASE, D1_R1_BYTES);
      2: return in_region(addr, end_addr, D2_R0_BASE, D2_R0_BYTES) ||
                in_region(addr, end_addr, D2_R1_BASE, D2_R1_BYTES) ||
                in_region(addr, end_addr, D2_R2_BASE, D2_R2_BYTES);
      3: return in_region(addr, end_addr, D3_R0_BASE, D3_R0_BYTES) ||
                in_region(addr, end_addr, D3_R1_BASE, D3_R1_BYTES) ||
                in_region(addr, end_addr, D3_R2_BASE, D3_R2_BYTES) ||
                in_region(addr, end_addr, D3_R3_BASE, D3_R3_BYTES);
      4: return in_region(addr, end_addr, D4_R0_BASE, D4_R0_BYTES) ||
                in_region(addr, end_addr, D4_R1_BASE, D4_R1_BYTES) ||
                in_region(addr, end_addr, D4_R2_BASE, D4_R2_BYTES) ||
                in_region(addr, end_addr, D4_R3_BASE, D4_R3_BYTES);
      5: return in_region(addr, end_addr, D5_R0_BASE, D5_R0_BYTES) ||
                in_region(addr, end_addr, D5_R1_BASE, D5_R1_BYTES) ||
                in_region(addr, end_addr, D5_R2_BASE, D5_R2_BYTES) ||
                in_region(addr, end_addr, D5_R3_BASE, D5_R3_BYTES);
      6: return in_region(addr, end_addr, D6_R0_BASE, D6_R0_BYTES) ||
                in_region(addr, end_addr, D6_R1_BASE, D6_R1_BYTES);
      7: return in_region(addr, end_addr, D7_R0_BASE, D7_R0_BYTES) ||
                in_region(addr, end_addr, D7_R1_BASE, D7_R1_BYTES);
      8: return in_region(addr, end_addr, D8_R0_BASE, D8_R0_BYTES) ||
                in_region(addr, end_addr, D8_R1_BASE, D8_R1_BYTES) ||
                in_region(addr, end_addr, D8_R2_BASE, D8_R2_BYTES) ||
                in_region(addr, end_addr, D8_R3_BASE, D8_R3_BYTES);
      9: return in_region(addr, end_addr, D9_R0_BASE, D9_R0_BYTES) ||
                in_region(addr, end_addr, D9_R1_BASE, D9_R1_BYTES) ||
                in_region(addr, end_addr, D9_R2_BASE, D9_R2_BYTES) ||
                in_region(addr, end_addr, D9_R3_BASE, D9_R3_BYTES);
      10: return in_region(addr, end_addr, D10_R0_BASE, D10_R0_BYTES) ||
                 in_region(addr, end_addr, D10_R1_BASE, D10_R1_BYTES) ||
                 in_region(addr, end_addr, D10_R2_BASE, D10_R2_BYTES);
      11: return in_region(addr, end_addr, D11_R0_BASE, D11_R0_BYTES) ||
                 in_region(addr, end_addr, D11_R1_BASE, D11_R1_BYTES) ||
                 in_region(addr, end_addr, D11_R2_BASE, D11_R2_BYTES);
      default: return 1'b0;
    endcase
  endfunction

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      desc_idx <= 8'd0;
      state_pulses <= 0;
    end else begin
      if (done) desc_idx <= desc_idx + 8'd1;
      if (state_update_valid) state_pulses <= state_pulses + 1;
      if (obs_cmd_valid) begin
        cmd_end = obs_cmd_addr + {13'd0, obs_cmd_len, 3'b000};
        if (!in_desc(int'(desc_idx), obs_cmd_addr, cmd_end)) begin
          $display("trace desc=%0d addr=%h len=%0d end=%h",
                   desc_idx, obs_cmd_addr, obs_cmd_len, cmd_end);
          tb_fatal("token selector memory trace outside declared regions");
        end
      end
    end
  end

  logic [63:0] local_exp [0:(197 * 96 + 7) / 8];
  logic [63:0] global_exp [0:12];
  logic [63:0] concat_exp [0:(197 * 192 + 7) / 8];
  logic [63:0] h1_exp [0:(197 * 96 + 7) / 8];
  logic [63:0] h2_exp [0:(197 * 48 + 7) / 8];
  logic [63:0] logits_exp [0:(197 * 6 + 7) / 8];
  logic [63:0] keep_exp [0:(197 * 12 + 7) / 8];
  logic [63:0] stats_exp [0:(197 * 3 + 7) / 8];
  logic [63:0] hw_hidden_exp [0:(197 * 3 + 7) / 8];
  logic [63:0] hw_exp [0:(197 * 12 + 7) / 8];
  logic [63:0] fused_exp [0:(197 * 4 + 7) / 8];
  logic [63:0] out_exp [0:(197 * 192 + 7) / 8];
  logic [7:0]  got_byte;
  string       vector_dir;
  string       input_file;
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
      tb_fatal("token selector descriptor reported an error");
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
          "local":     want_byte = local_exp[i / 8][8 * (i % 8) +: 8];
          "global":    want_byte = global_exp[i / 8][8 * (i % 8) +: 8];
          "concat":    want_byte = concat_exp[i / 8][8 * (i % 8) +: 8];
          "h1":        want_byte = h1_exp[i / 8][8 * (i % 8) +: 8];
          "h2":        want_byte = h2_exp[i / 8][8 * (i % 8) +: 8];
          "logits":    want_byte = logits_exp[i / 8][8 * (i % 8) +: 8];
          "keep":      want_byte = keep_exp[i / 8][8 * (i % 8) +: 8];
          "stats":     want_byte = stats_exp[i / 8][8 * (i % 8) +: 8];
          "hw_hidden": want_byte = hw_hidden_exp[i / 8][8 * (i % 8) +: 8];
          "hw":        want_byte = hw_exp[i / 8][8 * (i % 8) +: 8];
          "fused":     want_byte = fused_exp[i / 8][8 * (i % 8) +: 8];
          "out":       want_byte = out_exp[i / 8][8 * (i % 8) +: 8];
          default:     want_byte = 8'h00;
        endcase
        if (got_byte !== want_byte) begin
          $display("%s byte %0d mismatch: got=%h want=%h", name, i,
                   got_byte, want_byte);
          tb_fatal("token selector checkpoint mismatch");
        end
      end
    end
  endtask

  initial begin
    #10000000000;
    $display("WATCHDOG: desc_idx=%0d busy=%0d", desc_idx, busy);
    tb_fatal("tb_token_selector watchdog");
  end

  initial begin
    abort      = 1'b0;
    desc_valid = 1'b0;
    desc_d     = '0;
    dbg_valid  = 1'b0;
    dbg_addr   = 32'h00000000;
    dbg_w_valid = 1'b0;
    dbg_w_addr  = 32'h00000000;
    dbg_w_data  = 8'h00;
    load_valid = 1'b0;
    load_seg   = 2'd0;
    load_bytes = 32'd0;
    load_file  = "";

    if (!$value$plusargs("STALL_MASK=%s", stall_str)) stall_str = "0";
    if ($sscanf(stall_str, "%h", parsed_mask) != 1) parsed_mask = 16'h0000;
    stall_mask = parsed_mask;

    if (!$value$plusargs("VECTOR_DIR=%s", vector_dir))
      vector_dir = "build/vectors/selector_mixed";
    input_file  = {vector_dir, "/input.mem"};
    weight_file = {vector_dir, "/weight.mem"};

    begin
      int fd;
      fd = $fopen(input_file, "r");
      if (fd == 0) begin
        $display("missing vector file: %s", input_file);
        tb_fatal("token selector vectors missing");
      end
      $fclose(fd);
    end

    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
    #1;

    load_segment(2'd0, WEIGHT_BYTES, weight_file);
    load_segment(2'd1, N * 192, input_file);
    $readmemh({vector_dir, "/local_expected.mem"}, local_exp);
    $readmemh({vector_dir, "/global_expected.mem"}, global_exp);
    $readmemh({vector_dir, "/concat_expected.mem"}, concat_exp);
    $readmemh({vector_dir, "/h1_expected.mem"}, h1_exp);
    $readmemh({vector_dir, "/h2_expected.mem"}, h2_exp);
    $readmemh({vector_dir, "/logits_expected.mem"}, logits_exp);
    $readmemh({vector_dir, "/keep_expected.mem"}, keep_exp);
    $readmemh({vector_dir, "/stats_expected.mem"}, stats_exp);
    $readmemh({vector_dir, "/hw_hidden_expected.mem"}, hw_hidden_exp);
    $readmemh({vector_dir, "/hw_expected.mem"}, hw_exp);
    $readmemh({vector_dir, "/fused_expected.mem"}, fused_exp);
    $readmemh({vector_dir, "/output_expected.mem"}, out_exp);

    // Sentinel-fill the output tail rows beyond NEXT_TOKEN_COUNT.
    for (i = OUT_BYTES; i < N * 192; i++)
      dbg_w(SCRATCH_BASE + OUTPUT_OFF + i, 8'ha5);

    run_desc(DESC0);
    run_desc(DESC1);
    run_desc(DESC2);
    run_desc(DESC3);
    run_desc(DESC4);
    run_desc(DESC5);
    run_desc(DESC6);
    run_desc(DESC7);
    run_desc(DESC8);
    run_desc(DESC9);
    run_desc(DESC10);
    run_desc(DESC11);

    @(posedge clk);
    #1;
    if (desc_idx != 8'd12) begin
      $display("desc_idx=%0d expected=12", desc_idx);
      tb_fatal("expected exactly twelve descriptor completions");
    end
    if (state_pulses != 1) begin
      $display("state_update_valid pulses=%0d expected=1", state_pulses);
      tb_fatal("token selector state update pulse count mismatch");
    end
    if (next_token_count != NEXT_TOKEN_COUNT) begin
      $display("next_token_count=%0d expected=%0d", next_token_count,
               NEXT_TOKEN_COUNT);
      tb_fatal("token selector token count mismatch");
    end
    if (next_package_present != NEXT_PACKAGE_PRESENT) begin
      $display("next_package_present=%b expected=%b", next_package_present,
               NEXT_PACKAGE_PRESENT);
      tb_fatal("token selector package present mismatch");
    end

    check_region(LOCAL_OFF, LOCAL_BYTES, "local");
    check_region(GLOBAL_OFF, GLOBAL_BYTES, "global");
    check_region(CONCAT_OFF, CONCAT_BYTES, "concat");
    check_region(H1_OFF, H1_BYTES, "h1");
    check_region(H2_OFF, H2_BYTES, "h2");
    check_region(LOGITS_OFF, LOGITS_BYTES, "logits");
    check_region(KEEP_OFF, KEEP_BYTES, "keep");
    check_region(STATS_OFF, STATS_BYTES, "stats");
    check_region(HW_HIDDEN_OFF, HW_HIDDEN_BYTES, "hw_hidden");
    check_region(HW_OFF, HW_BYTES, "hw");
    check_region(FUSED_OFF, FUSED_BYTES, "fused");
    check_region(OUTPUT_OFF, OUT_BYTES, "out");
    // Sentinel tail untouched.
    for (i = OUT_BYTES; i < N * 192; i++) begin
      dbg_rd(SCRATCH_BASE + OUTPUT_OFF + i, got_byte);
      if (got_byte !== 8'ha5) begin
        $display("output tail byte %0d modified: got=%h", i, got_byte);
        tb_fatal("token selector output tail modified");
      end
    end

    $display("TEST_PASS tb_token_selector");
    $finish;
  end

endmodule
