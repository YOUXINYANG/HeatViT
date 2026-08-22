`timescale 1ns / 1ps

// Task 5: Pre-LN FFN with GELU post-op and residual through the executor.
//
// Runs LN2 -> GEMM(192x768, GELU) -> GEMM(768x192) -> RESIDUAL and compares
// the four checkpoints byte for byte; the dst tail rows beyond N keep their
// 0xA5 sentinel untouched. The memory trace is checked per descriptor.
module tb_ffn;
  import heatvit_pkg::*;
  import tb_pkg::*;
  import ffn_tb_config_pkg::*;

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
    .LFSR_INIT  (16'hABCD)
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
                in_region(addr, end_addr, D2_R1_BASE, D2_R1_BYTES) ||
                in_region(addr, end_addr, D2_R2_BASE, D2_R2_BYTES) ||
                in_region(addr, end_addr, D2_R3_BASE, D2_R3_BYTES);
      3: return in_region(addr, end_addr, D3_R0_BASE, D3_R0_BYTES) ||
                in_region(addr, end_addr, D3_R1_BASE, D3_R1_BYTES) ||
                in_region(addr, end_addr, D3_R2_BASE, D3_R2_BYTES);
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
          tb_fatal("FFN memory trace outside declared regions");
        end
      end
    end
  end

  logic [63:0] ln2_exp [0:LN2_BYTES / 8 - 1];
  logic [63:0] hidden_exp [0:HIDDEN_BYTES / 8 - 1];
  logic [63:0] ffn_out_exp [0:FFN_OUT_BYTES / 8 - 1];
  logic [63:0] z_exp [0:Z_BYTES / 8 - 1];
  int          i;
  logic [7:0]  got_byte;
  string       vector_dir;
  string       y_file;
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
      tb_fatal("FFN descriptor reported an error");
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
          "ln2":     want_byte = ln2_exp[i / 8][8 * (i % 8) +: 8];
          "hidden":  want_byte = hidden_exp[i / 8][8 * (i % 8) +: 8];
          "ffn_out": want_byte = ffn_out_exp[i / 8][8 * (i % 8) +: 8];
          "z":       want_byte = z_exp[i / 8][8 * (i % 8) +: 8];
          default:   want_byte = 8'h00;
        endcase
        if (got_byte !== want_byte) begin
          $display("%s byte %0d mismatch: got=%h want=%h", name, i, got_byte, want_byte);
          tb_fatal("FFN checkpoint mismatch");
        end
      end
    end
  endtask

  initial begin
    #500000000;
    $display("WATCHDOG: executor state=%0d busy=%0d desc_idx=%0d",
             dut.state, busy, desc_idx);
    tb_fatal("tb_ffn watchdog");
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
    for (i = 0; i < LN2_BYTES / 8; i++) ln2_exp[i] = 64'h0000000000000000;
    for (i = 0; i < HIDDEN_BYTES / 8; i++) hidden_exp[i] = 64'h0000000000000000;
    for (i = 0; i < FFN_OUT_BYTES / 8; i++) ffn_out_exp[i] = 64'h0000000000000000;
    for (i = 0; i < Z_BYTES / 8; i++) z_exp[i] = 64'h0000000000000000;

    if (!$value$plusargs("VECTOR_DIR=%s", vector_dir))
      vector_dir = "build/vectors/ffn13";
    y_file      = {vector_dir, "/y.mem"};
    weight_file = {vector_dir, "/weight.mem"};

    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
    #1;

    load_segment(2'd1, 192 * N, y_file);
    load_segment(2'd0, WEIGHT_BYTES, weight_file);
    $readmemh({vector_dir, "/ln2_expected.mem"}, ln2_exp);
    $readmemh({vector_dir, "/hidden_expected.mem"}, hidden_exp);
    $readmemh({vector_dir, "/ffn_out_expected.mem"}, ffn_out_exp);
    $readmemh({vector_dir, "/z_expected.mem"}, z_exp);

    // Sentinel-fill the dst tail rows beyond N.
    for (i = 0; i < Z_PAD_BYTES; i++)
      dbg_w(SCRATCH_BASE + Z_PAD_OFF + i, 8'ha5);

    run_desc(DESC0);
    run_desc(DESC1);
    run_desc(DESC2);
    run_desc(DESC3);

    @(posedge clk);
    #1;
    if (desc_idx != 8'd4) begin
      $display("desc_idx=%0d expected=4", desc_idx);
      tb_fatal("expected exactly four descriptor completions");
    end

    check_region(LN2_OFF, LN2_BYTES, "ln2");
    check_region(HIDDEN_OFF, HIDDEN_BYTES, "hidden");
    check_region(FFN_OUT_OFF, FFN_OUT_BYTES, "ffn_out");
    check_region(Z_OFF, Z_BYTES, "z");

    for (i = 0; i < Z_PAD_BYTES; i++) begin
      dbg_rd(SCRATCH_BASE + Z_PAD_OFF + i, got_byte);
      if (got_byte !== 8'ha5) begin
        $display("dst tail byte %0d changed: got=%h want=a5", i, got_byte);
        tb_fatal("FFN dst tail sentinel was overwritten");
      end
    end

    $display("TEST_PASS tb_ffn");
    $finish;
  end

endmodule
