`timescale 1ns / 1ps

// Task 1: address/burst bounds plus the behavioral external-memory model.
//
// Default run (+CASE absent or "normal") exercises the address guard and a
// write/read round trip against one locked region, optionally with
// deterministic pseudo-random backpressure selected by +STALL_MASK.
//
// Error-injection runs use +CASE=err_* and are expected to terminate with a
// $fatal from the behavioral model; run_xsim.ps1 must report a nonzero exit
// code for every one of those cases.
module tb_memory_path;
  import heatvit_pkg::*;
  import tb_pkg::*;

  localparam logic [31:0] REGION_BASE  = 32'h00001000;
  localparam logic [31:0] REGION_BYTES = 32'h00000100;
  localparam string MEM_FILE = "sim/vectors/gemm/path.mem";

  logic        clk   = 1'b0;
  logic        rst_n = 1'b0;
  logic [15:0] stall_mask = 16'h0000;

  // Locked external memory interface for the primary model.
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

  behavioral_memory #(
    .SEG_COUNT  (1),
    .SEG0_FILE  (MEM_FILE),
    .SEG0_BASE  (REGION_BASE),
    .SEG0_BYTES (REGION_BYTES),
    .LFSR_INIT  (16'hACE1)
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
    .dbg_w_data   (8'h00)
  );

  // Independent model used only by the read-length-mismatch injection.
  logic        err_cmd_valid;
  logic        err_cmd_ready;
  logic        err_cmd_write;
  logic [31:0] err_cmd_addr;
  logic [15:0] err_cmd_len;
  logic        err_w_valid;
  logic        err_w_ready;
  logic [63:0] err_w_data;
  logic [7:0]  err_w_strb;
  logic        err_w_last;
  logic        err_r_valid;
  logic        err_r_ready;
  logic [63:0] err_r_data;
  logic        err_r_last;

  behavioral_memory #(
    .SEG_COUNT       (1),
    .SEG0_FILE       (MEM_FILE),
    .SEG0_BASE       (REGION_BASE),
    .SEG0_BYTES      (REGION_BYTES),
    .LFSR_INIT       (16'h5EED),
    .EXPECTED_CMD_LEN(16'd3)
  ) dut_mem_err (
    .clk          (clk),
    .rst_n        (rst_n),
    .stall_mask   (16'h0000),
    .mem_cmd_valid(err_cmd_valid),
    .mem_cmd_ready(err_cmd_ready),
    .mem_cmd_write(err_cmd_write),
    .mem_cmd_addr (err_cmd_addr),
    .mem_cmd_len  (err_cmd_len),
    .mem_w_valid  (err_w_valid),
    .mem_w_ready  (err_w_ready),
    .mem_w_data   (err_w_data),
    .mem_w_strb   (err_w_strb),
    .mem_w_last   (err_w_last),
    .mem_r_valid  (err_r_valid),
    .mem_r_ready  (err_r_ready),
    .mem_r_data   (err_r_data),
    .mem_r_last   (err_r_last),
    .obs_cmd_valid(),
    .obs_cmd_write(),
    .obs_cmd_addr (),
    .obs_cmd_len  (),
    .dbg_valid    (1'b0),
    .dbg_ready    (),
    .dbg_addr     (32'h00000000),
    .dbg_data     (),
    .dbg_w_valid  (1'b0),
    .dbg_w_addr   (32'h00000000),
    .dbg_w_data   (8'h00)
  );

  // Address guard under test.
  logic [31:0] g_base;
  logic [31:0] g_bytes;
  logic [31:0] g_addr;
  logic [15:0] g_len;
  logic        g_ok;
  logic [7:0]  g_code;

  heatvit_addr_guard dut_guard (
    .region_base    (g_base),
    .region_bytes   (g_bytes),
    .cmd_addr       (g_addr),
    .cmd_len        (g_len),
    .addr_ok        (g_ok),
    .addr_error_code(g_code)
  );

  always #5 clk = ~clk;

  // Local byte-accurate mirror of the .mem image. The testbench applies the
  // same strobed writes so reads can be checked without trusting the model.
  logic [63:0] mirror [0:31];
  logic [63:0] wdata [8];
  logic [7:0]  wstrb [8];

  int i;
  int j;
  int beat;
  int got;
  string case_str;
  string stall_str;
  logic [15:0] parsed_mask;

  task automatic check_guard(
    input logic [31:0] base,
    input logic [31:0] bytes,
    input logic [31:0] addr,
    input logic [15:0] len,
    input logic        expected
  );
    g_base  = base;
    g_bytes = bytes;
    g_addr  = addr;
    g_len   = len;
    #1;
    if (g_ok !== expected) begin
      $display("guard mismatch: base=%h bytes=%h addr=%h len=%0d got=%0d expected=%0d",
               base, bytes, addr, len, g_ok, expected);
      tb_fatal("address guard boundary mismatch");
    end
    if (!g_ok && g_code != ERR_ADDRESS) begin
      $display("guard error code=%0d, expected=%0d", g_code, ERR_ADDRESS);
      tb_fatal("address guard wrong error code");
    end
  endtask

  task automatic send_command(input logic write, input logic [31:0] addr,
                              input logic [15:0] len);
    mem_cmd_valid = 1'b1;
    mem_cmd_write = write;
    mem_cmd_addr  = addr;
    mem_cmd_len   = len;
    wait (mem_cmd_ready);
    @(posedge clk);
    #1;
    mem_cmd_valid = 1'b0;
  endtask

  task automatic send_write_beats(
    input logic [31:0] addr,
    input logic [15:0] beats
  );
    send_command(1'b1, addr, beats);
    for (beat = 0; beat < beats; beat++) begin
      mem_w_valid = 1'b1;
      mem_w_data  = wdata[beat];
      mem_w_strb  = wstrb[beat];
      mem_w_last  = (beat == beats - 1);
      wait (mem_w_ready);
      @(posedge clk);
      #1;
      // Mirror the write exactly like the model applies strobes.
      for (j = 0; j < 8; j++) begin
        if (wstrb[beat][j]) begin
          mirror[(addr - REGION_BASE) / 8 + beat][8*j +: 8] = wdata[beat][8*j +: 8];
        end
      end
    end
    mem_w_valid = 1'b0;
    mem_w_data  = 64'h0000000000000000;
    mem_w_strb  = 8'h00;
    mem_w_last  = 1'b0;
  endtask

  task automatic check_read_beats(
    input logic [31:0] addr,
    input logic [15:0] beats
  );
    logic [63:0] observed;
    logic [63:0] expected_word;

    send_command(1'b0, addr, beats);
    mem_r_ready = 1'b0;
    got = 0;
    while (got < beats) begin
      wait (mem_r_valid);
      observed = mem_r_data;
      if (mem_r_last !== ((got == beats - 1) ? 1'b1 : 1'b0)) begin
        $display("read last framing mismatch at beat %0d", got);
        tb_fatal("read last framing mismatch");
      end
      // Deliberately stall the valid beat and require payload stability.
      repeat (3) begin
        #1;
        if (mem_r_data !== observed ||
            mem_r_last !== ((got == beats - 1) ? 1'b1 : 1'b0)) begin
          $display("read payload not stable during stall (beat=%0d)", got);
          tb_fatal("read payload unstable during stall");
        end
        @(posedge clk);
      end
      #1;
      if (mem_r_data !== observed ||
          mem_r_last !== ((got == beats - 1) ? 1'b1 : 1'b0)) begin
        $display("read payload not stable after stall (beat=%0d)", got);
        tb_fatal("read payload unstable after stall");
      end
      expected_word = mirror[(addr - REGION_BASE) / 8 + got];
      if (observed !== expected_word) begin
        $display("read beat %0d mismatch: got=%h expected=%h",
                 got, observed, expected_word);
        tb_fatal("behavioral memory read mismatch");
      end
      mem_r_ready = 1'b1;
      do @(posedge clk); while (!mem_r_valid);
      #1;
      mem_r_ready = 1'b0;
      got++;
    end
  endtask

  task automatic run_roundtrip();
    logic [63:0] init_data [8];

    for (i = 0; i < 8; i++) begin
      wdata[i] = 64'h0000000000000000;
      wstrb[i] = 8'h00;
      init_data[i] = 64'h0000000000000000;
    end

    // Verify the .mem image was loaded by reading it back untouched.
    check_read_beats(REGION_BASE, 16'd4);

    // Full-strobe write and read back.
    init_data[0] = 64'h0123456789abcdef;
    init_data[1] = 64'hfedcba9876543210;
    init_data[2] = 64'ha5a5a5a5a5a5a5a5;
    init_data[3] = 64'h5a5a5a5a5a5a5a5a;
    for (i = 0; i < 4; i++) begin
      wdata[i] = init_data[i];
      wstrb[i] = 8'hff;
    end
    send_write_beats(REGION_BASE, 16'd4);
    check_read_beats(REGION_BASE, 16'd4);

    // Partial strobe must only replace the enabled bytes.
    wdata[0] = 64'hffffffffffffffff;
    wstrb[0] = 8'h0f;
    send_write_beats(REGION_BASE + 32'd8, 16'd1);
    check_read_beats(REGION_BASE + 32'd8, 16'd1);

    // 31-byte write with a 7-byte final strobe, then read back.
    wdata[0] = 64'h1111111111111111;
    wdata[1] = 64'h2222222222222222;
    wdata[2] = 64'h3333333333333333;
    wdata[3] = 64'h4444444444444444;
    wstrb[0] = 8'hff;
    wstrb[1] = 8'hff;
    wstrb[2] = 8'hff;
    wstrb[3] = 8'h7f;
    send_write_beats(REGION_BASE + 32'h000000e0, 16'd4);
    check_read_beats(REGION_BASE + 32'h000000e0, 16'd4);

    // Exercise every observed command boundary while we are here.
    if (obs_cmd_valid) begin
      $display("observed command: write=%0d addr=%h len=%0d",
               obs_cmd_write, obs_cmd_addr, obs_cmd_len);
    end
  endtask

  initial begin
    mem_cmd_valid = 1'b0;
    mem_cmd_write = 1'b0;
    mem_cmd_addr  = 32'h00000000;
    mem_cmd_len   = 16'h0000;
    mem_w_valid   = 1'b0;
    mem_w_data    = 64'h0000000000000000;
    mem_w_strb    = 8'h00;
    mem_w_last    = 1'b0;
    mem_r_ready   = 1'b0;
    err_cmd_valid = 1'b0;
    err_cmd_write = 1'b0;
    err_cmd_addr  = 32'h00000000;
    err_cmd_len   = 16'h0000;
    err_w_valid   = 1'b0;
    err_w_data    = 64'h0000000000000000;
    err_w_strb    = 8'h00;
    err_w_last    = 1'b0;
    err_r_ready   = 1'b0;
    g_base        = 32'h00000000;
    g_bytes       = 32'h00000000;
    g_addr        = 32'h00000000;
    g_len         = 16'h0000;
    dbg_valid     = 1'b0;
    dbg_addr      = 32'h00000000;

    for (i = 0; i < 32; i++) mirror[i] = 64'h0000000000000000;
    $readmemh(MEM_FILE, mirror);

    if (!$value$plusargs("CASE=%s", case_str)) case_str = "normal";
    if (!$value$plusargs("STALL_MASK=%s", stall_str)) stall_str = "0";
    if ($sscanf(stall_str, "%h", parsed_mask) != 1) parsed_mask = 16'h0000;
    stall_mask = parsed_mask;

    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
    #1;

    case (case_str)
      "normal", "": begin
        // Directed address-guard acceptance/rejection checks.
        check_guard(32'h00001000, 32'h00000100, 32'h00001000, 16'd1, 1'b1);
        check_guard(32'h00001000, 32'h00000100, 32'h000010f8, 16'd1, 1'b1);
        check_guard(32'h00001000, 32'h00000100, 32'h00001000, 16'd32, 1'b1);
        check_guard(32'h00001000, 32'h00000100, 32'h00001001, 16'd1, 1'b0);
        check_guard(32'h00001000, 32'h00000100, 32'h00001000, 16'd0, 1'b0);
        check_guard(32'h00001000, 32'h00000100, 32'h000010f8, 16'd2, 1'b0);
        check_guard(32'h00001000, 32'h00000100, 32'hfffffff8, 16'd2, 1'b0);
        check_guard(32'h00001001, 32'h00000100, 32'h00001008, 16'd1, 1'b0);

        run_roundtrip();
        $display("TEST_PASS tb_memory_path");
        $finish;
      end
      "err_oob": begin
        send_command(1'b0, 32'h000010f8, 16'd2);
        repeat (100) @(posedge clk);
        tb_fatal("out-of-bounds command did not trigger model $fatal");
      end
      "err_unaligned": begin
        send_command(1'b0, 32'h00001001, 16'd1);
        repeat (100) @(posedge clk);
        tb_fatal("unaligned command did not trigger model $fatal");
      end
      "err_zero_len": begin
        send_command(1'b0, 32'h00001000, 16'd0);
        repeat (100) @(posedge clk);
        tb_fatal("zero-length command did not trigger model $fatal");
      end
      "err_early_last": begin
        send_command(1'b1, 32'h00001000, 16'd2);
        mem_w_valid = 1'b1;
        mem_w_data  = 64'h0000000000000001;
        mem_w_strb  = 8'hff;
        mem_w_last  = 1'b1;
        wait (mem_w_ready);
        @(posedge clk);
        repeat (100) @(posedge clk);
        tb_fatal("early mem_w_last did not trigger model $fatal");
      end
      "err_late_last": begin
        send_command(1'b1, 32'h00001000, 16'd2);
        for (beat = 0; beat < 2; beat++) begin
          mem_w_valid = 1'b1;
          mem_w_data  = 64'h0000000000000001;
          mem_w_strb  = 8'hff;
          mem_w_last  = 1'b0;
          wait (mem_w_ready);
          @(posedge clk);
          #1;
        end
        repeat (100) @(posedge clk);
        tb_fatal("late mem_w_last did not trigger model $fatal");
      end
      "err_read_len": begin
        err_cmd_valid = 1'b1;
        err_cmd_write = 1'b0;
        err_cmd_addr  = 32'h00001000;
        err_cmd_len   = 16'd2;
        wait (err_cmd_ready);
        @(posedge clk);
        #1;
        err_cmd_valid = 1'b0;
        repeat (100) @(posedge clk);
        tb_fatal("read length mismatch did not trigger model $fatal");
      end
      default: begin
        $display("unknown CASE=%s", case_str);
        tb_fatal("unknown tb_memory_path case");
      end
    endcase
  end

endmodule
