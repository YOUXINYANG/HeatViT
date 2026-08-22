`timescale 1ns / 1ps

// Task 2: single-outstanding burst master, inferred SDP RAM and decoupling
// ready/valid FIFO. The testbench drives a small deterministic slave on the
// external side so every stall, abort and framing error is injected exactly.
module tb_mem_master;
  import tb_pkg::*;

  localparam int MAX_BEATS = 8;

  logic        clk   = 1'b0;
  logic        rst_n = 1'b0;

  // Inner request/client stream.
  logic        req_valid;
  logic        req_ready;
  logic        req_write;
  logic [31:0] req_addr;
  logic [31:0] req_bytes;
  logic        req_w_valid;
  logic        req_w_ready;
  logic [63:0] req_w_data;
  logic [7:0]  req_w_strb;
  logic        req_w_last;
  logic        req_r_valid;
  logic        req_r_ready;
  logic [63:0] req_r_data;
  logic        req_r_last;
  logic        abort;
  logic        done;
  logic        protocol_error;
  logic        abort_done;

  // External locked interface, driven here by the deterministic slave.
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

  heatvit_mem_master dut (
    .clk            (clk),
    .rst_n          (rst_n),
    .req_valid      (req_valid),
    .req_ready      (req_ready),
    .req_write      (req_write),
    .req_addr       (req_addr),
    .req_bytes      (req_bytes),
    .req_w_valid    (req_w_valid),
    .req_w_ready    (req_w_ready),
    .req_w_data     (req_w_data),
    .req_w_strb     (req_w_strb),
    .req_w_last     (req_w_last),
    .req_r_valid    (req_r_valid),
    .req_r_ready    (req_r_ready),
    .req_r_data     (req_r_data),
    .req_r_last     (req_r_last),
    .abort          (abort),
    .done           (done),
    .protocol_error (protocol_error),
    .abort_done     (abort_done),
    .mem_cmd_valid  (mem_cmd_valid),
    .mem_cmd_ready  (mem_cmd_ready),
    .mem_cmd_write  (mem_cmd_write),
    .mem_cmd_addr   (mem_cmd_addr),
    .mem_cmd_len    (mem_cmd_len),
    .mem_w_valid    (mem_w_valid),
    .mem_w_ready    (mem_w_ready),
    .mem_w_data     (mem_w_data),
    .mem_w_strb     (mem_w_strb),
    .mem_w_last     (mem_w_last),
    .mem_r_valid    (mem_r_valid),
    .mem_r_ready    (mem_r_ready),
    .mem_r_data     (mem_r_data),
    .mem_r_last     (mem_r_last)
  );

  // Direct primitive checks.
  logic        ram_we;
  logic [2:0]  ram_waddr;
  logic [63:0] ram_wdata;
  logic [7:0]  ram_wstrb;
  logic [2:0]  ram_raddr;
  logic [63:0] ram_rdata;

  heatvit_sdp_ram #(.WIDTH(64), .DEPTH(8), .AW(3)) dut_ram (
    .clk   (clk),
    .we    (ram_we),
    .waddr (ram_waddr),
    .wdata (ram_wdata),
    .wstrb (ram_wstrb),
    .raddr (ram_raddr),
    .rdata (ram_rdata)
  );

  logic        fifo_s_valid;
  logic        fifo_s_ready;
  logic [63:0] fifo_s_data;
  logic        fifo_m_valid;
  logic        fifo_m_ready;
  logic [63:0] fifo_m_data;

  heatvit_rv_fifo #(.WIDTH(64), .DEPTH(4)) dut_fifo (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (fifo_s_valid),
    .s_ready (fifo_s_ready),
    .s_data  (fifo_s_data),
    .m_valid (fifo_m_valid),
    .m_ready (fifo_m_ready),
    .m_data  (fifo_m_data)
  );

  always #5 clk = ~clk;

  // Scratch for the command currently under test.
  logic [31:0] cur_addr;
  logic [15:0] cur_len;
  logic        cur_write;
  logic [63:0] captured_w_data;
  logic [7:0]  captured_w_strb;
  logic        captured_w_last;
  logic [63:0] captured_r_data;
  logic        captured_r_last;

  int beat;
  int idx;
  int stall;

  function automatic logic [63:0] read_pattern(
    input logic [31:0] addr,
    input int          beat_index
  );
    logic [15:0] b16;
    b16 = beat_index[15:0];
    return {addr, 16'h0000} + {48'h000000000000, b16};
  endfunction

  task automatic master_request(
    input logic        write,
    input logic [31:0] addr,
    input logic [31:0] bytes
  );
    req_valid = 1'b1;
    req_write = write;
    req_addr  = addr;
    req_bytes = bytes;
    wait (req_ready);
    @(posedge clk);
    #1;
    req_valid = 1'b0;
    req_write = 1'b0;
  endtask

  task automatic slave_cmd(input int stall_cycles);
    logic [31:0] exp_addr;
    logic [15:0] exp_len;
    logic        exp_write;

    mem_cmd_ready = 1'b0;
    wait (mem_cmd_valid);
    exp_addr  = mem_cmd_addr;
    exp_len   = mem_cmd_len;
    exp_write = mem_cmd_write;
    repeat (stall_cycles) begin
      @(posedge clk);
      #1;
      if (!mem_cmd_valid) tb_fatal("master dropped mem_cmd_valid during stall");
      if (mem_cmd_addr !== exp_addr || mem_cmd_len !== exp_len ||
          mem_cmd_write !== exp_write)
        tb_fatal("master command payload unstable during stall");
    end
    mem_cmd_ready = 1'b1;
    @(posedge clk);
    #1;
    mem_cmd_ready = 1'b0;
    cur_addr  = exp_addr;
    cur_len   = exp_len;
    cur_write = exp_write;
  endtask

  task automatic write_beat(
    input logic [63:0] data,
    input logic [7:0]  strb,
    input logic        last,
    input int          stall_cycles
  );
    logic [63:0] exp_data;
    logic [7:0]  exp_strb;
    logic        exp_last;

    req_w_valid = 1'b1;
    req_w_data  = data;
    req_w_strb  = strb;
    req_w_last  = last;
    mem_w_ready = 1'b0;
    wait (mem_w_valid);
    exp_data = mem_w_data;
    exp_strb = mem_w_strb;
    exp_last = mem_w_last;
    repeat (stall_cycles) begin
      @(posedge clk);
      #1;
      if (!mem_w_valid) tb_fatal("master dropped mem_w_valid during stall");
      if (mem_w_data !== exp_data || mem_w_strb !== exp_strb ||
          mem_w_last !== exp_last) begin
        $display("WBDBG unstable t=%0t data=%h->%h strb=%h->%h last=%0d->%0d",
                 $time, exp_data, mem_w_data, exp_strb, mem_w_strb,
                 exp_last, mem_w_last);
        tb_fatal("master write payload unstable during stall");
      end
    end
    mem_w_ready = 1'b1;
    @(posedge clk);
    #1;
    mem_w_ready = 1'b0;
    req_w_valid = 1'b0;
    req_w_data  = 64'h0000000000000000;
    req_w_strb  = 8'h00;
    req_w_last  = 1'b0;
    #1;
    captured_w_data = exp_data;
    captured_w_strb = exp_strb;
    captured_w_last = exp_last;
  endtask

  task automatic read_beat(
    input logic [15:0] total_beats,
    input int          stall_cycles
  );
    logic [63:0] exp_data;
    logic        exp_last;
    logic [63:0] want_data;
    logic        want_last;

    mem_r_valid = 1'b1;
    mem_r_data  = read_pattern(cur_addr, beat);
    mem_r_last  = (beat == total_beats - 1);
    req_r_ready = 1'b0;
    wait (req_r_valid);
    exp_data = req_r_data;
    exp_last = req_r_last;
    repeat (stall_cycles) begin
      @(posedge clk);
      #1;
      if (!req_r_valid) tb_fatal("master dropped req_r_valid during stall");
      if (req_r_data !== exp_data || req_r_last !== exp_last)
        tb_fatal("master read payload unstable during stall");
    end
    req_r_ready = 1'b1;
    @(posedge clk);
    #1;
    req_r_ready = 1'b0;
    mem_r_valid = 1'b0;
    mem_r_data  = 64'h0000000000000000;
    mem_r_last  = 1'b0;
    #1;
    want_data = read_pattern(cur_addr, beat);
    want_last = (beat == total_beats - 1) ? 1'b1 : 1'b0;
    if (exp_data !== want_data) begin
      $display("read beat %0d data mismatch: got=%h want=%h", beat, exp_data, want_data);
      tb_fatal("master read data mismatch");
    end
    if (exp_last !== want_last) tb_fatal("master read last mismatch");
    captured_r_data = exp_data;
    captured_r_last = exp_last;
  endtask

  task automatic run_ram_checks();
    ram_we    = 1'b1;
    ram_waddr = 3'd0;
    ram_wdata = 64'h0102030405060708;
    ram_wstrb = 8'hff;
    ram_raddr = 3'd0;
    @(posedge clk);
    #1;
    ram_we = 1'b0;
    @(posedge clk);
    #1;
    if (ram_rdata !== 64'h0102030405060708) tb_fatal("ram full write/read mismatch");

    // Read-after-write to the same address returns the old value this cycle.
    ram_we    = 1'b1;
    ram_waddr = 3'd1;
    ram_wdata = 64'haaaaaaaaaaaaaaaa;
    ram_wstrb = 8'hff;
    ram_raddr = 3'd1;
    @(posedge clk);
    #1;
    ram_we = 1'b0;
    @(posedge clk);
    #1;
    if (ram_rdata !== 64'haaaaaaaaaaaaaaaa) tb_fatal("ram RAW mismatch");

    // Byte enables replace only the selected bytes.
    ram_we    = 1'b1;
    ram_waddr = 3'd2;
    ram_wdata = 64'hffffffffffffffff;
    ram_wstrb = 8'b01010000;
    ram_raddr = 3'd2;
    @(posedge clk);
    #1;
    ram_we = 1'b0;
    @(posedge clk);
    #1;
    if (ram_rdata !== 64'h00ff00ff00000000)
      tb_fatal("ram byte strobe mismatch");
  endtask

  task automatic run_fifo_checks();
    logic [63:0] want;
    int          k;

    // Fill to depth 4 and verify first-word fall-through ordering.
    for (k = 1; k <= 4; k++) begin
      fifo_s_valid = 1'b1;
      fifo_s_data  = 64'(k);
      @(posedge clk);
      #1;
    end
    fifo_s_valid = 1'b0;
    if (fifo_s_ready) tb_fatal("fifo s_ready should be low when full");
    // Drain in order with a simultaneous push/pop in the middle.
    for (k = 1; k <= 4; k++) begin
      if (!fifo_m_valid) tb_fatal("fifo lost m_valid");
      want = 64'(k);
      if (fifo_m_data !== want) begin
        $display("fifo pop %0d: got=%0d want=%0d", k, fifo_m_data, want);
        tb_fatal("fifo order mismatch");
      end
      fifo_m_ready = 1'b1;
      @(posedge clk);
      #1;
    end
    fifo_m_ready = 1'b0;
    if (fifo_m_valid) tb_fatal("fifo m_valid should drop after drain");

    // Simultaneous push and pop must not lose the new word.
    fifo_s_valid = 1'b1;
    fifo_s_data  = 64'd77;
    fifo_m_ready = 1'b1;
    @(posedge clk);
    #1;
    if (!fifo_m_valid) tb_fatal("fifo simultaneous push/pop lost valid");
    if (fifo_m_data !== 64'd77) tb_fatal("fifo simultaneous push/pop data mismatch");
    fifo_s_valid = 1'b0;
    fifo_m_ready = 1'b0;
  endtask

  task automatic wait_done_pulse();
    @(posedge clk);
    #1;
    if (!done) tb_fatal("missing done pulse");
    if (protocol_error) tb_fatal("spurious protocol_error");
  endtask

  task automatic run_lengths();
    int lengths [5];
    int want_beats [5];
    int want_strb [5];
    int beats_n;
    logic [7:0] strb;
    logic       last;
    logic [63:0] wdata [8];
    int b;

    lengths    = '{1, 7, 8, 9, 31};
    want_beats = '{1, 1, 1, 2, 4};
    want_strb  = '{1, 16'h7f, 16'hff, 1, 16'h7f};
    for (idx = 0; idx < 5; idx++) begin
      beats_n = want_beats[idx];
      for (b = 0; b < 8; b++) wdata[b] = 64'h0000000000000000;
      for (b = 0; b < beats_n; b++) wdata[b] = 64'h0000000000000000 + b + 1;

      // Write direction.
      master_request(1'b1, 32'h00002000 + idx * 32'h00000100, 32'(lengths[idx]));
      slave_cmd(3 + idx);
      if (cur_len != beats_n) begin
        $display("len %0d: cmd beats=%0d want=%0d", lengths[idx], cur_len, beats_n);
        tb_fatal("wrong write command beat count");
      end
      for (b = 0; b < beats_n; b++) begin
        strb = (b == beats_n - 1) ? 8'(want_strb[idx]) : 8'hff;
        last = (b == beats_n - 1);
        write_beat(wdata[b], strb, last, 4 + idx);
        if (captured_w_data !== wdata[b]) tb_fatal("write data passthrough mismatch");
        if (captured_w_strb !== strb) tb_fatal("write strobe passthrough mismatch");
        if (captured_w_last !== last) tb_fatal("write last mismatch");
      end
      if (captured_w_strb !== 8'(want_strb[idx])) begin
        $display("len %0d: tail strobe=%h want=%h", lengths[idx],
                 captured_w_strb, 8'(want_strb[idx]));
        tb_fatal("wrong tail strobe");
      end
      wait_done_pulse();

      // Read direction.
      master_request(1'b0, 32'h00002000 + idx * 32'h00000100, 32'(lengths[idx]));
      slave_cmd(5 + idx);
      if (cur_len != beats_n) tb_fatal("wrong read command beat count");
      for (b = 0; b < beats_n; b++) begin
        beat = b;
        read_beat(16'(beats_n), 6 + idx);
      end
      wait_done_pulse();
    end
  endtask

  initial begin
    req_valid   = 1'b0;
    req_write   = 1'b0;
    req_addr    = 32'h00000000;
    req_bytes   = 32'h00000000;
    req_w_valid = 1'b0;
    req_w_data  = 64'h0000000000000000;
    req_w_strb  = 8'h00;
    req_w_last  = 1'b0;
    req_r_ready = 1'b0;
    abort       = 1'b0;
    mem_cmd_ready = 1'b0;
    mem_w_ready   = 1'b0;
    mem_r_valid   = 1'b0;
    mem_r_data    = 64'h0000000000000000;
    mem_r_last    = 1'b0;
    ram_we        = 1'b0;
    ram_waddr     = 3'd0;
    ram_wdata     = 64'h0000000000000000;
    ram_wstrb     = 8'h00;
    ram_raddr     = 3'd0;
    fifo_s_valid  = 1'b0;
    fifo_s_data   = 64'h0000000000000000;
    fifo_m_ready  = 1'b0;

    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
    #1;

    run_ram_checks();
    run_fifo_checks();
    run_lengths();

    // Oversize and zero-length requests are rejected without a command.
    req_valid = 1'b1;
    req_write = 1'b0;
    req_addr  = 32'h00001000;
    req_bytes = 32'd524281;
    wait (req_ready);
    @(posedge clk);
    #1;
    req_valid = 1'b0;
    req_write = 1'b0;
    if (!protocol_error) tb_fatal("oversize request did not assert protocol_error");
    if (mem_cmd_valid) tb_fatal("oversize request issued a command");
    @(posedge clk);
    #1;

    req_valid = 1'b1;
    req_write = 1'b0;
    req_addr  = 32'h00001000;
    req_bytes = 32'd0;
    wait (req_ready);
    @(posedge clk);
    #1;
    req_valid = 1'b0;
    req_write = 1'b0;
    if (!protocol_error) tb_fatal("zero-length request did not assert protocol_error");
    if (mem_cmd_valid) tb_fatal("zero-length request issued a command");
    @(posedge clk);
    #1;

    // Abort before the command handshake: no command may be issued.
    master_request(1'b0, 32'h00004000, 32'd16);
    mem_cmd_ready = 1'b0;
    wait (mem_cmd_valid);
    repeat (2) begin
      @(posedge clk);
      #1;
    end
    abort = 1'b1;
    @(posedge clk);
    #1;
    if (mem_cmd_valid) tb_fatal("command valid still high after pre-handshake abort");
    if (!abort_done) tb_fatal("pre-handshake abort did not pulse abort_done");
    abort = 1'b0;
    @(posedge clk);
    #1;
    if (!req_ready) tb_fatal("master not back in idle after pre-handshake abort");

    // Abort mid-read-burst: drain to last, mask local output, no new request.
    master_request(1'b0, 32'h00005000, 32'd32);
    slave_cmd(0);
    beat = 0;
    mem_r_valid = 1'b1;
    mem_r_data  = read_pattern(cur_addr, 0);
    mem_r_last  = 1'b0;
    req_r_ready = 1'b0;
    wait (req_r_valid);
    req_r_ready = 1'b1;
    @(posedge clk);
    #1;
    req_r_ready = 1'b0;
    mem_r_valid = 1'b0;
    mem_r_last  = 1'b0;
    abort = 1'b1;
    @(posedge clk);
    #1;
    if (req_r_valid) tb_fatal("master leaked read output during drain");
    req_valid = 1'b1;
    req_write = 1'b0;
    req_addr  = 32'h00006000;
    req_bytes = 32'd8;
    for (idx = 1; idx < 4; idx++) begin
      mem_r_valid = 1'b1;
      mem_r_data  = read_pattern(cur_addr, idx);
      mem_r_last  = (idx == 3);
      wait (mem_r_ready);
      @(posedge clk);
      #1;
      if (req_ready) tb_fatal("master accepted a request during drain");
    end
    mem_r_valid = 1'b0;
    mem_r_last  = 1'b0;
    if (!abort_done) tb_fatal("mid-burst read abort did not pulse abort_done");
    abort = 1'b0;
    req_valid = 1'b0;
    req_write = 1'b0;
    @(posedge clk);
    #1;
    if (!req_ready) tb_fatal("master not idle after read abort drain");

    // Abort mid-write-burst: remaining beats must be zero-strobed.
    master_request(1'b1, 32'h00007000, 32'd32);
    slave_cmd(0);
    write_beat(64'h00000000000000aa, 8'hff, 1'b0, 0);
    abort = 1'b1;
    @(posedge clk);
    #1;
    req_w_valid = 1'b0;
    for (idx = 1; idx < 4; idx++) begin
      wait (mem_w_valid);
      if (mem_w_data !== 64'h0000000000000000) tb_fatal("abort drain wrote data");
      if (mem_w_strb !== 8'h00) tb_fatal("abort drain wrote strobes");
      if (mem_w_last !== ((idx == 3) ? 1'b1 : 1'b0))
        tb_fatal("abort drain last framing mismatch");
      mem_w_ready = 1'b1;
      @(posedge clk);
      #1;
      mem_w_ready = 1'b0;
    end
    if (!abort_done) tb_fatal("mid-burst write abort did not pulse abort_done");
    abort = 1'b0;
    @(posedge clk);
    #1;

    // Client framing error: early last.
    master_request(1'b1, 32'h00008000, 32'd32);
    slave_cmd(0);
    req_w_valid = 1'b1;
    req_w_data  = 64'h0000000000000011;
    req_w_strb  = 8'hff;
    req_w_last  = 1'b1;
    mem_w_ready = 1'b0;
    wait (mem_w_valid);
    mem_w_ready = 1'b1;
    @(posedge clk);
    captured_w_strb = mem_w_strb;
    #1;
    mem_w_ready = 1'b0;
    req_w_valid = 1'b0;
    req_w_last  = 1'b0;
    if (captured_w_strb !== 8'h00) tb_fatal("early-last beat was not zero-strobed");
    for (idx = 1; idx < 4; idx++) begin
      wait (mem_w_valid);
      if (mem_w_strb !== 8'h00 || mem_w_data !== 64'h0000000000000000)
        tb_fatal("early-last drain beat not zeroed");
      if (mem_w_last !== ((idx == 3) ? 1'b1 : 1'b0))
        tb_fatal("early-last drain last mismatch");
      mem_w_ready = 1'b1;
      @(posedge clk);
      #1;
      mem_w_ready = 1'b0;
    end
    if (!protocol_error) tb_fatal("early last did not pulse protocol_error");
    @(posedge clk);
    #1;

    // Client framing error: late last (never asserted on the final beat).
    master_request(1'b1, 32'h00009000, 32'd32);
    slave_cmd(0);
    for (idx = 0; idx < 3; idx++)
      write_beat(64'h0000000000000022 + idx, 8'hff, 1'b0, 0);
    req_w_valid = 1'b1;
    req_w_data  = 64'h0000000000000033;
    req_w_strb  = 8'hff;
    req_w_last  = 1'b0;
    mem_w_ready = 1'b0;
    wait (mem_w_valid);
    mem_w_ready = 1'b1;
    @(posedge clk);
    captured_w_strb = mem_w_strb;
    captured_w_last = mem_w_last;
    #1;
    mem_w_ready = 1'b0;
    req_w_valid = 1'b0;
    if (captured_w_strb !== 8'h00) tb_fatal("late-last beat was not zero-strobed");
    if (captured_w_last !== 1'b1) tb_fatal("late-last beat lacked master last");
    @(posedge clk);
    #1;
    if (!protocol_error) tb_fatal("late last did not pulse protocol_error");
    @(posedge clk);
    #1;

    // External read framing error: early last.
    master_request(1'b0, 32'h0000a000, 32'd32);
    slave_cmd(0);
    mem_r_valid = 1'b1;
    mem_r_data  = read_pattern(cur_addr, 0);
    mem_r_last  = 1'b1;
    req_r_ready = 1'b1;
    @(posedge clk);
    #1;
    req_r_ready = 1'b0;
    mem_r_valid = 1'b0;
    mem_r_last  = 1'b0;
    if (!protocol_error) tb_fatal("external early read last did not pulse protocol_error");
    @(posedge clk);
    #1;

    // External read framing error: late last.
    master_request(1'b0, 32'h0000b000, 32'd32);
    slave_cmd(0);
    for (idx = 0; idx < 4; idx++) begin
      mem_r_valid = 1'b1;
      mem_r_data  = read_pattern(cur_addr, idx);
      mem_r_last  = 1'b0;
      req_r_ready = 1'b1;
      @(posedge clk);
      #1;
      req_r_ready = 1'b0;
    end
    mem_r_valid = 1'b0;
    if (!protocol_error) tb_fatal("external late read last did not pulse protocol_error");
    @(posedge clk);
    #1;

    $display("TEST_PASS tb_mem_master");
    $finish;
  end

endmodule
