// P7-1 debug TB: drive the rewritten layout engine through one QKV_UNPACK
// op and dump the write bursts against the golden qkv_val mapping.
//
// NOTE: development debug harness for the P7-1 bbuf->BRAM rewrite — NOT part
// of the regression matrix (the self-checking tb_tensor_executor/tb_mhsa/
// e2e suites are the authoritative gates). It embeds its own behavioral
// memory model with a 3-cycle backpressure so stall-dependent read-lookahead
// bugs are reproducible. Prints QKV_DBG_PASS / QKV_DBG_FAIL.
`timescale 1ns/1ps

module tb_p7_qkv_dbg;

  logic        clk;
  logic        rst_n;
  logic        start;
  logic        busy;
  logic        done;
  logic        error_valid;
  logic [7:0]  error_code;

  logic [1:0]  op;
  logic [15:0] m_eff;
  logic [15:0] n_eff;
  logic [31:0] src0_base;
  logic [31:0] src1_base;
  logic [31:0] aux_base;
  logic [31:0] dst_base;
  logic [5:0]  s0, s1, sa, sd;

  logic        req_valid, req_ready, req_write;
  logic [31:0] req_addr;
  logic [31:0] req_bytes;
  logic        req_w_valid, req_w_ready;
  logic [63:0] req_w_data;
  logic [7:0]  req_w_strb;
  logic        req_w_last;
  logic        req_r_valid, req_r_ready;
  logic [63:0] req_r_data;
  logic        req_r_last;

  heatvit_layout_engine dut (
    .clk(clk), .rst_n(rst_n), .start(start), .busy(busy), .done(done),
    .error_valid(error_valid), .error_code(error_code),
    .op(op), .m_eff(m_eff), .n_eff(n_eff),
    .src0_base(src0_base), .src1_base(src1_base),
    .aux_base(aux_base), .dst_base(dst_base),
    .src0_scale(s0), .src1_scale(s1), .aux_scale(sa), .dst_scale(sd),
    .req_valid(req_valid), .req_ready(req_ready), .req_write(req_write),
    .req_addr(req_addr), .req_bytes(req_bytes),
    .req_w_valid(req_w_valid), .req_w_ready(req_w_ready),
    .req_w_data(req_w_data), .req_w_strb(req_w_strb), .req_w_last(req_w_last),
    .req_r_valid(req_r_valid), .req_r_ready(req_r_ready),
    .req_r_data(req_r_data), .req_r_last(req_r_last)
  );

  // Behavioral memory: one region for src (13*576 bytes), one for dst.
  logic [7:0] mem [0:65535];
  integer ii;
  initial begin
    for (ii = 0; ii < 65536; ii++) mem[ii] = 8'h00;
  end

  function automatic logic [7:0] qkv_val(input int token, input int idx);
    return 8'(token * 5 + idx) * 8'd29 + 8'd3;
  endfunction

  // Read responder.
  logic [31:0] rd_base;
  logic [15:0] rd_cnt;
  logic        rd_active;
  logic [31:0] wr_base;
  logic [31:0] wr_off;
  logic [15:0] wr_cnt;
  logic        wr_active;
  logic [1:0]  stall_cnt;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      req_ready   <= 1'b0;
      req_r_valid <= 1'b0;
      req_w_ready <= 1'b0;
      rd_active   <= 1'b0;
      wr_active   <= 1'b0;
      rd_cnt      <= 16'd0;
      rd_base     <= 32'd0;
      wr_base     <= 32'd0;
      wr_off      <= 32'd0;
      wr_cnt      <= 16'd0;
      stall_cnt   <= 2'd0;
    end else begin
      req_r_valid <= 1'b0;
      req_w_ready <= 1'b0;
      stall_cnt   <= stall_cnt + 2'd1;
      if (req_valid && req_write) begin
        req_ready   <= 1'b1;
        req_w_ready <= 1'b1;
        wr_active   <= 1'b1;
        wr_base     <= req_addr;
        wr_cnt      <= req_bytes;
        wr_off      <= 32'd0;
      end else if (req_valid && !req_write && !rd_active) begin
        rd_base   <= req_addr;
        rd_cnt    <= req_bytes;
        req_ready <= 1'b1;
        rd_active <= 1'b1;
      end else begin
        req_ready <= 1'b0;
      end
      if (wr_active) req_w_ready <= (stall_cnt == 2'd0);
      // Deliver a read beat; advance the address only when the engine
      // accepts (req_r_valid && req_r_ready in the SAME cycle).
      if (rd_active && req_r_valid && req_r_ready) begin
        req_r_valid <= 1'b0;
        req_r_last  <= (rd_cnt == 16'd8);
        if (rd_cnt == 16'd8) begin
          rd_active <= 1'b0;
        end else begin
          rd_cnt  <= rd_cnt - 16'd8;
          rd_base <= rd_base + 32'd8;
        end
      end else if (rd_active && req_r_ready && (stall_cnt == 2'd0)) begin
        req_r_valid <= 1'b1;
      end
      if (req_w_valid && req_w_ready) begin
        for (int b = 0; b < 8; b++)
          if (req_w_strb[b]) mem[wr_base + wr_off + b] <= req_w_data[8*b +: 8];
        wr_off <= wr_off + 32'd8;
        if (req_w_last) wr_active <= 1'b0;
      end
    end
  end

  // Drive read data from mem.
  always_comb begin
    req_r_data = 64'd0;
    for (int b = 0; b < 8; b++)
      req_r_data[8*b +: 8] = mem[rd_base + b];
  end

  int errors;
  int kind;
  int head;
  int token;
  int lane;
  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

  initial begin
    #500000;
    $display("TB_TIMEOUT state=%0d busy=%0d req_valid=%0d req_ready=%0d req_write=%0d req_r_valid=%0d r_ready=%0d req_w_valid=%0d req_w_ready=%0d rd_bi=%0d wr_bi=%0d",
             dut.state, busy, req_valid, req_ready, req_write, req_r_valid,
             req_r_ready, req_w_valid, req_w_ready, dut.rd_bi, dut.wr_bi);
    $finish;
  end

  initial begin
    op = 2'd2;        // QKV_UNPACK
    m_eff = 16'd13;
    n_eff = 16'd576;
    src0_base = 32'd0;
    src1_base = 32'd0;
    aux_base  = 32'd0;
    dst_base  = 32'd20000;
    s0 = 6'sd0; s1 = 6'sd0; sa = 6'sd0; sd = 6'sd0;
    start = 1'b0;
    for (int i = 0; i < 13 * 576; i++) mem[i] = qkv_val(i / 576, i % 576);

    rst_n = 1'b0;
    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
    #1;
    start = 1'b1;
    @(posedge clk);
    #1;
    start = 1'b0;
    wait (done);
    #1;

    errors = 0;
    for (kind = 0; kind < 3; kind++)
      for (head = 0; head < 3; head++)
        for (token = 0; token < 13; token++)
          for (lane = 0; lane < 64; lane++) begin
            logic [7:0] got, want;
            got = mem[20000 + ((kind * 3 + head) * 13 + token) * 64 + lane];
            want = qkv_val(token, kind * 192 + head * 64 + lane);
            if (got !== want) begin
              if (errors < 8)
                $display("MISMATCH k=%0d h=%0d t=%0d l=%0d got=%h want=%h",
                         kind, head, token, lane, got, want);
              errors++;
            end
          end
    if (errors == 0) $display("QKV_DBG_PASS");
    else $display("QKV_DBG_FAIL errors=%0d", errors);
    $finish;
  end

endmodule
