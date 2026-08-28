// Memory-streaming vector engine for OP_LAYERNORM, OP_RESIDUAL and
// OP_ATTN_SOFTMAX. It adapts the Phase-1 arithmetic units to byte/word
// streams: rows are read with aligned windows, int32 scores are unpacked,
// rescaled to Q8.16 and saturated before Softmax, and results are written
// back through byte strobes. Both shared-divider clients (Softmax=0,
// LayerNorm=1) are exposed for the executor's arbiter.
//
// P7-1 (2026-08-28): the dynamic byte-addressed staging register array
// (`bbuf`) is replaced by byte-write-enable SDP RAMs (heatvit_sdp_ram):
//   ram_x/ram_g/ram_b  24x64   LN x/gamma/beta staging (RESIDUAL reuses
//                              x=main, g=aux, b=output staging)
//   ram_o              32x64   LN output staging / ATTN out_row stored with
//                              a +wr_e pre-offset so write-out beats read
//                              single aligned words
//   ram_s              128x64  ATTN raw score stream; unpacked sequentially
//                              through a 16-byte {current, previous} word
//                              window (one score per cycle), eliminating the
//                              single-cycle 788x dynamic byte reads of the
//                              old S_SM_PREP.
// All RAM reads are registered (1-cycle latency); the engine presents
// addresses in the same cycle the old code presented the value, so the
// registered input stages of the LN/Softmax/Residual units see identical
// timing. Bit-exact behaviour is unchanged.
module heatvit_vector_engine
  import heatvit_pkg::*;
#(
  parameter int MAX_ROW   = 197
)(
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  output logic        busy,
  output logic        done,
  output logic        error_valid,
  output logic [7:0]  error_code,
  output logic        warn_negative_variance,
  input  logic [1:0]  op,      // 0 LAYERNORM, 1 RESIDUAL, 2 ATTN_SOFTMAX
  input  logic [15:0] m_eff,
  input  logic [31:0] src0_base,
  input  logic [31:0] src1_base,
  input  logic [31:0] aux_base,
  input  logic [31:0] dst_base,
  input  logic [31:0] src0_end,
  input  logic [31:0] dst_end,
  input  heatvit_scale_t src0_scale,
  input  heatvit_scale_t src1_scale,
  input  heatvit_scale_t aux_scale,
  input  heatvit_scale_t dst_scale,
  // LayerNorm shared-divider client.
  output logic        ln_div_req_valid,
  input  logic        ln_div_req_ready,
  output logic [63:0] ln_div_num,
  output logic [63:0] ln_div_den,
  input  logic        ln_div_rsp_valid,
  input  logic [63:0] ln_div_quot,
  input  logic [63:0] ln_div_rem,
  input  logic        ln_div_div_zero,
  // Softmax shared-divider client.
  output logic        sm_div_req_valid,
  input  logic        sm_div_req_ready,
  output logic [63:0] sm_div_num,
  output logic [63:0] sm_div_den,
  input  logic        sm_div_rsp_valid,
  input  logic [63:0] sm_div_quot,
  input  logic [63:0] sm_div_rem,
  input  logic        sm_div_div_zero,
  // Memory master client.
  output logic        req_valid,
  input  logic        req_ready,
  output logic        req_write,
  output logic [31:0] req_addr,
  output logic [31:0] req_bytes,
  output logic        req_w_valid,
  input  logic        req_w_ready,
  output logic [63:0] req_w_data,
  output logic [7:0]  req_w_strb,
  output logic        req_w_last,
  input  logic        req_r_valid,
  output logic        req_r_ready,
  input  logic [63:0] req_r_data,
  input  logic        req_r_last
);

  typedef enum logic [3:0] {
    S_IDLE,
    S_PLAN,
    S_RD_REQ,
    S_RD_RECV,
    S_LN_CFG,
    S_LN_IN,
    S_LN_OUT,
    S_ELEM,
    S_ELEM_DRAIN,
    S_SM_PREP,
    S_SM_START,
    S_SM_IN,
    S_SM_OUT,
    S_WR_REQ,
    S_WR_BEAT,
    S_DONE
  } state_t;

  state_t state;

  logic [1:0]  op_r;
  logic [15:0] m_r;
  logic [31:0] src0_r;
  logic [31:0] src1_r;
  logic [31:0] aux_r;
  logic [31:0] dst_r;
  logic [31:0] src0_end_r;
  logic [31:0] dst_end_r;
  heatvit_scale_t src0_scale_r;
  heatvit_scale_t src1_scale_r;
  heatvit_scale_t aux_scale_r;
  heatvit_scale_t dst_scale_r;

  logic [7:0] token;
  logic [1:0] head;
  logic [7:0] idx;
  logic [7:0] out_idx;
  logic [8:0] pe;       // present element counter
  logic [2:0] step;     // LN: 0..2 row reads; softmax: not used

  logic [31:0] rd_addr;
  logic [15:0] rd_len;
  logic [2:0]  rd_e;
  logic [10:0] rd_bi;
  logic        r_ready_r;

  logic [31:0] wr_addr;
  logic [15:0] wr_len;
  logic [2:0]  wr_e;
  logic [15:0] wr_w;
  logic [10:0] wr_bi;
  logic [63:0] wr_data_c;
  logic [7:0]  wr_strb_c;

  // RAM filled by the receive burst: 0=x, 1=g, 2=b, 3=score.
  logic [1:0]  wr_sel;

  // ATTN_SOFTMAX streaming score unpack.
  logic [7:0]  sp_i;        // score index being written (0..m-1)
  logic [1:0]  prime_r;     // S_SM_PREP window priming countdown (2 -> 1 -> 0)
  logic [6:0]  s_raddr;     // score RAM read address (registered)
  logic [63:0] prev_word;   // previous score RAM word (16-byte window low half)
  logic [2:0]  wr_e_early;  // out_row byte offset, captured before S_SM_OUT

  // Staging RAMs.
  logic        we_x, we_g, we_b, we_o, we_s;
  logic [4:0]  waddr_x, waddr_g, waddr_b, waddr_o;
  logic [6:0]  waddr_s;
  logic [63:0] wdata_x, wdata_g, wdata_b, wdata_o, wdata_s;
  logic [7:0]  wstrb_x, wstrb_g, wstrb_b, wstrb_o, wstrb_s;
  logic [4:0]  raddr_x, raddr_g, raddr_b, raddr_o;
  logic [6:0]  raddr_s;
  logic [63:0] rdata_x, rdata_g, rdata_b, rdata_o, rdata_s;

  heatvit_sdp_ram #(.WIDTH(64), .DEPTH(24)) u_ram_x (
    .clk(clk), .we(we_x), .waddr(waddr_x), .wdata(wdata_x), .wstrb(wstrb_x),
    .raddr(raddr_x), .rdata(rdata_x)
  );
  heatvit_sdp_ram #(.WIDTH(64), .DEPTH(24)) u_ram_g (
    .clk(clk), .we(we_g), .waddr(waddr_g), .wdata(wdata_g), .wstrb(wstrb_g),
    .raddr(raddr_g), .rdata(rdata_g)
  );
  heatvit_sdp_ram #(.WIDTH(64), .DEPTH(24)) u_ram_b (
    .clk(clk), .we(we_b), .waddr(waddr_b), .wdata(wdata_b), .wstrb(wstrb_b),
    .raddr(raddr_b), .rdata(rdata_b)
  );
  heatvit_sdp_ram #(.WIDTH(64), .DEPTH(32)) u_ram_o (
    .clk(clk), .we(we_o), .waddr(waddr_o), .wdata(wdata_o), .wstrb(wstrb_o),
    .raddr(raddr_o), .rdata(rdata_o)
  );
  heatvit_sdp_ram #(.WIDTH(64), .DEPTH(128)) u_ram_s (
    .clk(clk), .we(we_s), .waddr(waddr_s), .wdata(wdata_s), .wstrb(wstrb_s),
    .raddr(raddr_s), .rdata(rdata_s)
  );

  heatvit_q8_16_t srow [0:MAX_ROW-1];

  // LayerNorm unit.
  logic        ln_cfg_valid;
  logic        ln_cfg_ready;
  logic        ln_in_valid;
  logic        ln_in_ready;
  heatvit_s8_t ln_in_x;
  heatvit_s8_t ln_in_gamma;
  heatvit_s8_t ln_in_beta;
  logic        ln_out_valid;
  logic        ln_out_ready;
  heatvit_s8_t ln_out_data;
  logic        ln_warn;

  heatvit_layernorm u_ln (
    .clk                    (clk),
    .rst_n                  (rst_n),
    .cfg_valid              (ln_cfg_valid),
    .cfg_ready              (ln_cfg_ready),
    .cfg_x_scale_exp        (src0_scale_r),
    .cfg_gamma_scale_exp    (src1_scale_r),
    .cfg_beta_scale_exp     (aux_scale_r),
    .cfg_out_scale_exp      (dst_scale_r),
    .busy                   (),
    .done                   (),
    .warn_negative_variance (ln_warn),
    .in_valid               (ln_in_valid),
    .in_ready               (ln_in_ready),
    .in_x                   (ln_in_x),
    .in_gamma               (ln_in_gamma),
    .in_beta                (ln_in_beta),
    .out_valid              (ln_out_valid),
    .out_ready              (ln_out_ready),
    .out_data               (ln_out_data),
    .div_req_valid          (ln_div_req_valid),
    .div_req_ready          (ln_div_req_ready),
    .div_num                (ln_div_num),
    .div_den                (ln_div_den),
    .div_rsp_valid          (ln_div_rsp_valid),
    .div_quot              (ln_div_quot),
    .div_rem                (ln_div_rem),
    .div_div_zero           (ln_div_div_zero)
  );

  // Attention softmax unit.
  logic        sm_start;
  logic        sm_busy;
  logic        sm_done;
  logic [7:0]  sm_row_len;
  logic        sm_s_valid;
  logic        sm_s_ready;
  heatvit_q8_16_t sm_s_data;
  logic        sm_m_valid;
  logic        sm_m_ready;
  logic [7:0]  sm_m_data;
  logic        sm_m_last;
  logic        sm_error_zero_sum;

  heatvit_softmax_attention u_sm (
    .clk            (clk),
    .rst_n          (rst_n),
    .start          (sm_start),
    .busy           (sm_busy),
    .done           (sm_done),
    .row_len        (sm_row_len),
    .s_valid        (sm_s_valid),
    .s_ready        (sm_s_ready),
    .s_data         (sm_s_data),
    .div_req_valid  (sm_div_req_valid),
    .div_req_ready  (sm_div_req_ready),
    .div_num        (sm_div_num),
    .div_den        (sm_div_den),
    .div_rsp_valid  (sm_div_rsp_valid),
    .div_quot       (sm_div_quot),
    .div_rem        (sm_div_rem),
    .div_div_zero   (sm_div_div_zero),
    .m_valid        (sm_m_valid),
    .m_ready        (sm_m_ready),
    .m_data         (sm_m_data),
    .m_last         (sm_m_last),
    .error_zero_sum (sm_error_zero_sum)
  );

  // Residual unit.
  logic        res_main_valid;
  logic        res_main_ready;
  heatvit_s8_t res_main_value;
  heatvit_scale_t res_main_scale;
  logic        res_aux_valid;
  logic        res_aux_ready;
  heatvit_s8_t res_aux_value;
  heatvit_scale_t res_aux_scale;
  heatvit_scale_t res_out_scale;
  logic        res_out_valid;
  logic        res_out_ready;
  heatvit_s8_t res_out_value;

  heatvit_residual u_residual (
    .clk            (clk),
    .rst_n          (rst_n),
    .main_valid     (res_main_valid),
    .main_ready     (res_main_ready),
    .main_value     (res_main_value),
    .main_scale_exp (res_main_scale),
    .aux_valid      (res_aux_valid),
    .aux_ready      (res_aux_ready),
    .aux_value      (res_aux_value),
    .aux_scale_exp  (res_aux_scale),
    .out_scale_exp  (res_out_scale),
    .out_valid      (res_out_valid),
    .out_ready      (res_out_ready),
    .out_value      (res_out_value)
  );

  integer bi;
  initial begin
    for (bi = 0; bi < MAX_ROW; bi++) srow[bi] = 24'sd0;
  end

  function automatic heatvit_q8_16_t score_q16(input logic [31:0] score);
    heatvit_s128_t wide;
    heatvit_s128_t scaled;
    wide   = $signed({{96{score[31]}}, score});
    scaled = scale_to_exp_s128(wide, src0_scale_r, -6'sd16);
    if (scaled > 128'sd8388607) return 24'sd8388607;
    if (scaled < -128'sd8388608) return -24'sd8388608;
    return heatvit_q8_16_t'(scaled);
  endfunction

  function automatic logic [63:0] clamp64(
    input logic [63:0] value,
    input logic [63:0] limit
  );
    return (value < limit) ? value : limit;
  endfunction

  // ------------------------------------------------------------------
  // RAM write ports.
  // ------------------------------------------------------------------
  logic accept_rd;
  assign accept_rd = (state == S_RD_RECV) && req_r_valid && r_ready_r;

  assign we_x = accept_rd && (wr_sel == 2'd0);
  assign we_g = accept_rd && (wr_sel == 2'd1);
  assign we_s = accept_rd && (wr_sel == 2'd3);
  assign waddr_x = rd_bi[4:0];
  assign waddr_g = rd_bi[4:0];
  assign waddr_s = rd_bi[6:0];
  assign wdata_x = req_r_data;
  assign wdata_g = req_r_data;
  assign wdata_s = req_r_data;
  assign wstrb_x = 8'hFF;
  assign wstrb_g = 8'hFF;
  assign wstrb_s = 8'hFF;

  // ram_b: LN beta staging (word writes) or RESIDUAL output staging
  // (byte writes from the residual unit).
  always_comb begin
    we_b    = accept_rd && (wr_sel == 2'd2);
    waddr_b = rd_bi[4:0];
    wdata_b = req_r_data;
    wstrb_b = 8'hFF;
    if (state == S_ELEM && res_out_valid) begin
      we_b    = 1'b1;
      waddr_b = idx[7:3];
      wdata_b = {56'd0, res_out_value} << (8*idx[2:0]);
      wstrb_b = 8'b1 << idx[2:0];
    end
  end

  // ram_o: LN output staging (byte writes at out_idx) or ATTN out_row
  // staging (byte writes at idx + wr_e_early, the pre-offset layout).
  always_comb begin
    we_o    = 1'b0;
    waddr_o = 5'd0;
    wdata_o = 64'd0;
    wstrb_o = 8'd0;
    if (state == S_LN_OUT && ln_out_valid) begin
      we_o    = 1'b1;
      waddr_o = out_idx[7:3];
      wdata_o = {56'd0, ln_out_data} << (8*out_idx[2:0]);
      wstrb_o = 8'b1 << out_idx[2:0];
    end else if (state == S_SM_OUT && sm_m_valid) begin
      logic [7:0] bidx;
      bidx    = idx + {5'd0, wr_e_early};
      we_o    = 1'b1;
      waddr_o = bidx[7:3];
      wdata_o = {56'd0, sm_m_data} << (8*bidx[2:0]);
      wstrb_o = 8'b1 << bidx[2:0];
    end
  end

  // ------------------------------------------------------------------
  // RAM read ports (registered reads; addresses issued one cycle ahead
  // of the presentation, matching the old combinational-read+register
  // timing exactly).
  // ------------------------------------------------------------------
  logic [7:0] lg;
  logic [7:0] lg2;
  logic [8:0] pe_g;
  logic [8:0] pe_g2;
  assign lg   = (idx == 8'd191) ? 8'd191 : idx + 8'd1;
  assign lg2  = (idx >= 8'd190) ? 8'd191 : idx + 8'd2;
  assign pe_g = (pe > 9'd191) ? 9'd191 : pe;
  assign pe_g2 = (pe >= 9'd191) ? 9'd191 : pe + 9'd1;

  // Registered RAM reads: rdata(T) = mem[raddr(T-1)], so the read address
  // must lead the byte index presented this cycle (S_LN_IN presents
  // byte lg = idx+1, hence the address needs word(idx+2); S_ELEM presents
  // byte pe, hence the address needs word(pe+1)). For the write-out, the
  // address advances ONLY on the accepted beat (req_w_valid && req_w_ready);
  // during backpressure stalls it holds the current word, otherwise the
  // lookahead would over-advance and shift the burst.
  wire w_accept = (state == S_WR_BEAT) && req_w_valid && req_w_ready;
  assign raddr_x = (state == S_LN_IN) ? lg2[7:3]
                 : (state == S_ELEM)  ? pe_g2[7:3]
                 : 5'd0;
  assign raddr_g = (state == S_LN_IN) ? lg2[7:3]
                 : (state == S_ELEM)  ? pe_g2[7:3]
                 : 5'd0;
  assign raddr_b = (state == S_LN_IN)              ? lg2[7:3]
                 : (state == S_WR_BEAT && op_r == 2'd1)
                   ? (w_accept ? ((wr_bi[4:0] == 5'd23) ? 5'd23 : wr_bi[4:0] + 5'd1)
                               : wr_bi[4:0])
                 : 5'd0;
  assign raddr_o = (state == S_WR_BEAT)
                   ? (w_accept ? (wr_bi[4:0] + 5'd1) : wr_bi[4:0])
                   : 5'd0;
  assign raddr_s = s_raddr;

  // ------------------------------------------------------------------
  // Write-out data path: whole 64-bit words from the staging RAMs.
  // ------------------------------------------------------------------
  always_comb begin
    wr_data_c = (op_r == 2'd1) ? rdata_b : rdata_o;
    wr_strb_c = 8'hFF;
    if (state == S_WR_BEAT && op_r == 2'd2) begin
      for (int j = 0; j < 8; j++) begin
        int p;
        p = int'(wr_bi) * 8 + j;
        if (!((p >= int'(wr_e)) && (p < int'(wr_e) + int'(wr_w))))
          wr_strb_c[j] = 1'b0;
      end
    end
  end

  assign req_valid   = (state == S_RD_REQ) || (state == S_WR_REQ);
  assign req_write   = (state == S_WR_REQ);
  assign req_addr    = (state == S_WR_REQ) ? wr_addr : rd_addr;
  assign req_bytes   = (state == S_WR_REQ) ? {16'd0, wr_len} : {16'd0, rd_len};
  assign req_r_ready = r_ready_r;
  assign req_w_valid = (state == S_WR_BEAT);
  assign req_w_data  = wr_data_c;
  assign req_w_strb  = wr_strb_c;
  assign req_w_last  = (wr_bi == ({5'd0, wr_len[10:3]} - 11'd1));
  assign warn_negative_variance = ln_warn;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state   <= S_IDLE;
      busy    <= 1'b0;
      done    <= 1'b0;
      error_valid <= 1'b0;
      error_code  <= 8'd0;
      r_ready_r <= 1'b0;
      wr_sel    <= 2'd0;
      sp_i      <= 8'd0;
      prime_r   <= 1'b0;
      s_raddr   <= 7'd0;
      prev_word <= 64'd0;
      wr_e_early <= 3'd0;
      ln_cfg_valid <= 1'b0;
      ln_in_valid  <= 1'b0;
      ln_out_ready <= 1'b0;
      sm_start     <= 1'b0;
      sm_s_valid   <= 1'b0;
      sm_m_ready   <= 1'b0;
      res_main_valid <= 1'b0;
      res_aux_valid  <= 1'b0;
      res_out_ready  <= 1'b0;
    end else begin
      done        <= 1'b0;
      error_valid <= 1'b0;
      r_ready_r   <= 1'b0;
      ln_cfg_valid <= 1'b0;
      ln_in_valid  <= 1'b0;
      ln_out_ready <= 1'b0;
      sm_start     <= 1'b0;
      sm_s_valid   <= 1'b0;
      sm_m_ready   <= 1'b0;
      res_main_valid <= 1'b0;
      res_aux_valid  <= 1'b0;
      res_out_ready  <= 1'b0;

      if (start) begin
        // Unconditional restart, also recovering an aborted child.
        op_r        <= op;
        m_r         <= m_eff;
        src0_r      <= src0_base;
        src1_r      <= src1_base;
        aux_r       <= aux_base;
        dst_r       <= dst_base;
        src0_end_r  <= src0_end;
        dst_end_r   <= dst_end;
        src0_scale_r <= src0_scale;
        src1_scale_r <= src1_scale;
        aux_scale_r  <= aux_scale;
        dst_scale_r  <= dst_scale;
        token <= 8'd0;
        head  <= 2'd0;
        idx   <= 8'd0;
        step  <= 3'd0;
        busy  <= 1'b1;
        done  <= 1'b0;
        state <= S_PLAN;
      end else if (sm_error_zero_sum) begin
        error_valid <= 1'b1;
        error_code  <= ERR_SOFTMAX_ZERO_SUM;
        busy        <= 1'b0;
        state       <= S_IDLE;
      end else case (state)
        S_IDLE: begin
        end

        S_PLAN: begin
          case (op_r)
            2'd0: begin  // LAYERNORM
              if (step == 3'd0) begin
                rd_addr <= src0_r + {8'd0, token} * 16'd192;
                rd_len  <= 16'd192;
                rd_e    <= 3'd0;
                                rd_bi   <= 11'd0;
                wr_sel  <= 2'd0;
                step    <= 3'd1;
                state   <= S_RD_REQ;
              end else if (step == 3'd1) begin
                rd_addr <= src1_r;
                rd_len  <= 16'd192;
                rd_e    <= 3'd0;
                                rd_bi   <= 11'd0;
                wr_sel  <= 2'd1;
                step    <= 3'd2;
                state   <= S_RD_REQ;
              end else if (step == 3'd2) begin
                rd_addr <= aux_r;
                rd_len  <= 16'd192;
                rd_e    <= 3'd0;
                                rd_bi   <= 11'd0;
                wr_sel  <= 2'd2;
                step    <= 3'd3;
                state   <= S_RD_REQ;
              end else begin
                ln_cfg_valid <= 1'b1;
                state        <= S_LN_CFG;
              end
            end
            2'd1: begin  // RESIDUAL
              if (step == 3'd0) begin
                rd_addr <= src0_r + {8'd0, token} * 16'd192;
                rd_len  <= 16'd192;
                rd_e    <= 3'd0;
                                rd_bi   <= 11'd0;
                wr_sel  <= 2'd0;
                step    <= 3'd1;
                state   <= S_RD_REQ;
              end else if (step == 3'd1) begin
                rd_addr <= aux_r + {8'd0, token} * 16'd192;
                rd_len  <= 16'd192;
                rd_e    <= 3'd0;
                                rd_bi   <= 11'd0;
                wr_sel  <= 2'd1;
                step    <= 3'd2;
                state   <= S_RD_REQ;
              end else begin
                idx   <= 8'd0;
                res_main_valid <= 1'b1;
                res_aux_valid  <= 1'b1;
                res_out_ready  <= 1'b1;
                res_main_value <= rdata_x[7:0];
                res_aux_value  <= rdata_g[7:0];
                pe <= 9'd1;
                res_main_scale <= src0_scale_r;
                res_aux_scale  <= aux_scale_r;
                res_out_scale  <= dst_scale_r;
                state <= S_ELEM;
              end
            end
            default: begin  // ATTN_SOFTMAX
              begin
                logic [63:0] a64;
                logic [63:0] aligned64;
                logic [63:0] cover64;
                logic [63:0] end64;
                a64 = {32'd0, src0_r} +
                      (64'({2'd0, head}) * 64'(m_r) + 64'({8'd0, token})) * 64'(m_r) * 64'd4;
                aligned64 = a64 & 64'hFFFFFFFFFFFFFFF8;
                cover64 = (aligned64 + 64'(m_r) * 64'd4 + 64'(a64[2:0]) + 64'd7) &
                          64'hFFFFFFFFFFFFFFF8;
                end64 = clamp64(cover64, {32'd0, src0_end_r});
                rd_addr <= aligned64[31:0];
                rd_len  <= 16'(end64 - aligned64);
                rd_e    <= a64[2:0];
                                rd_bi   <= 11'd0;
                wr_sel  <= 2'd3;
                // Hold the score RAM read at word 0 during the receive so
                // that the S_SM_PREP window priming latches the real word 0.
                s_raddr <= 7'd0;
                state   <= S_RD_REQ;
              end
            end
          endcase
        end

        S_RD_REQ: begin
          if (req_valid && req_ready) begin
            rd_bi     <= 11'd0;
            r_ready_r <= 1'b1;
            state     <= S_RD_RECV;
          end
        end

        S_RD_RECV: begin
          if (req_r_valid && r_ready_r) begin
            if (rd_bi == ({5'd0, rd_len[10:3]} - 11'd1)) begin
              r_ready_r <= 1'b0;
              if (op_r == 2'd2) begin
                s_raddr <= 7'd0;
                sp_i    <= 8'd0;
                prime_r <= 2'd2;
                state   <= S_SM_PREP;
              end else begin
                state <= S_PLAN;
              end
            end else begin
              rd_bi <= rd_bi + 11'd1;
            end
          end else begin
            r_ready_r <= 1'b1;
          end
        end

        S_SM_PREP: begin
          if (prime_r != 2'd0) begin
            // Prime the 16-byte window. The RAM read is registered:
            // rdata_s during cycle T = mem[s_raddr(T-1)], so the address
            // must lead the consuming cycle by one. P1 latches word 0 into
            // prev_word and stages raddr_0; P2 stages raddr_1 so the first
            // write cycle sees window = {word(raddr_0), word 0}.
            prev_word <= rdata_s;
            s_raddr   <= (prime_r == 2'd2)
                         ? (({8'd0, rd_e} + 11'd3) >> 3)
                         : (({8'd0, rd_e} + 11'd7) >> 3);
            prime_r   <= prime_r - 2'd1;
          end else begin
            begin
              logic [127:0] win;
              logic [10:0]  r_cur;
              logic [10:0]  b0, b1, b2, b3;
              logic [31:0]  w32;
              win   = {rdata_s, prev_word};
              r_cur = (({8'd0, rd_e} + {sp_i, 2'b00} + 11'd3) >> 3);
              b0    = {8'd0, rd_e} + {sp_i, 2'b00} + 11'd8 - {r_cur, 3'b000};
              b1    = b0 + 11'd1;
              b2    = b0 + 11'd2;
              b3    = b0 + 11'd3;
              w32   = {win[8*b3 +: 8], win[8*b2 +: 8],
                       win[8*b1 +: 8], win[8*b0 +: 8]};
              srow[sp_i] <= score_q16($signed(w32));
            end
            prev_word <= rdata_s;
            // Two-cycle lookahead: raddr_{k+2}, so that rdata_s during
            // write cycle k+1 already holds word(raddr_{k+1}).
            s_raddr   <= ({8'd0, rd_e} + {sp_i, 2'b00} + 11'd11) >> 3;
            sp_i      <= sp_i + 8'd1;
            if (sp_i == m_r[7:0] - 8'd1) begin
              begin
                logic [63:0] a64e;
                a64e = {32'd0, dst_r} +
                       (64'({2'd0, head}) * 64'(m_r) + 64'({8'd0, token})) * 64'(m_r);
                wr_e_early <= a64e[2:0];
              end
              idx        <= 8'd0;
              sm_start   <= 1'b1;
              sm_row_len <= m_r[7:0];
              state      <= S_SM_START;
            end
          end
        end

        S_LN_CFG: begin
          ln_in_x     <= rdata_x[7:0];
          ln_in_gamma <= rdata_g[7:0];
          ln_in_beta  <= rdata_b[7:0];
          if (ln_cfg_valid && ln_cfg_ready) begin
            idx          <= 8'd0;
            ln_in_valid  <= 1'b1;
            state        <= S_LN_IN;
          end else begin
            ln_cfg_valid <= 1'b1;
          end
        end

        S_LN_IN: begin
          ln_in_valid <= 1'b1;
          // Present the next channel on the cycle the current channel is
          // accepted, matching the LayerNorm unit's registered input stage.
          ln_in_x     <= rdata_x[8*lg[2:0] +: 8];
          ln_in_gamma <= rdata_g[8*lg[2:0] +: 8];
          ln_in_beta  <= rdata_b[8*lg[2:0] +: 8];
          if (ln_in_valid && ln_in_ready) begin
            if (idx == 8'd191) begin
              out_idx <= 8'd0;
              state   <= S_LN_OUT;
            end else begin
              idx <= idx + 8'd1;
            end
          end
        end

        S_LN_OUT: begin
          ln_out_ready <= 1'b1;
          if (ln_out_valid) begin
            if (out_idx == 8'd191) begin
              wr_addr <= dst_r + {8'd0, token} * 16'd192;
              wr_len  <= 16'd192;
              wr_e    <= 3'd0;
              wr_w    <= 16'd192;
              wr_bi   <= 11'd0;
              state   <= S_WR_REQ;
            end else begin
              out_idx <= out_idx + 8'd1;
            end
          end
        end

        S_ELEM: begin
          res_main_valid <= 1'b1;
          res_aux_valid  <= 1'b1;
          res_out_ready  <= 1'b1;
          // Present element pe while capturing element idx.
          res_main_value <= rdata_x[8*pe_g[2:0] +: 8];
          res_aux_value  <= rdata_g[8*pe_g[2:0] +: 8];
          pe <= pe + 9'd1;
          res_main_scale <= src0_scale_r;
          res_aux_scale  <= aux_scale_r;
          res_out_scale  <= dst_scale_r;
          if (res_out_valid) begin
            if (idx == 8'd191) begin
              idx   <= 8'd0;
              state <= S_ELEM_DRAIN;
            end else begin
              idx <= idx + 8'd1;
            end
          end
        end

        S_ELEM_DRAIN: begin
          res_out_ready <= 1'b1;
          wr_addr <= dst_r + {8'd0, token} * 16'd192;
          wr_len  <= 16'd192;
          wr_e    <= 3'd0;
          wr_w    <= 16'd192;
          wr_bi   <= 11'd0;
          state   <= S_WR_REQ;
        end

        S_SM_START: begin
          sm_s_valid <= 1'b1;
          sm_s_data  <= srow[0];
          idx        <= 8'd0;
          state      <= S_SM_IN;
        end

        S_SM_IN: begin
          sm_s_valid <= 1'b1;
          // Present the next element on the cycle that the current element
          // is accepted, matching the core's registered input stage.
          sm_s_data  <= (idx == m_r[7:0] - 8'd1) ? srow[idx] : srow[idx + 8'd1];
          if (sm_s_valid && sm_s_ready) begin
            if (idx == m_r[7:0] - 8'd1) begin
              idx   <= 8'd0;
              state <= S_SM_OUT;
            end else begin
              idx <= idx + 8'd1;
            end
          end
        end

        S_SM_OUT: begin
          sm_m_ready <= 1'b1;
          if (sm_m_valid) begin
            if (sm_m_last) begin
              begin
                logic [63:0] a64;
                logic [63:0] aligned64;
                logic [63:0] cover64;
                logic [63:0] end64;
                a64 = {32'd0, dst_r} +
                      (64'({2'd0, head}) * 64'(m_r) + 64'({8'd0, token})) * 64'(m_r);
                aligned64 = a64 & 64'hFFFFFFFFFFFFFFF8;
                cover64 = (aligned64 + 64'(m_r) + 64'(a64[2:0]) + 64'd7) &
                          64'hFFFFFFFFFFFFFFF8;
                end64 = clamp64(cover64, {32'd0, dst_end_r});
                wr_addr <= aligned64[31:0];
                wr_len  <= 16'(end64 - aligned64);
                wr_e    <= a64[2:0];
                wr_w    <= m_r;
                wr_bi   <= 11'd0;
                state   <= S_WR_REQ;
              end
            end else begin
              idx <= idx + 8'd1;
            end
          end
        end

        S_WR_REQ: begin
          if (req_valid && req_ready) begin
            wr_bi <= 11'd0;
            state <= S_WR_BEAT;
          end
        end

        S_WR_BEAT: begin
          if (req_w_valid && req_w_ready) begin
            if (wr_bi == ({5'd0, wr_len[10:3]} - 11'd1)) begin
              if (op_r == 2'd2) begin
                if (head == 2'd2) begin
                  head  <= 2'd0;
                  if (token == m_r - 16'd1) state <= S_DONE;
                  else begin token <= token + 8'd1; state <= S_PLAN; end
                end else begin
                  head  <= head + 2'd1;
                  state <= S_PLAN;
                end
              end else begin
                if (token == m_r - 16'd1) state <= S_DONE;
                else begin
                  token <= token + 8'd1;
                  step  <= 3'd0;
                  state <= S_PLAN;
                end
              end
            end else begin
              wr_bi <= wr_bi + 11'd1;
            end
          end
        end

        S_DONE: begin
          busy  <= 1'b0;
          done  <= 1'b1;
          state <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
