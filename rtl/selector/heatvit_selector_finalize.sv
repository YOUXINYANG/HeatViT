// Atomic Token Selector finalize for OP_SELECTOR_FINALIZE (Phase 4 Task 4).
//
// Single pass over the [N][192] input tokens with a buffered [C] fused-score
// vector (C = N-1):
//   * the CLS (token 0) is copied to output slot 0;
//   * a normal candidate with fused score >= 32768 is copied to the next
//     output slot in stable input order;
//   * a pruned normal candidate accumulates into the Package;
//   * the incoming Package (the last candidate, when current_package_present
//     is set) always accumulates and never becomes a normal output token.
//
// After the scan, if any participant exists the packager divides the
// per-channel weighted numerators (zero denominator falls back to the
// unweighted mean with a warning) and writes one Package row. The module
// then atomically produces next_token_count = 1 + kept + package and
// next_package_present.
//
// P7-2 (2026-08-28): the dynamic byte-addressed score buffer (`sbuf`,
// SCORE_BYTES bytes) is replaced by a 128x64 byte-write-enable SDP RAM
// (heatvit_sdp_ram). The score receive burst writes whole 64-bit words at
// waddr=rd_bi with a per-byte strobe masking the trailing bytes past
// score_field_bytes; each Q0.16 score sits inside one word (offset 4*k is
// 4k/8 = k/2 words and 4*(k&1) bytes in), so the S_TOKEN read presents a
// single registered word. The read address (t-1)>>1 is issued during
// S_CHILD_WAIT (and held through S_TOKEN), one cycle before consumption,
// matching the registered-read timing. Bit-exact behaviour is unchanged.
module heatvit_selector_finalize
  import heatvit_pkg::*;
#(
  parameter int MAX_C = 197,
  parameter int SCORE_BYTES = 4 * 197
)(
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  output logic        busy,
  output logic        done,
  output logic        error_valid,
  output logic [7:0]  error_code,
  input  logic [15:0] m_eff,               // N = current_token_count
  input  logic        current_package_present,
  input  logic [31:0] src0_base,           // input tokens [N][192]
  input  logic [31:0] src1_base,           // fused scores [C] Q0.16
  input  logic [31:0] dst_base,            // output tokens [N_next][192]
  output logic        warn_package_den_zero,
  output logic [7:0]  next_token_count,
  output logic        next_package_present,
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
  input  logic        req_r_last,
  // Divider client (arbiter client 2, shared).
  output logic        div_req_valid,
  input  logic        div_req_ready,
  output logic [63:0] div_num,
  output logic [63:0] div_den,
  input  logic        div_rsp_valid,
  input  logic [63:0] div_quot,
  input  logic [63:0] div_rem,
  input  logic        div_div_zero
);

  typedef enum logic [3:0] {
    S_IDLE,
    S_SC_RD_REQ,
    S_SC_RD_RECV,
    S_TOKEN,
    S_CHILD_WAIT,
    S_DONE,
    S_ERROR
  } state_t;

  typedef enum logic [1:0] {
    PH_SC,
    PH_CP,
    PH_PK
  } phase_t;

  state_t state;
  phase_t phase_sel;

  logic [15:0] n_r;
  logic        pkg_present_r;
  logic [31:0] src0_r;
  logic [31:0] src1_r;
  logic [31:0] dst_r;
  logic [8:0]  rd_bi;
  logic [8:0]  rd_beats;
  logic [15:0] score_field_bytes;
  logic        sc_r_ready;

  // Score buffer RAM (one 64-bit word holds two consecutive Q0.16 scores).
  logic        we_sbuf;
  logic [6:0]  waddr_sbuf;
  logic [63:0] wdata_sbuf;
  logic [7:0]  wstrb_sbuf;
  logic [6:0]  raddr_sbuf;
  logic [63:0] rdata_sbuf;

  heatvit_sdp_ram #(.WIDTH(64), .DEPTH(128)) u_ram_sbuf (
    .clk(clk), .we(we_sbuf), .waddr(waddr_sbuf), .wdata(wdata_sbuf),
    .wstrb(wstrb_sbuf), .raddr(raddr_sbuf), .rdata(rdata_sbuf)
  );

  logic [7:0]  t;         // current input token index 0..N-1
  logic [7:0]  kept;      // kept normal tokens so far
  logic [16:0] cur_score;
  logic        pkg_will_exist;

  // Child handshakes.
  logic        cp_start;
  logic        cp_done;
  logic [31:0] cp_src_addr;
  logic [7:0]  cp_slot;
  logic        pk_acc_start;
  logic        pk_acc_done;
  logic [31:0] pk_acc_addr;
  logic [16:0] pk_acc_score;
  logic        pk_div_start;
  logic        pk_div_done;
  logic [31:0] pk_div_addr;
  logic        pk_reset_acc;
  logic [8:0]  pk_participants;

  logic        cp_req_valid;
  logic        cp_req_ready;
  logic        cp_req_write;
  logic [31:0] cp_req_addr;
  logic [31:0] cp_req_bytes;
  logic        cp_req_w_valid;
  logic        cp_req_w_ready;
  logic [63:0] cp_req_w_data;
  logic [7:0]  cp_req_w_strb;
  logic        cp_req_w_last;
  logic        cp_req_r_valid;
  logic        cp_req_r_ready;
  logic [63:0] cp_req_r_data;
  logic        cp_req_r_last;

  logic        pk_req_valid;
  logic        pk_req_ready;
  logic        pk_req_write;
  logic [31:0] pk_req_addr;
  logic [31:0] pk_req_bytes;
  logic        pk_req_w_valid;
  logic        pk_req_w_ready;
  logic [63:0] pk_req_w_data;
  logic [7:0]  pk_req_w_strb;
  logic        pk_req_w_last;
  logic        pk_req_r_valid;
  logic        pk_req_r_ready;
  logic [63:0] pk_req_r_data;
  logic        pk_req_r_last;

  heatvit_token_compactor u_compactor (
    .clk         (clk),
    .rst_n       (rst_n),
    .start       (cp_start),
    .busy        (),
    .done        (cp_done),
    .error_valid (),
    .error_code  (),
    .src_addr    (cp_src_addr),
    .dst_base    (dst_r),
    .slot        (cp_slot),
    .req_valid   (cp_req_valid),
    .req_ready   (cp_req_ready),
    .req_write   (cp_req_write),
    .req_addr    (cp_req_addr),
    .req_bytes   (cp_req_bytes),
    .req_w_valid (cp_req_w_valid),
    .req_w_ready (cp_req_w_ready),
    .req_w_data  (cp_req_w_data),
    .req_w_strb  (cp_req_w_strb),
    .req_w_last  (cp_req_w_last),
    .req_r_valid (cp_req_r_valid),
    .req_r_ready (cp_req_r_ready),
    .req_r_data  (cp_req_r_data),
    .req_r_last  (cp_req_r_last)
  );

  heatvit_token_packager u_packager (
    .clk           (clk),
    .rst_n         (rst_n),
    .reset_acc     (pk_reset_acc),
    .acc_start     (pk_acc_start),
    .acc_busy      (),
    .acc_done      (pk_acc_done),
    .acc_error_valid(),
    .acc_error_code (),
    .acc_addr      (pk_acc_addr),
    .acc_score     (pk_acc_score),
    .div_start     (pk_div_start),
    .div_busy      (),
    .div_done      (pk_div_done),
    .div_addr      (pk_div_addr),
    .warn_package_den_zero(warn_package_den_zero),
    .participants  (pk_participants),
    .req_valid     (pk_req_valid),
    .req_ready     (pk_req_ready),
    .req_write     (pk_req_write),
    .req_addr      (pk_req_addr),
    .req_bytes     (pk_req_bytes),
    .req_w_valid   (pk_req_w_valid),
    .req_w_ready   (pk_req_w_ready),
    .req_w_data    (pk_req_w_data),
    .req_w_strb    (pk_req_w_strb),
    .req_w_last    (pk_req_w_last),
    .req_r_valid   (pk_req_r_valid),
    .req_r_ready   (pk_req_r_ready),
    .req_r_data    (pk_req_r_data),
    .req_r_last    (pk_req_r_last),
    .div_req_valid (div_req_valid),
    .div_req_ready (div_req_ready),
    .div_num       (div_num),
    .div_den       (div_den),
    .div_rsp_valid (div_rsp_valid),
    .div_quot      (div_quot),
    .div_rem       (div_rem),
    .div_div_zero  (div_div_zero)
  );

  // Score buffer RAM write port: word writes during the score receive burst,
  // masking the trailing bytes beyond score_field_bytes.
  wire accept_sc = (state == S_SC_RD_RECV) && req_r_valid && sc_r_ready;
  assign we_sbuf    = accept_sc;
  assign waddr_sbuf = rd_bi[6:0];
  assign wdata_sbuf = req_r_data;

  always_comb begin
    wstrb_sbuf = 8'd0;
    for (int j = 0; j < 8; j++) begin
      if (int'(rd_bi) * 8 + j < int'(score_field_bytes))
        wstrb_sbuf[j] = 1'b1;
    end
  end

  // Score buffer RAM read port (registered): the score for token t lives in
  // word (t-1)>>1, low half for odd t and high half for even t. The address
  // is issued during S_CHILD_WAIT (and held through S_TOKEN), one cycle
  // before S_TOKEN consumes rdata.
  assign raddr_sbuf = ((state == S_CHILD_WAIT) || (state == S_TOKEN))
                      ? ((t >= 8'd1) ? 7'((t - 8'd1) >> 1) : 7'd0)
                      : 7'd0;

  // Memory routing between the score-read phase and the two children.
  assign req_valid   = (phase_sel == PH_CP) ? cp_req_valid :
                       (phase_sel == PH_PK) ? pk_req_valid :
                       (state == S_SC_RD_REQ);
  assign req_write   = (phase_sel == PH_CP) ? cp_req_write : pk_req_write;
  assign req_addr    = (phase_sel == PH_CP) ? cp_req_addr :
                       (phase_sel == PH_PK) ? pk_req_addr : src1_r;
  assign req_bytes   = (phase_sel == PH_CP) ? cp_req_bytes :
                       (phase_sel == PH_PK) ? pk_req_bytes :
                       {16'd0, rd_beats} * 32'd8;
  assign req_w_valid = (phase_sel == PH_CP) ? cp_req_w_valid : pk_req_w_valid;
  assign req_w_data  = (phase_sel == PH_CP) ? cp_req_w_data : pk_req_w_data;
  assign req_w_strb  = (phase_sel == PH_CP) ? cp_req_w_strb : pk_req_w_strb;
  assign req_w_last  = (phase_sel == PH_CP) ? cp_req_w_last : pk_req_w_last;
  assign req_r_ready = (phase_sel == PH_CP) ? cp_req_r_ready :
                       (phase_sel == PH_PK) ? pk_req_r_ready : sc_r_ready;
  assign cp_req_ready = (phase_sel == PH_CP) ? req_ready : 1'b0;
  assign pk_req_ready = (phase_sel == PH_PK) ? req_ready : 1'b0;
  assign cp_req_w_ready = (phase_sel == PH_CP) ? req_w_ready : 1'b0;
  assign pk_req_w_ready = (phase_sel == PH_PK) ? req_w_ready : 1'b0;
  assign cp_req_r_valid = (phase_sel == PH_CP) ? req_r_valid : 1'b0;
  assign pk_req_r_valid = (phase_sel == PH_PK) ? req_r_valid : 1'b0;
  assign cp_req_r_data  = req_r_data;
  assign pk_req_r_data  = req_r_data;
  assign cp_req_r_last  = req_r_last;
  assign pk_req_r_last  = req_r_last;

  always_comb begin
    cp_start     = 1'b0;
    cp_src_addr  = src0_r;
    cp_slot      = 8'd0;
    pk_acc_start = 1'b0;
    pk_acc_addr  = src0_r;
    pk_acc_score = 17'd0;
    cur_score    = 17'd0;
    if (state == S_TOKEN && t != 8'd0) begin
      // Score for token t = Q0.16 at byte offset (t-1)*4 of the score field,
      // contained in word (t-1)>>1: low half for odd t, high half for even t.
      cur_score = t[0]
        ? (17'(rdata_sbuf[7:0])   | (17'(rdata_sbuf[15:8]) << 8) |
           (17'(rdata_sbuf[23:16]) << 16))
        : (17'(rdata_sbuf[39:32]) | (17'(rdata_sbuf[47:40]) << 8) |
           (17'(rdata_sbuf[55:48]) << 16));
      if (pkg_present_r && (int'(t) - 1 == int'(n_r) - 2)) begin
        pk_acc_start = 1'b1;
        pk_acc_addr  = src0_r + 32'({24'd0, t}) * 32'd192;
        pk_acc_score = cur_score;
      end else if (cur_score >= 17'd32768) begin
        cp_start     = 1'b1;
        cp_src_addr  = src0_r + 32'({24'd0, t}) * 32'd192;
        cp_slot      = kept + 8'd1;
      end else begin
        pk_acc_start = 1'b1;
        pk_acc_addr  = src0_r + 32'({24'd0, t}) * 32'd192;
        pk_acc_score = cur_score;
      end
    end else if (state == S_TOKEN && t == 8'd0) begin
      cp_start    = 1'b1;
      cp_src_addr = src0_r;
      cp_slot     = 8'd0;
    end
  end

  assign busy = (state != S_IDLE) && (state != S_ERROR);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state       <= S_IDLE;
      phase_sel   <= PH_SC;
      done        <= 1'b0;
      error_valid <= 1'b0;
      error_code  <= 8'd0;
      rd_bi       <= 9'd0;
      rd_beats    <= 9'd0;
      sc_r_ready  <= 1'b0;
      t           <= 8'd0;
      kept        <= 8'd0;
      pkg_will_exist <= 1'b0;
      next_token_count <= 8'd0;
      next_package_present <= 1'b0;
      pk_reset_acc <= 1'b0;
      pk_div_start <= 1'b0;
      pk_div_addr  <= 32'd0;
    end else begin
      done        <= 1'b0;
      error_valid <= 1'b0;
      sc_r_ready  <= 1'b0;
      pk_reset_acc <= 1'b0;
      pk_div_start <= 1'b0;

      case (state)
        S_IDLE: begin
          if (start) begin
            n_r           <= m_eff;
            pkg_present_r <= current_package_present;
            src0_r        <= src0_base;
            src1_r        <= src1_base;
            dst_r         <= dst_base;
            if (m_eff < 16'd2 || m_eff > 16'd197) begin
              error_code <= ERR_DIMENSION;
              state      <= S_ERROR;
            end else begin
              score_field_bytes <= 16'(m_eff - 16'd1) * 16'd4;
              rd_beats    <= 9'((int'(m_eff - 16'd1) * 4 + 7) / 8);
              rd_bi       <= 9'd0;
              phase_sel   <= PH_SC;
              pk_reset_acc <= 1'b1;
              state       <= S_SC_RD_REQ;
            end
          end
        end

        S_SC_RD_REQ: begin
          if (req_valid && req_ready) begin
            rd_bi      <= 9'd0;
            sc_r_ready <= 1'b1;
            state      <= S_SC_RD_RECV;
          end
        end

        S_SC_RD_RECV: begin
          if (req_r_valid && sc_r_ready) begin
            if (rd_bi == rd_beats - 9'd1) begin
              sc_r_ready <= 1'b0;
              t          <= 8'd0;
              kept       <= 8'd0;
              state      <= S_TOKEN;
            end else begin
              rd_bi <= rd_bi + 9'd1;
            end
          end else begin
            sc_r_ready <= 1'b1;
          end
        end

        S_TOKEN: begin
          // cp_start / pk_acc_start pulses fire this cycle (combinational);
          // latch the active child phase for the request routing.
          if (t == 8'd0) begin
            phase_sel <= PH_CP;
            t         <= 8'd1;
          end else if (pkg_present_r && (int'(t) - 1 == int'(n_r) - 2)) begin
            phase_sel <= PH_PK;
            t         <= t + 8'd1;
          end else if (cur_score >= 17'd32768) begin
            phase_sel <= PH_CP;
            kept      <= kept + 8'd1;
            t         <= t + 8'd1;
          end else begin
            phase_sel <= PH_PK;
            t         <= t + 8'd1;
          end
          state <= S_CHILD_WAIT;
        end

        S_CHILD_WAIT: begin
          if (phase_sel == PH_CP && cp_done) begin
            if (t == n_r[7:0]) begin
              // Scan complete.
              if (pk_participants > 9'd0) begin
                pkg_will_exist <= 1'b1;
                pk_div_addr    <= dst_r +
                                  32'({24'd0, kept + 8'd1}) * 32'd192;
                pk_div_start   <= 1'b1;
                phase_sel      <= PH_PK;
              end else begin
                pkg_will_exist <= 1'b0;
                next_token_count <= kept + 8'd1;
                next_package_present <= 1'b0;
                state <= S_DONE;
              end
            end else begin
              state <= S_TOKEN;
            end
          end else if (phase_sel == PH_PK && pk_acc_done) begin
            if (t == n_r[7:0]) begin
              if (pk_participants > 9'd0) begin
                pkg_will_exist <= 1'b1;
                pk_div_addr    <= dst_r +
                                  32'({24'd0, kept + 8'd1}) * 32'd192;
                pk_div_start   <= 1'b1;
                phase_sel      <= PH_PK;
              end else begin
                pkg_will_exist <= 1'b0;
                next_token_count <= kept + 8'd1;
                next_package_present <= 1'b0;
                state <= S_DONE;
              end
            end else begin
              state <= S_TOKEN;
            end
          end else if (phase_sel == PH_PK && pk_div_done) begin
            next_token_count <= kept + 8'd2;
            next_package_present <= 1'b1;
            state <= S_DONE;
          end
        end

        S_DONE: begin
          done  <= 1'b1;
          state <= S_IDLE;
        end

        S_ERROR: begin
          error_valid <= 1'b1;
          state       <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
