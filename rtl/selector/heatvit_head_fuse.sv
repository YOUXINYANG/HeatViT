// Weighted three-head keep-score fusion for OP_HEAD_FUSE (Phase 4 Task 3).
//
// Per candidate: three Q0.16 head keep scores (src0, [3][C]) and three Q0.16
// head weights (src1, [C][3]) are fused into one Q0.16 score (dst, [C]):
//
//   weighted_num = score0*weight0 + score1*weight1 + score2*weight2
//   weight_den   = weight0 + weight1 + weight2
//   fused        = round(weighted_num / weight_den), sat 0..65536
//
// A zero denominator falls back to the equal-weight mean of the three keep
// scores (denominator 3) and pulses warn_head_den_zero. Products are 34-bit,
// accumulated in 36 bits; the quotient uses the shared divider (client 2)
// with quotient/remainder ties-away rounding. All Q0.16 elements are 4-byte
// little-endian with bits 31:17 zero; whole tensors are moved as 8-byte
// aligned bursts so odd C is supported via strobe-masked padding bytes.
//
// P7-2: the three large staging register arrays become byte-write-enable
// SDP RAMs: u_ram_s / u_ram_w (296 x 64, score and weight tensors) and
// u_ram_o (128 x 64, fused output). The six dynamic three-byte reads of the
// old word_at() mux network are replaced by a 6-cycle S_FETCH lookahead
// per candidate: each fetch cycle issues one RAM address and captures the
// 3-byte field into a 17-bit register at the same cycle end (the RAM read
// is registered, and the capture register is the consumer). The divider
// handshake then consumes the captured registers. The shared divider's
// ~64-cycle latency hides the 6-cycle fetch. Output bytes are written with
// byte enables (the four bytes of a fused score always lie inside one
// 64-bit word, lanes 0..3 for even c, 4..7 for odd c); the write-out reads
// one aligned word per beat with an accept-gated lookahead so backpressure
// stalls cannot shift the burst. Bit-exact behaviour is unchanged.
module heatvit_head_fuse
  import heatvit_pkg::*;
#(
  parameter int MAX_C = 197,
  parameter int TENSOR_BYTES = 3 * 197 * 4,
  parameter int OUT_BYTES = 4 * 197
)(
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  output logic        busy,
  output logic        done,
  output logic        error_valid,
  output logic [7:0]  error_code,
  input  logic [15:0] m_eff,     // C = current_token_count - 1
  input  logic [31:0] src0_base, // head keep scores [3][C] Q0.16
  input  logic [31:0] src1_base, // head weights [C][3] Q0.16
  input  logic [31:0] dst_base,  // fused keep scores [C] Q0.16
  output logic        warn_head_den_zero,
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
  // Divider client (arbiter client 2, shared with reduce).
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
    S_RD1_REQ,
    S_RD1_RECV,
    S_RD2_REQ,
    S_RD2_RECV,
    S_FETCH,
    S_DIV_REQ,
    S_DIV_WAIT,
    S_WR_REQ,
    S_WR_BEAT,
    S_ERROR,
    S_DONE
  } state_t;

  state_t state;

  logic [15:0] m_r;
  logic [31:0] src0_r;
  logic [31:0] src1_r;
  logic [31:0] dst_r;
  logic [8:0]  rd_bi;
  logic [8:0]  rd_beats;
  logic [15:0] field_bytes_r;
  logic        r_ready_r;

  logic [7:0]  cc;         // candidate 0..C-1
  logic [2:0]  fetch_cnt;  // 0..5 lookahead fetch step
  logic [16:0] s0_r, s1_r, s2_r;
  logic [16:0] w0_r, w1_r, w2_r;
  logic        zero_den_r;
  logic [63:0] div_den_r;
  logic [10:0] wr_bi;
  logic [10:0] wr_beats;
  logic [15:0] out_bytes_r;

  logic [7:0]  wr_strb_c;

  // Staging RAMs.
  logic        we_s, we_w, we_o;
  logic [8:0]  waddr_s, waddr_w;
  logic [6:0]  waddr_o;
  logic [63:0] wdata_s, wdata_w, wdata_o;
  logic [7:0]  wstrb_s, wstrb_w, wstrb_o;
  logic [8:0]  raddr_s, raddr_w;
  logic [6:0]  raddr_o;
  logic [63:0] rdata_s, rdata_w, rdata_o;

  heatvit_sdp_ram #(.WIDTH(64), .DEPTH(296)) u_ram_s (
    .clk(clk), .we(we_s), .waddr(waddr_s), .wdata(wdata_s), .wstrb(wstrb_s),
    .raddr(raddr_s), .rdata(rdata_s)
  );
  heatvit_sdp_ram #(.WIDTH(64), .DEPTH(296)) u_ram_w (
    .clk(clk), .we(we_w), .waddr(waddr_w), .wdata(wdata_w), .wstrb(wstrb_w),
    .raddr(raddr_w), .rdata(rdata_w)
  );
  heatvit_sdp_ram #(.WIDTH(64), .DEPTH(128)) u_ram_o (
    .clk(clk), .we(we_o), .waddr(waddr_o), .wdata(wdata_o), .wstrb(wstrb_o),
    .raddr(raddr_o), .rdata(rdata_o)
  );

  logic accept_rd;
  logic w_accept;
  assign accept_rd = ((state == S_RD1_RECV) || (state == S_RD2_RECV)) &&
                     req_r_valid && r_ready_r;
  assign w_accept  = (state == S_WR_BEAT) && req_w_valid && req_w_ready;

  // Receive writes: whole aligned words, masked on the padding tail.
  always_comb begin
    wstrb_s = 8'hFF;
    wstrb_w = 8'hFF;
    for (int j = 0; j < 8; j++) begin
      if (int'(rd_bi) * 8 + j >= int'(field_bytes_r)) begin
        wstrb_s[j] = 1'b0;
        wstrb_w[j] = 1'b0;
      end
    end
  end
  assign we_s    = accept_rd && (state == S_RD1_RECV);
  assign we_w    = accept_rd && (state == S_RD2_RECV);
  assign waddr_s = rd_bi;
  assign waddr_w = rd_bi;
  assign wdata_s = req_r_data;
  assign wdata_w = req_r_data;

  // Output byte writes (one word per candidate; lanes depend on c parity).
  // fused is combinational off the divider response so the write cycle
  // latches the CURRENT candidate's value.
  logic [63:0] rounded_c;
  logic [16:0] fused_c;
  assign rounded_c = div_quot + ((64'd2 * div_rem >= div_den_r) ? 64'd1
                                                               : 64'd0);
  assign fused_c   = (rounded_c > 64'd65536) ? 17'd65536 : 17'(rounded_c);

  assign we_o    = (state == S_DIV_WAIT) && div_rsp_valid;
  assign waddr_o = cc[7:1];
  assign wdata_o = cc[0]
                   ? {{8'h00, fused_c[16], fused_c[15:8], fused_c[7:0]},
                      32'd0}
                   : {32'd0,
                      {8'h00, fused_c[16], fused_c[15:8], fused_c[7:0]}};
  assign wstrb_o = cc[0] ? 8'hF0 : 8'h0F;

  // Per-candidate lookahead fetch addresses. The RAM read is registered:
  // rdata at cycle k = mem[raddr(k-1)], so each address is issued one cycle
  // BEFORE the cycle that captures it (capture at fetch cycle 0 consumes
  // the address issued in the previous state; the S_DIV_WAIT response cycle
  // primes the next candidate's s0 word).
  logic [8:0] s0_wd, s1_wd, s2_wd;
  logic [8:0] w0_wd, w1_wd, w2_wd;
  assign s0_wd = {1'b0, cc[7:1]};
  assign s1_wd = (m_r[8:0] + {1'b0, cc}) >> 1;
  assign s2_wd = ({1'b0, m_r[7:0], 1'b0} + {1'b0, cc}) >> 1;
  assign w0_wd = {1'b0, cc} + {2'b0, cc[7:1]};
  assign w1_wd = w0_wd + {8'd0, cc[0]};
  assign w2_wd = w0_wd + 9'd1;

  assign raddr_s = (state == S_FETCH && fetch_cnt == 3'd0) ? s1_wd
                 : (state == S_FETCH && fetch_cnt == 3'd1) ? s2_wd
                 : (state == S_DIV_WAIT) ? (({1'b0, cc} + 9'd1) >> 1)
                 : 9'd0;
  assign raddr_w = (state == S_FETCH && fetch_cnt == 3'd2) ? w0_wd
                 : (state == S_FETCH && fetch_cnt == 3'd3) ? w1_wd
                 : (state == S_FETCH && fetch_cnt == 3'd4) ? w2_wd
                 : 9'd0;
  assign raddr_o = (state == S_WR_BEAT)
                   ? (w_accept ? ((wr_bi[6:0] == 7'd98) ? 7'd98
                                                      : wr_bi[6:0] + 7'd1)
                               : wr_bi[6:0])
                   : 7'd0;

  assign req_valid   = (state == S_RD1_REQ) || (state == S_RD2_REQ) ||
                       (state == S_WR_REQ);
  assign req_write   = (state == S_WR_REQ);
  assign req_addr    = (state == S_WR_REQ) ? dst_r :
                       (state == S_RD2_REQ || state == S_RD2_RECV)
                           ? src1_r : src0_r;
  assign req_bytes   = (state == S_WR_REQ)
                       ? {16'd0, wr_beats} * 32'd8
                       : {16'd0, (field_bytes_r + 16'd7) & 16'hFFF8};
  assign req_r_ready = r_ready_r;
  assign req_w_valid = (state == S_WR_BEAT);
  assign req_w_data  = rdata_o;
  assign req_w_strb  = wr_strb_c;
  assign req_w_last  = (wr_bi == (wr_beats - 11'd1));

  assign div_req_valid = (state == S_DIV_REQ);

  always_comb begin
    logic [35:0] weighted_num;
    logic [18:0] weight_den;
    logic [35:0] num_c;
    logic [18:0] den_c;
    div_num = 64'd0;
    div_den = 64'd1;
    if (state == S_DIV_REQ) begin
      weighted_num = {19'd0, s0_r} * {19'd0, w0_r} +
                     {19'd0, s1_r} * {19'd0, w1_r} +
                     {19'd0, s2_r} * {19'd0, w2_r};
      weight_den   = {2'd0, w0_r} + {2'd0, w1_r} + {2'd0, w2_r};
      if (weight_den == 19'd0) begin
        num_c = {19'd0, s0_r} + {19'd0, s1_r} + {19'd0, s2_r};
        den_c = 19'd3;
      end else begin
        num_c = weighted_num;
        den_c = weight_den;
      end
      div_num = {28'd0, num_c};
      div_den = {45'd0, den_c};
    end
  end

  // Write-out strobe mask (tail padding beyond out_bytes_r).
  always_comb begin
    wr_strb_c = 8'h00;
    if (state == S_WR_BEAT) begin
      for (int j = 0; j < 8; j++) begin
        if (int'(wr_bi) * 8 + j < int'(out_bytes_r))
          wr_strb_c[j] = 1'b1;
      end
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state       <= S_IDLE;
      busy        <= 1'b0;
      done        <= 1'b0;
      error_valid <= 1'b0;
      error_code  <= 8'd0;
      r_ready_r   <= 1'b0;
      rd_bi       <= 9'd0;
      rd_beats    <= 9'd0;
      field_bytes_r <= 16'd0;
      cc          <= 8'd0;
      fetch_cnt   <= 3'd0;
      s0_r        <= 17'd0;
      s1_r        <= 17'd0;
      s2_r        <= 17'd0;
      w0_r        <= 17'd0;
      w1_r        <= 17'd0;
      w2_r        <= 17'd0;
      zero_den_r  <= 1'b0;
      div_den_r   <= 64'd1;
      wr_bi       <= 11'd0;
      wr_beats    <= 11'd0;
      out_bytes_r <= 16'd0;
      warn_head_den_zero <= 1'b0;
    end else begin
      done        <= 1'b0;
      error_valid <= 1'b0;
      r_ready_r   <= 1'b0;
      warn_head_den_zero <= 1'b0;

      case (state)
        S_IDLE: begin
          if (start) begin
            m_r    <= m_eff;
            src0_r <= src0_base;
            src1_r <= src1_base;
            dst_r  <= dst_base;
            busy   <= 1'b1;
            if (m_eff == 16'd0 || m_eff > MAX_C[15:0]) begin
              error_code <= ERR_DIMENSION;
              state      <= S_ERROR;
            end else begin
              field_bytes_r <= 16'(m_eff) * 16'd12;  // 3 * C * 4 bytes
              rd_beats      <= 9'((int'(m_eff) * 12 + 7) / 8);
              rd_bi         <= 9'd0;
              state         <= S_RD1_REQ;
            end
          end
        end

        S_RD1_REQ: begin
          if (req_valid && req_ready) begin
            rd_bi     <= 9'd0;
            r_ready_r <= 1'b1;
            state     <= S_RD1_RECV;
          end
        end

        S_RD1_RECV: begin
          if (req_r_valid && r_ready_r) begin
            if (rd_bi == rd_beats - 9'd1) begin
              r_ready_r <= 1'b0;
              rd_bi     <= 9'd0;
              state     <= S_RD2_REQ;
            end else begin
              rd_bi <= rd_bi + 9'd1;
            end
          end else begin
            r_ready_r <= 1'b1;
          end
        end

        S_RD2_REQ: begin
          if (req_valid && req_ready) begin
            rd_bi     <= 9'd0;
            r_ready_r <= 1'b1;
            state     <= S_RD2_RECV;
          end
        end

        S_RD2_RECV: begin
          if (req_r_valid && r_ready_r) begin
            if (rd_bi == rd_beats - 9'd1) begin
              r_ready_r <= 1'b0;
              cc        <= 8'd0;
              fetch_cnt <= 3'd0;
              state     <= S_FETCH;
            end else begin
              rd_bi <= rd_bi + 9'd1;
            end
          end else begin
            r_ready_r <= 1'b1;
          end
        end

        S_FETCH: begin
          // Each cycle: capture the registered rdata of the address issued
          // one cycle earlier (the capture register IS the consumer).
          begin
            logic [2:0] off0;
            logic [2:0] off1;
            logic [2:0] off1w;
            off0  = {1'b0, cc[0], 2'b00};
            off1  = {1'b0, m_r[0] ^ cc[0], 2'b00};
            off1w = {1'b0, ~cc[0], 2'b00};
            case (fetch_cnt)
              3'd0: s0_r <= {rdata_s[8*off0 + 16 +: 8],
                             rdata_s[8*off0 + 8 +: 8],
                             rdata_s[8*off0 +: 8]};
              3'd1: s1_r <= {rdata_s[8*off1 + 16 +: 8],
                             rdata_s[8*off1 + 8 +: 8],
                             rdata_s[8*off1 +: 8]};
              3'd2: s2_r <= {rdata_s[8*off0 + 16 +: 8],
                             rdata_s[8*off0 + 8 +: 8],
                             rdata_s[8*off0 +: 8]};
              3'd3: w0_r <= {rdata_w[8*off0 + 16 +: 8],
                             rdata_w[8*off0 + 8 +: 8],
                             rdata_w[8*off0 +: 8]};
              3'd4: w1_r <= {rdata_w[8*off1w + 16 +: 8],
                             rdata_w[8*off1w + 8 +: 8],
                             rdata_w[8*off1w +: 8]};
              3'd5: w2_r <= {rdata_w[8*off0 + 16 +: 8],
                             rdata_w[8*off0 + 8 +: 8],
                             rdata_w[8*off0 +: 8]};
              default: ;
            endcase
          end
          if (fetch_cnt == 3'd5) begin
            state <= S_DIV_REQ;
          end else begin
            fetch_cnt <= fetch_cnt + 3'd1;
          end
        end

        S_DIV_REQ: begin
          if (div_req_valid && div_req_ready) begin
            begin
              logic [18:0] weight_den;
              weight_den  = {2'd0, w0_r} + {2'd0, w1_r} + {2'd0, w2_r};
              zero_den_r  <= (weight_den == 19'd0);
              div_den_r   <= (weight_den == 19'd0) ? 64'd3
                                                   : {45'd0, weight_den};
              if (weight_den == 19'd0)
                warn_head_den_zero <= 1'b1;
            end
            state <= S_DIV_WAIT;
          end
        end

        S_DIV_WAIT: begin
          if (div_rsp_valid) begin
            if (cc == m_r[7:0] - 8'd1) begin
              out_bytes_r <= 16'(m_r) * 16'd4;
              wr_beats    <= 11'((int'(m_r) * 4 + 7) / 8);
              wr_bi       <= 11'd0;
              state       <= S_WR_REQ;
            end else begin
              cc        <= cc + 8'd1;
              fetch_cnt <= 3'd0;
              state     <= S_FETCH;
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
            if (wr_bi == wr_beats - 11'd1) state <= S_DONE;
            else wr_bi <= wr_bi + 11'd1;
          end
        end

        S_ERROR: begin
          error_valid <= 1'b1;
          busy        <= 1'b0;
          state       <= S_IDLE;
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
