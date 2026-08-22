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

  logic [7:0]  sbuf [0:TENSOR_BYTES-1];
  logic [7:0]  wbuf [0:TENSOR_BYTES-1];
  logic [7:0]  obuf [0:OUT_BYTES-1];

  logic [7:0]  cc;         // candidate 0..C-1
  logic        zero_den_r;
  logic [63:0] div_den_r;
  logic [10:0] wr_bi;
  logic [10:0] wr_beats;
  logic [15:0] out_bytes_r;

  logic [63:0] wr_data_c;
  logic [7:0]  wr_strb_c;

  integer bi;
  initial begin
    for (bi = 0; bi < TENSOR_BYTES; bi++) begin
      sbuf[bi] = 8'h00;
      wbuf[bi] = 8'h00;
    end
    for (bi = 0; bi < OUT_BYTES; bi++) obuf[bi] = 8'h00;
  end

  // Little-endian 4-byte Q0.16 word read from a byte buffer (17-bit value).
  function automatic logic [16:0] word_at(
      input logic [7:0] barr [0:TENSOR_BYTES-1],
      input int base);
    logic [16:0] v;
    v = 17'(barr[base]) | (17'(barr[base + 1]) << 8) |
        (17'(barr[base + 2]) << 16);
    return v;
  endfunction

  always_comb begin
    wr_data_c = 64'h0000000000000000;
    wr_strb_c = 8'h00;
    if (state == S_WR_BEAT) begin
      for (int j = 0; j < 8; j++) begin
        if (int'(wr_bi) * 8 + j < int'(out_bytes_r)) begin
          wr_strb_c[j]        = 1'b1;
          wr_data_c[8*j +: 8] = obuf[int'(wr_bi) * 8 + j];
        end
      end
    end
  end

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
  assign req_w_data  = wr_data_c;
  assign req_w_strb  = wr_strb_c;
  assign req_w_last  = (wr_bi == (wr_beats - 11'd1));

  assign div_req_valid = (state == S_DIV_REQ);

  always_comb begin
    logic [16:0] s0, s1, s2;
    logic [16:0] w0, w1, w2;
    logic [35:0] weighted_num;
    logic [18:0] weight_den;
    logic [35:0] num_c;
    logic [18:0] den_c;
    div_num = 64'd0;
    div_den = 64'd1;
    if (state == S_DIV_REQ) begin
      s0 = word_at(sbuf, int'(cc) * 4);
      s1 = word_at(sbuf, (int'(m_r) + int'(cc)) * 4);
      s2 = word_at(sbuf, (2 * int'(m_r) + int'(cc)) * 4);
      w0 = word_at(wbuf, (3 * int'(cc) + 0) * 4);
      w1 = word_at(wbuf, (3 * int'(cc) + 1) * 4);
      w2 = word_at(wbuf, (3 * int'(cc) + 2) * 4);
      weighted_num = {19'd0, s0} * {19'd0, w0} +
                     {19'd0, s1} * {19'd0, w1} +
                     {19'd0, s2} * {19'd0, w2};
      weight_den   = {2'd0, w0} + {2'd0, w1} + {2'd0, w2};
      if (weight_den == 19'd0) begin
        num_c = {19'd0, s0} + {19'd0, s1} + {19'd0, s2};
        den_c = 19'd3;
      end else begin
        num_c = weighted_num;
        den_c = weight_den;
      end
      div_num = {28'd0, num_c};
      div_den = {45'd0, den_c};
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
            for (int j = 0; j < 8; j++) begin
              if (int'(rd_bi) * 8 + j < int'(field_bytes_r))
                sbuf[int'(rd_bi) * 8 + j] <= req_r_data[8*j +: 8];
            end
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
            for (int j = 0; j < 8; j++) begin
              if (int'(rd_bi) * 8 + j < int'(field_bytes_r))
                wbuf[int'(rd_bi) * 8 + j] <= req_r_data[8*j +: 8];
            end
            if (rd_bi == rd_beats - 9'd1) begin
              r_ready_r <= 1'b0;
              cc        <= 8'd0;
              state     <= S_DIV_REQ;
            end else begin
              rd_bi <= rd_bi + 9'd1;
            end
          end else begin
            r_ready_r <= 1'b1;
          end
        end

        S_DIV_REQ: begin
          if (div_req_valid && div_req_ready) begin
            begin
              logic [16:0] w0, w1, w2;
              logic [18:0] weight_den;
              w0 = word_at(wbuf, (3 * int'(cc) + 0) * 4);
              w1 = word_at(wbuf, (3 * int'(cc) + 1) * 4);
              w2 = word_at(wbuf, (3 * int'(cc) + 2) * 4);
              weight_den  = {2'd0, w0} + {2'd0, w1} + {2'd0, w2};
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
            begin
              logic [63:0] rounded;
              logic [16:0] fused;
              rounded = div_quot + ((64'd2 * div_rem >= div_den_r) ? 64'd1
                                                                  : 64'd0);
              fused   = (rounded > 64'd65536) ? 17'd65536 : 17'(rounded);
              obuf[int'(cc) * 4 + 0] <= fused[7:0];
              obuf[int'(cc) * 4 + 1] <= fused[15:8];
              obuf[int'(cc) * 4 + 2] <= fused[16];
              obuf[int'(cc) * 4 + 3] <= 8'h00;
            end
            if (cc == m_r[7:0] - 8'd1) begin
              out_bytes_r <= 16'(m_r) * 16'd4;
              wr_beats    <= 11'((int'(m_r) * 4 + 7) / 8);
              wr_bi       <= 11'd0;
              state       <= S_WR_REQ;
            end else begin
              cc    <= cc + 8'd1;
              state <= S_DIV_REQ;
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
