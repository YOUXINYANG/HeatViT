// Local/global feature concat for OP_CONCAT_LOCAL_GLOBAL (Phase 4 Task 2).
//
// For every (head, candidate): write 64 bytes whose first 32 come from
// local[h][c][0:32] and whose last 32 come from the head's own global
// features global[h][0:32]:
//
//   dst[head, c, 0:32]  = local[head, c, 0:32]
//   dst[head, c, 32:64] = global[head, 0:32]
//
// Layouts (row-major): local [3][C][32], global [3][32], dst [3][C][64].
// Each (head, candidate) slot performs two 32-byte reads then one 64-byte
// write; the two source halves never interleave.
module heatvit_feature_concat
  import heatvit_pkg::*;
#(
  parameter int MAX_C = 197
)(
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  output logic        busy,
  output logic        done,
  output logic        error_valid,
  output logic [7:0]  error_code,
  input  logic [15:0] m_eff,     // C = current_token_count - 1
  input  logic [31:0] src0_base, // local [3][C][32]
  input  logic [31:0] src1_base, // global [3][32]
  input  logic [31:0] dst_base,  // local_global [3][C][64]
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
    S_RD1_REQ,
    S_RD1_RECV,
    S_RD2_REQ,
    S_RD2_RECV,
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
  logic [1:0]  hh;
  logic [7:0]  cc;

  logic [31:0] rd_addr;
  logic [1:0]  rd_bi;
  logic        r_ready_r;

  logic [7:0]  bbuf [0:63];
  logic [2:0]  wr_bi;
  logic [63:0] wr_data_c;

  integer bi;
  initial begin
    for (bi = 0; bi < 64; bi++) bbuf[bi] = 8'h00;
  end

  always_comb begin
    wr_data_c = 64'h0000000000000000;
    if (state == S_WR_BEAT) begin
      for (int j = 0; j < 8; j++)
        wr_data_c[8*j +: 8] = bbuf[int'(wr_bi) * 8 + j];
    end
  end

  assign req_valid   = (state == S_RD1_REQ) || (state == S_RD2_REQ) ||
                       (state == S_WR_REQ);
  assign req_write   = (state == S_WR_REQ);
  assign req_addr    = rd_addr;
  assign req_bytes   = (state == S_WR_REQ) ? 32'd64 : 32'd32;
  assign req_r_ready = r_ready_r;
  assign req_w_valid = (state == S_WR_BEAT);
  assign req_w_data  = wr_data_c;
  assign req_w_strb  = 8'hff;
  assign req_w_last  = (wr_bi == 3'd7);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state       <= S_IDLE;
      busy        <= 1'b0;
      done        <= 1'b0;
      error_valid <= 1'b0;
      error_code  <= 8'd0;
      r_ready_r   <= 1'b0;
      rd_addr     <= 32'd0;
      rd_bi       <= 2'd0;
      wr_bi       <= 3'd0;
      hh          <= 2'd0;
      cc          <= 8'd0;
    end else begin
      done        <= 1'b0;
      error_valid <= 1'b0;
      r_ready_r   <= 1'b0;

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
              hh      <= 2'd0;
              cc      <= 8'd0;
              rd_addr <= src0_base;
              rd_bi   <= 2'd0;
              state   <= S_RD1_REQ;
            end
          end
        end

        S_RD1_REQ: begin
          if (req_valid && req_ready) begin
            rd_bi     <= 2'd0;
            r_ready_r <= 1'b1;
            state     <= S_RD1_RECV;
          end
        end

        S_RD1_RECV: begin
          if (req_r_valid && r_ready_r) begin
            for (int j = 0; j < 8; j++)
              bbuf[int'(rd_bi) * 8 + j] <= req_r_data[8*j +: 8];
            if (rd_bi == 2'd3) begin
              r_ready_r <= 1'b0;
              rd_addr   <= src1_r + 32'({6'd0, hh}) * 32'd32;
              rd_bi     <= 2'd0;
              state     <= S_RD2_REQ;
            end else begin
              rd_bi <= rd_bi + 2'd1;
            end
          end else begin
            r_ready_r <= 1'b1;
          end
        end

        S_RD2_REQ: begin
          if (req_valid && req_ready) begin
            rd_bi     <= 2'd0;
            r_ready_r <= 1'b1;
            state     <= S_RD2_RECV;
          end
        end

        S_RD2_RECV: begin
          if (req_r_valid && r_ready_r) begin
            for (int j = 0; j < 8; j++)
              bbuf[32 + int'(rd_bi) * 8 + j] <= req_r_data[8*j +: 8];
            if (rd_bi == 2'd3) begin
              r_ready_r <= 1'b0;
              rd_addr   <= dst_r +
                           32'(({6'd0, hh} * {8'd0, m_r} + {8'd0, cc}) *
                               7'd64);
              wr_bi     <= 3'd0;
              state     <= S_WR_REQ;
            end else begin
              rd_bi <= rd_bi + 2'd1;
            end
          end else begin
            r_ready_r <= 1'b1;
          end
        end

        S_WR_REQ: begin
          if (req_valid && req_ready) begin
            wr_bi <= 3'd0;
            state <= S_WR_BEAT;
          end
        end

        S_WR_BEAT: begin
          if (req_w_valid && req_w_ready) begin
            if (wr_bi == 3'd7) begin
              if (cc == m_r[7:0] - 8'd1) begin
                if (hh == 2'd2) begin
                  state <= S_DONE;
                end else begin
                  hh      <= hh + 2'd1;
                  cc      <= 8'd0;
                  rd_addr <= src0_r + 32'({6'd0, hh + 2'd1} *
                                          {8'd0, m_r}) * 32'd32;
                  rd_bi   <= 2'd0;
                  state   <= S_RD1_REQ;
                end
              end else begin
                cc      <= cc + 8'd1;
                rd_addr <= src0_r +
                           32'(({6'd0, hh} * {8'd0, m_r} +
                                {8'd0, cc + 8'd1}) * 32'd32);
                rd_bi   <= 2'd0;
                state   <= S_RD1_REQ;
              end
            end else begin
              wr_bi <= wr_bi + 3'd1;
            end
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
