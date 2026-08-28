// Signed nearest-mean reduction for OP_REDUCE_MEAN (Phase 4 Task 2).
//
// Two axis modes selected by ``axis`` (desc.param0[3:2]):
//   0 candidate axis: src [3][C][32] int8 -> dst [3][32] int8
//                     dst[h][j] = round(sum_c src[h][c][j] / C)
//   1 head-lane axis: src [C][3][64] int8 -> dst [C][3] int8
//                     dst[c][h] = round(sum_l src[c][h][l] / 64)
//
// The signed sum is reduced to a magnitude before requesting the shared
// divider (client 2); quotient/remainder round to nearest with ties away
// from zero, the sign is restored and the result saturated to int8.
// Both sources are row-major; every read burst is 8-byte aligned and every
// write burst is padded to whole 64-bit beats with strobe masking on the
// final partial beat.
//
// P7-2: the old candidate-axis receive computed (b+j)/(32*m_r) per byte per
// beat - a variable divider network that dominated LUTs. Because both 32 and
// 32*m_r are multiples of 8, every 64-bit beat lies entirely inside one
// (head, candidate) row, so row indices are kept in REGISTER counters
// (c2r/h2r advance once per accepted beat) and the channel offset is the
// low bits of the beat index: no division at all. The output staging array
// becomes a byte-write-enable SDP RAM (128 x 64): candidate-axis results
// are one byte write per divider response; lane-axis rows commit their
// three bytes over a 3-cycle S_ROW_COMMIT state. The write-out reads one
// aligned word per beat with an accept-gated lookahead so backpressure
// stalls cannot shift the burst. Bit-exact behaviour is unchanged.
module heatvit_reduce_mean
  import heatvit_pkg::*;
#(
  parameter int MAX_C = 197,
  parameter int OUT_BUF_BYTES = 1024
)(
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  output logic        busy,
  output logic        done,
  output logic        error_valid,
  output logic [7:0]  error_code,
  input  logic        axis,      // 0 candidate axis, 1 head-lane axis
  input  logic [15:0] m_eff,     // C = current_token_count - 1
  input  logic [31:0] src0_base,
  input  logic [31:0] dst_base,
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
  // Divider client (arbiter client 2).
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
    S_RD_REQ,
    S_RD_RECV,
    S_DIV_REQ,
    S_DIV_WAIT,
    S_ROW_COMMIT,
    S_WR_REQ,
    S_WR_BEAT,
    S_ERROR,
    S_DONE
  } state_t;

  state_t state;

  logic        axis_r;
  logic [15:0] m_r;
  logic [31:0] src0_r;
  logic [31:0] dst_r;

  logic [31:0] rd_addr;
  logic [15:0] rd_len;   // byte count (multiple of 8)
  logic [11:0] rd_bi;
  logic [11:0] rd_beats;
  logic        r_ready_r;

  // Candidate-axis accumulators: 3 heads x 32 channels.
  logic signed [31:0] acc [2:0][31:0];
  // Lane-axis accumulators: 3 heads of the current candidate row.
  logic signed [31:0] lane_acc [2:0];

  // Candidate-axis row counters (incremented once per accepted beat; every
  // beat lies inside one (h, c) row because 32 and 32*m are multiples of 8).
  logic [1:0]  h2r;
  logic [7:0]  c2r;

  logic [7:0]  div_idx;      // 0..95 candidate axis, 0..2 lane axis
  logic        sign_r;
  logic [63:0] div_den_r;
  logic [7:0]  cc_row;       // lane-axis candidate row counter
  logic [7:0]  row_res [2:0]; // lane-axis per-row division results
  logic [7:0]  res_r;        // lane-axis head-2 result held for the commit
  logic [1:0]  commit_cnt;

  logic [10:0] wr_bi;
  logic [10:0] wr_beats;
  logic [15:0] out_bytes_r;
  logic [7:0]  wr_strb_c;

  // Output staging RAM (128 x 64 bytes).
  logic        we_obuf;
  logic [6:0]  waddr_obuf;
  logic [63:0] wdata_obuf;
  logic [7:0]  wstrb_obuf;
  logic [6:0]  raddr_obuf;
  logic [63:0] rdata_obuf;

  heatvit_sdp_ram #(.WIDTH(64), .DEPTH(128)) u_ram_obuf (
    .clk(clk), .we(we_obuf), .waddr(waddr_obuf), .wdata(wdata_obuf),
    .wstrb(wstrb_obuf), .raddr(raddr_obuf), .rdata(rdata_obuf)
  );

  integer bi;
  initial begin
    for (bi = 0; bi < 3; bi++) begin
      lane_acc[bi] = 32'sd0;
      row_res[bi]  = 8'h00;
      for (int j = 0; j < 32; j++) acc[bi][j] = 32'sd0;
    end
  end

  // Rounded, sign-restored, saturated int8 result (combinational off the
  // divider response so the write cycle latches the current value).
  logic [63:0] rounded_c;
  logic [7:0]  res_c;
  assign rounded_c = div_quot + ((64'd2 * div_rem >= div_den_r) ? 64'd1
                                                               : 64'd0);
  assign res_c = (!sign_r && rounded_c > 64'd127)      ? 8'sd127
               : (sign_r && rounded_c > 64'd128)       ? -8'sd128
               : sign_r ? -$signed(rounded_c[7:0])
                        : $signed(rounded_c[7:0]);

  // obuf byte writes: candidate axis one byte per divider response; lane
  // axis three bytes over the S_ROW_COMMIT cycles. bidx is 10-bit: the
  // lane-axis byte index reaches 3*195+2 = 587 (beyond 8 bits).
  logic [7:0] obuf_byte;
  logic [9:0] obuf_bidx;
  always_comb begin
    obuf_byte = res_c;
    obuf_bidx = {2'd0, div_idx};
    if (axis_r) begin
      obuf_bidx = 10'(int'(cc_row) * 3 + int'(commit_cnt));
      obuf_byte = (commit_cnt == 2'd2) ? res_r
                : (commit_cnt == 2'd0) ? row_res[0] : row_res[1];
    end
  end
  assign we_obuf    = ((state == S_DIV_WAIT) && div_rsp_valid && !axis_r) ||
                      (state == S_ROW_COMMIT);
  assign waddr_obuf = obuf_bidx[9:3];
  assign wdata_obuf = {56'd0, obuf_byte} << (8*obuf_bidx[2:0]);
  assign wstrb_obuf = 8'b1 << obuf_bidx[2:0];

  // Registered read with accept-gated lookahead (backpressure-safe).
  wire w_accept = (state == S_WR_BEAT) && req_w_valid && req_w_ready;
  assign raddr_obuf = (state == S_WR_BEAT)
                      ? (w_accept ? wr_bi[6:0] + 7'd1 : wr_bi[6:0])
                      : 7'd0;

  always_comb begin
    wr_strb_c = 8'h00;
    for (int j = 0; j < 8; j++) begin
      if (int'(wr_bi) * 8 + j < int'(out_bytes_r))
        wr_strb_c[j] = 1'b1;
    end
  end
  assign req_valid   = (state == S_RD_REQ) || (state == S_WR_REQ);
  assign req_write   = (state == S_WR_REQ);
  assign req_addr    = (state == S_WR_REQ) ? dst_r : rd_addr;
  assign req_bytes   = (state == S_WR_REQ)
                       ? {16'd0, wr_beats} * 32'd8 : {16'd0, rd_len};
  assign req_r_ready = r_ready_r;
  assign req_w_valid = (state == S_WR_BEAT);
  assign req_w_data  = rdata_obuf;
  assign req_w_strb  = wr_strb_c;
  assign req_w_last  = (wr_bi == (wr_beats - 11'd1));

  assign div_req_valid = (state == S_DIV_REQ);

  always_comb begin
    logic signed [31:0] val0;
    div_num = 64'd0;
    div_den = 64'd1;
    if (state == S_DIV_REQ) begin
      if (axis_r) begin
        val0    = lane_acc[div_idx[1:0]];
        div_num = {32'd0, (val0 < 0) ? (-val0) : val0};
        div_den = 64'd64;
      end else begin
        val0    = acc[div_idx[7:5]][div_idx[4:0]];
        div_num = {32'd0, (val0 < 0) ? (-val0) : val0};
        div_den = {48'd0, m_r};
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
      rd_bi       <= 12'd0;
      rd_beats    <= 12'd0;
      rd_addr     <= 32'd0;
      rd_len      <= 16'd0;
      cc_row      <= 8'd0;
      div_idx     <= 8'd0;
      sign_r      <= 1'b0;
      div_den_r   <= 64'd1;
      h2r         <= 2'd0;
      c2r         <= 8'd0;
      res_r       <= 8'd0;
      commit_cnt  <= 2'd0;
      wr_bi       <= 11'd0;
      wr_beats    <= 11'd0;
      out_bytes_r <= 16'd0;
    end else begin
      done        <= 1'b0;
      error_valid <= 1'b0;
      r_ready_r   <= 1'b0;

      case (state)
        S_IDLE: begin
          if (start) begin
            axis_r  <= axis;
            m_r     <= m_eff;
            src0_r  <= src0_base;
            dst_r   <= dst_base;
            for (int h = 0; h < 3; h++) begin
              for (int j = 0; j < 32; j++) acc[h][j] <= 32'sd0;
            end
            // Explicit per-head clears: XSim executes loop-variable-indexed
            // NBAs to this unpacked array unreliably when mixed with the
            // per-beat accumulation below.
            lane_acc[0] <= 32'sd0;
            lane_acc[1] <= 32'sd0;
            lane_acc[2] <= 32'sd0;
            h2r  <= 2'd0;
            c2r  <= 8'd0;
            busy <= 1'b1;
            if (m_eff == 16'd0 || m_eff > MAX_C[15:0]) begin
              error_code <= ERR_DIMENSION;
              state      <= S_ERROR;
            end else if (axis) begin
              cc_row   <= 8'd0;
              rd_addr  <= src0_base;
              rd_len   <= 16'd192;
              rd_beats <= 12'd24;
              rd_bi    <= 12'd0;
              state    <= S_RD_REQ;
            end else begin
              rd_addr  <= src0_base;
              rd_len   <= 16'(m_eff) * 16'd96;          // 3 * C * 32 bytes
              rd_beats <= 12'({4'd0, m_eff} * 12'd12);  // C * 12 beats
              rd_bi    <= 12'd0;
              state    <= S_RD_REQ;
            end
          end
        end

        S_RD_REQ: begin
          if (req_valid && req_ready) begin
            rd_bi     <= 12'd0;
            r_ready_r <= 1'b1;
            state     <= S_RD_RECV;
          end
        end

        S_RD_RECV: begin
          if (req_r_valid && r_ready_r) begin
            if (axis_r) begin
              // One 192-byte candidate row per burst; each 64-bit beat lies
              // inside one head's 64-lane group, so accumulate the beat sum.
              begin
                int b;
                int h2;
                int bsum;
                b    = int'(rd_bi) * 8;
                h2   = b / 64;
                bsum = 0;
                for (int j = 0; j < 8; j++)
                  bsum = bsum + $signed(req_r_data[8*j +: 8]);
                lane_acc[h2] <= lane_acc[h2] + $signed(bsum[31:0]);
              end
            end else begin
              // Every beat lies inside one (head, candidate) row (32 and
              // 32*m are multiples of 8): eight consecutive channels of a
              // single row. Register counters replace the old per-byte
              // variable division; the candidate counter advances every
              // FOURTH beat (one row = 32 bytes = 4 beats).
              for (int j = 0; j < 8; j++)
                acc[h2r][{rd_bi[1:0], 3'b000} + j] <=
                  acc[h2r][{rd_bi[1:0], 3'b000} + j] +
                  $signed(req_r_data[8*j +: 8]);
              if (rd_bi[1:0] == 2'b11) begin
                if (c2r == m_r[7:0] - 8'd1) begin
                  c2r <= 8'd0;
                  h2r <= h2r + 2'd1;
                end else begin
                  c2r <= c2r + 8'd1;
                end
              end
            end
            if (rd_bi == rd_beats - 12'd1) begin
              r_ready_r <= 1'b0;
              div_idx   <= 8'd0;
              state     <= S_DIV_REQ;
            end else begin
              rd_bi <= rd_bi + 12'd1;
            end
          end else begin
            r_ready_r <= 1'b1;
          end
        end

        S_DIV_REQ: begin
          if (div_req_valid && div_req_ready) begin
            if (axis_r) begin
              sign_r    <= (lane_acc[div_idx[1:0]] < 0);
              div_den_r <= 64'd64;
            end else begin
              sign_r    <= (acc[div_idx[7:5]][div_idx[4:0]] < 0);
              div_den_r <= {48'd0, m_r};
            end
            state <= S_DIV_WAIT;
          end
        end

        S_DIV_WAIT: begin
          if (div_rsp_valid) begin
            if (axis_r) begin
              if (div_idx[1:0] == 2'd2) begin
                res_r      <= res_c;
                commit_cnt <= 2'd0;
                state      <= S_ROW_COMMIT;
              end else begin
                row_res[div_idx[1:0]] <= res_c;
                div_idx <= div_idx + 8'd1;
                state   <= S_DIV_REQ;
              end
            end else begin
              if (div_idx == 8'd95) begin
                out_bytes_r <= 16'd96;
                wr_beats    <= 11'd12;
                wr_bi       <= 11'd0;
                state       <= S_WR_REQ;
              end else begin
                div_idx <= div_idx + 8'd1;
                state   <= S_DIV_REQ;
              end
            end
          end
        end

        S_ROW_COMMIT: begin
          // The three head means are written byte-wise over three cycles
          // (the SDP RAM has one write port).
          if (commit_cnt == 2'd2) begin
            if (cc_row == m_r[7:0] - 8'd1) begin
              out_bytes_r <= 16'(m_r) * 16'd3;
              wr_beats    <= 11'((int'(m_r) * 3 + 7) / 8);
              wr_bi       <= 11'd0;
              state       <= S_WR_REQ;
            end else begin
              cc_row    <= cc_row + 8'd1;
              rd_addr   <= rd_addr + 32'd192;
              rd_beats  <= 12'd24;
              rd_bi     <= 12'd0;
              // Fresh accumulators for the next candidate row.
              lane_acc[0] <= 32'sd0;
              lane_acc[1] <= 32'sd0;
              lane_acc[2] <= 32'sd0;
              state     <= S_RD_REQ;
            end
          end else begin
            commit_cnt <= commit_cnt + 2'd1;
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
