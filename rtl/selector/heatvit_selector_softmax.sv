// Selector softmax for OP_SELECTOR_SOFTMAX (Phase 4 Task 5).
//
// Converts the head-mode int8 logits [3][C][2] into per-(head, candidate)
// Keep probabilities [3][C] stored as 4-byte little-endian Q0.16. Each
// 2-logit row is requantized from ``src0_scale`` to Q8.16 (saturated to
// 24 bits), streamed through the delta2=1.0 selector softmax core, and
// only the Keep column (column 1) is kept. The whole logit tensor is
// moved as one 8-byte-aligned burst so odd C stays supported.
//
// P7-2: the logit and output staging register arrays become byte-write-
// enable SDP RAMs (u_ram_l 148 x 64, u_ram_o 296 x 64). Each 2-logit row
// lives at an even byte offset, so both logits are always inside a single
// word; the read address is issued during S_SM_PREP (the state before
// S_SM_START) and held so the registered rdata presents the row during
// S_SM_START/S_SM_IN. Each 4-byte keep value also fits one word (lanes
// 0..3 or 4..7), so the S_SM_OUT store is one byte-enabled write. The
// write-out reads one aligned word per beat with an accept-gated
// lookahead so backpressure stalls cannot shift the burst. Bit-exact
// behaviour is unchanged.
module heatvit_selector_softmax
  import heatvit_pkg::*;
#(
  parameter int MAX_C = 197,
  parameter int LOGIT_BYTES = 3 * 197 * 2,
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
  input  logic [31:0] src0_base, // logits [3][C][2] int8
  input  logic [31:0] dst_base,  // keep scores [3][C] Q0.16
  input  heatvit_scale_t src0_scale,
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
  // Divider client (arbiter client 0, shared with attention softmax).
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
    S_SM_PREP,
    S_SM_START,
    S_SM_IN,
    S_SM_OUT,
    S_WR_REQ,
    S_WR_BEAT,
    S_ERROR,
    S_DONE
  } state_t;

  state_t state;

  logic [15:0] m_r;
  logic [31:0] src0_r;
  logic [31:0] dst_r;
  heatvit_scale_t src0_scale_r;

  logic [8:0]  rd_bi;
  logic [8:0]  rd_beats;
  logic [15:0] field_bytes_r;
  logic        r_ready_r;

  logic [1:0]  hh;
  logic [7:0]  cc;
  logic [7:0]  idx;         // softmax row element index / output index

  // Staging RAMs.
  logic        we_l, we_o;
  logic [7:0]  waddr_l;
  logic [8:0]  waddr_o;
  logic [63:0] wdata_l, wdata_o;
  logic [7:0]  wstrb_l, wstrb_o;
  logic [7:0]  raddr_l;
  logic [8:0]  raddr_o;
  logic [63:0] rdata_l, rdata_o;

  heatvit_sdp_ram #(.WIDTH(64), .DEPTH(148)) u_ram_l (
    .clk(clk), .we(we_l), .waddr(waddr_l), .wdata(wdata_l), .wstrb(wstrb_l),
    .raddr(raddr_l), .rdata(rdata_l)
  );
  heatvit_sdp_ram #(.WIDTH(64), .DEPTH(296)) u_ram_o (
    .clk(clk), .we(we_o), .waddr(waddr_o), .wdata(wdata_o), .wstrb(wstrb_o),
    .raddr(raddr_o), .rdata(rdata_o)
  );

  // Selector softmax core.
  logic        sm_start;
  logic        sm_busy;
  logic        sm_done;
  logic [7:0]  sm_row_len;
  logic        sm_s_valid;
  logic        sm_s_ready;
  heatvit_q8_16_t sm_s_data;
  logic        sm_m_valid;
  logic        sm_m_ready;
  heatvit_uq0_16_t sm_m_data;
  logic        sm_m_last;
  logic        sm_error_zero_sum;

  heatvit_softmax_selector u_sm (
    .clk            (clk),
    .rst_n          (rst_n),
    .start          (sm_start),
    .busy           (sm_busy),
    .done           (sm_done),
    .row_len        (sm_row_len),
    .s_valid        (sm_s_valid),
    .s_ready        (sm_s_ready),
    .s_data         (sm_s_data),
    .div_req_valid  (div_req_valid),
    .div_req_ready  (div_req_ready),
    .div_num        (div_num),
    .div_den        (div_den),
    .div_rsp_valid  (div_rsp_valid),
    .div_quot       (div_quot),
    .div_rem        (div_rem),
    .div_div_zero   (div_div_zero),
    .m_valid        (sm_m_valid),
    .m_ready        (sm_m_ready),
    .m_data         (sm_m_data),
    .m_last         (sm_m_last),
    .error_zero_sum (sm_error_zero_sum)
  );

  logic [10:0] wr_bi;
  logic [10:0] wr_beats;
  logic [15:0] out_bytes_r;
  logic [7:0]  wr_strb_c;

  // Row index hh*m + c (11 bits; max 2*196+196 = 588).
  logic [10:0] row_idx;
  assign row_idx = {1'b0, hh} * m_r + {3'd0, cc};

  // Logit pair: byte offset 2*row_idx, always inside one word; word
  // row_idx>>2, byte lane {row_idx[1:0], 1'b0}.
  assign waddr_l = rd_bi[7:0];

  // Receive write strobes (tail masked by field_bytes_r).
  always_comb begin
    wstrb_l = 8'hFF;
    for (int j = 0; j < 8; j++) begin
      if (int'(rd_bi) * 8 + j >= int'(field_bytes_r))
        wstrb_l[j] = 1'b0;
    end
  end
  assign we_l    = (state == S_RD_RECV) && req_r_valid && r_ready_r;
  assign wdata_l = req_r_data;

  // Logit read: address issued during S_SM_PREP, held through S_SM_START/
  // S_SM_IN (registered read: rdata at cycle T = mem[raddr(T-1)]).
  assign raddr_l = (state == S_SM_PREP) ? row_idx[10:2] :
                   ((state == S_SM_START) || (state == S_SM_IN))
                       ? row_idx[10:2] : 8'd0;

  // Keep-value store: one byte-enabled write per row (4 bytes always fit
  // one word, lanes 0..3 for even row, 4..7 for odd row).
  assign we_o    = (state == S_SM_OUT) && sm_m_valid && (idx == 8'd1);
  assign waddr_o = row_idx[10:1];
  assign wdata_o = row_idx[0]
                   ? {{8'h00, sm_m_data[16], sm_m_data[15:8],
                       sm_m_data[7:0]}, 32'd0}
                   : {32'd0, {8'h00, sm_m_data[16], sm_m_data[15:8],
                              sm_m_data[7:0]}};
  assign wstrb_o = row_idx[0] ? 8'hF0 : 8'h0F;

  // Output read: accept-gated lookahead (backpressure-safe).
  wire w_accept = (state == S_WR_BEAT) && req_w_valid && req_w_ready;
  assign raddr_o = (state == S_WR_BEAT)
                   ? (w_accept ? ((wr_bi[8:0] == 9'd295) ? 9'd295
                                                        : wr_bi[8:0] + 9'd1)
                               : wr_bi[8:0])
                   : 9'd0;

  function automatic heatvit_q8_16_t logit_q16(input logic [7:0] v);
    heatvit_s128_t wide;
    heatvit_s128_t scaled;
    wide = $signed({{120{v[7]}}, v});
    scaled = scale_to_exp_s128(wide, src0_scale_r, -6'sd16);
    if (scaled > 128'sd8388607) return 24'sd8388607;
    if (scaled < -128'sd8388608) return -24'sd8388608;
    return heatvit_q8_16_t'(scaled[23:0]);
  endfunction

  // Logit bytes for the current row: lane = {row_idx[1:0], 1'b0}.
  wire [2:0] lg_lane = {row_idx[1:0], 1'b0};
  wire [7:0] lg0 = rdata_l[8*lg_lane +: 8];
  wire [7:0] lg1 = rdata_l[8*lg_lane + 8 +: 8];

  always_comb begin
    wr_strb_c = 8'h00;
    if (state == S_WR_BEAT) begin
      for (int j = 0; j < 8; j++) begin
        if (int'(wr_bi) * 8 + j < int'(out_bytes_r))
          wr_strb_c[j] = 1'b1;
      end
    end
  end

  assign req_valid   = (state == S_RD_REQ) || (state == S_WR_REQ);
  assign req_write   = (state == S_WR_REQ);
  assign req_addr    = (state == S_WR_REQ) ? dst_r : src0_r;
  assign req_bytes   = (state == S_WR_REQ)
                       ? {16'd0, wr_beats} * 32'd8
                       : {16'd0, (field_bytes_r + 16'd7) & 16'hFFF8};
  assign req_r_ready = r_ready_r;
  assign req_w_valid = (state == S_WR_BEAT);
  assign req_w_data  = rdata_o;
  assign req_w_strb  = wr_strb_c;
  assign req_w_last  = (wr_bi == (wr_beats - 11'd1));

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
      wr_bi       <= 11'd0;
      wr_beats    <= 11'd0;
      sm_start    <= 1'b0;
      sm_s_valid  <= 1'b0;
      sm_m_ready  <= 1'b0;
    end else begin
      done        <= 1'b0;
      error_valid <= 1'b0;
      r_ready_r   <= 1'b0;
      sm_start    <= 1'b0;
      sm_s_valid  <= 1'b0;
      sm_m_ready  <= 1'b0;

      case (state)
        S_IDLE: begin
          if (start) begin
            m_r          <= m_eff;
            src0_r       <= src0_base;
            dst_r        <= dst_base;
            src0_scale_r <= src0_scale;
            busy         <= 1'b1;
            if (m_eff == 16'd0 || m_eff > MAX_C[15:0]) begin
              error_code <= ERR_DIMENSION;
              state      <= S_ERROR;
            end else begin
              field_bytes_r <= 16'(m_eff) * 16'd6;   // 3 * C * 2 bytes
              rd_beats      <= 9'((int'(m_eff) * 6 + 7) / 8);
              rd_bi         <= 9'd0;
              state         <= S_RD_REQ;
            end
          end
        end

        S_RD_REQ: begin
          if (req_valid && req_ready) begin
            rd_bi     <= 9'd0;
            r_ready_r <= 1'b1;
            state     <= S_RD_RECV;
          end
        end

        S_RD_RECV: begin
          if (req_r_valid && r_ready_r) begin
            if (rd_bi == rd_beats - 9'd1) begin
              r_ready_r <= 1'b0;
              hh        <= 2'd0;
              cc        <= 8'd0;
              state     <= S_SM_PREP;
            end else begin
              rd_bi <= rd_bi + 9'd1;
            end
          end else begin
            r_ready_r <= 1'b1;
          end
        end

        S_SM_PREP: begin
          // The read address for the current row is issued here; rdata
          // holds the row word during S_SM_START/S_SM_IN.
          sm_start   <= 1'b1;
          sm_row_len <= 8'd2;
          state      <= S_SM_START;
        end

        S_SM_START: begin
          sm_s_valid <= 1'b1;
          sm_s_data  <= logit_q16(lg0);
          idx        <= 8'd0;
          state      <= S_SM_IN;
        end

        S_SM_IN: begin
          sm_s_valid <= 1'b1;
          // Present the second logit while the first is accepted.
          sm_s_data  <= logit_q16(lg1);
          if (sm_s_valid && sm_s_ready) begin
            if (idx == 8'd1) begin
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
            if (sm_error_zero_sum) begin
              error_code <= ERR_SOFTMAX_ZERO_SUM;
              state      <= S_ERROR;
            end else begin
              if (sm_m_last) begin
                if (cc == m_r[7:0] - 8'd1) begin
                  if (hh == 2'd2) begin
                    out_bytes_r <= 16'(m_r) * 16'd12;  // 3 * C * 4
                    wr_beats <= 11'((int'(m_r) * 12 + 7) / 8);
                    wr_bi    <= 11'd0;
                    state    <= S_WR_REQ;
                  end else begin
                    hh    <= hh + 2'd1;
                    cc    <= 8'd0;
                    state <= S_SM_PREP;
                  end
                end else begin
                  cc    <= cc + 8'd1;
                  state <= S_SM_PREP;
                end
              end else begin
                idx <= idx + 8'd1;
              end
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
