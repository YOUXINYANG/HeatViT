// Three-pass row softmax core. Pass 1 buffers the Q8.16 row and finds its
// maximum; pass 2 computes the range-reduced exponentials and their sum and
// requests one Q0.32 reciprocal from the external shared divider; pass 3
// emits each scaled Q0.16 value. DELTA2_Q16 selects the 0.5/1.0 wrapper
// scaling; the wrappers perform the final width narrowing.
//
// P7-4 timing rewrite: the emit datapath (exp_element + scale_element, the
// former single-cycle chain of four 48-bit multiplies) is split into a
// 4-stage exp pipeline plus a 2-stage scale pipeline that streams to
// m_valid/m_data. The same exp pipeline also produces the pass-2 sum, so
// the sum pass reuses the identical arithmetic. Every stage performs the
// exact same integer operations as the old combinational functions, only
// registered between them: results are bit-identical for every input.
module heatvit_softmax_core
  import heatvit_pkg::*;
#(
  parameter int MAX_ROW    = 197,
  parameter int DELTA2_Q16 = 65536
)
(
  input  logic            clk,
  input  logic            rst_n,
  input  logic            start,
  output logic            busy,
  output logic            done,
  input  logic [7:0]      row_len,
  input  logic            s_valid,
  output logic            s_ready,
  input  heatvit_q8_16_t  s_data,
  output logic            div_req_valid,
  input  logic            div_req_ready,
  output logic [63:0]     div_num,
  output logic [63:0]     div_den,
  input  logic            div_rsp_valid,
  input  logic [63:0]     div_quot,
  input  logic [63:0]     div_rem,
  input  logic            div_div_zero,
  output logic            m_valid,
  input  logic            m_ready,
  output heatvit_uq0_16_t m_data,
  output logic            m_last,
  output logic            error_zero_sum
);

  localparam logic [3:0] S_IDLE     = 4'd0;
  localparam logic [3:0] S_LOAD_MAX = 4'd1;
  localparam logic [3:0] S_EXP_SUM  = 4'd2;
  localparam logic [3:0] S_DIV_WAIT = 4'd3;
  localparam logic [3:0] S_EMIT     = 4'd4;

  localparam int IDX_W = $clog2(MAX_ROW + 1);
  // ceil(2^32 / 45426), used with a one-step correction for exact floor
  // division by the ln(2) Q8.16 constant.
  localparam int LN2_DIV_MAGIC = 94548;

  logic [3:0]      state;
  logic [IDX_W-1:0] idx;
  logic [7:0]      row_len_r;
  heatvit_q8_16_t  row_max_r;
  heatvit_q8_16_t  buffer [MAX_ROW];
  logic [31:0]     sum_r;
  logic [32:0]     recip_r;
  logic            req_sent;

  // Pipeline flow control. During S_EXP_SUM the scale stages are idle and
  // the exp pipeline advances every cycle; during S_EMIT the whole chain
  // advances unless the output handshake stalls.
  wire feeding = (state == S_EXP_SUM || state == S_EMIT) &&
                 (idx < {1'b0, row_len_r});
  wire in_emit  = (state == S_EMIT);
  wire advance  = in_emit ? (!m_valid || m_ready) : 1'b1;

  heatvit_q8_16_t cur_x;
  // The pipeline keeps advancing for a few flush cycles after the last
  // feed, with idx parked at row_len_r; clamp the read index so the flush
  // never reads past the buffer (which would inject X into the pipeline).
  assign cur_x = buffer[feeding ? idx : {IDX_W{1'b0}}];

  // ------------------------------------------------------------------
  // Exp pipeline: stage0 -> xt/neg_xt/prod_z, stage1 -> z/p/shifted,
  // stage2 -> square, stage3 -> e (17-bit exp value). Valid and last
  // markers travel with the data (4 register hops, matching the 4 data
  // stages: xt -> shifted -> square -> e).
  // ------------------------------------------------------------------
  logic            e1_valid_r, e2_valid_r, e3_valid_r, e4_valid_r;
  logic            last1_r, last2_r, last3_r, last4_r;
  heatvit_s48_t    xt_r;
  logic [24:0]     neg_xt_r;
  logic [47:0]     prod_z_r;
  logic [8:0]      z_r;
  logic [8:0]      z2_r;
  heatvit_s48_t    shifted_r;
  heatvit_s48_t    square_r;
  logic [16:0]     e_r;

  // Stage 0 comb.
  heatvit_s48_t    xt_w;
  logic [24:0]     neg_xt_w;
  logic [47:0]     prod_z_w;
  assign xt_w     = heatvit_s48_t'(cur_x) - heatvit_s48_t'(row_max_r);
  assign neg_xt_w = 25'h0 - xt_w[24:0];
  assign prod_z_w = 48'(neg_xt_w) * 48'd94548;

  // Stage 1 comb.
  logic [8:0]   q0_w;
  logic [8:0]   z_w;
  heatvit_s48_t p_w;
  heatvit_s48_t shifted_w;
  assign q0_w      = prod_z_r[40:32];
  assign z_w       = (((q0_w + 9'd1) * 25'd45426) <= neg_xt_r)
                       ? (q0_w + 9'd1) : q0_w;
  assign p_w       = xt_r + heatvit_s48_t'(z_w) * 48'sd45426;
  assign shifted_w = p_w + 48'sd88670;

  // Stage 2 comb.
  heatvit_s48_t square_w;
  assign square_w = round_shift_away_s48(shifted_r * shifted_r, 6'd16);

  // Stage 3 comb (square_r and z2_r are both the stage2->stage3 boundary,
  // so e pairs each element's square with its own z).
  heatvit_s48_t exp_q16_w;
  logic [16:0]  e_w;
  assign exp_q16_w = round_shift_away_s48(48'sd23495 * square_r, 6'd16)
                     + 48'sd22544;
  assign e_w = (z2_r >= 9'd17) ? 17'd0 : (exp_q16_w[16:0] >> z2_r[4:0]);

  // ------------------------------------------------------------------
  // Scale pipeline: stage4 -> ratio, stage5 -> scaled output.
  // ------------------------------------------------------------------
  logic             s1_valid_r;
  logic             last5_r;
  logic             last6_r;
  heatvit_s48_t     ratio_r;
  heatvit_uq0_16_t  scaled_r;

  logic [49:0]    prod_s_w;
  heatvit_s48_t   ratio_w;
  assign prod_s_w = 50'(e_r) * 50'(recip_r);
  assign ratio_w  = heatvit_s48_t'((prod_s_w + 50'd32768) >> 16);

  logic [50:0]    scaled_prod_w;
  logic [50:0]    shifted_scaled_w;
  assign scaled_prod_w     = 51'(ratio_r) * 51'(DELTA2_Q16);
  assign shifted_scaled_w  = (scaled_prod_w + 51'd32768) >> 16;

  assign s_ready = (state == S_LOAD_MAX);
  assign m_data  = scaled_r;
  assign m_last  = m_valid && last6_r;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state          <= S_IDLE;
      busy           <= 1'b0;
      done           <= 1'b0;
      idx            <= {IDX_W{1'b0}};
      row_len_r      <= 8'd0;
      row_max_r      <= 24'sd0;
      sum_r          <= 32'd0;
      recip_r        <= 33'd0;
      div_req_valid  <= 1'b0;
      div_num        <= 64'd0;
      div_den        <= 64'd0;
      req_sent       <= 1'b0;
      m_valid        <= 1'b0;
      error_zero_sum <= 1'b0;
      e1_valid_r     <= 1'b0;
      e2_valid_r     <= 1'b0;
      e3_valid_r     <= 1'b0;
      e4_valid_r     <= 1'b0;
      last1_r        <= 1'b0;
      last2_r        <= 1'b0;
      last3_r        <= 1'b0;
      last4_r        <= 1'b0;
      s1_valid_r     <= 1'b0;
      last5_r        <= 1'b0;
      last6_r        <= 1'b0;
      xt_r           <= 48'sd0;
      neg_xt_r       <= 25'd0;
      prod_z_r       <= 48'd0;
      z_r            <= 9'd0;
      z2_r           <= 9'd0;
      shifted_r      <= 48'sd0;
      square_r       <= 48'sd0;
      e_r            <= 17'd0;
      ratio_r        <= 48'sd0;
      scaled_r       <= 17'd0;
      for (int i = 0; i < MAX_ROW; i++) buffer[i] <= 24'sd0;
    end else begin
      done <= 1'b0;
      error_zero_sum <= 1'b0;
      if (start && busy) $error("heatvit_softmax_core: start asserted while busy");
      case (state)
        S_IDLE: begin
          div_req_valid <= 1'b0;
          req_sent      <= 1'b0;
          m_valid       <= 1'b0;
          if (start) begin
            if (row_len == 8'd0 || row_len > MAX_ROW)
              $fatal(2, "heatvit_softmax_core: row_len=%0d out of range", row_len);
            row_len_r <= row_len;
            idx       <= {IDX_W{1'b0}};
            sum_r     <= 32'd0;
            busy      <= 1'b1;
            state     <= S_LOAD_MAX;
          end
        end
        S_LOAD_MAX: begin
          if (s_valid && s_ready) begin
            buffer[idx] <= s_data;
            if (idx == {IDX_W{1'b0}})
              row_max_r <= s_data;
            else if (s_data > row_max_r)
              row_max_r <= s_data;
            if (idx == row_len_r - 1) begin
              idx   <= {IDX_W{1'b0}};
              state <= S_EXP_SUM;
            end else begin
              idx <= idx + 1'b1;
            end
          end
        end
        S_EXP_SUM: begin
          // Sum completes when the last element's exp leaves stage 3
          // (e4_valid marks the exp output register, aligned with e_r).
          if (last4_r && e4_valid_r) begin
            div_num       <= 64'h1_0000_0000;
            div_den       <= 64'(sum_r) + 64'(e_r);
            div_req_valid <= 1'b1;
            req_sent      <= 1'b0;
            idx           <= {IDX_W{1'b0}};
            e1_valid_r    <= 1'b0;
            e2_valid_r    <= 1'b0;
            e3_valid_r    <= 1'b0;
            e4_valid_r    <= 1'b0;
            state         <= S_DIV_WAIT;
          end else begin
            // Advance the exp pipeline every cycle (no output backpressure
            // during the sum pass).
            if (advance) begin
              if (feeding) idx <= idx + 1'b1;
              e1_valid_r <= feeding;
              last1_r    <= feeding && (idx == row_len_r - 1);
              xt_r       <= xt_w;
              neg_xt_r   <= neg_xt_w;
              prod_z_r   <= prod_z_w;
              e2_valid_r <= e1_valid_r;
              last2_r    <= last1_r;
              z_r        <= z_w;
              shifted_r  <= shifted_w;
              e3_valid_r <= e2_valid_r;
              last3_r    <= last2_r;
              z2_r       <= z_r;
              square_r   <= square_w;
              e4_valid_r <= e3_valid_r;
              last4_r    <= last3_r;
              e_r        <= e_w;
            end
            if (e4_valid_r) sum_r <= sum_r + 32'(e_r);
          end
        end
        S_DIV_WAIT: begin
          if (!req_sent && div_req_ready) begin
            div_req_valid <= 1'b0;
            req_sent      <= 1'b1;
          end
          if (div_rsp_valid) begin
            if (div_div_zero) begin
              error_zero_sum <= 1'b1;
              busy           <= 1'b0;
              done           <= 1'b1;
              state          <= S_IDLE;
            end else begin
              recip_r  <= div_quot[32:0] + (((div_rem << 1) >= div_den) ? 33'd1 : 33'd0);
              idx      <= {IDX_W{1'b0}};
              e1_valid_r <= 1'b0;
              e2_valid_r <= 1'b0;
              e3_valid_r <= 1'b0;
              e4_valid_r <= 1'b0;
              s1_valid_r <= 1'b0;
              m_valid   <= 1'b0;
              state     <= S_EMIT;
            end
          end
        end
        S_EMIT: begin
          if (m_valid && m_ready && last6_r) begin
            m_valid <= 1'b0;
            busy    <= 1'b0;
            done    <= 1'b1;
            state   <= S_IDLE;
          end else begin
            if (advance) begin
              if (feeding) idx <= idx + 1'b1;
              e1_valid_r <= feeding;
              last1_r    <= feeding && (idx == row_len_r - 1);
              xt_r       <= xt_w;
              neg_xt_r   <= neg_xt_w;
              prod_z_r   <= prod_z_w;
              e2_valid_r <= e1_valid_r;
              last2_r    <= last1_r;
              z_r        <= z_w;
              shifted_r  <= shifted_w;
              e3_valid_r <= e2_valid_r;
              last3_r    <= last2_r;
              z2_r       <= z_r;
              square_r   <= square_w;
              e4_valid_r <= e3_valid_r;
              last4_r    <= last3_r;
              e_r        <= e_w;
              s1_valid_r <= e4_valid_r;
              last5_r    <= last4_r;
              ratio_r    <= ratio_w;
              m_valid    <= s1_valid_r;
              last6_r    <= last5_r;
              scaled_r   <= shifted_scaled_w[16:0];
            end
          end
        end
        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
