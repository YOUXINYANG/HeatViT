// Three-pass row softmax core. Pass 1 buffers the Q8.16 row and finds its
// maximum; pass 2 computes the range-reduced exponentials and their sum and
// requests one Q0.32 reciprocal from the external shared divider; pass 3
// emits each scaled Q0.16 value. DELTA2_Q16 selects the 0.5/1.0 wrapper
// scaling; the wrappers perform the final width narrowing.
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

  // Range-reduced Q8.16 exponential: exp(x - max) = exp_poly(p) >> z.
  function automatic logic [16:0] exp_element(
    input heatvit_q8_16_t x,
    input heatvit_q8_16_t row_max
  );
    heatvit_s48_t xt;
    logic [24:0]  neg_xt;
    logic [47:0]  prod_z;
    logic [8:0]   q0;
    logic [8:0]   z;
    heatvit_s48_t p;
    heatvit_s48_t shifted;
    heatvit_s48_t square;
    heatvit_s48_t exp_q16;
    begin
      xt = heatvit_s48_t'(x) - heatvit_s48_t'(row_max);
      neg_xt = 25'h0 - xt[24:0];
      prod_z = 48'(neg_xt) * 48'd94548;
      q0 = prod_z[40:32];
      z = (((q0 + 9'd1) * 25'd45426) <= neg_xt) ? (q0 + 9'd1) : q0;
      p = xt + heatvit_s48_t'(z) * 48'sd45426;
      shifted = p + 48'sd88670;
      square = round_shift_away_s48(shifted * shifted, 16);
      exp_q16 = round_shift_away_s48(48'sd23495 * square, 16) + 48'sd22544;
      return (z >= 9'd17) ? 17'd0 : (exp_q16[16:0] >> z[4:0]);
    end
  endfunction

  // ratio = round(E * recip_q32 / 2^16), scaled = round(ratio * delta2 / 2^16).
  function automatic logic [16:0] scale_element(
    input logic [16:0] exp_value,
    input logic [32:0] recip
  );
    logic [63:0] prod;
    heatvit_s48_t ratio;
    logic [63:0] scaled_prod;
    logic [63:0] shifted_scaled;
    begin
      prod = 64'(exp_value) * 64'(recip);
      ratio = heatvit_s48_t'((prod + 64'd32768) >> 16);
      scaled_prod = 64'(ratio) * 64'(DELTA2_Q16);
      shifted_scaled = (scaled_prod + 64'd32768) >> 16;
      return shifted_scaled[16:0];
    end
  endfunction

  logic [3:0]      state;
  logic [IDX_W-1:0] idx;
  logic [7:0]      row_len_r;
  heatvit_q8_16_t  row_max_r;
  heatvit_q8_16_t  buffer [MAX_ROW];
  logic [31:0]     sum_r;
  logic [32:0]     recip_r;
  logic            req_sent;

  heatvit_q8_16_t cur_x;
  logic [16:0]   cur_e;
  logic [16:0]   cur_scaled;

  assign cur_x = buffer[idx];
  assign cur_e = exp_element(cur_x, row_max_r);
  assign cur_scaled = scale_element(cur_e, recip_r);

  assign s_ready = (state == S_LOAD_MAX);
  assign m_data  = cur_scaled;
  assign m_last  = m_valid && (idx == row_len_r - 1);

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
          sum_r <= sum_r + 32'(cur_e);
          if (idx == row_len_r - 1) begin
            div_num       <= 64'h1_0000_0000;
            div_den       <= 64'(sum_r) + 64'(cur_e);
            div_req_valid <= 1'b1;
            req_sent      <= 1'b0;
            idx           <= {IDX_W{1'b0}};
            state         <= S_DIV_WAIT;
          end else begin
            idx <= idx + 1'b1;
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
              m_valid  <= 1'b1;
              state    <= S_EMIT;
            end
          end
        end
        S_EMIT: begin
          if (m_valid && m_ready) begin
            if (idx == row_len_r - 1) begin
              m_valid <= 1'b0;
              busy    <= 1'b0;
              done    <= 1'b1;
              state   <= S_IDLE;
            end else begin
              idx <= idx + 1'b1;
            end
          end
        end
        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
