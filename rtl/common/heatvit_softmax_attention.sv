// Attention softmax wrapper: delta2 = 1.0, output rounded right by 8 and
// saturated to unsigned Q0.8. delta2=1.0 keeps the probability mass at 1.0
// (attention output is not halved); the /sqrt(64) scaling is applied by the
// tensor executor when it requants the score to Q8.16 (ATTN_SOFTMAX s0 =
// score_scale - 3).
module heatvit_softmax_attention
  import heatvit_pkg::*;
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
  output logic [7:0]      m_data,
  output logic            m_last,
  output logic            error_zero_sum
);

  heatvit_uq0_16_t core_m_data;
  logic [8:0]      shifted;

  heatvit_softmax_core #(
    .MAX_ROW    (197),
    .DELTA2_Q16 (65536)
  ) u_core (
    .clk            (clk),
    .rst_n          (rst_n),
    .start          (start),
    .busy           (busy),
    .done           (done),
    .row_len        (row_len),
    .s_valid        (s_valid),
    .s_ready        (s_ready),
    .s_data         (s_data),
    .div_req_valid  (div_req_valid),
    .div_req_ready  (div_req_ready),
    .div_num        (div_num),
    .div_den        (div_den),
    .div_rsp_valid  (div_rsp_valid),
    .div_quot       (div_quot),
    .div_rem        (div_rem),
    .div_div_zero   (div_div_zero),
    .m_valid        (m_valid),
    .m_ready        (m_ready),
    .m_data         (core_m_data),
    .m_last         (m_last),
    .error_zero_sum (error_zero_sum)
  );

  assign shifted = (core_m_data + 17'd128) >> 8;
  assign m_data = (shifted > 9'd255) ? 8'd255 : shifted[7:0];

endmodule
