// One-stage registered residual adder: main + aux, scale aligned, int8 out.
module heatvit_residual
  import heatvit_pkg::*;
(
  input  logic           clk,
  input  logic           rst_n,

  input  logic           main_valid,
  output logic           main_ready,
  input  heatvit_s8_t    main_value,
  input  heatvit_scale_t main_scale_exp,

  input  logic           aux_valid,
  output logic           aux_ready,
  input  heatvit_s8_t    aux_value,
  input  heatvit_scale_t aux_scale_exp,

  input  heatvit_scale_t out_scale_exp,
  output logic           out_valid,
  input  logic           out_ready,
  output heatvit_s8_t    out_value
);

  heatvit_scale_t  common_exp;
  heatvit_s128_t   main_wide;
  heatvit_s128_t   aux_wide;
  heatvit_s128_t   sum_wide;
  heatvit_s128_t   scaled;
  heatvit_s8_t     out_value_next;
  logic            out_valid_q;
  heatvit_s8_t     out_value_q;

  always_comb begin
    common_exp = (main_scale_exp < aux_scale_exp) ? main_scale_exp : aux_scale_exp;
    main_wide  = $signed({{112 {main_value[7]}}, main_value}) <<<
                 ($signed({1'b0, main_scale_exp}) - $signed({1'b0, common_exp}));
    aux_wide   = $signed({{112 {aux_value[7]}}, aux_value}) <<<
                 ($signed({1'b0, aux_scale_exp}) - $signed({1'b0, common_exp}));
    sum_wide   = main_wide + aux_wide;
    scaled     = scale_to_exp_s128(sum_wide, common_exp, out_scale_exp);
    out_value_next = sat_s8(scaled);
  end

  wire fire = main_valid && aux_valid && (out_ready || !out_valid_q);

  assign main_ready = fire;
  assign aux_ready  = fire;
  assign out_valid  = out_valid_q;
  assign out_value  = out_value_q;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_valid_q <= 1'b0;
    end else begin
      if (fire) begin
        out_valid_q <= 1'b1;
        out_value_q <= out_value_next;
      end else if (out_ready) begin
        out_valid_q <= 1'b0;
      end
    end
  end

endmodule
