// Combinational 48-bit -> int8 rescaler with saturation flag.
module heatvit_requant
  import heatvit_pkg::*;
(
  input  heatvit_s48_t  in_value,
  input  heatvit_scale_t src_scale_exp,
  input  heatvit_scale_t dst_scale_exp,
  output heatvit_s8_t   out_value,
  output logic          saturated
);

  wire heatvit_s128_t wide = $signed({{(128 - 48) {in_value[47]}}, in_value});
  wire heatvit_s128_t scaled = scale_to_exp_s128(wide, src_scale_exp, dst_scale_exp);

  assign out_value = sat_s8(scaled);
  assign saturated = (scaled > $signed(8'sd127)) || (scaled < $signed(-8'sd128));

endmodule
