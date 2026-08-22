// Piecewise-linear PLAN sigmoid approximation. Input is signed Q8.16,
// output is unsigned Q0.16. Negative inputs use y = 1.0 - y(abs(x)).
module heatvit_plan_sigmoid
  import heatvit_pkg::*;
(
  input  logic            clk,
  input  logic            rst_n,
  input  logic            start,
  output logic            busy,
  output logic            done,
  input  heatvit_q8_16_t  x_in,
  output heatvit_uq0_16_t y_out
);

  localparam int PLAN_BP1_Q16 = 65536;   // 1.0
  localparam int PLAN_BP2_Q16 = 155648;  // 2.375
  localparam int PLAN_BP3_Q16 = 327680;  // 5.0
  localparam int PLAN_C0_Q16  = 32768;   // 1/2
  localparam int PLAN_C1_Q16  = 40960;   // 5/8
  localparam int PLAN_C2_Q16  = 55296;   // 27/32

  heatvit_q8_16_t x_r;
  logic signed [24:0] abs_x;
  heatvit_uq0_16_t y_abs;
  heatvit_uq0_16_t y_comb;

  assign abs_x = (x_r < 0) ? -$signed({x_r[23], x_r}) : $signed({x_r[23], x_r});

  always_comb begin
    if (abs_x >= PLAN_BP3_Q16)
      y_abs = 17'd65536;
    else if (abs_x >= PLAN_BP2_Q16)
      y_abs = heatvit_uq0_16_t'((abs_x >> 5) + PLAN_C2_Q16);
    else if (abs_x >= PLAN_BP1_Q16)
      y_abs = heatvit_uq0_16_t'((abs_x >> 3) + PLAN_C1_Q16);
    else
      y_abs = heatvit_uq0_16_t'((abs_x >> 2) + PLAN_C0_Q16);
  end

  assign y_comb = (x_r < 0) ? (17'd65536 - y_abs) : y_abs;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      x_r   <= 24'sd0;
      busy  <= 1'b0;
      done  <= 1'b0;
      y_out <= 17'd0;
    end else begin
      done <= 1'b0;
      if (start && busy) $error("heatvit_plan_sigmoid: start asserted while busy");
      if (start && !busy) begin
        x_r  <= x_in;
        busy <= 1'b1;
      end else if (busy) begin
        y_out <= y_comb;
        busy  <= 1'b0;
        done  <= 1'b1;
      end
    end
  end

endmodule
