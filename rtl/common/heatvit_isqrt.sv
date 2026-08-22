// Restoring integer square root. Two radicand bits are consumed per cycle.
module heatvit_isqrt #(
  parameter int RAD_W = 48
)(
  input  logic               clk,
  input  logic               rst_n,
  input  logic               start,
  output logic               busy,
  output logic               done,
  input  logic [RAD_W-1:0]   radicand,
  output logic [((RAD_W + 1) / 2) - 1:0] root,
  output logic [RAD_W-1:0]   remainder
);

  localparam int ROOT_W = (RAD_W + 1) / 2;
  localparam int CNT_W = $clog2(ROOT_W + 1);
  localparam logic S_IDLE = 1'b0;
  localparam logic S_RUN  = 1'b1;

  logic [RAD_W-1:0]  radicand_shift;
  logic [ROOT_W-1:0] root_reg;
  logic [RAD_W-1:0]  remainder_reg;
  logic [CNT_W-1:0]  count;
  logic              state;

  logic [RAD_W-1:0]  shifted_rem;
  logic [ROOT_W:0]   trial;
  logic              bit_one;
  logic [ROOT_W-1:0] next_root;
  logic [RAD_W-1:0]  next_remainder;

  assign shifted_rem = {remainder_reg[RAD_W-3:0], radicand_shift[RAD_W-1:RAD_W-2]};
  assign trial       = {root_reg[ROOT_W-2:0], 2'b00} | {{(ROOT_W - 1) {1'b0}}, 2'b01};
  assign bit_one     = shifted_rem >= trial;
  assign next_root       = {root_reg[ROOT_W-2:0], bit_one};
  assign next_remainder = bit_one ? (shifted_rem - trial) : shifted_rem;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state          <= S_IDLE;
      busy           <= 1'b0;
      done           <= 1'b0;
      count          <= {CNT_W{1'b0}};
      root_reg       <= {ROOT_W{1'b0}};
      remainder_reg  <= {RAD_W{1'b0}};
      radicand_shift <= {RAD_W{1'b0}};
      root           <= {ROOT_W{1'b0}};
      remainder      <= {RAD_W{1'b0}};
    end else begin
      done <= 1'b0;
      if (start && busy) $error("heatvit_isqrt: start asserted while busy");
      case (state)
        S_IDLE: begin
          if (start) begin
            radicand_shift <= radicand;
            root_reg       <= {ROOT_W{1'b0}};
            remainder_reg  <= {RAD_W{1'b0}};
            count          <= {CNT_W{1'b0}};
            busy           <= 1'b1;
            state          <= S_RUN;
          end
        end
        S_RUN: begin
          if (count == ROOT_W - 1) begin
            root           <= next_root;
            remainder      <= next_remainder;
            done           <= 1'b1;
            busy           <= 1'b0;
            state          <= S_IDLE;
          end else begin
            radicand_shift <= {radicand_shift[RAD_W-3:0], 2'b00};
            root_reg       <= next_root;
            remainder_reg  <= next_remainder;
            count          <= count + 1'b1;
          end
        end
        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
