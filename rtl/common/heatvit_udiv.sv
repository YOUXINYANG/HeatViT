// Restoring unsigned divider. One quotient bit is produced per cycle.
module heatvit_udiv #(
  parameter int NUM_W  = 64,
  parameter int DEN_W  = 64,
  parameter int QUOT_W = 64
)(
  input  logic               clk,
  input  logic               rst_n,
  input  logic               start,
  output logic               busy,
  output logic               done,
  output logic               divide_by_zero,
  input  logic [NUM_W-1:0]   numerator,
  input  logic [DEN_W-1:0]   denominator,
  output logic [QUOT_W-1:0]  quotient,
  output logic [DEN_W-1:0]   remainder
);

  localparam logic S_IDLE = 1'b0;
  localparam logic S_RUN  = 1'b1;
  localparam int CNT_W = $clog2(QUOT_W + 1);

  logic [NUM_W-1:0]  numerator_shift;
  logic [DEN_W-1:0]  denominator_reg;
  logic [DEN_W-1:0]  remainder_reg;
  logic [QUOT_W-1:0] quotient_shift;
  logic [CNT_W-1:0]  count;
  logic              state;
  logic              div_zero_reg;

  logic [DEN_W-1:0]  shifted_rem;
  logic              bit_one;
  logic [QUOT_W-1:0] next_quotient;
  logic [DEN_W-1:0]  next_remainder;

  assign shifted_rem = {remainder_reg[DEN_W-2:0], numerator_shift[NUM_W-1]};
  assign bit_one = shifted_rem >= denominator_reg;
  assign next_quotient  = {quotient_shift[QUOT_W-2:0], bit_one};
  assign next_remainder = bit_one ? (shifted_rem - denominator_reg) : shifted_rem;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state           <= S_IDLE;
      busy            <= 1'b0;
      done            <= 1'b0;
      divide_by_zero  <= 1'b0;
      div_zero_reg    <= 1'b0;
      count           <= {CNT_W{1'b0}};
      quotient_shift  <= {QUOT_W{1'b0}};
      remainder_reg   <= {DEN_W{1'b0}};
      numerator_shift <= {NUM_W{1'b0}};
      denominator_reg <= {DEN_W{1'b0}};
      quotient        <= {QUOT_W{1'b0}};
      remainder       <= {DEN_W{1'b0}};
    end else begin
      done <= 1'b0;
      if (start && busy) $error("heatvit_udiv: start asserted while busy");
      case (state)
        S_IDLE: begin
          if (start) begin
            numerator_shift <= numerator;
            denominator_reg <= denominator;
            remainder_reg   <= {DEN_W{1'b0}};
            quotient_shift  <= {QUOT_W{1'b0}};
            count           <= {CNT_W{1'b0}};
            div_zero_reg    <= (denominator == {DEN_W{1'b0}});
            divide_by_zero  <= (denominator == {DEN_W{1'b0}});
            busy            <= 1'b1;
            state           <= S_RUN;
          end
        end
        S_RUN: begin
          if (div_zero_reg) begin
            done           <= 1'b1;
            busy           <= 1'b0;
            state          <= S_IDLE;
          end else if (count == QUOT_W - 1) begin
            quotient       <= next_quotient;
            remainder      <= next_remainder;
            done           <= 1'b1;
            busy           <= 1'b0;
            state          <= S_IDLE;
          end else begin
            numerator_shift <= {numerator_shift[NUM_W-2:0], 1'b0};
            remainder_reg   <= next_remainder;
            quotient_shift  <= next_quotient;
            count           <= count + 1'b1;
          end
        end
        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
