// Fixed-priority (0 > 1 > 2) arbiter over three divider clients.
module heatvit_div_arbiter #(
  parameter int NUM_W  = 64,
  parameter int DEN_W  = 64,
  parameter int QUOT_W = 64
)(
  input  logic                 clk,
  input  logic                 rst_n,
  input  logic [2:0]           req_valid,
  output logic [2:0]           req_ready,
  input  logic [NUM_W-1:0]     num [2:0],
  input  logic [DEN_W-1:0]     den [2:0],
  output logic [2:0]           rsp_valid,
  output logic [QUOT_W-1:0]    quot [2:0],
  output logic [DEN_W-1:0]     rem [2:0],
  output logic [2:0]           div_zero
);

  logic                 start;
  logic                 busy;
  logic                 done;
  logic                 div_zero_q;
  logic [NUM_W-1:0]     num_q;
  logic [DEN_W-1:0]     den_q;
  logic [QUOT_W-1:0]    quot_q;
  logic [DEN_W-1:0]     rem_q;
  logic [1:0]           grant;
  logic                 serving;

  function automatic logic [1:0] pick(input logic [2:0] v);
    if (v[0]) return 2'd0;
    if (v[1]) return 2'd1;
    return 2'd2;
  endfunction

  heatvit_udiv #(
    .NUM_W  (NUM_W),
    .DEN_W  (DEN_W),
    .QUOT_W (QUOT_W)
  ) u_udiv (
    .clk            (clk),
    .rst_n          (rst_n),
    .start          (start),
    .busy           (busy),
    .done           (done),
    .divide_by_zero (div_zero_q),
    .numerator      (num_q),
    .denominator    (den_q),
    .quotient       (quot_q),
    .remainder      (rem_q)
  );

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      grant   <= 2'd0;
      serving <= 1'b0;
      start   <= 1'b0;
      num_q   <= {NUM_W{1'b0}};
      den_q   <= {DEN_W{1'b0}};
    end else begin
      start <= 1'b0;
      if (!serving && !busy) begin
        if (req_valid[0] || req_valid[1] || req_valid[2]) begin
          grant   <= pick(req_valid);
          num_q   <= num[pick(req_valid)];
          den_q   <= den[pick(req_valid)];
          start   <= 1'b1;
          serving <= 1'b1;
        end
      end else if (serving && done) begin
        serving <= 1'b0;
      end
    end
  end

  always_comb begin
    req_ready = 3'b000;
    if (!serving && !busy && (req_valid[0] || req_valid[1] || req_valid[2]))
      req_ready[pick(req_valid)] = 1'b1;
  end

  always_comb begin
    rsp_valid = 3'b000;
    if (serving && done) rsp_valid[grant] = 1'b1;
  end

  genvar g;
  generate
    for (g = 0; g < 3; g++) begin : g_rsp
      assign quot[g]     = rsp_valid[g] ? quot_q : {QUOT_W{1'b0}};
      assign rem[g]      = rsp_valid[g] ? rem_q  : {DEN_W{1'b0}};
      assign div_zero[g] = rsp_valid[g] ? div_zero_q : 1'b0;
    end
  endgenerate

endmodule
