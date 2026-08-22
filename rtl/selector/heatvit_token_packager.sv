// Token packager for OP_SELECTOR_FINALIZE (Phase 4 Task 4).
//
// Accumulates pruned normal tokens and the incoming Package into one
// Package token. Each accumulated row adds ``score * feature[d]`` to a
// 48-bit signed per-channel numerator and the raw feature to an unweighted
// fallback sum; the denominator accumulates the scores and the participant
// counter increments.
//
// On div_start the package row is produced: with a non-zero denominator
// each channel rounds ``wnum[d] / den`` to nearest (ties away from zero);
// a zero denominator falls back to the unweighted feature mean
// ``fsum[d] / participants`` and pulses warn_package_den_zero. The row is
// then written to div_addr and the accumulators are cleared for the next
// descriptor.
module heatvit_token_packager
  import heatvit_pkg::*;
#(
  parameter int MAX_C = 197
)(
  input  logic        clk,
  input  logic        rst_n,
  input  logic        reset_acc,
  // Row accumulation command.
  input  logic        acc_start,
  output logic        acc_busy,
  output logic        acc_done,
  output logic        acc_error_valid,
  output logic [7:0]  acc_error_code,
  input  logic [31:0] acc_addr,
  input  logic [16:0] acc_score,
  // Package division/write command.
  input  logic        div_start,
  output logic        div_busy,
  output logic        div_done,
  input  logic [31:0] div_addr,
  output logic        warn_package_den_zero,
  output logic [8:0]  participants,
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
  // Divider client (arbiter client 2, shared).
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
    S_ACC_RD_REQ,
    S_ACC_RD_RECV,
    S_ACC_DONE,
    S_DIV_REQ,
    S_DIV_WAIT,
    S_WR_REQ,
    S_WR_BEAT,
    S_DIV_DONE
  } state_t;

  state_t state;

  logic [31:0] acc_addr_r;
  logic [16:0] acc_score_r;
  logic [31:0] div_addr_r;

  logic signed [47:0] wnum [0:191];
  logic signed [31:0] fsum [0:191];
  logic [31:0] den_r;
  logic [8:0]  count_r;

  logic [4:0]  rd_bi;
  logic        r_ready_r;
  logic [7:0]  d;            // channel 0..191
  logic        sign_r;
  logic [63:0] div_den_r;
  logic [7:0]  obuf [0:191];
  logic [4:0]  wr_bi;
  logic [63:0] wr_data_c;

  integer bi;
  initial begin
    for (bi = 0; bi < 192; bi++) begin
      wnum[bi] = 48'sd0;
      fsum[bi] = 32'sd0;
      obuf[bi] = 8'h00;
    end
  end

  always_comb begin
    wr_data_c = 64'h0000000000000000;
    if (state == S_WR_BEAT) begin
      for (int j = 0; j < 8; j++)
        wr_data_c[8*j +: 8] = obuf[int'(wr_bi) * 8 + j];
    end
  end

  assign req_valid   = (state == S_ACC_RD_REQ) || (state == S_WR_REQ);
  assign req_write   = (state == S_WR_REQ);
  assign req_addr    = (state == S_WR_REQ) ? div_addr_r : acc_addr_r;
  assign req_bytes   = 32'd192;
  assign req_r_ready = r_ready_r;
  assign req_w_valid = (state == S_WR_BEAT);
  assign req_w_data  = wr_data_c;
  assign req_w_strb  = 8'hff;
  assign req_w_last  = (wr_bi == 5'd23);

  assign participants   = count_r;
  assign acc_busy       = (state == S_ACC_RD_REQ) || (state == S_ACC_RD_RECV);
  assign div_busy       = (state == S_DIV_REQ) || (state == S_DIV_WAIT) ||
                          (state == S_WR_REQ) || (state == S_WR_BEAT);
  assign div_req_valid  = (state == S_DIV_REQ);

  always_comb begin
    logic [47:0] num_c;
    logic [31:0] den_c;
    div_num = 64'd0;
    div_den = 64'd1;
    if (state == S_DIV_REQ) begin
      num_c = (den_r == 32'd0) ? {{16{fsum[d][31]}}, fsum[d]} : wnum[d];
      den_c = (den_r == 32'd0) ? {23'd0, count_r} : den_r;
      div_num = {16'd0, (num_c[47] ? -num_c : num_c)};
      div_den = {32'd0, den_c};
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state       <= S_IDLE;
      acc_done    <= 1'b0;
      div_done    <= 1'b0;
      acc_error_valid <= 1'b0;
      acc_error_code  <= 8'd0;
      r_ready_r   <= 1'b0;
      rd_bi       <= 5'd0;
      wr_bi       <= 5'd0;
      d           <= 8'd0;
      sign_r      <= 1'b0;
      div_den_r   <= 64'd1;
      den_r       <= 32'd0;
      count_r     <= 9'd0;
      warn_package_den_zero <= 1'b0;
      for (int k = 0; k < 192; k++) begin
        wnum[k] <= 48'sd0;
        fsum[k] <= 32'sd0;
      end
    end else begin
      acc_done    <= 1'b0;
      div_done    <= 1'b0;
      acc_error_valid <= 1'b0;
      r_ready_r   <= 1'b0;
      warn_package_den_zero <= 1'b0;
      if (reset_acc) begin
        den_r   <= 32'd0;
        count_r <= 9'd0;
        for (int k = 0; k < 192; k++) begin
          wnum[k] <= 48'sd0;
          fsum[k] <= 32'sd0;
        end
      end

      case (state)
        S_IDLE: begin
          if (acc_start) begin
            acc_addr_r <= acc_addr;
            acc_score_r <= acc_score;
            rd_bi <= 5'd0;
            state <= S_ACC_RD_REQ;
          end else if (div_start) begin
            div_addr_r <= div_addr;
            d          <= 8'd0;
            if (den_r == 32'd0) begin
              warn_package_den_zero <= 1'b1;
              div_den_r <= {32'd0, {23'd0, count_r}};
            end else begin
              div_den_r <= {32'd0, den_r};
            end
            state <= S_DIV_REQ;
          end
        end

        S_ACC_RD_REQ: begin
          if (req_valid && req_ready) begin
            rd_bi     <= 5'd0;
            r_ready_r <= 1'b1;
            state     <= S_ACC_RD_RECV;
          end
        end

        S_ACC_RD_RECV: begin
          if (req_r_valid && r_ready_r) begin
            for (int j = 0; j < 8; j++) begin
              wnum[int'(rd_bi) * 8 + j] <=
                  wnum[int'(rd_bi) * 8 + j] +
                  $signed({31'd0, acc_score_r}) *
                  $signed(req_r_data[8*j +: 8]);
              fsum[int'(rd_bi) * 8 + j] <=
                  fsum[int'(rd_bi) * 8 + j] +
                  $signed(req_r_data[8*j +: 8]);
            end
            if (rd_bi == 5'd23) begin
              r_ready_r <= 1'b0;
              state     <= S_ACC_DONE;
            end else begin
              rd_bi <= rd_bi + 5'd1;
            end
          end else begin
            r_ready_r <= 1'b1;
          end
        end

        S_ACC_DONE: begin
          den_r   <= den_r + {15'd0, acc_score_r};
          count_r <= count_r + 9'd1;
          acc_done <= 1'b1;
          state   <= S_IDLE;
        end

        S_DIV_REQ: begin
          if (div_req_valid && div_req_ready) begin
            sign_r <= (den_r == 32'd0) ? fsum[d][31] : wnum[d][47];
            state  <= S_DIV_WAIT;
          end
        end

        S_DIV_WAIT: begin
          if (div_rsp_valid) begin
            begin
              logic [63:0] rounded;
              logic [7:0]  res;
              rounded = div_quot + ((64'd2 * div_rem >= div_den_r) ? 64'd1
                                                                  : 64'd0);
              if (!sign_r && rounded > 64'd127) res = 8'sd127;
              else if (sign_r && rounded > 64'd128) res = -8'sd128;
              else res = sign_r ? -$signed(rounded[7:0])
                                : $signed(rounded[7:0]);
              obuf[d] <= res;
            end
            if (d == 8'd191) begin
              wr_bi <= 5'd0;
              state <= S_WR_REQ;
            end else begin
              d     <= d + 8'd1;
              state <= S_DIV_REQ;
            end
          end
        end

        S_WR_REQ: begin
          if (req_valid && req_ready) begin
            wr_bi <= 5'd0;
            state <= S_WR_BEAT;
          end
        end

        S_WR_BEAT: begin
          if (req_w_valid && req_w_ready) begin
            if (wr_bi == 5'd23) state <= S_DIV_DONE;
            else wr_bi <= wr_bi + 5'd1;
          end
        end

        S_DIV_DONE: begin
          // Clear for the next descriptor.
          den_r   <= 32'd0;
          count_r <= 9'd0;
          div_done <= 1'b1;
          state   <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
