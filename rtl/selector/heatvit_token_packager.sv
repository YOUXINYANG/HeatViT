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
//
// P7-2 (2026-08-28): the three dynamic byte-addressed register arrays
// (`wnum`, `fsum`, `obuf`) are replaced by byte-write-enable SDP RAMs:
//   ram_wnum 192x64  48-bit signed weighted numerator per channel
//   ram_fsum 192x64  32-bit signed unweighted fallback sum per channel
//   ram_obuf  24x64  package output byte buffer (one word per beat)
// The receive burst is serialised: each accepted 64-bit beat is latched and
// its 8 lanes are folded into ram_wnum/ram_fsum one channel per cycle via a
// read-then-write pipeline (9 cycles/beat, READ_FIRST guarantees the read
// sees the pre-update value because raddr leads waddr by one). The divide
// loop reads one channel per cycle with a one-cycle lookahead address that
// advances only when div_req is accepted (holding the presentation stable
// under arbiter backpressure), and clears the consumed wnum/fsum entry in
// the response cycle, so the accumulators are zeroed by the time the
// package write-out begins. The write-out reads ram_obuf as whole 64-bit
// words with a one-beat lookahead. Bit-exact behaviour is unchanged.
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
    S_ACC_ELEM,
    S_ACC_DONE,
    S_DIV_PRIME,
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

  logic [31:0] den_r;
  logic [8:0]  count_r;

  logic [4:0]  rd_bi;
  logic        r_ready_r;
  logic [7:0]  d;            // channel 0..191
  logic        sign_r;
  logic [63:0] div_den_r;
  logic [4:0]  wr_bi;

  // Serialised accumulate lane state.
  logic [63:0] acc_beat;     // latched receive beat
  logic [3:0]  acc_step;     // 0 = prime, 1..8 = write lane (step-1)
  
  // Staging RAMs.
  logic        we_wnum, we_fsum, we_obuf;
  logic [7:0]  waddr_wnum, waddr_fsum;
  logic [4:0]  waddr_obuf;
  logic [63:0] wdata_wnum, wdata_fsum, wdata_obuf;
  logic [7:0]  wstrb_obuf;
  logic [7:0]  raddr_wnum, raddr_fsum;
  logic [4:0]  raddr_obuf;
  logic [63:0] rdata_wnum, rdata_fsum, rdata_obuf;

  heatvit_sdp_ram #(.WIDTH(64), .DEPTH(192)) u_ram_wnum (
    .clk(clk), .we(we_wnum), .waddr(waddr_wnum), .wdata(wdata_wnum),
    .wstrb(8'hFF), .raddr(raddr_wnum), .rdata(rdata_wnum)
  );
  heatvit_sdp_ram #(.WIDTH(64), .DEPTH(192)) u_ram_fsum (
    .clk(clk), .we(we_fsum), .waddr(waddr_fsum), .wdata(wdata_fsum),
    .wstrb(8'hFF), .raddr(raddr_fsum), .rdata(rdata_fsum)
  );
  heatvit_sdp_ram #(.WIDTH(64), .DEPTH(24)) u_ram_obuf (
    .clk(clk), .we(we_obuf), .waddr(waddr_obuf), .wdata(wdata_obuf),
    .wstrb(wstrb_obuf), .raddr(raddr_obuf), .rdata(rdata_obuf)
  );

  // ------------------------------------------------------------------
  // Accumulate lane arithmetic (S_ACC_ELEM write cycles).
  // ------------------------------------------------------------------
  logic [7:0]  acc_ch_w;     // channel being written this cycle
  logic signed [47:0] wnum_next;
  logic signed [31:0] fsum_next;

  always_comb begin
    logic signed [7:0] b;
    if (acc_step >= 4'd1) begin
      acc_ch_w = 8'(int'(rd_bi) * 8 + int'(acc_step) - 1);
      b        = acc_beat[8*(acc_step - 4'd1) +: 8];
    end else begin
      acc_ch_w = 8'd0;
      b        = 8'sd0;
    end
    wnum_next = $signed(rdata_wnum[47:0]) +
                $signed({1'b0, acc_score_r}) * b;
    fsum_next = $signed(rdata_fsum[31:0]) + b;
  end

  // ------------------------------------------------------------------
  // RAM write ports.
  // ------------------------------------------------------------------
  assign we_wnum = ((state == S_ACC_ELEM) && (acc_step >= 4'd1)) ||
                   ((state == S_DIV_WAIT) && div_rsp_valid);
  assign we_fsum = ((state == S_ACC_ELEM) && (acc_step >= 4'd1)) ||
                   ((state == S_DIV_WAIT) && div_rsp_valid);

  assign waddr_wnum = (state == S_ACC_ELEM) ? acc_ch_w : d;
  assign waddr_fsum = (state == S_ACC_ELEM) ? acc_ch_w : d;

  assign wdata_wnum = (state == S_ACC_ELEM)
                      ? {{16{wnum_next[47]}}, wnum_next} : 64'd0;
  assign wdata_fsum = (state == S_ACC_ELEM)
                      ? {{32{fsum_next[31]}}, fsum_next} : 64'd0;

  // obuf: byte writes during the divide response.
  logic [7:0] obuf_byte;
  always_comb begin
    logic [63:0] rounded;
    logic [7:0]  res;
    rounded = div_quot + ((64'd2 * div_rem >= div_den_r) ? 64'd1 : 64'd0);
    if (!sign_r && rounded > 64'd127)      res = 8'sd127;
    else if (sign_r && rounded > 64'd128)  res = -8'sd128;
    else res = sign_r ? -$signed(rounded[7:0]) : $signed(rounded[7:0]);
    obuf_byte = res;
  end

  assign we_obuf    = (state == S_DIV_WAIT) && div_rsp_valid;
  assign waddr_obuf = d[7:3];
  assign wdata_obuf = {56'd0, obuf_byte} << (8 * int'(d[2:0]));
  assign wstrb_obuf = 8'b1 << d[2:0];

  // ------------------------------------------------------------------
  // RAM read ports (registered reads; addresses issued one cycle ahead).
  // ------------------------------------------------------------------
  always_comb begin
    raddr_wnum = 8'd0;
    if (state == S_ACC_ELEM) begin
      raddr_wnum = (int'(rd_bi) * 8 + int'(acc_step) > 191)
                   ? 8'd191 : 8'(int'(rd_bi) * 8 + int'(acc_step));
    end else if (state == S_DIV_PRIME) begin
      raddr_wnum = 8'd0;
    end else if (state == S_DIV_REQ) begin
      if (div_req_valid && div_req_ready)
        raddr_wnum = (d == 8'd191) ? 8'd191 : d + 8'd1;
      else
        raddr_wnum = d;
    end else if (state == S_DIV_WAIT) begin
      raddr_wnum = (d == 8'd191) ? 8'd191 : d + 8'd1;
    end
  end
  assign raddr_fsum = raddr_wnum;

  wire w_accept = (state == S_WR_BEAT) && req_w_valid && req_w_ready;
  assign raddr_obuf = (state == S_WR_BEAT)
                      ? (w_accept ? ((wr_bi == 5'd23) ? 5'd23 : wr_bi + 5'd1)
                                  : wr_bi)
                      : 5'd0;

  assign req_valid   = (state == S_ACC_RD_REQ) || (state == S_WR_REQ);
  assign req_write   = (state == S_WR_REQ);
  assign req_addr    = (state == S_WR_REQ) ? div_addr_r : acc_addr_r;
  assign req_bytes   = 32'd192;
  assign req_r_ready = r_ready_r;
  assign req_w_valid = (state == S_WR_BEAT);
  assign req_w_data  = rdata_obuf;
  assign req_w_strb  = 8'hff;
  assign req_w_last  = (wr_bi == 5'd23);

  assign participants   = count_r;
  assign acc_busy       = (state == S_ACC_RD_REQ) || (state == S_ACC_RD_RECV) ||
                          (state == S_ACC_ELEM);
  assign div_busy       = (state == S_DIV_PRIME) || (state == S_DIV_REQ) ||
                          (state == S_DIV_WAIT) ||
                          (state == S_WR_REQ) || (state == S_WR_BEAT);
  assign div_req_valid  = (state == S_DIV_REQ);

  always_comb begin
    logic [47:0] num_c;
    div_num = 64'd0;
    div_den = 64'd1;
    if (state == S_DIV_REQ) begin
      if (den_r == 32'd0)
        num_c = {{16{rdata_fsum[31]}}, rdata_fsum[31:0]};
      else
        num_c = rdata_wnum[47:0];
      div_num = {16'd0, (num_c[47] ? -num_c : num_c)};
      div_den = {32'd0, ((den_r == 32'd0) ? {23'd0, count_r} : den_r)};
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
      acc_beat    <= 64'd0;
      acc_step    <= 4'd0;
      warn_package_den_zero <= 1'b0;
    end else begin
      acc_done    <= 1'b0;
      div_done    <= 1'b0;
      acc_error_valid <= 1'b0;
      r_ready_r   <= 1'b0;
      warn_package_den_zero <= 1'b0;
      if (reset_acc) begin
        den_r   <= 32'd0;
        count_r <= 9'd0;
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
            state <= S_DIV_PRIME;
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
            acc_beat <= req_r_data;
            acc_step <= 4'd0;
            r_ready_r <= 1'b0;
            state     <= S_ACC_ELEM;
          end else begin
            r_ready_r <= 1'b1;
          end
        end

        S_ACC_ELEM: begin
          if (acc_step == 4'd8) begin
            if (rd_bi == 5'd23) begin
              state <= S_ACC_DONE;
            end else begin
              rd_bi     <= rd_bi + 5'd1;
              acc_step  <= 4'd0;
              r_ready_r <= 1'b1;
              state     <= S_ACC_RD_RECV;
            end
          end else begin
            acc_step <= acc_step + 4'd1;
          end
        end

        S_ACC_DONE: begin
          den_r   <= den_r + {15'd0, acc_score_r};
          count_r <= count_r + 9'd1;
          acc_done <= 1'b1;
          state   <= S_IDLE;
        end

        S_DIV_PRIME: begin
          state <= S_DIV_REQ;
        end

        S_DIV_REQ: begin
          if (div_req_valid && div_req_ready) begin
            sign_r <= (den_r == 32'd0) ? rdata_fsum[31] : rdata_wnum[47];
            state  <= S_DIV_WAIT;
          end
        end

        S_DIV_WAIT: begin
          if (div_rsp_valid) begin
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
          // Clear for the next descriptor. wnum/fsum were already cleared
          // channel-by-channel during the divide loop.
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
