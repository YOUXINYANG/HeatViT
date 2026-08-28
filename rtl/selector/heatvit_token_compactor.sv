// Stable token compactor for OP_SELECTOR_FINALIZE (Phase 4 Task 4).
//
// Copies one [192]-byte input token row to a destination slot:
//
//   dst[slot][0:192] = src[0:192]
//
// The selector finalize orchestrator drives one copy for the CLS (slot 0)
// and one for every kept normal token (slots 1.. in input order), which
// implements the stable compaction. Each copy is one 192-byte read burst
// followed by one 192-byte write burst.
//
// P7-2 (2026-08-28): the dynamic byte-addressed staging register array
// (`rbuf`) is replaced by a single 24x64 byte-write-enable SDP RAM
// (heatvit_sdp_ram). The receive burst writes whole 64-bit words at
// waddr=rd_bi; the write-out burst reads whole 64-bit words with a
// one-beat lookahead (rdata(T) = mem[raddr(T-1)]), matching the P7-1
// write-out idiom. Bit-exact behaviour is unchanged.
module heatvit_token_compactor
  import heatvit_pkg::*;
(
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  output logic        busy,
  output logic        done,
  output logic        error_valid,
  output logic [7:0]  error_code,
  input  logic [31:0] src_addr,  // source token row
  input  logic [31:0] dst_base,  // output activation base
  input  logic [7:0]  slot,      // destination row index
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
  input  logic        req_r_last
);

  typedef enum logic [2:0] {
    S_IDLE,
    S_RD_REQ,
    S_RD_RECV,
    S_WR_REQ,
    S_WR_BEAT,
    S_DONE
  } state_t;

  state_t state;

  logic [31:0] src_r;
  logic [31:0] dst_r;
  logic [7:0]  slot_r;
  logic [4:0]  rd_bi;
  logic        r_ready_r;
  logic [4:0]  wr_bi;

  // Staging RAM: one 64-bit word per burst beat.
  logic        we_buf;
  logic [4:0]  waddr_buf;
  logic [63:0] wdata_buf;
  logic [7:0]  wstrb_buf;
  logic [4:0]  raddr_buf;
  logic [63:0] rdata_buf;

  heatvit_sdp_ram #(.WIDTH(64), .DEPTH(24)) u_ram_buf (
    .clk(clk), .we(we_buf), .waddr(waddr_buf), .wdata(wdata_buf),
    .wstrb(wstrb_buf), .raddr(raddr_buf), .rdata(rdata_buf)
  );

  // RAM write port: word writes during the receive burst.
  wire accept_rd = (state == S_RD_RECV) && req_r_valid && r_ready_r;
  assign we_buf    = accept_rd;
  assign waddr_buf = rd_bi;
  assign wdata_buf = req_r_data;
  assign wstrb_buf = 8'hFF;

  // RAM read port: registered read with one-beat lookahead. The address
  // advances ONLY on the accepted beat (req_w_valid && req_w_ready); during
  // backpressure stalls it holds the current word, otherwise the lookahead
  // would over-advance and shift the burst.
  wire w_accept = (state == S_WR_BEAT) && req_w_valid && req_w_ready;
  assign raddr_buf = (state == S_WR_BEAT)
                     ? (w_accept ? ((wr_bi == 5'd23) ? 5'd23 : wr_bi + 5'd1)
                                 : wr_bi)
                     : 5'd0;

  assign req_valid   = (state == S_RD_REQ) || (state == S_WR_REQ);
  assign req_write   = (state == S_WR_REQ);
  assign req_addr    = (state == S_WR_REQ)
                       ? dst_r + 32'({24'd0, slot_r}) * 32'd192 : src_r;
  assign req_bytes   = 32'd192;
  assign req_r_ready = r_ready_r;
  assign req_w_valid = (state == S_WR_BEAT);
  assign req_w_data  = rdata_buf;
  assign req_w_strb  = 8'hff;
  assign req_w_last  = (wr_bi == 5'd23);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state       <= S_IDLE;
      busy        <= 1'b0;
      done        <= 1'b0;
      error_valid <= 1'b0;
      error_code  <= 8'd0;
      r_ready_r   <= 1'b0;
      rd_bi       <= 5'd0;
      wr_bi       <= 5'd0;
    end else begin
      done        <= 1'b0;
      error_valid <= 1'b0;
      r_ready_r   <= 1'b0;

      case (state)
        S_IDLE: begin
          if (start) begin
            src_r  <= src_addr;
            dst_r  <= dst_base;
            slot_r <= slot;
            busy   <= 1'b1;
            rd_bi  <= 5'd0;
            state  <= S_RD_REQ;
          end
        end

        S_RD_REQ: begin
          if (req_valid && req_ready) begin
            rd_bi     <= 5'd0;
            r_ready_r <= 1'b1;
            state     <= S_RD_RECV;
          end
        end

        S_RD_RECV: begin
          if (req_r_valid && r_ready_r) begin
            if (rd_bi == 5'd23) begin
              r_ready_r <= 1'b0;
              wr_bi     <= 5'd0;
              state     <= S_WR_REQ;
            end else begin
              rd_bi <= rd_bi + 5'd1;
            end
          end else begin
            r_ready_r <= 1'b1;
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
            if (wr_bi == 5'd23) state <= S_DONE;
            else wr_bi <= wr_bi + 5'd1;
          end
        end

        S_DONE: begin
          busy  <= 1'b0;
          done  <= 1'b1;
          state <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
