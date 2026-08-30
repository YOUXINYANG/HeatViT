// Memory-streaming layout engine for the four layout opcodes:
// OP_PATCHIFY, OP_COPY_ADD_POS, OP_QKV_UNPACK and OP_HEAD_CONCAT.
//
// Inputs are resolved absolute region bases plus effective dimensions; every
// memory access is an 8-byte-aligned burst and results are written with full
// strobes because all layout tensors here are whole 64-bit multiples per row
// or per patch. COPY_ADD_POS aligns its two sources with the shared residual
// unit before writing.
//
// P7-1 (2026-08-28): the dynamic byte-addressed staging register array
// (`bbuf`) is replaced by six byte-write-enable SDP RAMs
// (heatvit_sdp_ram):
//   ram_p   1344x64  PATCHIFY 16-row image window (16 x 84 words)
//   ram_q   72x64    QKV_UNPACK row (576 bytes)
//   ram_h   24x64    HEAD_CONCAT row (192 bytes)
//   ram_m/a/o 24x64  COPY_ADD_POS main / aux / output staging
// The PATCHIFY write-out maps each output beat to a single aligned 64-bit
// word: beat b covers input bytes [48*(b/6) + pc*48 + 8*(b%6) .. +7], i.e.
// word (b/6)*84 + pc*6 + (b%6). The wr_byte() mux network is gone. Reads
// are registered (1-cycle latency) with the address issued one cycle ahead,
// preserving the registered input timing of the residual unit; write-out
// beats present whole words read with a one-beat lookahead. Bit-exact
// behaviour is unchanged.
module heatvit_layout_engine
  import heatvit_pkg::*;
(
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  output logic        busy,
  output logic        done,
  output logic        error_valid,
  output logic [7:0]  error_code,
  input  logic [1:0]  op,      // 0 PATCHIFY, 1 COPY_ADD_POS, 2 QKV_UNPACK, 3 HEAD_CONCAT
  input  logic [15:0] m_eff,
  input  logic [15:0] n_eff,
  input  logic [31:0] src0_base,
  input  logic [31:0] src1_base,
  input  logic [31:0] aux_base,
  input  logic [31:0] dst_base,
  input  heatvit_scale_t src0_scale,
  input  heatvit_scale_t src1_scale,
  input  heatvit_scale_t aux_scale,
  input  heatvit_scale_t dst_scale,
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

  typedef enum logic [3:0] {
    S_IDLE,
    S_PLAN,
    S_RD_REQ,
    S_RD_RECV,
    S_WR_REQ,
    S_WR_BEAT,
    S_ELEM,
    S_ELEM_DRAIN,
    S_ELEM_DRAIN2,
    S_DONE
  } state_t;

  state_t state;

  logic [1:0]  op_r;
  logic [15:0] m_r;
  logic [31:0] src0_r;
  logic [31:0] src1_r;
  logic [31:0] aux_r;
  logic [31:0] dst_r;
  heatvit_scale_t src0_scale_r;
  heatvit_scale_t src1_scale_r;
  heatvit_scale_t aux_scale_r;
  heatvit_scale_t dst_scale_r;

  // Macro-loop counters.
  logic [3:0] pr;      // patch row 0..13
  logic [3:0] pc;      // patch col 0..13
  logic [4:0] row;     // image row within patch row 0..16 (16 = done)
  logic [7:0] token;   // token 0..196
  logic [1:0] kind;
  logic [1:0] head;
  logic [7:0] c;       // element 0..191
  logic [8:0] pe;      // present element counter
  logic [1:0] step;    // COPY_ADD_POS sub-step

  logic [10:0] rd_wbase;  // word offset of the receive burst (was rd_off/8)
  logic [31:0] rd_addr;
  logic [15:0] rd_len;
  logic [10:0] rd_bi;
  logic        r_ready_r;

  logic [31:0] wr_addr;
  logic [15:0] wr_len;
  logic [10:0] wr_bi;

  logic        wr_sel;    // COPY_ADD_POS: 0 = main, 1 = aux
  logic [3:0]  p_row;     // PATCHIFY write-out row within window (0..15)
  logic [2:0]  p_seg;     // PATCHIFY write-out segment within row (0..5)

  // Staging RAMs.
  logic         we_p, we_q, we_h, we_m, we_a, we_o;
  logic [10:0]  waddr_p;
  logic [6:0]   waddr_q;
  logic [4:0]   waddr_h, waddr_m, waddr_a, waddr_o;
  logic [63:0]  wdata_p, wdata_q, wdata_h, wdata_m, wdata_a, wdata_o;
  logic [7:0]   wstrb_p, wstrb_q, wstrb_h, wstrb_m, wstrb_a, wstrb_o;
  logic [10:0]  raddr_p;
  logic [6:0]   raddr_q;
  logic [4:0]   raddr_h, raddr_m, raddr_a, raddr_o;
  logic [63:0]  rdata_p, rdata_q, rdata_h, rdata_m, rdata_a, rdata_o;

  heatvit_sdp_ram #(.WIDTH(64), .DEPTH(1344)) u_ram_p (
    .clk(clk), .we(we_p), .waddr(waddr_p), .wdata(wdata_p), .wstrb(wstrb_p),
    .raddr(raddr_p), .rdata(rdata_p)
  );
  heatvit_sdp_ram #(.WIDTH(64), .DEPTH(72)) u_ram_q (
    .clk(clk), .we(we_q), .waddr(waddr_q), .wdata(wdata_q), .wstrb(wstrb_q),
    .raddr(raddr_q), .rdata(rdata_q)
  );
  heatvit_sdp_ram #(.WIDTH(64), .DEPTH(24)) u_ram_h (
    .clk(clk), .we(we_h), .waddr(waddr_h), .wdata(wdata_h), .wstrb(wstrb_h),
    .raddr(raddr_h), .rdata(rdata_h)
  );
  heatvit_sdp_ram #(.WIDTH(64), .DEPTH(24)) u_ram_m (
    .clk(clk), .we(we_m), .waddr(waddr_m), .wdata(wdata_m), .wstrb(wstrb_m),
    .raddr(raddr_m), .rdata(rdata_m)
  );
  heatvit_sdp_ram #(.WIDTH(64), .DEPTH(24)) u_ram_a (
    .clk(clk), .we(we_a), .waddr(waddr_a), .wdata(wdata_a), .wstrb(wstrb_a),
    .raddr(raddr_a), .rdata(rdata_a)
  );
  heatvit_sdp_ram #(.WIDTH(64), .DEPTH(24)) u_ram_o (
    .clk(clk), .we(we_o), .waddr(waddr_o), .wdata(wdata_o), .wstrb(wstrb_o),
    .raddr(raddr_o), .rdata(rdata_o)
  );

  // Residual unit for COPY_ADD_POS.
  logic        res_main_valid;
  logic        res_main_ready;
  heatvit_s8_t res_main_value;
  heatvit_scale_t res_main_scale;
  logic        res_aux_valid;
  logic        res_aux_ready;
  heatvit_s8_t res_aux_value;
  heatvit_scale_t res_aux_scale;
  heatvit_scale_t res_out_scale;
  logic        res_out_valid;
  logic        res_out_ready;
  heatvit_s8_t res_out_value;

  heatvit_residual u_residual (
    .clk            (clk),
    .rst_n          (rst_n),
    .main_valid     (res_main_valid),
    .main_ready     (res_main_ready),
    .main_value     (res_main_value),
    .main_scale_exp (res_main_scale),
    .aux_valid      (res_aux_valid),
    .aux_ready      (res_aux_ready),
    .aux_value      (res_aux_value),
    .aux_scale_exp  (res_aux_scale),
    .out_scale_exp  (res_out_scale),
    .out_valid      (res_out_valid),
    .out_ready      (res_out_ready),
    .out_value      (res_out_value)
  );

  // ------------------------------------------------------------------
  // RAM write ports.
  // ------------------------------------------------------------------
  logic accept_rd;
  assign accept_rd = (state == S_RD_RECV) && req_r_valid && r_ready_r;

  assign we_p = accept_rd && (op_r == 2'd0);
  assign we_q = accept_rd && (op_r == 2'd2);
  assign we_h = accept_rd && (op_r == 2'd3);
  assign we_m = accept_rd && (op_r == 2'd1) && (wr_sel == 1'b0);
  assign we_a = accept_rd && (op_r == 2'd1) && (wr_sel == 1'b1);

  assign waddr_p = rd_wbase + rd_bi;
  assign waddr_q = rd_bi[6:0];
  assign waddr_h = rd_wbase[4:0] + rd_bi[4:0];
  assign waddr_m = rd_bi[4:0];
  assign waddr_a = rd_bi[4:0];
  assign wdata_p = req_r_data;
  assign wdata_q = req_r_data;
  assign wdata_h = req_r_data;
  assign wdata_m = req_r_data;
  assign wdata_a = req_r_data;
  assign wstrb_p = 8'hFF;
  assign wstrb_q = 8'hFF;
  assign wstrb_h = 8'hFF;
  assign wstrb_m = 8'hFF;
  assign wstrb_a = 8'hFF;

  // ram_o: COPY_ADD_POS output staging, written byte-wise by the residual.
  always_comb begin
    we_o    = 1'b0;
    waddr_o = 5'd0;
    wdata_o = 64'd0;
    wstrb_o = 8'd0;
    if (state == S_ELEM && res_out_valid) begin
      we_o    = 1'b1;
      waddr_o = c[7:3];
      wdata_o = {56'd0, res_out_value} << (8*c[2:0]);
      wstrb_o = 8'b1 << c[2:0];
    end
  end

  // ------------------------------------------------------------------
  // RAM read ports (registered reads with one-beat lookahead where the
  // old code read the array combinationally).
  // ------------------------------------------------------------------
  logic [8:0] pe_g;
  logic [8:0] pe_g2;
  logic [3:0] p_row_n;
  logic [2:0] p_seg_n;
  logic [11:0] p_lookahead;

  assign pe_g = (pe > 9'd191) ? 9'd191 : pe;
  assign pe_g2 = (pe >= 9'd191) ? 9'd191 : pe + 9'd1;
  assign p_seg_n = (p_seg == 3'd5) ? 3'd0 : p_seg + 3'd1;
  assign p_row_n = (p_seg == 3'd5) ? p_row + 4'd1 : p_row;
  assign p_lookahead = {8'd0, p_row_n} * 12'd84 + {8'd0, pc} * 12'd6 +
                       {9'd0, p_seg_n};

  // Registered RAM reads: rdata(T) = mem[raddr(T-1)], so the read address
  // must lead the byte index presented this cycle (S_ELEM presents byte pe,
  // hence the address needs word(pe+1)). For the write-out, the address
  // advances ONLY on the accepted beat (req_w_valid && req_w_ready);
  // during backpressure stalls it holds the current word, otherwise the
  // lookahead would over-advance and shift the burst.
  wire w_accept = (state == S_WR_BEAT) && req_w_valid && req_w_ready;
  wire wr_last_beat = (wr_bi == ({5'd0, wr_len[10:3]} - 11'd1));
  assign raddr_m = (state == S_ELEM) ? pe_g2[7:3] : 5'd0;
  assign raddr_a = (state == S_ELEM) ? pe_g2[7:3] : 5'd0;
  assign raddr_o = (state == S_WR_BEAT && op_r == 2'd1)
                   ? (w_accept ? ((wr_bi[4:0] == 5'd23) ? 5'd23 : wr_bi[4:0] + 5'd1)
                               : wr_bi[4:0])
                   : 5'd0;
  assign raddr_h = (state == S_WR_BEAT && op_r == 2'd3)
                   ? (w_accept ? ((wr_bi[4:0] == 5'd23) ? 5'd23 : wr_bi[4:0] + 5'd1)
                               : wr_bi[4:0])
                   : 5'd0;
  assign raddr_q = (state == S_WR_REQ)  ? ({6'd0, kind} * 8'd24 + {6'd0, head} * 8'd8)
                 : (state == S_WR_BEAT) ? ({6'd0, kind} * 8'd24 + {6'd0, head} * 8'd8 +
                                           (w_accept ? wr_bi[6:0] + 7'd1 : wr_bi[6:0]))
                 : 7'd0;
  assign raddr_p = (op_r == 2'd0 && state == S_WR_REQ)
                   ? ({7'd0, pc} * 11'd6)
                 : (op_r == 2'd0 && state == S_WR_BEAT)
                   ? ((w_accept && !wr_last_beat) ? p_lookahead[10:0]
                                                  : ({8'd0, p_row} * 11'd84 +
                                                     {7'd0, pc} * 11'd6 +
                                                     {8'd0, p_seg}))
                 : 11'd0;

  assign req_valid   = (state == S_RD_REQ) || (state == S_WR_REQ);
  assign req_write   = (state == S_WR_REQ);
  assign req_addr    = (state == S_WR_REQ) ? wr_addr : rd_addr;
  assign req_bytes   = (state == S_WR_REQ) ? {16'd0, wr_len} : {16'd0, rd_len};
  assign req_r_ready = r_ready_r;
  assign req_w_valid = (state == S_WR_BEAT);
  assign req_w_data  = (op_r == 2'd0) ? rdata_p :
                       (op_r == 2'd1) ? rdata_o :
                       (op_r == 2'd2) ? rdata_q : rdata_h;
  assign req_w_strb  = 8'hff;
  assign req_w_last  = (wr_bi == ({5'd0, wr_len[10:3]} - 11'd1));

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state   <= S_IDLE;
      busy    <= 1'b0;
      done    <= 1'b0;
      error_valid <= 1'b0;
      error_code  <= 8'd0;
      r_ready_r <= 1'b0;
      rd_wbase  <= 11'd0;
      wr_sel    <= 1'b0;
      p_row     <= 4'd0;
      p_seg     <= 3'd0;
      res_main_valid <= 1'b0;
      res_aux_valid  <= 1'b0;
      res_out_ready  <= 1'b0;
    end else begin
      done        <= 1'b0;
      error_valid <= 1'b0;
      res_main_valid <= 1'b0;
      res_aux_valid  <= 1'b0;
      res_out_ready  <= 1'b0;
      r_ready_r   <= 1'b0;

      case (state)
        S_IDLE: begin
          if (start) begin
            op_r        <= op;
            m_r         <= m_eff;
            src0_r      <= src0_base;
            src1_r      <= src1_base;
            aux_r       <= aux_base;
            dst_r       <= dst_base;
            src0_scale_r <= src0_scale;
            src1_scale_r <= src1_scale;
            aux_scale_r  <= aux_scale;
            dst_scale_r  <= dst_scale;
            pr     <= 4'd0;
            pc     <= 4'd0;
            row    <= 4'd0;
            token  <= 8'd0;
            kind   <= 2'd0;
            head   <= 2'd0;
            c      <= 8'd0;
            step   <= 2'd0;
            busy   <= 1'b1;
            state  <= S_PLAN;
          end
        end

        S_PLAN: begin
          case (op_r)
            2'd0: begin  // PATCHIFY
              if (row != 5'd16) begin
                rd_wbase <= 11'({5'd0, row} * 16'd84);
                rd_addr <= src0_r +
                           32'({4'd0, pr} * 5'd16 + {5'd0, row}) * 32'd672;
                rd_len  <= 16'd672;
                rd_bi   <= 11'd0;
                row     <= row + 4'd1;
                state   <= S_RD_REQ;
              end else begin
                wr_addr <= dst_r +
                           32'({4'd0, pr} * 4'd14 + {4'd0, pc}) * 32'd768;
                wr_len  <= 16'd768;
                wr_bi   <= 11'd0;
                state   <= S_WR_REQ;
              end
            end
            2'd1: begin  // COPY_ADD_POS
              if (step == 2'd0) begin
                rd_wbase <= 11'd0;
                rd_addr <= aux_r;                 // CLS row
                rd_len  <= 16'd192;
                rd_bi   <= 11'd0;
                wr_sel  <= 1'b0;
                step    <= 2'd1;
                state   <= S_RD_REQ;
              end else if (step == 2'd1) begin
                rd_wbase <= 11'd0;
                rd_addr <= src1_r + {8'd0, token} * 16'd192;  // position row
                rd_len  <= 16'd192;
                rd_bi   <= 11'd0;
                wr_sel  <= 1'b1;
                step    <= 2'd2;
                state   <= S_RD_REQ;
              end else if (step == 2'd2) begin
                rd_wbase <= 11'd0;
                if (token == 8'd0) rd_addr <= aux_r;
                else rd_addr <= src0_r + (token - 8'd1) * 16'd192;
                rd_len  <= 16'd192;
                rd_bi   <= 11'd0;
                wr_sel  <= 1'b0;
                step    <= 2'd3;
                state   <= S_RD_REQ;
              end else begin
                c      <= 8'd0;
                res_main_valid <= 1'b1;
                res_aux_valid  <= 1'b1;
                res_out_ready  <= 1'b1;
                res_main_value <= rdata_m[7:0];
                res_aux_value  <= rdata_a[7:0];
                pe <= 9'd1;
                if (token == 8'd0) res_main_scale <= aux_scale_r;
                else res_main_scale <= src0_scale_r;
                res_aux_scale  <= src1_scale_r;
                res_out_scale  <= dst_scale_r;
                state  <= S_ELEM;
              end
            end
            2'd2: begin  // QKV_UNPACK
              if (step == 2'd0) begin
                rd_wbase <= 11'd0;
                rd_addr <= src0_r + 32'({8'd0, token}) * 32'd576;
                rd_len  <= 16'd576;
                rd_bi   <= 11'd0;
                step    <= 2'd1;
                state   <= S_RD_REQ;
              end else begin
                wr_addr <= dst_r +
                           32'(({2'd0, kind} * 2'd3 + {2'd0, head}) *
                               {8'd0, m_r} + {8'd0, token}) * 32'd64;
                wr_len  <= 16'd64;
                wr_bi   <= 11'd0;
                state   <= S_WR_REQ;
              end
            end
            default: begin  // HEAD_CONCAT
              if (head != 2'd3) begin
                rd_wbase <= 11'({4'd0, head} * 16'd8);
                rd_addr <= src0_r + ({2'd0, head} * {8'd0, m_r} + {8'd0, token}) * 16'd64;
                rd_len  <= 16'd64;
                rd_bi   <= 11'd0;
                head    <= head + 2'd1;
                state   <= S_RD_REQ;
              end else begin
                wr_addr <= dst_r + {8'd0, token} * 16'd192;
                wr_len  <= 16'd192;
                wr_bi   <= 11'd0;
                state   <= S_WR_REQ;
              end
            end
          endcase
        end

        S_RD_REQ: begin
          if (req_valid && req_ready) begin
            rd_bi     <= 11'd0;
            r_ready_r <= 1'b1;
            state     <= S_RD_RECV;
          end
        end

        S_RD_RECV: begin
          if (req_r_valid && r_ready_r) begin
            if (rd_bi == ({5'd0, rd_len[10:3]} - 11'd1)) begin
              r_ready_r <= 1'b0;
              state     <= S_PLAN;
            end else begin
              rd_bi <= rd_bi + 11'd1;
            end
          end else begin
            r_ready_r <= 1'b1;
          end
        end

        S_WR_REQ: begin
          if (req_valid && req_ready) begin
            wr_bi <= 11'd0;
            p_row <= 4'd0;
            p_seg <= 3'd0;
            state <= S_WR_BEAT;
          end
        end

        S_WR_BEAT: begin
          if (req_w_valid && req_w_ready) begin
            if (op_r == 2'd0) begin
              if (p_seg == 3'd5) begin
                p_seg <= 3'd0;
                p_row <= p_row + 4'd1;
              end else begin
                p_seg <= p_seg + 3'd1;
              end
            end
            if (wr_bi == ({5'd0, wr_len[10:3]} - 11'd1)) begin
              case (op_r)
                2'd0: begin
                  if (pc == 4'd13) begin
                    pc  <= 4'd0;
                    row <= 5'd0;
                    if (pr == 4'd13) state <= S_DONE;
                    else begin pr <= pr + 4'd1; state <= S_PLAN; end
                  end else begin
                    pc    <= pc + 4'd1;
                    state <= S_PLAN;
                  end
                end
                2'd1: begin
                  if (token == m_r - 16'd1) state <= S_DONE;
                  else begin
                    token <= token + 8'd1;
                    step  <= 2'd0;
                    state <= S_PLAN;
                  end
                end
                2'd2: begin
                  if (head == 2'd2) begin
                    if (kind == 2'd2) begin
                      kind  <= 2'd0;
                      head  <= 2'd0;
                      step  <= 2'd0;
                      if (token == m_r - 16'd1) state <= S_DONE;
                      else begin token <= token + 8'd1; state <= S_PLAN; end
                    end else begin
                      kind  <= kind + 2'd1;
                      head  <= 2'd0;
                      state <= S_PLAN;
                    end
                  end else begin
                    head  <= head + 2'd1;
                    state <= S_PLAN;
                  end
                end
                default: begin
                  if (token == m_r - 16'd1) state <= S_DONE;
                  else begin
                    token <= token + 8'd1;
                    head  <= 2'd0;
                    state <= S_PLAN;
                  end
                end
              endcase
            end else begin
              wr_bi <= wr_bi + 11'd1;
            end
          end
        end

        S_ELEM: begin
          res_main_valid <= 1'b1;
          res_aux_valid  <= 1'b1;
          res_out_ready  <= 1'b1;
          // The residual has registered inputs and outputs: present element
          // pe while capturing element c (capture lags presentation by two).
          res_main_value <= rdata_m[8*pe_g[2:0] +: 8];
          res_aux_value  <= rdata_a[8*pe_g[2:0] +: 8];
          pe <= pe + 9'd1;
          if (token == 8'd0) res_main_scale <= aux_scale_r;
          else res_main_scale <= src0_scale_r;
          res_aux_scale  <= src1_scale_r;
          res_out_scale  <= dst_scale_r;
          if (res_out_valid) begin
            if (c == 8'd191) begin
              c      <= 8'd0;
              state  <= S_ELEM_DRAIN;
            end else begin
              c <= c + 8'd1;
            end
          end
        end

        S_ELEM_DRAIN: begin
          // P7-5: the residual now has a two-stage pipeline, so its last
          // output trails the last presented element by one extra cycle.
          // The drain keeps out_ready asserted for two cycles so the
          // trailing output is consumed before the write burst starts.
          res_out_ready <= 1'b1;
          state <= S_ELEM_DRAIN2;
        end

        S_ELEM_DRAIN2: begin
          res_out_ready <= 1'b1;
          wr_addr <= dst_r + {8'd0, token} * 16'd192;
          wr_len  <= 16'd192;
          wr_bi   <= 11'd0;
          state   <= S_WR_REQ;
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
