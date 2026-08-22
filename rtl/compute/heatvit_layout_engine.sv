// Memory-streaming layout engine for the four layout opcodes:
// OP_PATCHIFY, OP_COPY_ADD_POS, OP_QKV_UNPACK and OP_HEAD_CONCAT.
//
// Inputs are resolved absolute region bases plus effective dimensions; every
// memory access is an 8-byte-aligned burst and results are written with full
// strobes because all layout tensors here are whole 64-bit multiples per row
// or per patch. COPY_ADD_POS aligns its two sources with the shared residual
// unit before writing.
module heatvit_layout_engine
  import heatvit_pkg::*;
#(
  parameter int BUF_BYTES = 10752
)(
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

  logic [15:0] rd_off;
  logic [31:0] rd_addr;
  logic [15:0] rd_len;
  logic [10:0] rd_bi;
  logic [63:0] rd_beat;
  logic        r_ready_r;

  logic [31:0] wr_addr;
  logic [15:0] wr_len;
  logic [10:0] wr_bi;
  logic [63:0] wr_data_c;

  logic [7:0] bbuf [0:BUF_BYTES-1];

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

  integer bi;
  initial begin
    for (bi = 0; bi < BUF_BYTES; bi++) bbuf[bi] = 8'h00;
  end

  function automatic logic [7:0] wr_byte(input int idx);
    int in_row;
    int rem;
    int in_col;
    int ch;
    case (op_r)
      2'd0: begin
        in_row = idx / 48;
        rem    = idx % 48;
        in_col = rem / 3;
        ch     = rem % 3;
        return bbuf[in_row * 672 + int'(pc) * 48 + in_col * 3 + ch];
      end
      2'd1: return bbuf[384 + idx];
      2'd2: return bbuf[int'(kind) * 192 + int'(head) * 64 + idx];
      default: return bbuf[idx];
    endcase
  endfunction

  always_comb begin
    wr_data_c = 64'h0000000000000000;
    if (state == S_WR_BEAT) begin
      for (int j = 0; j < 8; j++)
        wr_data_c[8*j +: 8] = wr_byte(int'(wr_bi) * 8 + j);
    end
  end

  assign req_valid   = (state == S_RD_REQ) || (state == S_WR_REQ);
  assign req_write   = (state == S_WR_REQ);
  assign req_addr    = (state == S_WR_REQ) ? wr_addr : rd_addr;
  assign req_bytes   = (state == S_WR_REQ) ? {16'd0, wr_len} : {16'd0, rd_len};
  assign req_r_ready = r_ready_r;
  assign req_w_valid = (state == S_WR_BEAT);
  assign req_w_data  = wr_data_c;
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
                rd_off  <= {4'd0, row} * 16'd672;
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
                rd_off  <= 16'd0;
                rd_addr <= aux_r;                 // CLS row
                rd_len  <= 16'd192;
                rd_bi   <= 11'd0;
                step    <= 2'd1;
                state   <= S_RD_REQ;
              end else if (step == 2'd1) begin
                rd_off  <= 16'd192;
                rd_addr <= src1_r + {8'd0, token} * 16'd192;  // position row
                rd_len  <= 16'd192;
                rd_bi   <= 11'd0;
                step    <= 2'd2;
                state   <= S_RD_REQ;
              end else if (step == 2'd2) begin
                rd_off  <= 16'd0;
                if (token == 8'd0) rd_addr <= aux_r;
                else rd_addr <= src0_r + (token - 8'd1) * 16'd192;
                rd_len  <= 16'd192;
                rd_bi   <= 11'd0;
                step    <= 2'd3;
                state   <= S_RD_REQ;
              end else begin
                c      <= 8'd0;
                res_main_valid <= 1'b1;
                res_aux_valid  <= 1'b1;
                res_out_ready  <= 1'b1;
                res_main_value <= bbuf[0];
                res_aux_value  <= bbuf[192];
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
                rd_off  <= 16'd0;
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
                rd_off  <= {4'd0, head} * 16'd64;
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
            for (int j = 0; j < 8; j++)
              bbuf[rd_off + rd_bi * 8 + j] <= req_r_data[8*j +: 8];
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
            state <= S_WR_BEAT;
          end
        end

        S_WR_BEAT: begin
          if (req_w_valid && req_w_ready) begin
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
          res_main_value <= bbuf[pe[7:0]];
          res_aux_value  <= bbuf[192 + pe[7:0]];
          pe <= pe + 9'd1;
          if (token == 8'd0) res_main_scale <= aux_scale_r;
          else res_main_scale <= src0_scale_r;
          res_aux_scale  <= src1_scale_r;
          res_out_scale  <= dst_scale_r;
          if (res_out_valid) begin
            bbuf[384 + c] <= res_out_value;
            if (c == 8'd191) begin
              c      <= 8'd0;
              state  <= S_ELEM_DRAIN;
            end else begin
              c <= c + 8'd1;
            end
          end
        end

        S_ELEM_DRAIN: begin
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
