// Single-outstanding read/write burst protocol engine.
//
// Inner side is one request plus 64-bit ready/valid/last/strb streams; the
// outer side is the locked external memory interface. The FSM is fixed as
// IDLE -> COMMAND -> DATA -> COMPLETE with a DRAIN_ABORT state. A committed
// burst is always finished legally: writes are completed with zero strobes
// and reads are consumed to last, so abort/error never corrupts memory.
module heatvit_mem_master (
  input  logic        clk,
  input  logic        rst_n,
  // Inner request.
  input  logic        req_valid,
  output logic        req_ready,
  input  logic        req_write,
  input  logic [31:0] req_addr,
  input  logic [31:0] req_bytes,
  // Inner write stream (client to master).
  input  logic        req_w_valid,
  output logic        req_w_ready,
  input  logic [63:0] req_w_data,
  input  logic [7:0]  req_w_strb,
  input  logic        req_w_last,
  // Inner read stream (master to client).
  output logic        req_r_valid,
  input  logic        req_r_ready,
  output logic [63:0] req_r_data,
  output logic        req_r_last,
  // Control/status.
  input  logic        abort,
  output logic        done,
  output logic        protocol_error,
  output logic        abort_done,
  // Locked external memory interface.
  output logic        mem_cmd_valid,
  input  logic        mem_cmd_ready,
  output logic        mem_cmd_write,
  output logic [31:0] mem_cmd_addr,
  output logic [15:0] mem_cmd_len,
  output logic        mem_w_valid,
  input  logic        mem_w_ready,
  output logic [63:0] mem_w_data,
  output logic [7:0]  mem_w_strb,
  output logic        mem_w_last,
  input  logic        mem_r_valid,
  output logic        mem_r_ready,
  input  logic [63:0] mem_r_data,
  input  logic        mem_r_last
);

  localparam int MAX_REQ_BYTES = 65535 * 8;

  typedef enum logic [2:0] {
    ST_IDLE,
    ST_COMMAND,
    ST_DATA,
    ST_COMPLETE,
    ST_DRAIN_ABORT
  } state_t;

  state_t      state;
  logic        write_req;
  logic [31:0] addr_req;
  logic [31:0] bytes_req;
  logic [15:0] cmd_len;
  logic [15:0] beat_count;
  logic [15:0] drain_remaining;
  logic        drain_error;

  logic        expected_last;
  logic        framing_error;

  // A client write framing violation turns the current beat into the first
  // zero-strobed drain beat so malformed client data never reaches memory.
  assign expected_last = (beat_count == cmd_len - 16'd1);
  assign framing_error =
      (state == ST_DATA) && write_req && req_w_valid && mem_w_ready &&
      ((expected_last && !req_w_last) || (!expected_last && req_w_last));

  assign mem_w_valid =
      ((state == ST_DATA) && write_req) ? req_w_valid :
      ((state == ST_DRAIN_ABORT) && write_req && (drain_remaining != 16'd0)) ? 1'b1 :
      1'b0;
  assign req_w_ready = ((state == ST_DATA) && write_req) ? mem_w_ready : 1'b0;
  assign mem_w_data  = ((state == ST_DATA) && write_req && !framing_error) ?
                       req_w_data : 64'h0000000000000000;
  assign mem_w_strb  = ((state == ST_DATA) && write_req && !framing_error) ?
                       req_w_strb : 8'h00;
  assign mem_w_last  =
      ((state == ST_DATA) && write_req) ? expected_last :
      ((state == ST_DRAIN_ABORT) && write_req) ? (drain_remaining == 16'd1) :
      1'b0;

  assign req_r_valid = ((state == ST_DATA) && !write_req) ? mem_r_valid : 1'b0;
  assign req_r_data  = mem_r_data;
  assign req_r_last  = mem_r_last;
  assign mem_r_ready =
      ((state == ST_DATA) && !write_req) ? req_r_ready :
      ((state == ST_DRAIN_ABORT) && !write_req) ? 1'b1 : 1'b0;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state           <= ST_IDLE;
      write_req       <= 1'b0;
      addr_req        <= 32'h00000000;
      bytes_req       <= 32'h00000000;
      cmd_len         <= 16'h0000;
      beat_count      <= 16'h0000;
      drain_remaining <= 16'h0000;
      drain_error     <= 1'b0;
      req_ready       <= 1'b0;
      mem_cmd_valid   <= 1'b0;
      mem_cmd_write   <= 1'b0;
      mem_cmd_addr    <= 32'h00000000;
      mem_cmd_len     <= 16'h0000;
      done            <= 1'b0;
      protocol_error  <= 1'b0;
      abort_done      <= 1'b0;
    end else begin
      done           <= 1'b0;
      protocol_error <= 1'b0;
      abort_done     <= 1'b0;

      case (state)
        ST_IDLE: begin
          req_ready     <= 1'b1;
          mem_cmd_valid <= 1'b0;
          if (req_valid && req_ready) begin
            if ((req_bytes == 32'd0) || (req_bytes > MAX_REQ_BYTES)) begin
              protocol_error <= 1'b1;
            end else begin
              write_req  <= req_write;
              addr_req   <= req_addr;
              bytes_req  <= req_bytes;
              cmd_len    <= 16'((req_bytes + 32'd7) >> 3);
              beat_count <= 16'h0000;
              state      <= ST_COMMAND;
            end
          end
        end

        ST_COMMAND: begin
          req_ready     <= 1'b0;
          mem_cmd_valid <= 1'b1;
          mem_cmd_write <= write_req;
          mem_cmd_addr  <= addr_req;
          mem_cmd_len   <= cmd_len;
          if (abort) begin
            mem_cmd_valid <= 1'b0;
            abort_done    <= 1'b1;
            state         <= ST_IDLE;
          end else if (mem_cmd_valid && mem_cmd_ready) begin
            state <= ST_DATA;
          end
        end

        ST_DATA: begin
          req_ready     <= 1'b0;
          mem_cmd_valid <= 1'b0;
          if (write_req) begin
            if (framing_error) begin
              drain_error     <= 1'b1;
              drain_remaining <= cmd_len - beat_count - 16'd1;
              state           <= ST_DRAIN_ABORT;
            end else if (req_w_valid && mem_w_ready) begin
              if (expected_last) begin
                state <= ST_COMPLETE;
              end else begin
                beat_count <= beat_count + 16'd1;
              end
            end else if (abort) begin
              drain_error     <= 1'b0;
              drain_remaining <= cmd_len - beat_count;
              state           <= ST_DRAIN_ABORT;
            end
          end else begin
            if (mem_r_valid && mem_r_ready) begin
              if (expected_last) begin
                if (!mem_r_last) begin
                  protocol_error <= 1'b1;
                  state          <= ST_IDLE;
                end else begin
                  state <= ST_COMPLETE;
                end
              end else begin
                if (mem_r_last) begin
                  protocol_error <= 1'b1;
                  state          <= ST_IDLE;
                end else begin
                  beat_count <= beat_count + 16'd1;
                end
              end
            end else if (abort) begin
              drain_error <= 1'b0;
              state       <= ST_DRAIN_ABORT;
            end
          end
        end

        ST_COMPLETE: begin
          req_ready <= 1'b0;
          done      <= 1'b1;
          state     <= ST_IDLE;
        end

        ST_DRAIN_ABORT: begin
          req_ready     <= 1'b0;
          mem_cmd_valid <= 1'b0;
          if (write_req) begin
            if (drain_remaining == 16'd0) begin
              if (drain_error) protocol_error <= 1'b1;
              else abort_done <= 1'b1;
              state <= ST_IDLE;
            end else if (mem_w_ready) begin
              if (drain_remaining == 16'd1) begin
                if (drain_error) protocol_error <= 1'b1;
                else abort_done <= 1'b1;
                state <= ST_IDLE;
              end else begin
                drain_remaining <= drain_remaining - 16'd1;
              end
            end
          end else begin
            if (mem_r_valid && mem_r_last) begin
              if (drain_error) protocol_error <= 1'b1;
              else abort_done <= 1'b1;
              state <= ST_IDLE;
            end
          end
        end

        default: state <= ST_IDLE;
      endcase
    end
  end

endmodule
