`timescale 1ns / 1ps

// Byte-accurate behavioral model of the locked 64-bit external memory.
//
// One to four byte segments are loaded little-endian from .mem files (one
// 64-bit word per line). The model independently rechecks alignment, burst
// bounds, zero length and read-length expectations, and injects deterministic
// pseudo-random backpressure on the command/read/write channels selected by
// stall_mask: bit 0 = command, bit 1 = read data, bit 2 = write data.
//
// Protocol violations (early/late mem_w_last, out-of-bounds access) end the
// simulation with $fatal so error-injection testbenches can observe them.
module behavioral_memory #(
  parameter int          SEG_COUNT   = 1,
  parameter string       SEG0_FILE   = "",
  parameter logic [31:0] SEG0_BASE   = 32'h00000000,
  parameter int          SEG0_BYTES  = 0,
  parameter string       SEG1_FILE   = "",
  parameter logic [31:0] SEG1_BASE   = 32'h00000000,
  parameter int          SEG1_BYTES  = 0,
  parameter string       SEG2_FILE   = "",
  parameter logic [31:0] SEG2_BASE   = 32'h00000000,
  parameter int          SEG2_BYTES  = 0,
  parameter string       SEG3_FILE   = "",
  parameter logic [31:0] SEG3_BASE   = 32'h00000000,
  parameter int          SEG3_BYTES  = 0,
  parameter logic [15:0] LFSR_INIT   = 16'hACE1,
  parameter logic [15:0] EXPECTED_CMD_LEN = 16'h0000
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic [15:0] stall_mask,
  // Command channel.
  input  logic        mem_cmd_valid,
  output logic        mem_cmd_ready,
  input  logic        mem_cmd_write,
  input  logic [31:0] mem_cmd_addr,
  input  logic [15:0] mem_cmd_len,
  // Write channel.
  input  logic        mem_w_valid,
  output logic        mem_w_ready,
  input  logic [63:0] mem_w_data,
  input  logic [7:0]  mem_w_strb,
  input  logic        mem_w_last,
  // Read channel.
  output logic        mem_r_valid,
  input  logic        mem_r_ready,
  output logic [63:0] mem_r_data,
  output logic        mem_r_last,
  // One-cycle observation pulse on command acceptance (trace checking).
  output logic        obs_cmd_valid,
  output logic        obs_cmd_write,
  output logic [31:0] obs_cmd_addr,
  output logic [15:0] obs_cmd_len,
  // Testbench backdoor byte read, used to verify stored results.
  input  logic        dbg_valid,
  output logic        dbg_ready,
  input  logic [31:0] dbg_addr,
  output logic [7:0]  dbg_data,
  // Testbench backdoor byte write, used to initialize memory images.
  input  logic        dbg_w_valid,
  input  logic [31:0] dbg_w_addr,
  input  logic [7:0]  dbg_w_data,
  // Runtime segment reload: one handshake loads a .mem image into a segment.
  input  logic        load_valid,
  output logic        load_ready,
  input  logic [1:0]  load_seg,
  input  logic [31:0] load_bytes,
  input  string       load_file
);

  localparam int TOTAL_BYTES =
      SEG0_BYTES + SEG1_BYTES + SEG2_BYTES + SEG3_BYTES;
  localparam int MEM_DEPTH = (TOTAL_BYTES > 0) ? TOTAL_BYTES : 1;
  localparam int TMP_WORDS =
      (((TOTAL_BYTES + 7) / 8) > 0) ? ((TOTAL_BYTES + 7) / 8) : 1;

  logic [7:0]  mem [0:MEM_DEPTH-1];
  logic [63:0] tmp [0:TMP_WORDS-1];

  typedef enum logic [1:0] {
    ST_IDLE,
    ST_WRITE,
    ST_READ
  } state_t;

  state_t      state;
  logic        cmd_write_reg;
  logic [31:0] cmd_addr_reg;
  logic [15:0] cmd_len_reg;
  logic [31:0] cmd_flat_off;
  logic [15:0] beat_count;
  logic [15:0] lfsr;
  int          chk_seg;
  int          chk_off;
  logic [31:0] chk_base;

  function automatic logic [31:0] seg_base(input int s);
    case (s)
      0: return SEG0_BASE;
      1: return SEG1_BASE;
      2: return SEG2_BASE;
      3: return SEG3_BASE;
      default: return 32'h00000000;
    endcase
  endfunction

  function automatic int seg_bytes(input int s);
    case (s)
      0: return SEG0_BYTES;
      1: return SEG1_BYTES;
      2: return SEG2_BYTES;
      3: return SEG3_BYTES;
      default: return 0;
    endcase
  endfunction

  function automatic string seg_file(input int s);
    case (s)
      0: return SEG0_FILE;
      1: return SEG1_FILE;
      2: return SEG2_FILE;
      3: return SEG3_FILE;
      default: return "";
    endcase
  endfunction

  function automatic int seg_flat_offset(input int s);
    case (s)
      0: return 0;
      1: return SEG0_BYTES;
      2: return SEG0_BYTES + SEG1_BYTES;
      3: return SEG0_BYTES + SEG1_BYTES + SEG2_BYTES;
      default: return 0;
    endcase
  endfunction

  function automatic int locate_seg(
    input logic [31:0] addr,
    output int         seg,
    output int         flat_off
  );
    for (int s = 0; s < SEG_COUNT; s++) begin
      if ((addr >= seg_base(s)) &&
          ({32'd0, addr} < {32'd0, seg_base(s)} + 64'(seg_bytes(s)))) begin
        seg      = s;
        flat_off = seg_flat_offset(s) + int'(addr - seg_base(s));
        return 1;
      end
    end
    seg      = 0;
    flat_off = 0;
    return 0;
  endfunction

  function automatic logic [63:0] read_word(input logic [31:0] addr);
    int seg;
    int off;
    if (!locate_seg(addr, seg, off)) begin
      $fatal(1, "behavioral_memory: read outside configured segments");
      return 64'h0000000000000000;
    end
    return {mem[off + 7], mem[off + 6], mem[off + 5], mem[off + 4],
            mem[off + 3], mem[off + 2], mem[off + 1], mem[off + 0]};
  endfunction

  function automatic logic [7:0] mem_byte(input logic [31:0] addr);
    int seg;
    int off;
    if ($isunknown(addr)) return 8'h00;
    if (!locate_seg(addr, seg, off)) begin
      $fatal(1, "behavioral_memory: debug read outside configured segments");
      return 8'h00;
    end
    return mem[off];
  endfunction

  assign load_ready = 1'b1;

  initial begin
    if (SEG_COUNT < 1 || SEG_COUNT > 4)
      $fatal(1, "behavioral_memory: SEG_COUNT must be between 1 and 4");
    for (int s = 0; s < SEG_COUNT; s++) begin
      chk_base = seg_base(s);
      if (chk_base[2:0] != 3'd0)
        $fatal(1, "behavioral_memory: segment base must be 8-byte aligned");
      if ((seg_bytes(s) % 8) != 0)
        $fatal(1, "behavioral_memory: segment byte count must be 8-byte aligned");
    end
    for (int i = 0; i < MEM_DEPTH; i++) mem[i] = 8'h00;
    for (int s = 0; s < SEG_COUNT; s++) begin
      if (seg_file(s) != "") begin
        $readmemh(seg_file(s), tmp);
        for (int w = 0; w < seg_bytes(s) / 8; w++) begin
          for (int j = 0; j < 8; j++)
            mem[seg_flat_offset(s) + w * 8 + j] = tmp[w][8*j +: 8];
        end
      end
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state          <= ST_IDLE;
      cmd_write_reg  <= 1'b0;
      cmd_addr_reg   <= 32'h00000000;
      cmd_len_reg    <= 16'h0000;
      cmd_flat_off   <= 32'h00000000;
      beat_count     <= 16'h0000;
      lfsr           <= LFSR_INIT;
      mem_cmd_ready  <= 1'b0;
      mem_w_ready    <= 1'b0;
      mem_r_valid    <= 1'b0;
      mem_r_data     <= 64'h0000000000000000;
      mem_r_last     <= 1'b0;
      obs_cmd_valid  <= 1'b0;
      obs_cmd_write  <= 1'b0;
      obs_cmd_addr   <= 32'h00000000;
      obs_cmd_len    <= 16'h0000;
      dbg_ready      <= 1'b0;
      dbg_data       <= 8'h00;
    end else begin
      lfsr          <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};
      obs_cmd_valid <= 1'b0;
      dbg_ready     <= 1'b0;

      if (dbg_valid) begin
        dbg_data  <= mem_byte(dbg_addr);
        dbg_ready <= 1'b1;
      end

      if (rst_n && dbg_w_valid && !$isunknown(dbg_w_addr)) begin
        if (locate_seg(dbg_w_addr, chk_seg, chk_off))
          mem[chk_off] <= dbg_w_data;
        else
          $fatal(1, "behavioral_memory: debug write outside configured segments");
      end

      if (load_valid) begin
        if (load_seg >= SEG_COUNT)
          $fatal(1, "behavioral_memory: reload segment out of range");
        if ((load_bytes[2:0] != 3'd0) ||
            (load_bytes > seg_bytes(load_seg)))
          $fatal(1, "behavioral_memory: reload byte count invalid");
        $readmemh(load_file, tmp);
        for (int i = 0; i < seg_bytes(load_seg); i++)
          mem[seg_flat_offset(load_seg) + i] <= 8'h00;
        for (int w = 0; w < load_bytes / 8; w++) begin
          for (int j = 0; j < 8; j++)
            mem[seg_flat_offset(load_seg) + w * 8 + j] <= tmp[w][8*j +: 8];
        end
      end

      case (state)
        ST_IDLE: begin
          mem_cmd_ready <= stall_mask[0] ? lfsr[0] : 1'b1;
          mem_w_ready   <= 1'b0;
          mem_r_valid   <= 1'b0;
          if (mem_cmd_valid && mem_cmd_ready) begin
            // Independent two-layer bounds/length/alignment recheck.
            if (mem_cmd_len == 16'd0)
              $fatal(1, "behavioral_memory: zero-length command");
            if (mem_cmd_addr[2:0] != 3'd0)
              $fatal(1, "behavioral_memory: unaligned command address");
            if ((EXPECTED_CMD_LEN != 16'd0) && !mem_cmd_write &&
                (mem_cmd_len != EXPECTED_CMD_LEN))
              $fatal(1, "behavioral_memory: read length mismatch");
            if (!locate_seg(mem_cmd_addr, chk_seg, chk_off))
              $fatal(1, "behavioral_memory: command outside configured segments");
            if ((64'(mem_cmd_addr) + (64'(mem_cmd_len) << 3) - 64'd1) >=
                (64'(seg_base(chk_seg)) + 64'(seg_bytes(chk_seg))))
              $fatal(1, "behavioral_memory: burst exceeds configured segment");
            cmd_flat_off <= chk_off;
            cmd_write_reg <= mem_cmd_write;
            cmd_addr_reg  <= mem_cmd_addr;
            cmd_len_reg   <= mem_cmd_len;
            beat_count    <= 16'h0000;
            obs_cmd_valid <= 1'b1;
            obs_cmd_write <= mem_cmd_write;
            obs_cmd_addr  <= mem_cmd_addr;
            obs_cmd_len   <= mem_cmd_len;
            mem_r_data    <= read_word(mem_cmd_addr);
            mem_r_last    <= (mem_cmd_len == 16'd1);
            state         <= mem_cmd_write ? ST_WRITE : ST_READ;
          end
        end

        ST_WRITE: begin
          mem_w_ready <= stall_mask[2] ? lfsr[2] : 1'b1;
          if (mem_w_valid && mem_w_ready) begin
            if (beat_count == cmd_len_reg - 16'd1) begin
              if (!mem_w_last) begin
                $fatal(1, "behavioral_memory: late mem_w_last");
              end
              for (int j = 0; j < 8; j++)
                if (mem_w_strb[j])
                  mem[cmd_flat_off + beat_count * 8 + j] <= mem_w_data[8*j +: 8];
              state <= ST_IDLE;
            end else begin
              if (mem_w_last) begin
                $fatal(1, "behavioral_memory: early mem_w_last");
              end
              for (int j = 0; j < 8; j++)
                if (mem_w_strb[j])
                  mem[cmd_flat_off + beat_count * 8 + j] <= mem_w_data[8*j +: 8];
              beat_count <= beat_count + 16'd1;
            end
          end
        end

        ST_READ: begin
          mem_r_valid <= stall_mask[1] ? lfsr[1] : 1'b1;
          if (mem_r_valid && mem_r_ready) begin
            if (beat_count == cmd_len_reg - 16'd1) begin
              state <= ST_IDLE;
            end else begin
              beat_count <= beat_count + 16'd1;
              mem_r_data <= read_word(
                  cmd_addr_reg + {13'd0, (beat_count + 16'd1), 3'b000});
              mem_r_last <= ((beat_count + 16'd1) == (cmd_len_reg - 16'd1));
            end
          end
        end

        default: state <= ST_IDLE;
      endcase
    end
  end

endmodule
