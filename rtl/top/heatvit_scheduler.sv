// Dynamic descriptor scheduler (Phase 5 Task 2).
//
// Serializes the 198-descriptor ROM into the single Tensor Executor. The
// FSM is IDLE -> ROM_REQ -> ROM_WAIT -> EXEC_ISSUE -> EXEC_WAIT ->
// ADVANCE -> COMPLETE/ERROR. Token/Package state starts at
// (197, 0) and only OP_SELECTOR_FINALIZE state updates are accepted; an
// out-of-range (not in 2..197, or growing) next N reports error 4.
//
// Activation double-buffering: descriptor operands encoded as slot-relative
// offsets (< ACT_SLOT) are rebased onto the current activation buffer;
// flag-4 descriptors (Block Residual2 / Selector Finalize) write the
// inactive slot and flip the active select on completion.
module heatvit_scheduler
  import heatvit_pkg::*;
#(
  parameter int ACT_SLOT = 37824,  // 197 * 192 bytes per activation slot
  parameter string DESC_MEM_FILE = "rtl/generated/heatvit_descriptors.mem"
)(
  input  logic          clk,
  input  logic          rst_n,
  input  logic          start,
  input  logic          abort,
  // Executor command/status.
  output logic          exec_desc_valid,
  input  logic          exec_desc_ready,
  output heatvit_desc_t exec_desc,
  input  logic          exec_done,
  input  logic          exec_error_valid,
  input  logic [7:0]    exec_error_code,
  input  logic          exec_state_update_valid,
  input  logic [7:0]    exec_next_token_count,
  input  logic          exec_next_package_present,
  // Token/Package state fed back to the executor.
  output logic [7:0]    current_token_count,
  output logic          current_package_present,
  // Status.
  output logic          busy,
  output logic          done,
  output logic          error_valid,
  output logic [7:0]    error_code,
  output logic [15:0]   current_desc_index
);

  typedef enum logic [2:0] {
    S_IDLE,
    S_ROM_REQ,
    S_ROM_WAIT,
    S_EXEC_ISSUE,
    S_EXEC_WAIT,
    S_ADVANCE,
    S_COMPLETE,
    S_ERROR
  } state_t;

  state_t state;

  logic [15:0] desc_index;
  logic [15:0] rom_addr;
  heatvit_desc_t rom_desc;
  heatvit_desc_t exec_desc_r;
  heatvit_desc_t patched_c;
  logic        active;
  logic [7:0]  n_r;
  logic        pkg_r;
  logic [7:0]  err_code_r;

  heatvit_descriptor_rom #(
    .DESC_MEM_FILE(DESC_MEM_FILE)
  ) u_rom (
    .clk   (clk),
    .rst_n (rst_n),
    .addr  (rom_addr),
    .desc  (rom_desc)
  );

  // Activation slot rebasing. Only scratch-region tensors are slot-relative:
  // input-region src0, weight-region src1/aux and output-region dst carry
  // region-relative offsets (FLAG_SRC0_INPUT / FLAG_SRC1_SCRATCH /
  // FLAG_AUX_WEIGHT / FLAG_DST_OUTPUT select the region) and must not be
  // rebased, even when their offset happens to be below the slot size.
  always_comb begin
    logic [31:0] active_base;
    logic [31:0] inactive_base;
    patched_c      = rom_desc;
    active_base    = active ? ACT_SLOT[31:0] : 32'd0;
    inactive_base  = active ? 32'd0 : ACT_SLOT[31:0];
    if (rom_desc.src0_offset < ACT_SLOT[31:0] &&
        !rom_desc.flags[FLAG_SRC0_INPUT])
      patched_c.src0_offset = rom_desc.src0_offset + active_base;
    if (rom_desc.src1_offset < ACT_SLOT[31:0] &&
        rom_desc.flags[FLAG_SRC1_SCRATCH])
      patched_c.src1_offset = rom_desc.src1_offset + active_base;
    if (rom_desc.aux_offset < ACT_SLOT[31:0] &&
        !rom_desc.flags[FLAG_AUX_WEIGHT])
      patched_c.aux_offset = rom_desc.aux_offset + active_base;
    if (rom_desc.dst_offset < ACT_SLOT[31:0] &&
        !rom_desc.flags[FLAG_DST_OUTPUT])
      patched_c.dst_offset = rom_desc.dst_offset +
          (rom_desc.flags[FLAG_SWAP_ACTIVATION] ? inactive_base
                                                : active_base);
    // N-scaled kind blocks inside the shared qkv tensor: the K slice sits
    // at qkv + 3*N*64 and the V slice at qkv + 6*N*64, so the QK^T and
    // Attention*V descriptors encode only the qkv base and the scheduler
    // adds the runtime-N stride.
    if (rom_desc.flags[FLAG_SRC1_SCRATCH] &&
        rom_desc.flags[FLAG_HEAD_MODE]) begin
      if (rom_desc.flags[FLAG_RHS_TRANSPOSE])
        patched_c.src1_offset = rom_desc.src1_offset +
            {16'd0, n_r} * 16'd192;
      else if (rom_desc.flags[FLAG_SRC0_UNSIGNED])
        patched_c.src1_offset = rom_desc.src1_offset +
            {16'd0, n_r} * 16'd384;
    end
  end

  assign exec_desc         = exec_desc_r;
  assign current_desc_index = desc_index;
  assign current_token_count = n_r;
  assign current_package_present = pkg_r;
  assign busy = (state != S_IDLE);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state       <= S_IDLE;
      desc_index  <= 16'd0;
      rom_addr    <= 16'd0;
      exec_desc_r <= '0;
      exec_desc_valid <= 1'b0;
      active      <= 1'b0;
      n_r         <= 8'd0;
      pkg_r       <= 1'b0;
      err_code_r  <= 8'd0;
      done        <= 1'b0;
      error_valid <= 1'b0;
      error_code  <= 8'd0;
    end else begin
      done        <= 1'b0;
      error_valid <= 1'b0;

      if (abort && state != S_IDLE) begin
        state           <= S_IDLE;
        exec_desc_valid <= 1'b0;
        done            <= 1'b0;
      end else case (state)
        S_IDLE: begin
          if (start) begin
            desc_index  <= 16'd0;
            n_r         <= 8'd197;
            pkg_r       <= 1'b0;
            active      <= 1'b0;
            state       <= S_ROM_REQ;
          end
        end

        S_ROM_REQ: begin
          rom_addr <= desc_index;
          state    <= S_ROM_WAIT;
        end

        S_ROM_WAIT: begin
          if (rom_desc.opcode == OP_FINISH) begin
            state <= S_COMPLETE;
          end else begin
            exec_desc_r     <= patched_c;
            exec_desc_valid <= 1'b1;
            state           <= S_EXEC_ISSUE;
          end
        end

        S_EXEC_ISSUE: begin
          if (exec_desc_ready) begin
            exec_desc_valid <= 1'b0;
            state           <= S_EXEC_WAIT;
          end
        end

        S_EXEC_WAIT: begin
          if (exec_error_valid) begin
            err_code_r <= exec_error_code;
            state      <= S_ERROR;
          end else if (exec_done) begin
            begin
              logic go_advance;
              go_advance = 1'b1;
              if (exec_state_update_valid) begin
                if (rom_desc.opcode != OP_SELECTOR_FINALIZE) begin
                  $warning("scheduler: state update from non-finalize descriptor");
                end else if (exec_next_token_count < 8'd2 ||
                             exec_next_token_count > 8'd197 ||
                             exec_next_token_count > n_r) begin
                  err_code_r <= ERR_TOKEN_COUNT;
                  state      <= S_ERROR;
                  go_advance = 1'b0;
                end else begin
                  n_r   <= exec_next_token_count;
                  pkg_r <= exec_next_package_present;
                end
              end
              if (go_advance) begin
                if (rom_desc.flags[FLAG_SWAP_ACTIVATION])
                  active <= ~active;
                state <= S_ADVANCE;
              end
            end
          end
        end

        S_ADVANCE: begin
          desc_index <= desc_index + 16'd1;
          state      <= S_ROM_REQ;
        end

        S_COMPLETE: begin
          done  <= 1'b1;
          state <= S_IDLE;
        end

        S_ERROR: begin
          error_valid <= 1'b1;
          error_code  <= err_code_r;
          state       <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
