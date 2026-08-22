// HeatViT inference top (Phase 5 Task 3).
//
// Instantiates one descriptor scheduler and one tensor executor. A legal
// start latches the four region base/byte registers and clears the error
// and warning state; warnings OR-latch into warning_flags[2:0] until the
// next legal start or reset. Any error (busy start, scheduler, executor)
// aborts both children and holds busy until the executor drains its
// accepted burst (protocol-safe point). The classifier completion latches
// its dst scale exponent into output_scale_exp.
module heatvit_top
  import heatvit_pkg::*;
#(
  parameter string DESC_MEM_FILE = "rtl/generated/heatvit_descriptors.mem"
)(
  input  logic                 clk,
  input  logic                 rst_n,
  input  logic                 start,
  input  logic [31:0]          input_base,
  input  logic [31:0]          input_bytes,
  input  logic [31:0]          weight_base,
  input  logic [31:0]          weight_bytes,
  input  logic [31:0]          scratch_base,
  input  logic [31:0]          scratch_bytes,
  input  logic [31:0]          output_base,
  input  logic [31:0]          output_bytes,
  output logic                 busy,
  output logic                 done,
  output logic                 error_valid,
  output logic [7:0]           error_code,
  output logic [7:0]           warning_flags,
  output heatvit_scale_t       output_scale_exp,
  output logic                 mem_cmd_valid,
  input  logic                 mem_cmd_ready,
  output logic                 mem_cmd_write,
  output logic [31:0]          mem_cmd_addr,
  output logic [15:0]          mem_cmd_len,
  output logic                 mem_w_valid,
  input  logic                 mem_w_ready,
  output logic [63:0]          mem_w_data,
  output logic [7:0]           mem_w_strb,
  output logic                 mem_w_last,
  input  logic                 mem_r_valid,
  output logic                 mem_r_ready,
  input  logic [63:0]          mem_r_data,
  input  logic                 mem_r_last
);

  // Scheduler <-> Executor.
  logic        sched_start;
  logic        sched_abort;
  logic        sched_busy;
  logic        sched_done;
  logic        sched_error_valid;
  logic [7:0]  sched_error_code;
  logic [15:0] sched_index;
  logic        exec_desc_valid;
  logic        exec_desc_ready;
  heatvit_desc_t exec_desc;
  logic [7:0]  cur_token_count;
  logic        cur_package_present;
  logic        exec_done;
  logic        exec_error_valid;
  logic [7:0]  exec_error_code;
  logic        exec_state_update_valid;
  logic [7:0]  exec_next_token_count;
  logic        exec_next_package_present;
  logic [2:0]  exec_warning_pulse;
  logic        exec_abort;
  logic        exec_abort_done;

  // Latched region registers.
  logic [31:0] input_base_r;
  logic [31:0] input_bytes_r;
  logic [31:0] weight_base_r;
  logic [31:0] weight_bytes_r;
  logic [31:0] scratch_base_r;
  logic [31:0] scratch_bytes_r;
  logic [31:0] output_base_r;
  logic [31:0] output_bytes_r;

  logic [7:0]  warning_r;
  logic [7:0]  err_code_r;
  logic        abort_pending;
  logic        exec_abort_hold;
  heatvit_scale_t out_scale_r;

  heatvit_scheduler #(
    .DESC_MEM_FILE(DESC_MEM_FILE)
  ) u_scheduler (
    .clk                    (clk),
    .rst_n                  (rst_n),
    .start                  (sched_start),
    .abort                  (sched_abort),
    .exec_desc_valid        (exec_desc_valid),
    .exec_desc_ready        (exec_desc_ready),
    .exec_desc              (exec_desc),
    .exec_done              (exec_done),
    .exec_error_valid       (exec_error_valid),
    .exec_error_code        (exec_error_code),
    .exec_state_update_valid(exec_state_update_valid),
    .exec_next_token_count  (exec_next_token_count),
    .exec_next_package_present(exec_next_package_present),
    .current_token_count    (cur_token_count),
    .current_package_present(cur_package_present),
    .busy                   (sched_busy),
    .done                   (sched_done),
    .error_valid            (sched_error_valid),
    .error_code             (sched_error_code),
    .current_desc_index     (sched_index)
  );

  heatvit_tensor_executor u_executor (
    .clk                   (clk),
    .rst_n                 (rst_n),
    .abort                 (exec_abort),
    .desc_valid            (exec_desc_valid),
    .desc_ready            (exec_desc_ready),
    .desc                  (exec_desc),
    .current_token_count   (cur_token_count),
    .current_package_present(cur_package_present),
    .input_base            (input_base_r),
    .input_bytes           (input_bytes_r),
    .weight_base           (weight_base_r),
    .weight_bytes          (weight_bytes_r),
    .scratch_base          (scratch_base_r),
    .scratch_bytes         (scratch_bytes_r),
    .output_base           (output_base_r),
    .output_bytes          (output_bytes_r),
    .busy                  (),
    .done                  (exec_done),
    .error_valid           (exec_error_valid),
    .error_code            (exec_error_code),
    .abort_done            (exec_abort_done),
    .warning_pulse         (exec_warning_pulse),
    .state_update_valid    (exec_state_update_valid),
    .next_token_count      (exec_next_token_count),
    .next_package_present  (exec_next_package_present),
    .mem_cmd_valid         (mem_cmd_valid),
    .mem_cmd_ready         (mem_cmd_ready),
    .mem_cmd_write         (mem_cmd_write),
    .mem_cmd_addr          (mem_cmd_addr),
    .mem_cmd_len           (mem_cmd_len),
    .mem_w_valid           (mem_w_valid),
    .mem_w_ready           (mem_w_ready),
    .mem_w_data            (mem_w_data),
    .mem_w_strb            (mem_w_strb),
    .mem_w_last            (mem_w_last),
    .mem_r_valid           (mem_r_valid),
    .mem_r_ready           (mem_r_ready),
    .mem_r_data            (mem_r_data),
    .mem_r_last            (mem_r_last)
  );

  assign busy              = sched_busy | abort_pending;
  assign warning_flags     = warning_r;
  assign output_scale_exp  = out_scale_r;
  assign sched_start       = start;
  assign exec_abort        = exec_abort_hold;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      warning_r       <= 8'd0;
      error_valid     <= 1'b0;
      error_code      <= ERR_NONE;
      err_code_r      <= ERR_NONE;
      done            <= 1'b0;
      sched_abort     <= 1'b0;
      exec_abort_hold <= 1'b0;
      abort_pending   <= 1'b0;
      out_scale_r     <= 6'sd0;
      input_base_r    <= 32'd0;
      input_bytes_r   <= 32'd0;
      weight_base_r   <= 32'd0;
      weight_bytes_r  <= 32'd0;
      scratch_base_r  <= 32'd0;
      scratch_bytes_r <= 32'd0;
      output_base_r   <= 32'd0;
      output_bytes_r  <= 32'd0;
    end else begin
      done        <= 1'b0;
      error_valid <= 1'b0;
      sched_abort <= 1'b0;
      if (exec_abort_done) begin
        exec_abort_hold <= 1'b0;
        abort_pending   <= 1'b0;
      end
      if (start && !busy) begin
        warning_r   <= 8'd0;
        error_valid <= 1'b0;
        error_code  <= ERR_NONE;
        err_code_r  <= ERR_NONE;
        input_base_r    <= input_base;
        input_bytes_r   <= input_bytes;
        weight_base_r   <= weight_base;
        weight_bytes_r  <= weight_bytes;
        scratch_base_r  <= scratch_base;
        scratch_bytes_r <= scratch_bytes;
        output_base_r   <= output_base;
        output_bytes_r  <= output_bytes;
      end else begin
        warning_r[2:0] <= warning_r[2:0] | exec_warning_pulse;
      end
      if (start && busy) begin
        err_code_r      <= ERR_BUSY_START;
        error_valid     <= 1'b1;
        error_code      <= ERR_BUSY_START;
        sched_abort     <= 1'b1;
        exec_abort_hold <= 1'b1;
        abort_pending   <= 1'b1;
      end
      if (exec_error_valid) begin
        err_code_r      <= exec_error_code;
        error_valid     <= 1'b1;
        error_code      <= exec_error_code;
        sched_abort     <= 1'b1;
        exec_abort_hold <= 1'b1;
        abort_pending   <= 1'b1;
      end
      if (sched_error_valid) begin
        err_code_r      <= sched_error_code;
        error_valid     <= 1'b1;
        error_code      <= sched_error_code;
        sched_abort     <= 1'b1;
        exec_abort_hold <= 1'b1;
        abort_pending   <= 1'b1;
      end
      if (sched_done) done <= 1'b1;
      // Classifier completion latches its dst scale exponent.
      if (sched_index == 16'd196 && exec_done)
        out_scale_r <= exec_desc.dst_scale_exp;
    end
  end

endmodule
