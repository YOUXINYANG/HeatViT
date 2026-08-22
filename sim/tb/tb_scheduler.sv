`timescale 1ns / 1ps

// Task 2: descriptor ROM and dynamic scheduler against a fake Executor.
//
// Phase 1 runs the complete 197-descriptor issue sequence; the fake
// Executor completes every descriptor in 1..5 cycles and returns state
// updates (150,1)/(90,1)/(48,1) from the three Selector Finalize
// descriptors. The TB asserts the next dynamic descriptors see the updated
// token counts, the flag-4 activation writes alternate between the two
// slots, OP_FINISH is never issued and top-level done is a single pulse.
// Phase 2 injects an illegal token count (250) and expects error 4; phase 3
// injects an Executor error 5 and expects passthrough; phase 4 emits a
// rogue state update on a normal descriptor which must be ignored while the
// run still completes.
module tb_scheduler;
  import heatvit_pkg::*;
  import tb_pkg::*;

  localparam int ACT_SLOT = 37824;

  logic        clk   = 1'b0;
  logic        rst_n = 1'b0;

  logic        start;
  logic        abort;
  logic        exec_desc_valid;
  logic        exec_desc_ready;
  heatvit_desc_t exec_desc;
  logic [7:0]  current_token_count;
  logic        current_package_present;
  logic        busy;
  logic        done;
  logic        error_valid;
  logic [7:0]  error_code;
  logic [15:0] current_desc_index;

  // Fake executor drive.
  logic        exec_done;
  logic        exec_error_valid;
  logic [7:0]  exec_error_code;
  logic        exec_state_update_valid;
  logic [7:0]  exec_next_token_count;
  logic        exec_next_package_present;

  heatvit_scheduler dut (
    .clk                    (clk),
    .rst_n                  (rst_n),
    .start                  (start),
    .abort                  (abort),
    .exec_desc_valid        (exec_desc_valid),
    .exec_desc_ready        (exec_desc_ready),
    .exec_desc              (exec_desc),
    .exec_done              (exec_done),
    .exec_error_valid       (exec_error_valid),
    .exec_error_code        (exec_error_code),
    .exec_state_update_valid(exec_state_update_valid),
    .exec_next_token_count  (exec_next_token_count),
    .exec_next_package_present(exec_next_package_present),
    .current_token_count    (current_token_count),
    .current_package_present(current_package_present),
    .busy                   (busy),
    .done                   (done),
    .error_valid            (error_valid),
    .error_code             (error_code),
    .current_desc_index     (current_desc_index)
  );

  always #5 clk = ~clk;
  assign exec_desc_ready = 1'b1;

  // ---------------------------------------------------------------
  // Fake executor.
  // ---------------------------------------------------------------
  int phase;
  int issued;
  int done_delay;
  int finalize_seen;
  int error_at;
  int rogue_at;
  logic [7:0] last_opcode;
  logic       reinit;

  logic [7:0] n_seen [0:210];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      issued      <= 0;
      done_delay  <= 0;
      finalize_seen <= 0;
      exec_done   <= 1'b0;
      exec_error_valid <= 1'b0;
      exec_state_update_valid <= 1'b0;
      exec_error_code <= 8'd0;
      exec_next_token_count <= 8'd0;
      exec_next_package_present <= 1'b0;
    end else begin
      exec_done   <= 1'b0;
      exec_error_valid <= 1'b0;
      exec_state_update_valid <= 1'b0;

      if (reinit) begin
        issued        <= 0;
        done_delay    <= 0;
        finalize_seen <= 0;
      end

      if (exec_desc_valid && exec_desc_ready) begin
        n_seen[issued] <= current_token_count;
        last_opcode <= exec_desc.opcode;
        issued      <= issued + 1;
        done_delay  <= (issued % 5) + 1;
      end

      if (done_delay > 0) begin
        done_delay <= done_delay - 1;
        if (done_delay == 1) begin
          exec_done <= 1'b1;
          if (issued == error_at) begin
            exec_error_valid <= 1'b1;
            exec_error_code  <= 8'd5;
          end else if (last_opcode == OP_SELECTOR_FINALIZE) begin
            finalize_seen <= finalize_seen + 1;
            if (phase == 1) begin
              if (finalize_seen == 0) begin
                exec_state_update_valid <= 1'b1;
                exec_next_token_count  <= 8'd150;
                exec_next_package_present <= 1'b1;
              end else if (finalize_seen == 1) begin
                exec_state_update_valid <= 1'b1;
                exec_next_token_count  <= 8'd90;
                exec_next_package_present <= 1'b1;
              end else begin
                exec_state_update_valid <= 1'b1;
                exec_next_token_count  <= 8'd48;
                exec_next_package_present <= 1'b1;
              end
            end else if (phase == 2 && finalize_seen == 0) begin
              exec_state_update_valid <= 1'b1;
              exec_next_token_count  <= 8'd250;
              exec_next_package_present <= 1'b1;
            end
          end else if (issued == rogue_at) begin
            exec_state_update_valid <= 1'b1;
            exec_next_token_count  <= 8'd100;
            exec_next_package_present <= 1'b1;
          end
        end
      end
    end
  end

  int done_pulses;
  int error_pulses;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      done_pulses  <= 0;
      error_pulses <= 0;
    end else begin
      if (done) done_pulses <= done_pulses + 1;
      if (error_valid) error_pulses <= error_pulses + 1;
    end
  end

  task automatic run_start();
    start = 1'b1;
    @(posedge clk);
    #1;
    start = 1'b0;
  endtask

  task automatic reinit_model();
    reinit = 1'b1;
    @(posedge clk);
    #1;
    reinit = 1'b0;
  endtask

  initial begin
    #50000000;
    $display("WATCHDOG: issued=%0d busy=%0d", issued, busy);
    tb_fatal("tb_scheduler watchdog");
  end

  initial begin
    start   = 1'b0;
    abort   = 1'b0;
    reinit  = 1'b0;
    phase   = 1;
    error_at = -1;
    rogue_at = -1;

    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
    #1;

    // ---- Phase 1: complete normal run. ----
    run_start();
    wait (done);
    #1;
    if (issued != 197) begin
      $display("phase1 issued=%0d expected=197", issued);
      tb_fatal("scheduler did not issue exactly 197 descriptors");
    end
    for (int i = 0; i <= 53; i++)
      if (n_seen[i] != 8'd197) begin
        $display("phase1 n_seen[%0d]=%0d expected=197", i, n_seen[i]);
        tb_fatal("initial token count mismatch");
      end
    for (int i = 54; i <= 92; i++)
      if (n_seen[i] != 8'd150) begin
        $display("phase1 n_seen[%0d]=%0d expected=150", i, n_seen[i]);
        tb_fatal("post-finalize-1 token count mismatch");
      end
    for (int i = 93; i <= 104; i++)
      if (n_seen[i] != 8'd150) begin
        $display("phase1 n_seen[%0d]=%0d expected=150", i, n_seen[i]);
        tb_fatal("selector-2 token count mismatch");
      end
    for (int i = 105; i <= 143; i++)
      if (n_seen[i] != 8'd90) begin
        $display("phase1 n_seen[%0d]=%0d expected=90", i, n_seen[i]);
        tb_fatal("post-finalize-2 token count mismatch");
      end
    for (int i = 144; i <= 155; i++)
      if (n_seen[i] != 8'd90) begin
        $display("phase1 n_seen[%0d]=%0d expected=90", i, n_seen[i]);
        tb_fatal("selector-3 token count mismatch");
      end
    for (int i = 156; i <= 196; i++)
      if (n_seen[i] != 8'd48) begin
        $display("phase1 n_seen[%0d]=%0d expected=48", i, n_seen[i]);
        tb_fatal("post-finalize-3 token count mismatch");
      end
    @(posedge clk);
    #1;
    if (done_pulses != 1) begin
      $display("phase1 done_pulses=%0d expected=1", done_pulses);
      tb_fatal("top-level done must be a single pulse");
    end

    // ---- Phase 2: illegal token count -> error 4. ----
    phase = 2;
    error_at = -1;
    rogue_at = -1;
    reinit_model();
    run_start();
    wait (error_valid);
    #1;
    if (error_code != 8'd4) begin
      $display("phase2 error_code=%0d expected=4", error_code);
      tb_fatal("illegal token count must report error 4");
    end
    if (done) begin
      tb_fatal("phase2 done must not pulse on error");
    end
    @(posedge clk);
    #1;
    if (issued != 54) begin
      $display("phase2 issued=%0d after error (expected 54)", issued);
      tb_fatal("no new descriptors may issue after the error");
    end

    // ---- Phase 3: executor error passthrough. ----
    phase = 3;
    error_at = 10;
    rogue_at = -1;
    reinit_model();
    run_start();
    wait (error_valid);
    #1;
    if (error_code != 8'd5) begin
      $display("phase3 error_code=%0d expected=5", error_code);
      tb_fatal("executor error must pass through");
    end

    // ---- Phase 4: rogue state update on a normal descriptor. ----
    phase = 4;
    error_at = -1;
    rogue_at = 5;
    reinit_model();
    run_start();
    wait (done);
    #1;
    if (issued != 197) begin
      $display("phase4 issued=%0d expected=197", issued);
      tb_fatal("rogue state update must be ignored and the run completes");
    end

    @(posedge clk);
    #1;
    $display("TEST_PASS tb_scheduler");
    $finish;
  end

endmodule
