`timescale 1ns / 1ps

// Task 7: error 1..7 and warning 0..2 coverage.
//
// Each case is its own heatvit_top with a minimal injection ROM from
// build/vectors/e2e/errors/*.mem and, where memory traffic is needed, a
// small behavioral memory with crafted backdoor contents. Injection lives
// only here: hierarchical forces for the softmax zero-sum, the read-last
// protocol violation and the illegal token update. Error cases must stop
// new commands, clear busy, pulse error_valid with the exact code and
// never assert done; warning cases complete with done, latch the right
// bit, and a later legal start clears the latch.
module tb_heatvit_errors;
  import heatvit_pkg::*;
  import tb_pkg::*;

  localparam string ERR1 = "build/vectors/e2e/errors/err1_opcode.mem";
  localparam string ERR2 = "build/vectors/e2e/errors/err2_dimension.mem";
  localparam string ERR3 = "build/vectors/e2e/errors/err3_address.mem";
  localparam string ERR4 = "build/vectors/e2e/errors/err4_token.mem";
  localparam string ERR5 = "build/vectors/e2e/errors/err5_protocol.mem";
  localparam string ERR6 = "build/vectors/e2e/errors/err6_softmax.mem";
  localparam string WARN0 = "build/vectors/e2e/errors/warn0_head_den.mem";
  localparam string WARN1 = "build/vectors/e2e/errors/warn1_pkg_den.mem";
  localparam string WARN2 = "build/vectors/e2e/errors/warn2_ln_var.mem";

  logic clk = 1'b0;
  always #5 clk = ~clk;

  // ---------------------------------------------------------------
  // Shared per-case signal bundle helpers (none: XSim flags task ref
  // arguments as extra drivers, so every case is inlined below).
  // ---------------------------------------------------------------

  // Cases 1..3: validation errors without memory traffic.
  logic r1, s1, b1, d1, e1; logic [7:0] c1;
  logic r2, s2, b2, d2, e2; logic [7:0] c2;
  logic r3, s3, b3, d3, e3; logic [7:0] c3;

  heatvit_top #(.DESC_MEM_FILE(ERR1)) top1 (
    .clk(clk), .rst_n(r1), .start(s1),
    .input_base(32'h0), .input_bytes(4096),
    .weight_base(32'h01000000), .weight_bytes(4096),
    .scratch_base(32'h02000000), .scratch_bytes(65536),
    .output_base(32'h03000000), .output_bytes(4096),
    .busy(b1), .done(d1), .error_valid(e1), .error_code(c1),
    .warning_flags(), .output_scale_exp(),
    .mem_cmd_valid(), .mem_cmd_ready(1'b0), .mem_cmd_write(),
    .mem_cmd_addr(), .mem_cmd_len(), .mem_w_valid(), .mem_w_ready(1'b0),
    .mem_w_data(), .mem_w_strb(), .mem_w_last(),
    .mem_r_valid(1'b0), .mem_r_ready(), .mem_r_data(64'd0),
    .mem_r_last(1'b0));

  heatvit_top #(.DESC_MEM_FILE(ERR2)) top2 (
    .clk(clk), .rst_n(r2), .start(s2),
    .input_base(32'h0), .input_bytes(4096),
    .weight_base(32'h01000000), .weight_bytes(4096),
    .scratch_base(32'h02000000), .scratch_bytes(65536),
    .output_base(32'h03000000), .output_bytes(4096),
    .busy(b2), .done(d2), .error_valid(e2), .error_code(c2),
    .warning_flags(), .output_scale_exp(),
    .mem_cmd_valid(), .mem_cmd_ready(1'b0), .mem_cmd_write(),
    .mem_cmd_addr(), .mem_cmd_len(), .mem_w_valid(), .mem_w_ready(1'b0),
    .mem_w_data(), .mem_w_strb(), .mem_w_last(),
    .mem_r_valid(1'b0), .mem_r_ready(), .mem_r_data(64'd0),
    .mem_r_last(1'b0));

  heatvit_top #(.DESC_MEM_FILE(ERR3)) top3 (
    .clk(clk), .rst_n(r3), .start(s3),
    .input_base(32'h0), .input_bytes(4096),
    .weight_base(32'h01000000), .weight_bytes(4096),
    .scratch_base(32'h02000000), .scratch_bytes(65536),
    .output_base(32'h03000000), .output_bytes(4096),
    .busy(b3), .done(d3), .error_valid(e3), .error_code(c3),
    .warning_flags(), .output_scale_exp(),
    .mem_cmd_valid(), .mem_cmd_ready(1'b0), .mem_cmd_write(),
    .mem_cmd_addr(), .mem_cmd_len(), .mem_w_valid(), .mem_w_ready(1'b0),
    .mem_w_data(), .mem_w_strb(), .mem_w_last(),
    .mem_r_valid(1'b0), .mem_r_ready(), .mem_r_data(64'd0),
    .mem_r_last(1'b0));

  // ---------------------------------------------------------------
  // Case 4: illegal token update (force) with a functional memory.
  // ---------------------------------------------------------------
  logic r4, s4, b4, d4, e4; logic [7:0] c4, w4;
  logic        cm4v, cm4w, cm4rdy; logic [31:0] cm4a; logic [15:0] cm4l;
  logic        wm4v, wm4rdy; logic [63:0] wm4d; logic [7:0] wm4s;
  logic        wm4l, rm4v, rm4rdy, rm4l; logic [63:0] rm4d;
  logic        dbg4v, dbg4w; logic [31:0] dbg4a; logic [7:0] dbg4d;

  heatvit_top #(.DESC_MEM_FILE(ERR4)) top4 (
    .clk(clk), .rst_n(r4), .start(s4),
    .input_base(32'h0), .input_bytes(4096),
    .weight_base(32'h01000000), .weight_bytes(4096),
    .scratch_base(32'h02000000), .scratch_bytes(65536),
    .output_base(32'h03000000), .output_bytes(4096),
    .busy(b4), .done(d4), .error_valid(e4), .error_code(c4),
    .warning_flags(w4), .output_scale_exp(),
    .mem_cmd_valid(cm4v), .mem_cmd_ready(cm4rdy), .mem_cmd_write(cm4w),
    .mem_cmd_addr(cm4a), .mem_cmd_len(cm4l),
    .mem_w_valid(wm4v), .mem_w_ready(wm4rdy), .mem_w_data(wm4d),
    .mem_w_strb(wm4s), .mem_w_last(wm4l),
    .mem_r_valid(rm4v), .mem_r_ready(rm4rdy), .mem_r_data(rm4d),
    .mem_r_last(rm4l));
  behavioral_memory #(
    .SEG_COUNT(1), .SEG0_BASE(32'h02000000), .SEG0_BYTES(65536)
  ) mem4 (
    .clk(clk), .rst_n(r4), .stall_mask(16'h0),
    .mem_cmd_valid(cm4v), .mem_cmd_ready(cm4rdy), .mem_cmd_write(cm4w),
    .mem_cmd_addr(cm4a), .mem_cmd_len(cm4l),
    .mem_w_valid(wm4v), .mem_w_ready(wm4rdy), .mem_w_data(wm4d),
    .mem_w_strb(wm4s), .mem_w_last(wm4l),
    .mem_r_valid(rm4v), .mem_r_ready(rm4rdy), .mem_r_data(rm4d),
    .mem_r_last(rm4l),
    .obs_cmd_valid(), .obs_cmd_write(), .obs_cmd_addr(), .obs_cmd_len(),
    .dbg_valid(1'b0), .dbg_ready(), .dbg_addr(32'h0), .dbg_data(),
    .dbg_w_valid(dbg4w), .dbg_w_addr(dbg4a), .dbg_w_data(dbg4d),
    .load_valid(1'b0), .load_ready(), .load_seg(2'd0),
    .load_bytes(32'd0), .load_file(""));

  // ---------------------------------------------------------------
  // Case 5: read-last protocol violation (force).
  // ---------------------------------------------------------------
  logic r5, s5, b5, d5, e5; logic [7:0] c5;
  logic        cm5v, cm5w, cm5rdy; logic [31:0] cm5a; logic [15:0] cm5l;
  logic        wm5v, wm5rdy; logic [63:0] wm5d; logic [7:0] wm5s;
  logic        wm5l, rm5v, rm5rdy, rm5l; logic [63:0] rm5d;

  heatvit_top #(.DESC_MEM_FILE(ERR5)) top5 (
    .clk(clk), .rst_n(r5), .start(s5),
    .input_base(32'h0), .input_bytes(150528),
    .weight_base(32'h01000000), .weight_bytes(4096),
    .scratch_base(32'h02000000), .scratch_bytes(524288),
    .output_base(32'h03000000), .output_bytes(4096),
    .busy(b5), .done(d5), .error_valid(e5), .error_code(c5),
    .warning_flags(), .output_scale_exp(),
    .mem_cmd_valid(cm5v), .mem_cmd_ready(cm5rdy), .mem_cmd_write(cm5w),
    .mem_cmd_addr(cm5a), .mem_cmd_len(cm5l),
    .mem_w_valid(wm5v), .mem_w_ready(wm5rdy), .mem_w_data(wm5d),
    .mem_w_strb(wm5s), .mem_w_last(wm5l),
    .mem_r_valid(rm5v), .mem_r_ready(rm5rdy), .mem_r_data(rm5d),
    .mem_r_last(rm5l));
  behavioral_memory #(
    .SEG_COUNT(2),
    .SEG0_BASE(32'h0), .SEG0_BYTES(150528),
    .SEG1_BASE(32'h02000000), .SEG1_BYTES(524288)
  ) mem5 (
    .clk(clk), .rst_n(r5), .stall_mask(16'h0),
    .mem_cmd_valid(cm5v), .mem_cmd_ready(cm5rdy), .mem_cmd_write(cm5w),
    .mem_cmd_addr(cm5a), .mem_cmd_len(cm5l),
    .mem_w_valid(wm5v), .mem_w_ready(wm5rdy), .mem_w_data(wm5d),
    .mem_w_strb(wm5s), .mem_w_last(wm5l),
    .mem_r_valid(rm5v), .mem_r_ready(rm5rdy), .mem_r_data(rm5d),
    .mem_r_last(rm5l),
    .obs_cmd_valid(), .obs_cmd_write(), .obs_cmd_addr(), .obs_cmd_len(),
    .dbg_valid(1'b0), .dbg_ready(), .dbg_addr(32'h0), .dbg_data(),
    .dbg_w_valid(1'b0), .dbg_w_addr(32'h0), .dbg_w_data(8'h0),
    .load_valid(1'b0), .load_ready(), .load_seg(2'd0),
    .load_bytes(32'd0), .load_file(""));

  // ---------------------------------------------------------------
  // Case 6: softmax zero-sum (force).
  // ---------------------------------------------------------------
  logic r6, s6, b6, d6, e6; logic [7:0] c6;
  logic        cm6v, cm6w, cm6rdy; logic [31:0] cm6a; logic [15:0] cm6l;
  logic        wm6v, wm6rdy; logic [63:0] wm6d; logic [7:0] wm6s;
  logic        wm6l, rm6v, rm6rdy, rm6l; logic [63:0] rm6d;

  heatvit_top #(.DESC_MEM_FILE(ERR6)) top6 (
    .clk(clk), .rst_n(r6), .start(s6),
    .input_base(32'h0), .input_bytes(4096),
    .weight_base(32'h01000000), .weight_bytes(4096),
    .scratch_base(32'h02000000), .scratch_bytes(524288),
    .output_base(32'h03000000), .output_bytes(4096),
    .busy(b6), .done(d6), .error_valid(e6), .error_code(c6),
    .warning_flags(), .output_scale_exp(),
    .mem_cmd_valid(cm6v), .mem_cmd_ready(cm6rdy), .mem_cmd_write(cm6w),
    .mem_cmd_addr(cm6a), .mem_cmd_len(cm6l),
    .mem_w_valid(wm6v), .mem_w_ready(wm6rdy), .mem_w_data(wm6d),
    .mem_w_strb(wm6s), .mem_w_last(wm6l),
    .mem_r_valid(rm6v), .mem_r_ready(rm6rdy), .mem_r_data(rm6d),
    .mem_r_last(rm6l));
  behavioral_memory #(
    .SEG_COUNT(1), .SEG0_BASE(32'h02000000), .SEG0_BYTES(524288)
  ) mem6 (
    .clk(clk), .rst_n(r6), .stall_mask(16'h0),
    .mem_cmd_valid(cm6v), .mem_cmd_ready(cm6rdy), .mem_cmd_write(cm6w),
    .mem_cmd_addr(cm6a), .mem_cmd_len(cm6l),
    .mem_w_valid(wm6v), .mem_w_ready(wm6rdy), .mem_w_data(wm6d),
    .mem_w_strb(wm6s), .mem_w_last(wm6l),
    .mem_r_valid(rm6v), .mem_r_ready(rm6rdy), .mem_r_data(rm6d),
    .mem_r_last(rm6l),
    .obs_cmd_valid(), .obs_cmd_write(), .obs_cmd_addr(), .obs_cmd_len(),
    .dbg_valid(1'b0), .dbg_ready(), .dbg_addr(32'h0), .dbg_data(),
    .dbg_w_valid(1'b0), .dbg_w_addr(32'h0), .dbg_w_data(8'h0),
    .load_valid(1'b0), .load_ready(), .load_seg(2'd0),
    .load_bytes(32'd0), .load_file(""));

  // ---------------------------------------------------------------
  // Warning cases 8..10 with functional memories and crafted data.
  // ---------------------------------------------------------------
  logic r8, s8, b8, d8, e8; logic [7:0] c8, w8;
  logic        cm8v, cm8w, cm8rdy; logic [31:0] cm8a; logic [15:0] cm8l;
  logic        wm8v, wm8rdy; logic [63:0] wm8d; logic [7:0] wm8s;
  logic        wm8l, rm8v, rm8rdy, rm8l; logic [63:0] rm8d;
  logic        dbg8w; logic [31:0] dbg8a; logic [7:0] dbg8d;

  heatvit_top #(.DESC_MEM_FILE(WARN0)) top8 (
    .clk(clk), .rst_n(r8), .start(s8),
    .input_base(32'h0), .input_bytes(4096),
    .weight_base(32'h01000000), .weight_bytes(4096),
    .scratch_base(32'h02000000), .scratch_bytes(65536),
    .output_base(32'h03000000), .output_bytes(4096),
    .busy(b8), .done(d8), .error_valid(e8), .error_code(c8),
    .warning_flags(w8), .output_scale_exp(),
    .mem_cmd_valid(cm8v), .mem_cmd_ready(cm8rdy), .mem_cmd_write(cm8w),
    .mem_cmd_addr(cm8a), .mem_cmd_len(cm8l),
    .mem_w_valid(wm8v), .mem_w_ready(wm8rdy), .mem_w_data(wm8d),
    .mem_w_strb(wm8s), .mem_w_last(wm8l),
    .mem_r_valid(rm8v), .mem_r_ready(rm8rdy), .mem_r_data(rm8d),
    .mem_r_last(rm8l));
  behavioral_memory #(
    .SEG_COUNT(1), .SEG0_BASE(32'h02000000), .SEG0_BYTES(65536)
  ) mem8 (
    .clk(clk), .rst_n(r8), .stall_mask(16'h0),
    .mem_cmd_valid(cm8v), .mem_cmd_ready(cm8rdy), .mem_cmd_write(cm8w),
    .mem_cmd_addr(cm8a), .mem_cmd_len(cm8l),
    .mem_w_valid(wm8v), .mem_w_ready(wm8rdy), .mem_w_data(wm8d),
    .mem_w_strb(wm8s), .mem_w_last(wm8l),
    .mem_r_valid(rm8v), .mem_r_ready(rm8rdy), .mem_r_data(rm8d),
    .mem_r_last(rm8l),
    .obs_cmd_valid(), .obs_cmd_write(), .obs_cmd_addr(), .obs_cmd_len(),
    .dbg_valid(1'b0), .dbg_ready(), .dbg_addr(32'h0), .dbg_data(),
    .dbg_w_valid(dbg8w), .dbg_w_addr(dbg8a), .dbg_w_data(dbg8d),
    .load_valid(1'b0), .load_ready(), .load_seg(2'd0),
    .load_bytes(32'd0), .load_file(""));

  logic r9, s9, b9, d9, e9; logic [7:0] c9, w9;
  logic        cm9v, cm9w, cm9rdy; logic [31:0] cm9a; logic [15:0] cm9l;
  logic        wm9v, wm9rdy; logic [63:0] wm9d; logic [7:0] wm9s;
  logic        wm9l, rm9v, rm9rdy, rm9l; logic [63:0] rm9d;

  heatvit_top #(.DESC_MEM_FILE(WARN1)) top9 (
    .clk(clk), .rst_n(r9), .start(s9),
    .input_base(32'h0), .input_bytes(4096),
    .weight_base(32'h01000000), .weight_bytes(4096),
    .scratch_base(32'h02000000), .scratch_bytes(65536),
    .output_base(32'h03000000), .output_bytes(4096),
    .busy(b9), .done(d9), .error_valid(e9), .error_code(c9),
    .warning_flags(w9), .output_scale_exp(),
    .mem_cmd_valid(cm9v), .mem_cmd_ready(cm9rdy), .mem_cmd_write(cm9w),
    .mem_cmd_addr(cm9a), .mem_cmd_len(cm9l),
    .mem_w_valid(wm9v), .mem_w_ready(wm9rdy), .mem_w_data(wm9d),
    .mem_w_strb(wm9s), .mem_w_last(wm9l),
    .mem_r_valid(rm9v), .mem_r_ready(rm9rdy), .mem_r_data(rm9d),
    .mem_r_last(rm9l));
  behavioral_memory #(
    .SEG_COUNT(1), .SEG0_BASE(32'h02000000), .SEG0_BYTES(65536)
  ) mem9 (
    .clk(clk), .rst_n(r9), .stall_mask(16'h0),
    .mem_cmd_valid(cm9v), .mem_cmd_ready(cm9rdy), .mem_cmd_write(cm9w),
    .mem_cmd_addr(cm9a), .mem_cmd_len(cm9l),
    .mem_w_valid(wm9v), .mem_w_ready(wm9rdy), .mem_w_data(wm9d),
    .mem_w_strb(wm9s), .mem_w_last(wm9l),
    .mem_r_valid(rm9v), .mem_r_ready(rm9rdy), .mem_r_data(rm9d),
    .mem_r_last(rm9l),
    .obs_cmd_valid(), .obs_cmd_write(), .obs_cmd_addr(), .obs_cmd_len(),
    .dbg_valid(1'b0), .dbg_ready(), .dbg_addr(32'h0), .dbg_data(),
    .dbg_w_valid(1'b0), .dbg_w_addr(32'h0), .dbg_w_data(8'h0),
    .load_valid(1'b0), .load_ready(), .load_seg(2'd0),
    .load_bytes(32'd0), .load_file(""));

  logic r10, s10, b10, d10, e10; logic [7:0] c10, w10;
  logic        cm10v, cm10w, cm10rdy; logic [31:0] cm10a;
  logic [15:0] cm10l;
  logic        wm10v, wm10rdy; logic [63:0] wm10d; logic [7:0] wm10s;
  logic        wm10l, rm10v, rm10rdy, rm10l; logic [63:0] rm10d;
  logic        dbg10w; logic [31:0] dbg10a; logic [7:0] dbg10d;

  heatvit_top #(.DESC_MEM_FILE(WARN2)) top10 (
    .clk(clk), .rst_n(r10), .start(s10),
    .input_base(32'h0), .input_bytes(4096),
    .weight_base(32'h01000000), .weight_bytes(4096),
    .scratch_base(32'h02000000), .scratch_bytes(65536),
    .output_base(32'h03000000), .output_bytes(4096),
    .busy(b10), .done(d10), .error_valid(e10), .error_code(c10),
    .warning_flags(w10), .output_scale_exp(),
    .mem_cmd_valid(cm10v), .mem_cmd_ready(cm10rdy),
    .mem_cmd_write(cm10w), .mem_cmd_addr(cm10a), .mem_cmd_len(cm10l),
    .mem_w_valid(wm10v), .mem_w_ready(wm10rdy), .mem_w_data(wm10d),
    .mem_w_strb(wm10s), .mem_w_last(wm10l),
    .mem_r_valid(rm10v), .mem_r_ready(rm10rdy), .mem_r_data(rm10d),
    .mem_r_last(rm10l));
  behavioral_memory #(
    .SEG_COUNT(2),
    .SEG0_BASE(32'h01000000), .SEG0_BYTES(4096),
    .SEG1_BASE(32'h02000000), .SEG1_BYTES(65536)
  ) mem10 (
    .clk(clk), .rst_n(r10), .stall_mask(16'h0),
    .mem_cmd_valid(cm10v), .mem_cmd_ready(cm10rdy),
    .mem_cmd_write(cm10w), .mem_cmd_addr(cm10a), .mem_cmd_len(cm10l),
    .mem_w_valid(wm10v), .mem_w_ready(wm10rdy), .mem_w_data(wm10d),
    .mem_w_strb(wm10s), .mem_w_last(wm10l),
    .mem_r_valid(rm10v), .mem_r_ready(rm10rdy), .mem_r_data(rm10d),
    .mem_r_last(rm10l),
    .obs_cmd_valid(), .obs_cmd_write(), .obs_cmd_addr(), .obs_cmd_len(),
    .dbg_valid(1'b0), .dbg_ready(), .dbg_addr(32'h0), .dbg_data(),
    .dbg_w_valid(dbg10w), .dbg_w_addr(dbg10a), .dbg_w_data(dbg10d),
    .load_valid(1'b0), .load_ready(), .load_seg(2'd0),
    .load_bytes(32'd0), .load_file(""));

  initial begin
    #100000000;
    $display("WATCHDOG b1=%b b2=%b b3=%b b4=%b b6=%b b8=%b b9=%b b10=%b",
             b1, b2, b3, b4, b6, b8, b9, b10);
    $display("WATCHDOG e1=%b e2=%b e3=%b e4=%b e5=%b e6=%b e8=%b e9=%b e10=%b",
             e1, e2, e3, e4, e5, e6, e8, e9, e10);
    $display("WATCHDOG c1=%0d c2=%0d c3=%0d c4=%0d c5=%0d c6=%0d c8=%0d c9=%0d c10=%0d",
             c1, c2, c3, c4, c5, c6, c8, c9, c10);
    $display("WATCHDOG top1 sched_state=%0d sched_idx=%0d exec_state=%0d",
             top1.u_scheduler.state, top1.u_scheduler.current_desc_index,
             top1.u_executor.state);
    tb_fatal("tb_heatvit_errors watchdog");
  end

  // Progress trace: print when each case starts so a hang can be localized.
  always @(posedge s1) $display("case1 start");
  always @(posedge s2) $display("case2 start");
  always @(posedge s3) $display("case3 start");
  always @(posedge s4) $display("case4 start");
  always @(posedge s5) $display("case5 start");
  always @(posedge s6) $display("case6 start");
  always @(posedge s8) $display("case8 start");
  always @(posedge s9) $display("case9 start");
  always @(posedge s10) $display("case10 start");
  always @(posedge e1) $display("case1 error pulse code=%0d", c1);
  always @(posedge e2) $display("case2 error pulse code=%0d", c2);
  always @(posedge e3) $display("case3 error pulse code=%0d", c3);
  always @(posedge e4) $display("case4 error pulse code=%0d", c4);
  always @(posedge e5) $display("case5 error pulse code=%0d", c5);
  always @(posedge e6) $display("case6 error pulse code=%0d", c6);

  // Trace top4's abort/busy internals around the token-count error.
  always @(b4) $display("t%0t b4=%b sched_state=%0d abort_pend=%b exec_hold=%b exec_state=%0d exec_abort=%b",
                        $time, b4, top4.u_scheduler.state, top4.abort_pending,
                        top4.exec_abort_hold, top4.u_executor.state,
                        top4.exec_abort);
  always @(top4.exec_abort_done) $display("t%0t exec_abort_done=%b", $time,
                                          top4.exec_abort_done);
  always @(top4.u_executor.state) $display("t%0t top4 exec_state=%0d", $time,
                                           top4.u_executor.state);
  always @(top4.u_scheduler.state) $display("t%0t top4 sched_state=%0d", $time,
                                            top4.u_scheduler.state);

  initial begin
    // ---- Cases 1..3: validation errors. ----
    r1 = 1'b0; r2 = 1'b0; r3 = 1'b0;
    s1 = 1'b0; s2 = 1'b0; s3 = 1'b0;
    repeat (4) @(posedge clk);
    r1 = 1'b1; r2 = 1'b1; r3 = 1'b1;
    @(posedge clk);
    #1;

    s1 = 1'b1;
    @(posedge clk); #1;
    s1 = 1'b0;
    wait (e1);
    #1;
    if (c1 != 8'd1) begin
      $display("case1 error_code=%0d expected=1", c1);
      tb_fatal("case1 code mismatch");
    end
    wait (!b1);
    if (d1) tb_fatal("case1 done pulsed");

    s2 = 1'b1;
    @(posedge clk); #1;
    s2 = 1'b0;
    wait (e2);
    #1;
    if (c2 != 8'd2) begin
      $display("case2 error_code=%0d expected=2", c2);
      tb_fatal("case2 code mismatch");
    end
    wait (!b2);
    if (d2) tb_fatal("case2 done pulsed");

    s3 = 1'b1;
    @(posedge clk); #1;
    s3 = 1'b0;
    wait (e3);
    #1;
    if (c3 != 8'd3) begin
      $display("case3 error_code=%0d expected=3", c3);
      tb_fatal("case3 code mismatch");
    end
    wait (!b3);
    if (d3) tb_fatal("case3 done pulsed");

    // ---- Case 4: forced illegal token count. ----
    r4 = 1'b0; s4 = 1'b0;
    repeat (4) @(posedge clk);
    r4 = 1'b1;
    @(posedge clk);
    #1;
    force top4.u_scheduler.exec_next_token_count = 8'd250;
    s4 = 1'b1;
    @(posedge clk); #1;
    s4 = 1'b0;
    wait (e4);
    #1;
    if (c4 != 8'd4) begin
      $display("case4 error_code=%0d expected=4", c4);
      tb_fatal("case4 code mismatch");
    end
    wait (!b4);
    if (d4) tb_fatal("case4 done pulsed");
    release top4.u_scheduler.exec_next_token_count;

    // ---- Case 5: forced read-last protocol violation. ----
    r5 = 1'b0; s5 = 1'b0;
    repeat (4) @(posedge clk);
    r5 = 1'b1;
    @(posedge clk);
    #1;
    force mem5.mem_r_last = 1'b0;
    s5 = 1'b1;
    @(posedge clk); #1;
    s5 = 1'b0;
    wait (e5);
    #1;
    if (c5 != 8'd5) begin
      $display("case5 error_code=%0d expected=5", c5);
      tb_fatal("case5 code mismatch");
    end
    wait (!b5);
    if (d5) tb_fatal("case5 done pulsed");
    release mem5.mem_r_last;

    // ---- Case 6: forced softmax zero-sum. ----
    r6 = 1'b0; s6 = 1'b0;
    repeat (4) @(posedge clk);
    r6 = 1'b1;
    @(posedge clk);
    #1;
    force top6.u_executor.sm_div_div_zero = 1'b1;
    s6 = 1'b1;
    @(posedge clk); #1;
    s6 = 1'b0;
    wait (e6);
    #1;
    if (c6 != 8'd6) begin
      $display("case6 error_code=%0d expected=6", c6);
      tb_fatal("case6 code mismatch");
    end
    wait (!b6);
    if (d6) tb_fatal("case6 done pulsed");
    release top6.u_executor.sm_div_div_zero;

    // ---- Case 7: busy start. ----
    r1 = 1'b0; s1 = 1'b0;
    repeat (4) @(posedge clk);
    r1 = 1'b1;
    @(posedge clk);
    #1;
    s1 = 1'b1;
    @(posedge clk); #1;
    s1 = 1'b0;
    wait (b1);
    s1 = 1'b1;
    @(posedge clk); #1;
    s1 = 1'b0;
    wait (e1);
    #1;
    if (c1 != 8'd7) begin
      $display("case7 error_code=%0d expected=7", c1);
      tb_fatal("case7 code mismatch");
    end
    wait (!b1);
    if (d1) tb_fatal("case7 done pulsed");

    // ---- Case 8: head zero denominator warning; latch cleared on the
    // next legal start. ----
    begin
      int c;
      r8 = 1'b0; s8 = 1'b0;
      repeat (4) @(posedge clk);
      r8 = 1'b1;
      @(posedge clk);
      #1;
      // Keep scores: all 32768 across the three heads.
      for (c = 0; c < 3 * 196; c++) begin
        dbg8w = 1'b1; dbg8a = 32'h02000000 + c * 4 + 0; dbg8d = 8'h00;
        @(posedge clk); #1; dbg8w = 1'b0;
        dbg8w = 1'b1; dbg8a = 32'h02000000 + c * 4 + 1; dbg8d = 8'h80;
        @(posedge clk); #1; dbg8w = 1'b0;
        dbg8w = 1'b1; dbg8a = 32'h02000000 + c * 4 + 2; dbg8d = 8'h00;
        @(posedge clk); #1; dbg8w = 1'b0;
        dbg8w = 1'b1; dbg8a = 32'h02000000 + c * 4 + 3; dbg8d = 8'h00;
        @(posedge clk); #1; dbg8w = 1'b0;
      end
      s8 = 1'b1;
      @(posedge clk); #1;
      s8 = 1'b0;
      wait (d8);
      #1;
      if (w8[0] != 1'b1) begin
        $display("case8 warning_flags=%b expected bit0", w8);
        tb_fatal("case8 warning mismatch");
      end
      if (e8) begin
        $display("case8 error_code=%0d", c8);
        tb_fatal("case8 errored");
      end
      s8 = 1'b1;
      @(posedge clk); #1;
      s8 = 1'b0;
      @(posedge clk);
      #1;
      if (w8[0] != 1'b0) begin
        $display("case8 latch not cleared: %b", w8);
        tb_fatal("case8 latch clear failed");
      end
    end

    // ---- Case 9: package zero denominator warning. ----
    r9 = 1'b0; s9 = 1'b0;
    repeat (4) @(posedge clk);
    r9 = 1'b1;
    @(posedge clk);
    #1;
    s9 = 1'b1;
    @(posedge clk); #1;
    s9 = 1'b0;
    wait (d9);
    #1;
    if (w9[1] != 1'b1) begin
      $display("case9 warning_flags=%b expected bit1", w9);
      tb_fatal("case9 warning mismatch");
    end
    if (e9) begin
      $display("case9 error_code=%0d", c9);
      tb_fatal("case9 errored");
    end

    // ---- Case 10: LayerNorm negative variance warning. ----
    begin
      int c;
      r10 = 1'b0; s10 = 1'b0;
      repeat (4) @(posedge clk);
      r10 = 1'b1;
      @(posedge clk);
      #1;
      // Crafted vector: 95 x 127 then 97 x 90 at x_scale=-23.
      for (c = 0; c < 95; c++) begin
        dbg10w = 1'b1; dbg10a = 32'h02000000 + c; dbg10d = 8'd127;
        @(posedge clk); #1; dbg10w = 1'b0;
      end
      for (c = 95; c < 192; c++) begin
        dbg10w = 1'b1; dbg10a = 32'h02000000 + c; dbg10d = 8'd90;
        @(posedge clk); #1; dbg10w = 1'b0;
      end
      for (c = 0; c < 192; c++) begin
        dbg10w = 1'b1; dbg10a = 32'h01000000 + c; dbg10d = 8'd64;
        @(posedge clk); #1; dbg10w = 1'b0;
      end
      for (c = 0; c < 192; c++) begin
        dbg10w = 1'b1; dbg10a = 32'h01000000 + 192 + c; dbg10d = 8'd0;
        @(posedge clk); #1; dbg10w = 1'b0;
      end
      s10 = 1'b1;
      @(posedge clk); #1;
      s10 = 1'b0;
      wait (d10);
      #1;
      if (w10[2] != 1'b1) begin
        $display("case10 warning_flags=%b expected bit2", w10);
        tb_fatal("case10 warning mismatch");
      end
      if (e10) begin
        $display("case10 error_code=%0d", c10);
        tb_fatal("case10 errored");
      end
    end

    $display("TEST_PASS tb_heatvit_errors");
    $finish;
  end

endmodule
