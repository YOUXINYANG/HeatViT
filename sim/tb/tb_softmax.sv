`timescale 1ns / 1ps

module tb_softmax;
  import heatvit_pkg::*;
  import tb_pkg::*;

  localparam int VEC_SLOTS = 65536;
  localparam string VEC_PATH = "sim/vectors/softmax/softmax.mem";

  logic          clk   = 1'b0;
  logic          rst_n = 1'b0;

  // Selector wrapper control/data.
  logic           sel_start;
  logic           sel_busy;
  logic           sel_done;
  logic [7:0]     sel_row_len;
  logic           sel_s_valid;
  logic           sel_s_ready;
  heatvit_q8_16_t sel_s_data;
  logic           sel_m_valid;
  logic           sel_m_ready;
  heatvit_uq0_16_t sel_m_data;
  logic           sel_m_last;
  logic           sel_error_zero_sum;

  // Attention wrapper control/data.
  logic           att_start;
  logic           att_busy;
  logic           att_done;
  logic [7:0]     att_row_len;
  logic           att_s_valid;
  logic           att_s_ready;
  heatvit_q8_16_t att_s_data;
  logic           att_m_valid;
  logic           att_m_ready;
  logic [7:0]     att_m_data;
  logic           att_m_last;
  logic           att_error_zero_sum;

  // Shared divider arbiter clients.
  logic [2:0]  a_req_valid;
  logic [2:0]  a_req_ready;
  logic [63:0] a_num [2:0];
  logic [63:0] a_den [2:0];
  logic [2:0]  a_rsp_valid;
  logic [63:0] a_quot [2:0];
  logic [63:0] a_rem [2:0];
  logic [2:0]  a_div_zero;

  heatvit_softmax_selector dut_sel (
    .clk            (clk),
    .rst_n          (rst_n),
    .start          (sel_start),
    .busy           (sel_busy),
    .done           (sel_done),
    .row_len        (sel_row_len),
    .s_valid        (sel_s_valid),
    .s_ready        (sel_s_ready),
    .s_data         (sel_s_data),
    .div_req_valid  (a_req_valid[0]),
    .div_req_ready  (a_req_ready[0]),
    .div_num        (a_num[0]),
    .div_den        (a_den[0]),
    .div_rsp_valid  (a_rsp_valid[0]),
    .div_quot       (a_quot[0]),
    .div_rem        (a_rem[0]),
    .div_div_zero   (a_div_zero[0]),
    .m_valid        (sel_m_valid),
    .m_ready        (sel_m_ready),
    .m_data         (sel_m_data),
    .m_last         (sel_m_last),
    .error_zero_sum (sel_error_zero_sum)
  );

  heatvit_softmax_attention dut_att (
    .clk            (clk),
    .rst_n          (rst_n),
    .start          (att_start),
    .busy           (att_busy),
    .done           (att_done),
    .row_len        (att_row_len),
    .s_valid        (att_s_valid),
    .s_ready        (att_s_ready),
    .s_data         (att_s_data),
    .div_req_valid  (a_req_valid[1]),
    .div_req_ready  (a_req_ready[1]),
    .div_num        (a_num[1]),
    .div_den        (a_den[1]),
    .div_rsp_valid  (a_rsp_valid[1]),
    .div_quot       (a_quot[1]),
    .div_rem        (a_rem[1]),
    .div_div_zero   (a_div_zero[1]),
    .m_valid        (att_m_valid),
    .m_ready        (att_m_ready),
    .m_data         (att_m_data),
    .m_last         (att_m_last),
    .error_zero_sum (att_error_zero_sum)
  );

  heatvit_div_arbiter #(.NUM_W(64), .DEN_W(64), .QUOT_W(64)) dut_arb (
    .clk       (clk),
    .rst_n     (rst_n),
    .req_valid (a_req_valid),
    .req_ready (a_req_ready),
    .num       (a_num),
    .den       (a_den),
    .rsp_valid (a_rsp_valid),
    .quot      (a_quot),
    .rem       (a_rem),
    .div_zero  (a_div_zero)
  );

  always #5 clk = ~clk;

  // 16-bit backpressure LFSR; bit 0 gives a ~50% random output stall.
  logic [15:0] lfsr = 16'hACE1;
  always_ff @(posedge clk) lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};

  heatvit_q8_16_t  row_x [197];
  heatvit_uq0_16_t exp_sel [197];
  logic [7:0]      exp_att [197];

  task automatic run_selector_row(input int len, input int stall_cycles);
    int sent = 0;
    int got  = 0;
    sel_row_len = 8'(len);
    sel_start = 1'b1;
    @(posedge clk);
    #1;
    sel_start = 1'b0;
    sel_s_valid = 1'b1;
    while (sent < len) begin
      sel_s_data = row_x[sent];
      wait (sel_s_ready);
      @(posedge clk);
      #1;
      sent++;
    end
    sel_s_valid = 1'b0;

    wait (sel_m_valid);
    sel_m_ready = 1'b0;
    repeat (stall_cycles) begin
      @(posedge clk);
      #1;
      if (!sel_m_valid) tb_fatal("selector: output lost during stall");
      if (sel_m_data !== exp_sel[0]) tb_fatal("selector: output unstable during stall");
    end
    while (got < len) begin
      sel_m_ready = lfsr[0];
      if (sel_m_valid && sel_m_ready) begin
        if (sel_m_data !== exp_sel[got]) begin
          $display("selector beat[%0d] mismatch: got=%0d expected=%0d x=%0d",
                   got, sel_m_data, exp_sel[got], row_x[got]);
          tb_fatal("selector output mismatch");
        end
        if (((got == len - 1) ? 1'b1 : 1'b0) !== sel_m_last)
          tb_fatal("selector: m_last framing mismatch");
        got++;
      end
      @(posedge clk);
      #1;
    end
    sel_m_ready = 1'b0;
    wait (sel_done);
    #1;
    if (sel_busy) tb_fatal("selector: busy not cleared after done");
    if (sel_error_zero_sum) tb_fatal("selector: spurious zero-sum error");
  endtask

  task automatic run_attention_row(input int len, input int stall_cycles);
    int sent = 0;
    int got  = 0;
    att_row_len = 8'(len);
    att_start = 1'b1;
    @(posedge clk);
    #1;
    att_start = 1'b0;
    att_s_valid = 1'b1;
    while (sent < len) begin
      att_s_data = row_x[sent];
      wait (att_s_ready);
      @(posedge clk);
      #1;
      sent++;
    end
    att_s_valid = 1'b0;

    wait (att_m_valid);
    att_m_ready = 1'b0;
    repeat (stall_cycles) begin
      @(posedge clk);
      #1;
      if (!att_m_valid) tb_fatal("attention: output lost during stall");
      if (att_m_data !== exp_att[0]) tb_fatal("attention: output unstable during stall");
    end
    while (got < len) begin
      att_m_ready = lfsr[0];
      if (att_m_valid && att_m_ready) begin
        if (att_m_data !== exp_att[got]) begin
          $display("attention beat[%0d] mismatch: got=%0d expected=%0d",
                   got, att_m_data, exp_att[got]);
          tb_fatal("attention output mismatch");
        end
        if (((got == len - 1) ? 1'b1 : 1'b0) !== att_m_last)
          tb_fatal("attention: m_last framing mismatch");
        got++;
      end
      @(posedge clk);
      #1;
    end
    att_m_ready = 1'b0;
    wait (att_done);
    #1;
    if (att_busy) tb_fatal("attention: busy not cleared after done");
    if (att_error_zero_sum) tb_fatal("attention: spurious zero-sum error");
  endtask

  logic [56:0] vec [VEC_SLOTS];
  int total_words;
  int pos;
  int len;
  int j;
  int rows_done;

  initial begin
    sel_start = 1'b0;
    sel_row_len = 8'd0;
    sel_s_valid = 1'b0;
    sel_s_data = 24'sd0;
    sel_m_ready = 1'b0;
    att_start = 1'b0;
    att_row_len = 8'd0;
    att_s_valid = 1'b0;
    att_s_data = 24'sd0;
    att_m_ready = 1'b0;
    for (j = 0; j < 197; j++) begin
      row_x[j]  = 24'sd0;
      exp_sel[j] = 17'd0;
      exp_att[j] = 8'd0;
    end

    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
    #1;

    // Directed checks: selector equal pair halves, attention unit row.
    row_x[0] = 24'sd0;
    row_x[1] = 24'sd0;
    exp_sel[0] = 17'd32768;
    exp_sel[1] = 17'd32768;
    exp_att[0] = 8'd128;
    exp_att[1] = 8'd128;
    run_selector_row(2, 4);
    run_attention_row(2, 4);

    row_x[0] = 24'sd12345;
    exp_sel[0] = 17'd65536;
    exp_att[0] = 8'd255;
    run_selector_row(1, 3);
    run_attention_row(1, 3);

    // Vector-driven sweep.
    $readmemh(VEC_PATH, vec);
    total_words = vec[0][15:0];
    if (total_words <= 0) begin
      $display("could not read %s", VEC_PATH);
      tb_fatal("missing softmax vectors");
    end
    pos = 1;
    rows_done = 0;
    while (pos <= total_words) begin
      len = vec[pos][7:0];
      if (len <= 0 || len > 197) begin
        $display("bad row_len=%0d at word %0d", len, pos);
        tb_fatal("softmax vector framing error");
      end
      for (j = 0; j < len; j++) begin
        row_x[j]  = $signed(vec[pos + j][31:8]);
        exp_att[j] = vec[pos + j][39:32];
        exp_sel[j] = vec[pos + j][56:40];
      end
      run_selector_row(len, 0);
      run_attention_row(len, 0);
      rows_done++;
      pos += len;
    end
    if (rows_done < 256) tb_fatal("softmax: fewer than 256 rows checked");

    $display("TEST_PASS tb_softmax");
    $finish;
  end

endmodule
