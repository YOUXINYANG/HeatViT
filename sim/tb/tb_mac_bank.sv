`timescale 1ns / 1ps

// Task 3: per-cycle 8x8 signed MAC bank.
//
// Directed cases check outer-product accumulation, row/column masks, clear
// priority, signed extremes and the unsigned-src0 mode. A generated vector
// sweep covers random masks and K=768 worst-case accumulation, all compared
// against the Python integer reference.
module tb_mac_bank;
  import heatvit_pkg::*;
  import tb_pkg::*;

  localparam int VEC_SLOTS = 65536;
  localparam string VEC_PATH = "sim/vectors/gemm/mac_bank.mem";

  logic        clk   = 1'b0;
  logic        rst_n = 1'b0;

  logic        clear_accum;
  logic        accum_valid;
  logic [7:0]  a_lane [0:7];
  heatvit_s8_t b_lane [0:7];
  logic        a_unsigned;
  logic [7:0]  row_mask;
  logic [7:0]  col_mask;
  heatvit_s32_t mac_accum [0:7][0:7];
  logic        accum_done;

  heatvit_mac_bank dut (
    .clk         (clk),
    .rst_n       (rst_n),
    .clear_accum (clear_accum),
    .accum_valid (accum_valid),
    .a_lane      (a_lane),
    .b_lane      (b_lane),
    .a_unsigned  (a_unsigned),
    .row_mask    (row_mask),
    .col_mask    (col_mask),
    .accum       (mac_accum),
    .accum_done  (accum_done)
  );

  always #5 clk = ~clk;

  logic [7:0]  case_a [8];
  logic [7:0]  case_b [8];
  heatvit_s32_t case_exp [8][8];
  logic [63:0] vec [VEC_SLOTS];
  int i;
  int r;
  int c;
  int k;
  int total_words;
  int base;
  int cases_done;
  heatvit_s32_t want;
  heatvit_s32_t got;

  task automatic do_clear();
    clear_accum = 1'b1;
    accum_valid = 1'b0;
    @(posedge clk);
    #1;
    clear_accum = 1'b0;
    if (!accum_done) tb_fatal("mac bank accum_done missing after clear");
    for (r = 0; r < 8; r++)
      for (c = 0; c < 8; c++)
        if (mac_accum[r][c] !== 32'sd0)
          tb_fatal("mac bank not zero after clear");
  endtask

  task automatic check_all(input heatvit_s32_t exp [8][8]);
    for (r = 0; r < 8; r++) begin
      for (c = 0; c < 8; c++) begin
        if (mac_accum[r][c] !== exp[r][c]) begin
          $display("accum[%0d][%0d] mismatch: got=%0d expected=%0d",
                   r, c, mac_accum[r][c], exp[r][c]);
          tb_fatal("mac bank accumulator mismatch");
        end
      end
    end
  endtask

  initial begin
    clear_accum = 1'b0;
    accum_valid = 1'b0;
    a_unsigned  = 1'b0;
    row_mask    = 8'hff;
    col_mask    = 8'hff;
    for (i = 0; i < 8; i++) begin
      a_lane[i] = 8'd0;
      b_lane[i] = 8'sd0;
    end

    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
    #1;
    for (r = 0; r < 8; r++)
      for (c = 0; c < 8; c++)
        if (mac_accum[r][c] !== 32'sd0)
          tb_fatal("accumulator not zero after reset");

    // Directed outer-product accumulation over two different K cycles.
    do_clear();
    a_lane[0] = 8'd1; a_lane[1] = 8'd2; a_lane[2] = 8'd3; a_lane[3] = 8'd4;
    a_lane[4] = 8'd5; a_lane[5] = 8'd6; a_lane[6] = 8'd7; a_lane[7] = 8'd8;
    b_lane[0] = 8'sd1; b_lane[1] = -8'sd1; b_lane[2] = 8'sd2; b_lane[3] = -8'sd2;
    b_lane[4] = 8'sd3; b_lane[5] = -8'sd3; b_lane[6] = 8'sd4; b_lane[7] = -8'sd4;
    accum_valid = 1'b1;
    @(posedge clk);
    #1;
    a_lane[0] = 8'd1; a_lane[1] = 8'd1; a_lane[2] = 8'd1; a_lane[3] = 8'd1;
    a_lane[4] = 8'd1; a_lane[5] = 8'd1; a_lane[6] = 8'd1; a_lane[7] = 8'd1;
    b_lane[0] = 8'sd2; b_lane[1] = 8'sd2; b_lane[2] = 8'sd2; b_lane[3] = 8'sd2;
    b_lane[4] = 8'sd2; b_lane[5] = 8'sd2; b_lane[6] = 8'sd2; b_lane[7] = 8'sd2;
    @(posedge clk);
    #1;
    accum_valid = 1'b0;
    case_exp[0][0] = 1 + 2;   case_exp[0][1] = -1 + 2;
    case_exp[0][2] = 2 + 2;   case_exp[0][3] = -2 + 2;
    case_exp[0][4] = 3 + 2;   case_exp[0][5] = -3 + 2;
    case_exp[0][6] = 4 + 2;   case_exp[0][7] = -4 + 2;
    case_exp[1][0] = 2 + 2;   case_exp[1][1] = -2 + 2;
    case_exp[1][2] = 4 + 2;   case_exp[1][3] = -4 + 2;
    case_exp[1][4] = 6 + 2;   case_exp[1][5] = -6 + 2;
    case_exp[1][6] = 8 + 2;   case_exp[1][7] = -8 + 2;
    case_exp[2][0] = 3 + 2;   case_exp[2][1] = -3 + 2;
    case_exp[2][2] = 6 + 2;   case_exp[2][3] = -6 + 2;
    case_exp[2][4] = 9 + 2;   case_exp[2][5] = -9 + 2;
    case_exp[2][6] = 12 + 2;  case_exp[2][7] = -12 + 2;
    case_exp[3][0] = 4 + 2;   case_exp[3][1] = -4 + 2;
    case_exp[3][2] = 8 + 2;   case_exp[3][3] = -8 + 2;
    case_exp[3][4] = 12 + 2;  case_exp[3][5] = -12 + 2;
    case_exp[3][6] = 16 + 2;  case_exp[3][7] = -16 + 2;
    case_exp[4][0] = 5 + 2;   case_exp[4][1] = -5 + 2;
    case_exp[4][2] = 10 + 2;  case_exp[4][3] = -10 + 2;
    case_exp[4][4] = 15 + 2;  case_exp[4][5] = -15 + 2;
    case_exp[4][6] = 20 + 2;  case_exp[4][7] = -20 + 2;
    case_exp[5][0] = 6 + 2;   case_exp[5][1] = -6 + 2;
    case_exp[5][2] = 12 + 2;  case_exp[5][3] = -12 + 2;
    case_exp[5][4] = 18 + 2;  case_exp[5][5] = -18 + 2;
    case_exp[5][6] = 24 + 2;  case_exp[5][7] = -24 + 2;
    case_exp[6][0] = 7 + 2;   case_exp[6][1] = -7 + 2;
    case_exp[6][2] = 14 + 2;  case_exp[6][3] = -14 + 2;
    case_exp[6][4] = 21 + 2;  case_exp[6][5] = -21 + 2;
    case_exp[6][6] = 28 + 2;  case_exp[6][7] = -28 + 2;
    case_exp[7][0] = 8 + 2;   case_exp[7][1] = -8 + 2;
    case_exp[7][2] = 16 + 2;  case_exp[7][3] = -16 + 2;
    case_exp[7][4] = 24 + 2;  case_exp[7][5] = -24 + 2;
    case_exp[7][6] = 32 + 2;  case_exp[7][7] = -32 + 2;
    check_all(case_exp);

    // Row/column masks leave every masked-out lane at zero.
    do_clear();
    row_mask = 8'h0f;
    col_mask = 8'h03;
    for (i = 0; i < 8; i++) begin
      a_lane[i] = 8'd3;
      b_lane[i] = 8'sd4;
    end
    accum_valid = 1'b1;
    @(posedge clk);
    #1;
    accum_valid = 1'b0;
    row_mask = 8'hff;
    col_mask = 8'hff;
    for (r = 0; r < 8; r++) begin
      for (c = 0; c < 8; c++) begin
        if ((r < 4) && (c < 2)) case_exp[r][c] = 32'sd12;
        else case_exp[r][c] = 32'sd0;
      end
    end
    check_all(case_exp);

    // Clear has priority over a simultaneous valid accumulation.
    clear_accum = 1'b1;
    accum_valid = 1'b1;
    for (i = 0; i < 8; i++) begin
      a_lane[i] = 8'd7;
      b_lane[i] = 8'sd8;
    end
    @(posedge clk);
    #1;
    clear_accum = 1'b0;
    accum_valid = 1'b0;
    for (r = 0; r < 8; r++)
      for (c = 0; c < 8; c++)
        if (mac_accum[r][c] !== 32'sd0)
          tb_fatal("clear did not win over accum_valid");

    // Signed extremes and unsigned-src0 mode.
    do_clear();
    a_unsigned = 1'b0;
    for (i = 0; i < 8; i++) begin
      a_lane[i] = 8'sd127;
      b_lane[i] = 8'sd127;
    end
    accum_valid = 1'b1;
    @(posedge clk);
    #1;
    accum_valid = 1'b0;
    if (mac_accum[0][0] !== 32'sd16129) tb_fatal("127*127 mismatch");

    do_clear();
    a_unsigned = 1'b1;
    for (i = 0; i < 8; i++) begin
      a_lane[i] = 8'h80;
      b_lane[i] = -8'sd128;
    end
    accum_valid = 1'b1;
    @(posedge clk);
    #1;
    accum_valid = 1'b0;
    if (mac_accum[0][0] !== -32'sd16384) begin
      $display("unsigned 128*-128 got %0d", mac_accum[0][0]);
      tb_fatal("unsigned-src0 product mismatch");
    end

    // Generated sweep, including K=768 worst-case accumulation.
    $readmemh(VEC_PATH, vec);
    total_words = vec[0][15:0];
    if (total_words <= 0) begin
      $display("could not read %s", VEC_PATH);
      tb_fatal("missing mac bank vectors");
    end
    cases_done = 0;
    for (base = 1; base <= total_words; base += 35) begin
      for (i = 0; i < 8; i++) begin
        case_a[i] = vec[base][8*i +: 8];
        case_b[i] = $signed(vec[base + 1][8*i +: 8]);
      end
      row_mask   = vec[base + 2][7:0];
      col_mask   = vec[base + 2][15:8];
      a_unsigned = vec[base + 2][16];
      k          = vec[base + 2][30:17];
      if (k <= 0) begin
        $display("bad K=%0d at word %0d", k, base);
        tb_fatal("mac bank vector framing error");
      end
      for (r = 0; r < 8; r++) begin
        for (c = 0; c < 8; c++) begin
          case_exp[r][c] = $signed(
              vec[base + 3 + (r * 8 + c) / 2][32 * ((r * 8 + c) % 2) +: 32]);
        end
      end
      do_clear();
      for (i = 0; i < 8; i++) begin
        a_lane[i] = case_a[i];
        b_lane[i] = case_b[i];
      end
      accum_valid = 1'b1;
      repeat (k) begin
        @(posedge clk);
        #1;
      end
      accum_valid = 1'b0;
      check_all(case_exp);
      cases_done++;
    end
    if (cases_done < 8) tb_fatal("fewer than 8 mac bank vector cases checked");

    $display("TEST_PASS tb_mac_bank");
    $finish;
  end

endmodule
