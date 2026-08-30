// Descriptor-driven unified int8 GEMM engine: TH=3, TI=8, TO=8.
//
// One command owns the single-outstanding memory master. Each (m tile, column
// group) loads A, up to three B tiles and the Bias slice, accumulates K
// cycles across three 8x8 MAC banks, then writes int8 or int32 results back.
// Normal mode maps banks to three adjacent N tiles; Head mode fixes bank
// 0/1/2 to Attention head 0/1/2 with per-head A/B addressing. All external
// access goes through heatvit_addr_guard before a command is issued.
module heatvit_gemm_engine
  import heatvit_pkg::*;
#(
  parameter int A_BYTES   = 6144,
  parameter int B_BYTES   = 6144,
  parameter int BIAS_COLS = 24
)(
  input  logic         clk,
  input  logic         rst_n,
  input  logic         cmd_valid,
  output logic         cmd_ready,
  input  heatvit_desc_t desc,
  input  logic [31:0]  input_base,
  input  logic [31:0]  input_bytes,
  input  logic [31:0]  weight_base,
  input  logic [31:0]  weight_bytes,
  input  logic [31:0]  scratch_base,
  input  logic [31:0]  scratch_bytes,
  input  logic [31:0]  output_base,
  input  logic [31:0]  output_bytes,
  output logic         busy,
  output logic         done,
  output logic         error_valid,
  output logic [7:0]   error_code,
  output logic [2:0][31:0] mac_active_cycles,
  // Locked external memory client interface.
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

  localparam int AW = $clog2(A_BYTES);

  typedef enum logic [4:0] {
    S_IDLE,
    S_CHECK,
    S_ERROR,
    S_LOAD_SETUP,
    S_LOAD_REQ,
    S_LOAD_RECV,
    S_LOAD_SCAT,
    S_COMPUTE_PRE,
    S_COMPUTE_WARM,
    S_COMPUTE_ACC,
    S_GELU_SETTLE,
    S_GELU_NEXT,
    S_GELU_DRAIN,
    S_PLAN_SETTLE,
    S_PLAN_NEXT,
    S_PLAN_WAIT,
    S_WB_NEXT,
    S_WB_REQ,
    S_WB_COMPOSE,
    S_WB_BEAT,
    S_DONE
  } state_t;

  state_t state;

  // Latched command configuration.
  heatvit_desc_t desc_reg;
  logic [15:0]   m_eff;
  logic [15:0]   n_eff;
  logic [15:0]   k_eff;
  logic [15:0]   nph;
  logic          head_mode;
  logic          rhs_transpose;
  logic          bias_en;
  logic          out_int32;
  logic          a_unsigned;
  logic          src0_cand_major;
  logic [31:0]   src0_abs;
  logic [31:0]   src1_abs;
  logic [31:0]   bias_abs;
  logic [31:0]   dst_abs;
  logic [31:0]   src0_region_base;
  logic [31:0]   src0_region_bytes;
  logic [31:0]   src1_region_base;
  logic [31:0]   src1_region_bytes;
  logic [31:0]   bias_region_base;
  logic [31:0]   bias_region_bytes;
  logic [31:0]   dst_region_base;
  logic [31:0]   dst_region_bytes;
  heatvit_scale_t src0_scale;
  heatvit_scale_t src1_scale;
  heatvit_scale_t dst_scale;
  logic [2:0]   post_op;

  // Tile / load / writeback counters.
  logic [15:0] m0;
  logic [15:0] n0;
  logic [1:0]  ld_kind;
  logic [1:0]  ld_bank;
  logic [15:0] ld_idx;
  logic [2:0]  ld_fill_bank;
  logic [31:0] ld_addr;
  logic [15:0] ld_len;
  logic [2:0]  ld_e;
  logic [15:0] ld_w;
  logic [63:0] ld_beat_data;
  logic [6:0]  ld_bi;
  logic [2:0]  scat_j;
  logic [15:0] kc;
  logic [1:0]  wb_b;
  logic [3:0]  wb_r;
  logic [31:0] wb_addr;
  logic [15:0] wb_len;
  logic [2:0]  wb_e;
  logic [5:0]  wb_w;
  logic [2:0]  wb_bi;
  // P7-5: write-back composition uses three registered stages before the
  // staging RAM: address decode, source-value selection, then requant/pack.
  // The added value register cuts the route-dominated MAC-bank-to-RAM cone
  // between the scattered accumulator mux and the 128-bit requant logic.
  logic [7:0]  wb_hit_a;        // stage A comb: per-byte in-range flags
  logic [4:0]  wb_c_a [0:7];    // stage A comb: output column index per byte
  logic [1:0]  wb_lane_a [0:7]; // stage A comb: byte lane within 32-bit word
  logic [1:0]  wb_kind_a [0:7]; // 0=int8, 1=gelu, 2=int32, 3=plan
  logic [7:0]  wb_hit_r;
  logic [4:0]  wb_c_r [0:7];
  logic [1:0]  wb_lane_r [0:7];
  logic [1:0]  wb_kind_r [0:7];
  logic        wb_comp_valid_r;   // stage B has a decoded beat to select
  logic [2:0]  wb_comp_idx_r;
  logic [7:0]  wb_value_hit_r;
  logic [1:0]  wb_value_lane_r [0:7];
  logic [1:0]  wb_value_kind_r [0:7];
  logic signed [32:0] wb_sum_c [0:7];
  logic signed [23:0] wb_gelu_c [0:7];
  logic [16:0] wb_plan_c [0:7];
  logic signed [32:0] wb_sum_r [0:7];
  logic signed [23:0] wb_gelu_r [0:7];
  logic [16:0] wb_plan_r [0:7];
  logic        wb_value_valid_r;  // stage C has registered values to scale
  logic [2:0]  wb_value_idx_r;
  logic [71:0] wb_ram_wdata;      // {64-bit data, 8-bit strb}
  logic [2:0]  wb_ram_raddr;
  logic [71:0] wb_ram_rdata;
  logic        wb_valid_r;
  logic [1:0]  wb_exit_pipe;      // drain select/scale/write-port stages
  logic [2:0]  wb_rd;             // streaming read index
  logic [2:0]  wb_last_idx;       // last beat index (wb_len-1, clamped to 3 bits)
  // Registered staging-RAM write port keeps the final scale/pack cone local
  // to its destination RAM cells.
  logic        wb_ram_we_r;
  logic [2:0]  wb_ram_waddr_r;
  logic [71:0] wb_ram_wdata_r;
  // Guard settle registers: the load/write-back address guards are deep
  // (multiplies + bounds checks), so the state machine pipelines each
  // window check into four phases (P7-5): stage 0 registers the address
  // multiply cone, stage 1 the cover/end/len 64-bit add chain, stage 2 the
  // addr-guard compare, stage 3 decides.  Each stage is register-to-
  // register and stays within the 100 MHz budget.
  logic        g_ok_r;
  logic [7:0]  g_code_r;
  logic [63:0] wb_addr64_r;    // wb stage 0: aligned window start
  logic [15:0] wb_len_c_r;     // wb stage 1: derived burst length
  logic [2:0]  wb_e_c_r;       // wb stage 0: captured byte offset
  logic [5:0]  wb_w_c_r;       // wb stage 0: captured useful width
  logic [31:0] wb_rbase_r;     // wb stage 0: captured dst region base
  logic [31:0] wb_rbytes_r;    // wb stage 0: captured dst region bytes
  logic [1:0]  wb_settle;      // wb guard pipeline phase
  logic [63:0] ld_aligned64_r; // ld stage 0: aligned window start
  logic [2:0]  ld_e_c_r;       // ld stage 0: captured byte offset
  logic [15:0] ld_w_c_r;       // ld stage 0: captured useful width
  logic [31:0] ld_rbase_r;     // ld stage 0: captured region base
  logic [31:0] ld_rbytes_r;    // ld stage 0: captured region bytes
  logic [15:0] ld_len_r;       // ld stage 1: derived burst length
  logic [2:0]  ld_fill_bank_r;
  logic [1:0]  ld_settle;      // ld guard pipeline phase
  // P7-5: S_CHECK registers the validation result before deciding, so the
  // deep descriptor -> addr_ok_all -> v_error cone no longer fans out to
  // the config-register clock enables.
  logic [7:0]  v_error_r;
  logic        v_shallow_r;
  logic        s_check_phase;
  logic [1:0]  gb;
  logic [2:0]  gr;
  logic [2:0]  gc;
  // Collect pointers for the pipelined GELU post-op: lanes are fed
  // back-to-back on (gb,gr,gc) and results are stored in feed order on
  // (cb,cr,cc), so the drain can overlap the feed of the same tile.
  logic [1:0]  cb;
  logic [2:0]  cr;
  logic [2:0]  cc;
  // Activation input pipeline.  The dynamic bank/row/column selection ends
  // at these registers; bias addition, scale conversion and saturation run
  // in the following stage before the GELU/PLAN unit samples the value.
  heatvit_s32_t act_acc_r;
  heatvit_s32_t act_bias_r;
  logic         act_fetch_valid_r;

  // Tile buffer interface.
  logic [12:0]  tb_fill_addr;
  logic [7:0]   tb_fill_data;
  logic         tb_fill_valid;
  logic [2:0]   tb_fill_bank;
  logic [12:0]  a_rd_addr [0:2];
  logic [63:0]  a_rd_data [0:2];
  logic [12:0]  b_rd_addr [0:2];
  logic [63:0]  b_rd_data [0:2];
  heatvit_s32_t bias_all [0:BIAS_COLS-1];

  heatvit_tile_buffer #(
    .A_BYTES(A_BYTES), .B_BYTES(B_BYTES), .BIAS_COLS(BIAS_COLS)
  ) u_tile (
    .clk         (clk),
    .rst_n       (rst_n),
    .fill_bank   (tb_fill_bank),
    .fill_valid  (tb_fill_valid),
    .fill_addr   (tb_fill_addr),
    .fill_data   (tb_fill_data),
    .a_rd_addr   (a_rd_addr),
    .a_rd_data   (a_rd_data),
    .b_rd_addr   (b_rd_addr),
    .b_rd_data   (b_rd_data),
    .bias_mem_out(bias_all)
  );

  // MAC banks.
  logic        clear_accum;
  logic        accum_valid;
  logic [7:0]  a_lane [0:2][0:7];
  heatvit_s8_t b_lane [0:2][0:7];
  logic [7:0]  row_mask;
  logic [7:0]  col_mask [0:2];
  heatvit_s32_t bank_accum [0:2][0:7][0:7];
  logic [2:0]  bank_accum_done;
  heatvit_q8_16_t gelu_buf [0:2][0:7][0:7];
  heatvit_uq0_16_t plan_buf [0:2][0:7][0:7];

  genvar g;
  generate
    for (g = 0; g < 3; g++) begin : gen_bank
      heatvit_mac_bank u_bank (
        .clk         (clk),
        .rst_n       (rst_n),
        .clear_accum (clear_accum),
        .accum_valid (accum_valid),
        .a_lane      (a_lane[g]),
        .b_lane      (b_lane[g]),
        .a_unsigned  (a_unsigned),
        .row_mask    (row_mask),
        .col_mask    (col_mask[g]),
        .accum       (bank_accum[g]),
        .accum_done  (bank_accum_done[g])
      );
    end
  endgenerate

  // Single sequential GELU post-op unit.
  logic        gelu_start;
  logic        gelu_busy;
  logic        gelu_done;
  heatvit_q8_16_t gelu_in;
  heatvit_q8_16_t gelu_out;
  // Shared registered Q8.16 input.  GELU lanes are fed back-to-back while
  // PLAN consumes the same pipeline serially.
  heatvit_q8_16_t act_q16_r;

  heatvit_gelu u_gelu (
    .clk    (clk),
    .rst_n  (rst_n),
    .start  (gelu_start),
    .busy   (gelu_busy),
    .done   (gelu_done),
    .x_in   (gelu_in),
    .y_out  (gelu_out)
  );

  // Single sequential PLAN sigmoid post-op unit (Q8.16 -> UQ0.16).
  logic        plan_start;
  logic        plan_busy;
  logic        plan_done;
  heatvit_q8_16_t plan_in;
  heatvit_uq0_16_t plan_out;

  heatvit_plan_sigmoid u_plan (
    .clk    (clk),
    .rst_n  (rst_n),
    .start  (plan_start),
    .busy   (plan_busy),
    .done   (plan_done),
    .x_in   (plan_in),
    .y_out  (plan_out)
  );

  // ------------------------------------------------------------------
  // P7-5: narrow bit-exact saturators / clamps.
  //
  // The original requant code ran on 128-bit containers, which the
  // synthesizer turned into ~65-CARRY4 shift/add cones (the P7-4 residual
  // violation).  Every value range proves a far narrower container (see the
  // width proofs below), so each saturation below is bit-exact with
  // sat_s8/sat_s32 of the equivalent 128-bit value.
  // ------------------------------------------------------------------
  function automatic heatvit_s8_t sat_s8_from_s33(input logic signed [32:0] v);
    if (v > 33'sd127)  return 8'sd127;
    if (v < -33'sd128) return -8'sd128;
    return heatvit_s8_t'(v[7:0]);
  endfunction

  function automatic heatvit_s8_t sat_s8_from_s24(input logic signed [23:0] v);
    if (v > 24'sd127)  return 8'sd127;
    if (v < -24'sd128) return -8'sd128;
    return heatvit_s8_t'(v[7:0]);
  endfunction

  function automatic heatvit_s8_t sat_s8_from_s40(input logic signed [39:0] v);
    if (v > 40'sd127)  return 8'sd127;
    if (v < -40'sd128) return -8'sd128;
    return heatvit_s8_t'(v[7:0]);
  endfunction

  function automatic heatvit_s32_t sat_s32_from_s33(input logic signed [32:0] v);
    if (v > 33'sd2147483647)  return 32'sd2147483647;
    if (v < -33'sd2147483648) return -32'sd2147483648;
    return heatvit_s32_t'(v[31:0]);
  endfunction

  function automatic heatvit_s32_t sat_s32_from_s64(input logic signed [63:0] v);
    if (v > 64'sd2147483647)  return 32'sd2147483647;
    if (v < -64'sd2147483648) return -32'sd2147483648;
    return heatvit_s32_t'(v[31:0]);
  endfunction

  function automatic heatvit_q8_16_t clamp_q8_16_from_s33(
    input logic signed [32:0] v
  );
    if (v > 33'sd8388607)  return 24'sd8388607;
    if (v < -33'sd8388608) return -24'sd8388608;
    return heatvit_q8_16_t'(v[23:0]);
  endfunction

  function automatic heatvit_q8_16_t clamp_q8_16_from_s55(
    input logic signed [54:0] v
  );
    if (v > 55'sd8388607)  return 24'sd8388607;
    if (v < -55'sd8388608) return -24'sd8388608;
    return heatvit_q8_16_t'(v[23:0]);
  endfunction

  // P7-5 narrow rewrite of activation_q16 (bit-exact width proof):
  //   wide = acc + bias is 33-bit exact (|wide| <= 2^32).  scale_to_exp
  //   takes src_exp = src0+src1 (6-bit wrap) and dst_exp = -16, so
  //   diff = -16 - src_exp in [-47..16].
  //   diff == 0 -> clamp24(wide)
  //   diff >  0 (<=16): round-away -> 34-bit cone -> clamp24
  //   diff <  0: k = -diff in [1..47]; wide<<k then clamp24.  k >= 23
  //     saturates (|wide| >= 1 -> |wide<<k| >= 2^23); k <= 22 -> 55-bit
  //     cone (32+22 = 54 < 55, exact).  The 128-bit left-shift overflow
  //     check in scale_to_exp_s128 is dead here: 32+47 < 127, so the
  //     shifted value always fits its container (identical in 55 bits).
  function automatic heatvit_q8_16_t activation_q16(
    input heatvit_s32_t accum_value,
    input heatvit_s32_t bias_value
  );
    logic signed [32:0] wide;
    logic signed [5:0]  src_exp;
    int diff;
    logic [5:0]  d;
    logic [33:0] mag_u;
    logic [33:0] rounded;
    logic signed [33:0] rsigned;
    wide = $signed({accum_value[31], accum_value}) +
           (bias_en ? $signed({bias_value[31], bias_value}) : 33'sd0);
    src_exp = src0_scale + src1_scale;
    diff = -16 - int'(src_exp);
    if (diff == 0) begin
      return clamp_q8_16_from_s33(wide);
    end else if (diff > 0) begin
      d = diff[5:0];  // diff in [1..16] -> d == diff
      mag_u = (wide < 33'sd0) ? (34'h200000000 - {1'b0, wide}) : {1'b0, wide};
      rounded = (mag_u + (34'd1 << (d - 6'd1))) >> d;
      rsigned = (wide < 33'sd0) ? -$signed(rounded) : $signed(rounded);
      return clamp_q8_16_from_s33(rsigned);
    end else begin
      if (-diff >= 23) begin
        return (wide == 33'sd0) ? 24'sd0 :
               ((wide < 33'sd0) ? -24'sd8388608 : 24'sd8388607);
      end else begin
        return clamp_q8_16_from_s55(
          $signed({{22{wide[32]}}, wide}) <<< (-diff));
      end
    end
  endfunction

  always_comb begin
    gelu_in = act_q16_r;
    plan_in = act_q16_r;
  end

  // Single-outstanding memory master.
  logic        req_valid;
  logic        req_ready;
  logic        req_write;
  logic [31:0] req_addr;
  logic [31:0] req_bytes;
  logic        req_w_valid;
  logic        req_w_ready;
  logic [63:0] req_w_data;
  logic [7:0]  req_w_strb;
  logic        req_w_last;
  logic        req_r_valid;
  logic        req_r_ready;
  logic        req_r_ready_r;
  logic [63:0] req_r_data;
  logic        req_r_last;
  logic        mm_done;
  logic        mm_perr;
  logic        mm_abort_done;

  heatvit_mem_master u_master (
    .clk            (clk),
    .rst_n          (rst_n),
    .req_valid      (req_valid),
    .req_ready      (req_ready),
    .req_write      (req_write),
    .req_addr       (req_addr),
    .req_bytes      (req_bytes),
    .req_w_valid    (req_w_valid),
    .req_w_ready    (req_w_ready),
    .req_w_data     (req_w_data),
    .req_w_strb     (req_w_strb),
    .req_w_last     (req_w_last),
    .req_r_valid    (req_r_valid),
    .req_r_ready    (req_r_ready),
    .req_r_data     (req_r_data),
    .req_r_last     (req_r_last),
    .abort          (1'b0),
    .done           (mm_done),
    .protocol_error (mm_perr),
    .abort_done     (mm_abort_done),
    .mem_cmd_valid  (mem_cmd_valid),
    .mem_cmd_ready  (mem_cmd_ready),
    .mem_cmd_write  (mem_cmd_write),
    .mem_cmd_addr   (mem_cmd_addr),
    .mem_cmd_len    (mem_cmd_len),
    .mem_w_valid    (mem_w_valid),
    .mem_w_ready    (mem_w_ready),
    .mem_w_data     (mem_w_data),
    .mem_w_strb     (mem_w_strb),
    .mem_w_last     (mem_w_last),
    .mem_r_valid    (mem_r_valid),
    .mem_r_ready    (mem_r_ready),
    .mem_r_data     (mem_r_data),
    .mem_r_last     (mem_r_last)
  );

  // Address guard.
  logic [31:0] g_base;
  logic [31:0] g_bytes;
  logic [31:0] g_addr;
  logic [15:0] g_len;
  logic        g_ok;
  logic [7:0]  g_code;

  heatvit_addr_guard u_guard (
    .region_base    (g_base),
    .region_bytes   (g_bytes),
    .cmd_addr       (g_addr),
    .cmd_len        (g_len),
    .addr_ok        (g_ok),
    .addr_error_code(g_code)
  );

  // Combinational validation and derived layout.
  logic [7:0]  v_error;
  logic [7:0]  v_error_shallow;
  logic [3:0]  a_rows;
  logic [3:0]  b_cols [0:2];
  logic [15:0] col_global [0:2];
  logic [15:0] col_local  [0:2];
  logic [15:0] col_stride;
  logic [15:0] b_count [0:2];

  logic [31:0] src0_region_base_c;
  logic [31:0] src0_region_bytes_c;
  logic [31:0] src1_region_base_c;
  logic [31:0] src1_region_bytes_c;
  logic [31:0] bias_region_base_c;
  logic [31:0] bias_region_bytes_c;
  logic [31:0] dst_region_base_c;
  logic [31:0] dst_region_bytes_c;
  logic [63:0] src0_abs64;
  logic [63:0] src1_abs64;
  logic [63:0] bias_abs64;
  logic [63:0] dst_abs64;
  logic [63:0] a_total64;
  logic [63:0] b_total64;
  logic [63:0] bias_total64;
  logic [63:0] dst_total64;
  logic [15:0] a_total_beats;
  logic [15:0] b_total_beats;
  logic [15:0] bias_total_beats;
  logic [15:0] dst_total_beats;
  logic        addr_ok_all;

  // Guard instances for the four full-tensor pre-checks.
  logic        ga_ok, gb_ok, gbi_ok, gd_ok;
  logic [7:0]  ga_code, gb_code, gbi_code, gd_code;
  heatvit_addr_guard u_guard_a (
    .region_base    (src0_region_base_c),
    .region_bytes   (src0_region_bytes_c),
    .cmd_addr       (src0_abs64[31:0]),
    .cmd_len        (a_total_beats),
    .addr_ok        (ga_ok),
    .addr_error_code(ga_code)
  );
  heatvit_addr_guard u_guard_b (
    .region_base    (src1_region_base_c),
    .region_bytes   (src1_region_bytes_c),
    .cmd_addr       (src1_abs64[31:0]),
    .cmd_len        (b_total_beats),
    .addr_ok        (gb_ok),
    .addr_error_code(gb_code)
  );
  heatvit_addr_guard u_guard_bias (
    .region_base    (bias_region_base_c),
    .region_bytes   (bias_region_bytes_c),
    .cmd_addr       (bias_abs64[31:0]),
    .cmd_len        (bias_total_beats),
    .addr_ok        (gbi_ok),
    .addr_error_code(gbi_code)
  );
  heatvit_addr_guard u_guard_dst (
    .region_base    (dst_region_base_c),
    .region_bytes   (dst_region_bytes_c),
    .cmd_addr       (dst_abs64[31:0]),
    .cmd_len        (dst_total_beats),
    .addr_ok        (gd_ok),
    .addr_error_code(gd_code)
  );

  // Load window math.
  logic [63:0] ld_addr64_c;
  logic [2:0]  ld_e_c;
  logic [15:0] ld_len_c;
  logic [15:0] ld_w_c;
  logic [2:0]  ld_fill_bank_c;
  logic [31:0] ld_region_base_c;
  logic [31:0] ld_region_bytes_c;

  // Writeback window math and beat composition.
  logic [63:0] wb_addr64_c;
  logic [2:0]  wb_e_c;
  logic [15:0] wb_len_c;
  logic [5:0]  wb_w_c;
  logic [63:0] wb_data_c;
  logic [7:0]  wb_strb_c;

  assign col_stride = head_mode ? nph : n_eff;

  always_comb begin
    a_rows = (m_eff - m0 >= 16'd8) ? 4'd8 : 4'(m_eff - m0);
    for (int b = 0; b < 3; b++) begin
      if (head_mode) begin
        col_global[b] = b * nph + n0;
        col_local[b]  = n0;
        if (nph >= n0 + 16'd8) b_cols[b] = 4'd8;
        else b_cols[b] = 4'(nph - n0);
      end else begin
        col_global[b] = n0 + 8 * b;
        col_local[b]  = n0 + 8 * b;
        if (n_eff >= n0 + 8 * b + 16'd8) b_cols[b] = 4'd8;
        else if (n_eff > n0 + 8 * b) b_cols[b] = 4'(n_eff - n0 - 8 * b);
        else b_cols[b] = 4'd0;
      end
      b_count[b] = rhs_transpose ? {12'd0, b_cols[b]} : k_eff;
    end
  end

  always_comb begin
    // src0 defaults to the Scratch region (activations); flag 11 selects the
    // Input region for image-origin tensors such as patch matrices.
    src0_region_base_c  = desc_reg.flags[FLAG_SRC0_INPUT] ? input_base : scratch_base;
    src0_region_bytes_c = desc_reg.flags[FLAG_SRC0_INPUT] ? input_bytes : scratch_bytes;
    src1_region_base_c  = desc_reg.flags[FLAG_SRC1_SCRATCH] ? scratch_base : weight_base;
    src1_region_bytes_c = desc_reg.flags[FLAG_SRC1_SCRATCH] ? scratch_bytes : weight_bytes;
    bias_region_base_c  = desc_reg.flags[FLAG_BIAS_SCRATCH] ? scratch_base : weight_base;
    bias_region_bytes_c = desc_reg.flags[FLAG_BIAS_SCRATCH] ? scratch_bytes : weight_bytes;
    dst_region_base_c   = desc_reg.flags[FLAG_DST_OUTPUT] ? output_base : scratch_base;
    dst_region_bytes_c  = desc_reg.flags[FLAG_DST_OUTPUT] ? output_bytes : scratch_bytes;

    src0_abs64  = {32'd0, src0_region_base_c} + {32'd0, desc_reg.src0_offset};
    src1_abs64  = {32'd0, src1_region_base_c} + {32'd0, desc_reg.src1_offset};
    bias_abs64  = {32'd0, bias_region_base_c} + {32'd0, desc_reg.bias_offset};
    dst_abs64   = {32'd0, dst_region_base_c} + {32'd0, desc_reg.dst_offset};

    a_total64 = 64'(desc_reg.m) * 64'(desc_reg.k) *
                (desc_reg.flags[FLAG_HEAD_MODE] ? 64'd3 : 64'd1);
    b_total64    = 64'(desc_reg.k) * 64'(desc_reg.n);
    bias_total64 = 64'd4 * 64'(desc_reg.n);
    dst_total64  = 64'(desc_reg.m) * 64'(desc_reg.n) *
                   ((desc_reg.flags[FLAG_OUTPUT_INT32] ||
                     desc_reg.flags[10:8] == POST_PLAN) ? 64'd4 : 64'd1);

    a_total_beats    = 16'((a_total64 + 64'd7) >> 3);
    b_total_beats    = 16'((b_total64 + 64'd7) >> 3);
    bias_total_beats = 16'((bias_total64 + 64'd7) >> 3);
    dst_total_beats  = 16'((dst_total64 + 64'd7) >> 3);

    addr_ok_all = (src0_abs64[63:32] == 32'd0) && ga_ok &&
                  (src1_abs64[63:32] == 32'd0) && gb_ok &&
                  (!desc_reg.flags[FLAG_BIAS_ENABLE] ||
                   ((bias_abs64[63:32] == 32'd0) && gbi_ok)) &&
                  (dst_abs64[63:32] == 32'd0) && gd_ok;
  end

  always_comb begin
    v_error = ERR_NONE;
    if (desc_reg.opcode != OP_GEMM) v_error = ERR_OPCODE;
    else if (desc_reg.reserved != 4'd0) v_error = ERR_DIMENSION;
    else if (desc_reg.m == 16'd0 || desc_reg.n == 16'd0 || desc_reg.k == 16'd0)
      v_error = ERR_DIMENSION;
    else if (desc_reg.flags[FLAG_HEAD_MODE] && desc_reg.heads != 4'd3)
      v_error = ERR_DIMENSION;
    else if (!desc_reg.flags[FLAG_HEAD_MODE] && desc_reg.heads != 4'd0)
      v_error = ERR_DIMENSION;
    else if (desc_reg.flags[FLAG_SRC0_UNSIGNED] &&
             !(desc_reg.flags[FLAG_HEAD_MODE] && desc_reg.heads == 4'd3))
      v_error = ERR_DIMENSION;
    else if (desc_reg.flags[FLAG_SRC0_CAND_MAJOR] &&
             !(desc_reg.flags[FLAG_HEAD_MODE] && desc_reg.heads == 4'd3))
      v_error = ERR_DIMENSION;
    else if (desc_reg.flags[10:8] != POST_NONE &&
             desc_reg.flags[10:8] != POST_GELU &&
             desc_reg.flags[10:8] != POST_PLAN)
      v_error = ERR_DIMENSION;
    else if (desc_reg.flags[10:8] == POST_GELU &&
             desc_reg.flags[FLAG_OUTPUT_INT32])
      v_error = ERR_DIMENSION;
    else if (desc_reg.flags[10:8] == POST_PLAN &&
             desc_reg.flags[FLAG_OUTPUT_INT32])
      v_error = ERR_DIMENSION;
    else if (desc_reg.k > (A_BYTES / 8) || desc_reg.k > (B_BYTES / 8))
      v_error = ERR_DIMENSION;
    else if (!addr_ok_all) v_error = ERR_ADDRESS;
  end

  // P7-5: shallow validation — the same priority-encoded descriptor checks
  // WITHOUT the deep addr_ok_all clause.  Shallow violations drive
  // error_valid/error_code/busy in the first S_CHECK phase (a ~12-level
  // cone), while the full v_error settles into v_error_r for the state
  // decision and for address errors one phase later.  When a shallow check
  // fires, the address clause is irrelevant, so v_error_shallow equals
  // v_error for every shallow violation.
  always_comb begin
    v_error_shallow = ERR_NONE;
    if (desc_reg.opcode != OP_GEMM) v_error_shallow = ERR_OPCODE;
    else if (desc_reg.reserved != 4'd0) v_error_shallow = ERR_DIMENSION;
    else if (desc_reg.m == 16'd0 || desc_reg.n == 16'd0 || desc_reg.k == 16'd0)
      v_error_shallow = ERR_DIMENSION;
    else if (desc_reg.flags[FLAG_HEAD_MODE] && desc_reg.heads != 4'd3)
      v_error_shallow = ERR_DIMENSION;
    else if (!desc_reg.flags[FLAG_HEAD_MODE] && desc_reg.heads != 4'd0)
      v_error_shallow = ERR_DIMENSION;
    else if (desc_reg.flags[FLAG_SRC0_UNSIGNED] &&
             !(desc_reg.flags[FLAG_HEAD_MODE] && desc_reg.heads == 4'd3))
      v_error_shallow = ERR_DIMENSION;
    else if (desc_reg.flags[FLAG_SRC0_CAND_MAJOR] &&
             !(desc_reg.flags[FLAG_HEAD_MODE] && desc_reg.heads == 4'd3))
      v_error_shallow = ERR_DIMENSION;
    else if (desc_reg.flags[10:8] != POST_NONE &&
             desc_reg.flags[10:8] != POST_GELU &&
             desc_reg.flags[10:8] != POST_PLAN)
      v_error_shallow = ERR_DIMENSION;
    else if (desc_reg.flags[10:8] == POST_GELU &&
             desc_reg.flags[FLAG_OUTPUT_INT32])
      v_error_shallow = ERR_DIMENSION;
    else if (desc_reg.flags[10:8] == POST_PLAN &&
             desc_reg.flags[FLAG_OUTPUT_INT32])
      v_error_shallow = ERR_DIMENSION;
    else if (desc_reg.k > (A_BYTES / 8) || desc_reg.k > (B_BYTES / 8))
      v_error_shallow = ERR_DIMENSION;
  end

  // Load window computation (used in S_LOAD_SETUP).
  always_comb begin
    ld_addr64_c   = 64'h0000000000000000;
    ld_e_c        = 3'd0;
    ld_w_c        = 16'd0;
    ld_fill_bank_c = 3'd0;
    ld_region_base_c  = 32'h00000000;
    ld_region_bytes_c = 32'h00000000;
    if (state == S_LOAD_SETUP) begin
      case (ld_kind)
        2'd0: begin
          ld_region_base_c  = src0_region_base;
          ld_region_bytes_c = src0_region_bytes;
          ld_w_c            = k_eff;
          ld_fill_bank_c    = head_mode ? {1'b0, ld_bank} : 3'd7;
          if (src0_cand_major) begin
            // Candidate-major head-mode A: rows are [M][3][K] with head
            // offset K and row stride 3*K (selector local MLP).
            ld_addr64_c = {32'd0, src0_abs} +
                          (64'(m0) + 64'(ld_idx)) * (64'(k_eff) * 64'd3) +
                          64'(ld_bank) * 64'(k_eff);
          end else begin
            ld_addr64_c = {32'd0, src0_abs} +
                          (head_mode ? (64'(ld_bank) * 64'(m_eff) *
                                        64'(k_eff)) : 64'd0) +
                          (64'(m0) + 64'(ld_idx)) * 64'(k_eff);
          end
        end
        2'd1: begin
          ld_region_base_c  = src1_region_base;
          ld_region_bytes_c = src1_region_bytes;
          ld_fill_bank_c    = 3 + ld_bank;
          if (rhs_transpose) begin
            ld_w_c = k_eff;
            ld_addr64_c = {32'd0, src1_abs} +
                          (head_mode ? (64'(ld_bank) * 64'(k_eff) * 64'(nph)) : 64'd0) +
                          (64'(col_local[ld_bank]) + 64'(ld_idx)) * 64'(k_eff);
          end else begin
            ld_w_c = {12'd0, b_cols[ld_bank]};
            ld_addr64_c = {32'd0, src1_abs} +
                          (head_mode ? (64'(ld_bank) * 64'(k_eff) * 64'(nph)) : 64'd0) +
                          64'(ld_idx) * 64'(col_stride) + 64'(col_local[ld_bank]);
          end
        end
        default: begin
          ld_region_base_c  = bias_region_base;
          ld_region_bytes_c = bias_region_bytes;
          ld_w_c            = {12'd0, b_cols[ld_bank]} * 16'd4;
          ld_fill_bank_c    = 3'd6;
          ld_addr64_c = {32'd0, bias_abs} + 64'(col_global[ld_bank]) * 64'd4;
        end
      endcase
      ld_e_c = ld_addr64_c[2:0];
    end
  end

  logic [63:0] ld_aligned64_c;
  logic [63:0] ld_cover64_c;
  logic [63:0] ld_end64_c;
  logic [63:0] ld_region_end64_c;
  assign ld_aligned64_c     = ld_addr64_c & 64'hFFFFFFFFFFFFFFF8;
  assign ld_cover64_c       = (ld_aligned64_c + 64'(ld_w_c) + 64'(ld_e_c) + 64'd7) &
                              64'hFFFFFFFFFFFFFFF8;
  assign ld_region_end64_c  = {32'd0, ld_region_base_c} + {32'd0, ld_region_bytes_c};
  assign ld_end64_c = (ld_cover64_c < ld_region_end64_c) ? ld_cover64_c : ld_region_end64_c;
  assign ld_len_c = 16'((ld_end64_c - ld_aligned64_c) >> 3);

  // P7-5 stage 1: the same cover/end/len chain re-evaluated from the
  // registered stage-0 window math (bit-identical to ld_len_c).
  logic [63:0] ld_cover64_p1;
  logic [63:0] ld_end64_p1;
  logic [63:0] ld_region_end64_p1;
  assign ld_cover64_p1 = (ld_aligned64_r + 64'(ld_w_c_r) + 64'(ld_e_c_r) + 64'd7) &
                         64'hFFFFFFFFFFFFFFF8;
  assign ld_region_end64_p1 = {32'd0, ld_rbase_r} + {32'd0, ld_rbytes_r};
  assign ld_end64_p1 = (ld_cover64_p1 < ld_region_end64_p1) ? ld_cover64_p1
                                                            : ld_region_end64_p1;

  // Writeback window computation (used in S_WB_NEXT).
  always_comb begin
    wb_addr64_c = 64'h0000000000000000;
    wb_e_c      = 3'd0;
    wb_w_c      = 6'd0;
    if (state == S_WB_NEXT && wb_b < 3 && b_cols[wb_b] != 0 && wb_r < a_rows) begin
      wb_w_c = {2'd0, b_cols[wb_b]} *
               ((out_int32 || post_op == POST_PLAN) ? 6'd4 : 6'd1);
      if (head_mode) begin
        wb_addr64_c = {32'd0, dst_abs} +
                      (64'(wb_b) * 64'(m_eff) * 64'(nph) +
                       (64'(m0) + 64'(wb_r)) * 64'(nph) + 64'(n0)) *
                      ((out_int32 || post_op == POST_PLAN) ? 64'd4 : 64'd1);
      end else begin
        wb_addr64_c = {32'd0, dst_abs} +
                      ((64'(m0) + 64'(wb_r)) * 64'(n_eff) +
                       64'(col_local[wb_b])) *
                      ((out_int32 || post_op == POST_PLAN) ? 64'd4 : 64'd1);
      end
      wb_e_c = wb_addr64_c[2:0];
    end
  end

  logic [63:0] wb_aligned64_c;
  logic [63:0] wb_cover64_c;
  logic [63:0] wb_end64_c;
  logic [63:0] wb_region_end64_c;
  assign wb_aligned64_c    = wb_addr64_c & 64'hFFFFFFFFFFFFFFF8;
  assign wb_cover64_c      = (wb_aligned64_c + 64'(wb_w_c) + 64'(wb_e_c) + 64'd7) &
                             64'hFFFFFFFFFFFFFFF8;
  assign wb_region_end64_c = {32'd0, dst_region_base} + {32'd0, dst_region_bytes};
  assign wb_end64_c = (wb_cover64_c < wb_region_end64_c) ? wb_cover64_c : wb_region_end64_c;
  assign wb_len_c = 16'((wb_end64_c - wb_aligned64_c) >> 3);

  // P7-5 stage 1: the same chain from the registered stage-0 window math
  // (bit-identical to wb_len_c).
  logic [63:0] wb_cover64_p1;
  logic [63:0] wb_end64_p1;
  logic [63:0] wb_region_end64_p1;
  assign wb_cover64_p1 = (wb_addr64_r + 64'(wb_w_c_r) + 64'(wb_e_c_r) + 64'd7) &
                         64'hFFFFFFFFFFFFFFF8;
  assign wb_region_end64_p1 = {32'd0, wb_rbase_r} + {32'd0, wb_rbytes_r};
  assign wb_end64_p1 = (wb_cover64_p1 < wb_region_end64_p1) ? wb_cover64_p1
                                                            : wb_region_end64_p1;

  // Per-byte writeback composition: stage A computes each byte's column
  // index / lane / kind for the current beat and registers them; stage B
  // re-selects the values from the accumulator and buffer arrays with the
  // registered indices and runs the original requant functions verbatim
  // (bit-exact by construction).

  always_comb begin
    wb_hit_a = 8'd0;
    for (int j = 0; j < 8; j++) begin
      int p;
      int c;
      wb_c_a[j]    = 5'd0;
      wb_lane_a[j] = 2'd0;
      wb_kind_a[j] = 2'd0;
      p = int'(wb_bi) * 8 + j;
      if ((p >= int'(wb_e)) && (p < int'(wb_e) + int'(wb_w))) begin
        wb_hit_a[j] = 1'b1;
        if (out_int32) begin
          c = (p - int'(wb_e)) / 4;
          wb_c_a[j]    = 5'(c);
          wb_lane_a[j] = 2'((p - int'(wb_e)) % 4);
          wb_kind_a[j] = 2'd2;
        end else if (post_op == POST_PLAN) begin
          c = (p - int'(wb_e)) / 4;
          wb_c_a[j]    = 5'(c);
          wb_lane_a[j] = 2'((p - int'(wb_e)) % 4);
          wb_kind_a[j] = 2'd3;
        end else begin
          c = p - int'(wb_e);
          wb_c_a[j]    = 5'(c);
          wb_kind_a[j] = (post_op == POST_GELU) ? 2'd1 : 2'd0;
        end
      end
    end
  end

  // P7-5 narrow rewrite of gemm_out8_from_sum (bit-exact width proof):
  //   shift = dst - src0 - src1 in [-95..95]; |sum| <= 2^32.
  //   shift >= 0: s = shift[6:0] in [0..95] (shift < 128, so the 7-bit
  //     slice is shift itself);
  //     round-away = (|sum| + 2^(s-1)) >> s.
  //       s == 0  -> sat8(sum)
  //       s <= 32 -> 34-bit cone (|sum|+2^(s-1) < 2^33)
  //       s == 33 -> -1 iff sum == -2^32 (|sum| == 2^32 == 2^(s-1)),
  //                  else 0
  //       s >= 34 -> 0                 (|sum|+2^(s-1) < 2^s)
  //   shift <  0: k = -shift in [1..95]; wide<<k then sat8.
  //       k >= 8  -> sign-saturates (|sum| >= 1 -> |sum<<k| >= 256)
  //       k <= 7  -> 40-bit cone (32+7 = 39 < 40, exact)
  function automatic heatvit_s8_t gemm_out8_from_sum(
    input logic signed [32:0] sum
  );
    int shift;
    logic [6:0] s;
    logic [33:0] mag_u;
    logic [33:0] rounded;
    shift = int'(dst_scale) - int'(src0_scale) - int'(src1_scale);
    if (shift >= 0) begin
      s = shift[6:0];
      if (s == 7'd0) begin
        return sat_s8_from_s33(sum);
      end else if (s <= 7'd32) begin
        mag_u = (sum < 33'sd0) ? (34'h200000000 - {1'b0, sum}) : {1'b0, sum};
        rounded = (mag_u + (34'd1 << (s - 7'd1))) >> s;
        if (sum < 33'sd0) begin
          return (rounded > 34'd128) ? -8'sd128 :
                 (8'sd0 - heatvit_s8_t'(rounded[7:0]));
        end else begin
          return (rounded > 34'd127) ? 8'sd127 : heatvit_s8_t'(rounded[7:0]);
        end
      end else if (s == 7'd33) begin
        return (sum == 33'sh100000000) ? -8'sd1 : 8'sd0;
      end else begin
        return 8'sd0;
      end
    end else begin
      if (-shift >= 8) begin
        return (sum == 33'sd0) ? 8'sd0 :
               ((sum < 33'sd0) ? -8'sd128 : 8'sd127);
      end else begin
        return sat_s8_from_s40($signed({{7{sum[32]}}, sum}) <<< (-shift));
      end
    end
  endfunction

  // P7-5 narrow rewrite of gemm_out32_from_sum (bit-exact width proof):
  //   src_exp = src0 + src1 (6-bit wrap); diff = dst - src_exp in [-63..62];
  //   |sum| <= 2^32.
  //   diff == 0 -> sat32(sum)
  //   diff >  0: round-away by diff (d = diff[5:0] == diff since diff < 64):
  //       d >= 34 -> 0; d == 33 -> -1 iff sum == -2^32;
  //       d <= 32 -> 34-bit cone, then sat32.
  //   diff <  0: k = -diff in [1..63]; sum<<k then sat32.
  //       k >= 31 -> sign-saturates (|sum| >= 1 -> |sum<<k| >= 2^31)
  //       k <= 30 -> 64-bit cone (32+30 = 62 < 64, exact).  The 128-bit
  //       left-shift overflow check is dead here too (32+63 = 95 < 127).
  function automatic heatvit_s32_t gemm_out32_from_sum(
    input logic signed [32:0] sum
  );
    logic signed [5:0] src_exp;
    int diff;
    logic [5:0] d;
    logic [33:0] mag_u;
    logic [33:0] rounded;
    src_exp = src0_scale + src1_scale;
    diff = int'(dst_scale) - int'(src_exp);
    if (diff == 0) begin
      return sat_s32_from_s33(sum);
    end else if (diff > 0) begin
      d = diff[5:0];
      if (d >= 6'd34) begin
        return 32'sd0;
      end else if (d == 6'd33) begin
        return (sum == 33'sh100000000) ? -32'sd1 : 32'sd0;
      end else begin
        mag_u = (sum < 33'sd0) ? (34'h200000000 - {1'b0, sum}) : {1'b0, sum};
        rounded = (mag_u + (34'd1 << (d - 6'd1))) >> d;
        if (sum < 33'sd0) begin
          // rounded <= 2^31 (max at d == 1), so -rounded >= -2^31 and the
          // sat_s32 collapses to the two's-complement negation.
          return -heatvit_s32_t'(rounded[31:0]);
        end else begin
          return (rounded > 34'd2147483647) ? 32'sd2147483647 :
                 heatvit_s32_t'(rounded[31:0]);
        end
      end
    end else begin
      if (-diff >= 31) begin
        return (sum == 33'sd0) ? 32'sd0 :
               ((sum < 33'sd0) ? -32'sd2147483648 : 32'sd2147483647);
      end else begin
        return sat_s32_from_s64($signed({{31{sum[32]}}, sum}) <<< (-diff));
      end
    end
  endfunction

  // P7-5 narrow rewrite of gelu_out8_from_value (bit-exact width proof):
  //   diff = dst - (-16) = dst + 16 in [-16..47]; |value| <= 2^23.
  //   diff == 0 -> sat8(value)
  //   diff >  0: round-away by diff (d == diff since diff <= 47 < 64):
  //       d >= 25 -> 0; d == 24 -> -1 iff value == -2^23;
  //       d <= 23 -> 24-bit cone (|value|+2^(d-1) < 2^24).
  //   diff <  0: k = -diff in [1..16]; value<<k then sat8.
  //       k >= 8  -> sign-saturates (|value| >= 1 -> >= 256)
  //       k <= 7  -> 40-bit cone (23+7 = 30 < 40, exact)
  function automatic heatvit_s8_t gelu_out8_from_value(
    input logic signed [23:0] value
  );
    int diff;
    logic [5:0] d;
    logic [23:0] mag;
    logic [23:0] rounded;
    diff = int'(dst_scale) + 16;
    if (diff == 0) begin
      return sat_s8_from_s24(value);
    end else if (diff > 0) begin
      d = diff[5:0];
      if (d >= 6'd25) begin
        return 8'sd0;
      end else if (d == 6'd24) begin
        return (value == 24'sh800000) ? -8'sd1 : 8'sd0;
      end else begin
        mag = (value < 24'sd0) ? (24'h800000 - {1'b0, value}) : {1'b0, value};
        rounded = (mag + (24'd1 << (d - 6'd1))) >> d;
        if (value < 24'sd0) begin
          return (rounded > 24'd128) ? -8'sd128 :
                 (8'sd0 - heatvit_s8_t'(rounded[7:0]));
        end else begin
          return (rounded > 24'd127) ? 8'sd127 : heatvit_s8_t'(rounded[7:0]);
        end
      end
    end else begin
      if (-diff >= 8) begin
        return (value == 24'sd0) ? 8'sd0 :
               ((value < 24'sd0) ? -8'sd128 : 8'sd127);
      end else begin
        return sat_s8_from_s40($signed({{16{value[23]}}, value}) <<< (-diff));
      end
    end
  endfunction

  // Stage B selects the source values using the registered decode. The
  // 33-bit sum is exact for signed int32 accumulator + signed int32 bias.
  always_comb begin
    for (int j = 0; j < 8; j++) begin
      wb_sum_c[j] = 33'sd0;
      wb_gelu_c[j] = 24'sd0;
      wb_plan_c[j] = 17'd0;
      if (wb_hit_r[j]) begin
        case (wb_kind_r[j])
          2'd0, 2'd2: begin
            wb_sum_c[j] =
              $signed({bank_accum[int'(wb_b)][int'(wb_r)][int'(wb_c_r[j])][31],
                       bank_accum[int'(wb_b)][int'(wb_r)][int'(wb_c_r[j])]}) +
              (bias_en
                ? $signed({bias_all[8*int'(wb_b) + int'(wb_c_r[j])][31],
                           bias_all[8*int'(wb_b) + int'(wb_c_r[j])]})
                : 33'sd0);
          end
          2'd1: begin
            wb_gelu_c[j] = gelu_buf[int'(wb_b)][int'(wb_r)][int'(wb_c_r[j])];
          end
          default: begin
            wb_plan_c[j] = plan_buf[int'(wb_b)][int'(wb_r)][int'(wb_c_r[j])];
          end
        endcase
      end
    end
  end

  // Stage C scales and packs only registered, locally placed values.
  always_comb begin
    logic [7:0]  strb;
    logic [63:0] data;
    strb = 8'd0;
    data = 64'd0;
    for (int j = 0; j < 8; j++) begin
      heatvit_s8_t  byte_value;
      heatvit_s32_t word_value;
      if (wb_value_hit_r[j]) begin
        strb[j] = 1'b1;
        case (wb_value_kind_r[j])
          2'd0: begin
            byte_value = gemm_out8_from_sum(wb_sum_r[j]);
            data[8*j +: 8] = byte_value;
          end
          2'd1: begin
            byte_value = gelu_out8_from_value(wb_gelu_r[j]);
            data[8*j +: 8] = byte_value;
          end
          2'd2: begin
            word_value = gemm_out32_from_sum(wb_sum_r[j]);
            data[8*j +: 8] = word_value[8*wb_value_lane_r[j] +: 8];
          end
          default: begin
            word_value = {15'd0, wb_plan_r[j]};
            data[8*j +: 8] = word_value[8*wb_value_lane_r[j] +: 8];
          end
        endcase
      end
    end
    wb_ram_wdata = {strb, data};
  end

  // Scatter one beat byte per cycle into the tile buffer.
  function automatic logic [12:0] fill_dest(
    input int kind,
    input int bank,
    input int idx,
    input int useful,
    input logic transposed
  );
    case (kind)
      0: return useful * 8 + idx;
      1: begin
        if (transposed) return useful * 8 + idx;
        else return idx * 8 + useful;
      end
      2: return bank * 32 + useful;
      default: return 13'd0;
    endcase
  endfunction

  assign tb_fill_valid = (state == S_LOAD_SCAT) &&
      (int'(ld_bi) * 8 + int'(scat_j) >= int'(ld_e)) &&
      (int'(ld_bi) * 8 + int'(scat_j) < int'(ld_e) + int'(ld_w));
  assign tb_fill_bank = ld_fill_bank;
  assign tb_fill_addr = fill_dest(int'(ld_kind), int'(ld_bank), int'(ld_idx),
                                  int'(ld_bi) * 8 + int'(scat_j) - int'(ld_e),
                                  rhs_transpose);
  assign tb_fill_data = ld_beat_data[8*scat_j +: 8];

  // Memory master request/read/write drive.
  assign req_valid   = (state == S_LOAD_REQ) || (state == S_WB_REQ);
  assign req_write   = (state == S_WB_REQ);
  assign req_addr    = (state == S_WB_REQ) ? wb_addr : ld_addr;
  assign req_bytes   = (state == S_WB_REQ) ? (32'(wb_len) << 3) :
                                              (32'(ld_len) << 3);
  assign req_r_ready = req_r_ready_r;
  assign req_w_valid = wb_valid_r;
  assign req_w_data  = wb_ram_rdata[63:0];
  assign req_w_strb  = wb_ram_rdata[71:64];
  assign req_w_last  = (state == S_WB_BEAT) && (wb_rd == wb_last_idx);

  // Write-back staging RAM: composed beats are written during
  // S_WB_COMPOSE and streamed combinationally during S_WB_BEAT (the RAM
  // read is registered, so rdata = mem[raddr of the previous cycle]).
  // raddr points at the beat being presented; it advances only on the
  // accepted beat (backpressure-safe) and warms up as 0 during the
  // compose phase, so the first S_WB_BEAT cycle already shows mem[0].
  heatvit_sdp_ram #(.WIDTH(72), .DEPTH(8)) u_wb_ram (
    .clk   (clk),
    .we    (wb_ram_we_r),
    .waddr (wb_ram_waddr_r),
    .wdata (wb_ram_wdata_r),
    .wstrb (9'h1FF),
    .raddr (wb_ram_raddr),
    .rdata (wb_ram_rdata)
  );

  assign wb_last_idx = wb_len[2:0] - 3'd1;
  wire wb_accept_w = (state == S_WB_BEAT) && wb_valid_r && req_w_ready;
  assign wb_ram_raddr = (state == S_WB_BEAT)
    ? (wb_accept_w ? ((wb_rd == wb_last_idx) ? wb_last_idx : (wb_rd + 3'd1))
                   : wb_rd)
    : 3'd0;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wb_ram_we_r    <= 1'b0;
      wb_ram_waddr_r <= 3'd0;
      wb_ram_wdata_r <= 72'd0;
    end else begin
      wb_ram_we_r    <= (state == S_WB_COMPOSE) && wb_value_valid_r;
      wb_ram_waddr_r <= wb_value_idx_r;
      wb_ram_wdata_r <= wb_ram_wdata;
    end
  end

  // Guard input mux for per-window runtime checks.  P7-5: both sides feed
  // the guard from their registered stage-0/1 values, so the guard's comb
  // cone is register-to-register.  The inputs are only sampled by g_ok_r /
  // g_code_r during each side's stage-2 phase; the other phases may present
  // stale registered values, which is harmless.
  assign g_base  = (state == S_WB_NEXT) ? wb_rbase_r : ld_rbase_r;
  assign g_bytes = (state == S_WB_NEXT) ? wb_rbytes_r : ld_rbytes_r;
  assign g_addr  = (state == S_WB_NEXT) ? wb_addr64_r[31:0] : ld_aligned64_r[31:0];
  assign g_len   = (state == S_WB_NEXT) ? wb_len_c_r : ld_len_r;

  assign cmd_ready = (state == S_IDLE);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state            <= S_IDLE;
      busy             <= 1'b0;
      done             <= 1'b0;
      error_valid      <= 1'b0;
      error_code       <= 8'd0;
      desc_reg         <= '0;
      src0_cand_major  <= 1'b0;
      mac_active_cycles <= '0;
      clear_accum      <= 1'b0;
      accum_valid      <= 1'b0;
      row_mask         <= 8'hff;
      post_op          <= 3'd0;
      gb               <= 2'd0;
      gr               <= 3'd0;
      gc               <= 3'd0;
      cb               <= 2'd0;
      cr               <= 3'd0;
      cc               <= 3'd0;
      act_acc_r        <= 32'sd0;
      act_bias_r       <= 32'sd0;
      act_fetch_valid_r <= 1'b0;
      gelu_start       <= 1'b0;
      act_q16_r        <= 24'sd0;
      plan_start       <= 1'b0;
      req_r_ready_r    <= 1'b0;
      wb_hit_r         <= 8'd0;
      for (int j = 0; j < 8; j++) begin
        wb_c_r[j]    <= 5'd0;
        wb_lane_r[j] <= 2'd0;
        wb_kind_r[j] <= 2'd0;
        wb_value_lane_r[j] <= 2'd0;
        wb_value_kind_r[j] <= 2'd0;
        wb_sum_r[j]  <= 33'sd0;
        wb_gelu_r[j] <= 24'sd0;
        wb_plan_r[j] <= 17'd0;
      end
      wb_comp_valid_r  <= 1'b0;
      wb_comp_idx_r    <= 3'd0;
      wb_value_hit_r   <= 8'd0;
      wb_value_valid_r <= 1'b0;
      wb_value_idx_r   <= 3'd0;
      wb_valid_r       <= 1'b0;
      wb_exit_pipe     <= 2'b00;
      wb_rd            <= 3'd0;
      wb_settle        <= 2'd0;
      wb_addr64_r      <= 64'd0;
      wb_len_c_r       <= 16'd0;
      wb_e_c_r         <= 3'd0;
      wb_w_c_r         <= 6'd0;
      wb_rbase_r       <= 32'd0;
      wb_rbytes_r      <= 32'd0;
      g_ok_r           <= 1'b0;
      g_code_r         <= 8'd0;
      ld_settle        <= 2'd0;
      ld_aligned64_r   <= 64'd0;
      ld_e_c_r         <= 3'd0;
      ld_w_c_r         <= 16'd0;
      ld_rbase_r       <= 32'd0;
      ld_rbytes_r      <= 32'd0;
      ld_len_r         <= 16'd0;
      ld_fill_bank_r   <= 3'd0;
      v_error_r        <= 8'd0;
      v_shallow_r      <= 1'b0;
      s_check_phase    <= 1'b0;
      for (int b = 0; b < 3; b++) begin
        col_mask[b]  <= 8'hff;
        a_rd_addr[b] <= 13'd0;
        b_rd_addr[b] <= 13'd0;
        for (int r = 0; r < 8; r++) a_lane[b][r] <= 8'd0;
        for (int c = 0; c < 8; c++) b_lane[b][c] <= 8'sd0;
        for (int r = 0; r < 8; r++)
          for (int c = 0; c < 8; c++) begin
            gelu_buf[b][r][c] <= 24'sd0;
            plan_buf[b][r][c] <= 17'd0;
          end
      end
    end else begin
      done        <= 1'b0;
      error_valid <= 1'b0;
      clear_accum <= 1'b0;
      accum_valid <= 1'b0;
      // Convert the raw value selected on the preceding cycle.  start and
      // x_in are registered together, then sampled by the activation unit on
      // the next edge under normal nonblocking-assignment semantics.
      gelu_start <= act_fetch_valid_r && (post_op == POST_GELU);
      plan_start <= act_fetch_valid_r && (post_op == POST_PLAN);
      if (act_fetch_valid_r) begin
        act_q16_r <= activation_q16(act_acc_r, act_bias_r);
      end
      act_fetch_valid_r <= 1'b0;
      req_r_ready_r <= 1'b0;

      if (mm_perr && state != S_IDLE) begin
        error_valid <= 1'b1;
        error_code  <= ERR_MEMORY_PROTOCOL;
        busy        <= 1'b0;
        state       <= S_IDLE;
      end else case (state)
        S_IDLE: begin
          if (cmd_valid && cmd_ready) begin
            desc_reg <= desc;
            busy     <= 1'b1;
            state    <= S_CHECK;
          end
        end

        S_CHECK: begin
          if (!s_check_phase) begin
            // P7-5: phase 0 registers the validation result and loads the
            // command configuration unconditionally.  The deep
            // desc_reg -> addr_ok_all -> v_error cone ends at v_error_r, so
            // the config-register clock enables only decode the state.  On
            // the error path the loaded values are never consumed (the next
            // command reloads them).  Shallow descriptor violations pulse
            // error_valid already in this phase (keeps the testbench's
            // one-cycle error window); address errors pulse one phase later
            // from the registered v_error_r.
            s_check_phase <= 1'b1;
            v_error_r     <= v_error;
            v_shallow_r   <= (v_error_shallow != ERR_NONE);
            error_valid   <= (v_error_shallow != ERR_NONE);
            error_code    <= v_error_shallow;
            busy          <= (v_error_shallow == ERR_NONE);
            m_eff          <= desc_reg.m;
            // In Head mode the descriptor n is the per-head width; the full
            // output width is heads * n so flag 19 can express dynamic N.
            n_eff          <= desc_reg.flags[FLAG_HEAD_MODE] ?
                              (desc_reg.n * desc_reg.heads) : desc_reg.n;
            k_eff          <= desc_reg.k;
            nph            <= desc_reg.flags[FLAG_HEAD_MODE] ?
                              desc_reg.n : 16'd0;
            head_mode      <= desc_reg.flags[FLAG_HEAD_MODE];
            src0_cand_major <= desc_reg.flags[FLAG_SRC0_CAND_MAJOR];
            rhs_transpose  <= desc_reg.flags[FLAG_RHS_TRANSPOSE];
            bias_en        <= desc_reg.flags[FLAG_BIAS_ENABLE];
            out_int32      <= desc_reg.flags[FLAG_OUTPUT_INT32];
            a_unsigned     <= desc_reg.flags[FLAG_SRC0_UNSIGNED];
            src0_abs       <= src0_abs64[31:0];
            src1_abs       <= src1_abs64[31:0];
            bias_abs       <= bias_abs64[31:0];
            dst_abs        <= dst_abs64[31:0];
            src0_region_base  <= src0_region_base_c;
            src0_region_bytes <= src0_region_bytes_c;
            src1_region_base  <= src1_region_base_c;
            src1_region_bytes <= src1_region_bytes_c;
            bias_region_base  <= bias_region_base_c;
            bias_region_bytes <= bias_region_bytes_c;
            dst_region_base   <= dst_region_base_c;
            dst_region_bytes  <= dst_region_bytes_c;
            src0_scale        <= desc_reg.src0_scale_exp;
            src1_scale        <= desc_reg.src1_scale_exp;
            dst_scale         <= desc_reg.dst_scale_exp;
            post_op           <= desc_reg.flags[10:8];
            m0                <= 16'd0;
            n0                <= 16'd0;
            ld_kind           <= 2'd0;
            ld_bank           <= 2'd0;
            ld_idx            <= 16'd0;
            mac_active_cycles <= '0;
          end else begin
            // Phase 1: decide from the registered validation result.
            s_check_phase <= 1'b0;
            if (v_error_r != ERR_NONE) begin
              if (!v_shallow_r) begin
                // Address error: the shallow phase stayed quiet, so pulse
                // the registered code here.
                error_valid <= 1'b1;
                error_code  <= v_error_r;
              end
              busy  <= 1'b0;
              state <= S_IDLE;
            end else begin
              state <= S_LOAD_SETUP;
            end
          end
        end

        // Load window selection with a four-phase guard pipeline (P7-5):
        // stage 0 registers the address-multiply window math, stage 1 the
        // cover/end/len add chain, stage 2 the addr-guard compare, stage 3
        // decides from registered values only.
        S_LOAD_SETUP: begin
          case (ld_settle)
          2'd0: begin
            ld_settle <= 2'd1;
            ld_aligned64_r <= ld_aligned64_c;
            ld_e_c_r    <= ld_e_c;
            ld_w_c_r    <= ld_w_c;
            ld_rbase_r  <= ld_region_base_c;
            ld_rbytes_r <= ld_region_bytes_c;
            ld_fill_bank_r <= ld_fill_bank_c;
          end
          2'd1: begin
            ld_settle <= 2'd2;
            ld_len_r  <= 16'((ld_end64_p1 - ld_aligned64_r) >> 3);
          end
          2'd2: begin
            ld_settle <= 2'd3;
            g_ok_r    <= g_ok;
            g_code_r  <= g_code;
          end
          default: begin
          ld_settle <= 2'd0;
          case (ld_kind)
            2'd0: begin
              if (head_mode ? (ld_bank == 2'd3) : (ld_idx == {12'd0, a_rows})) begin
                ld_kind <= 2'd1;
                ld_bank <= 2'd0;
                ld_idx  <= 16'd0;
              end else begin
                if (g_ok_r) begin
                  ld_addr <= ld_aligned64_r[31:0];
                  ld_len  <= ld_len_r;
                  ld_e    <= ld_e_c_r;
                  ld_w    <= ld_w_c_r;
                  ld_fill_bank <= ld_fill_bank_r;
                  state   <= S_LOAD_REQ;
                end else begin
                  error_code  <= g_code_r;
                  error_valid <= 1'b1;
                  busy        <= 1'b0;
                  state       <= S_IDLE;
                end
              end
            end
            2'd1: begin
              if (ld_bank == 2'd3) begin
                if (bias_en) begin
                  ld_kind <= 2'd2;
                  ld_bank <= 2'd0;
                end else begin
                  state <= S_COMPUTE_PRE;
                end
              end else if (b_cols[ld_bank] == 4'd0) begin
                ld_bank <= ld_bank + 2'd1;
              end else begin
                if (g_ok_r) begin
                  ld_addr <= ld_aligned64_r[31:0];
                  ld_len  <= ld_len_r;
                  ld_e    <= ld_e_c_r;
                  ld_w    <= ld_w_c_r;
                  ld_fill_bank <= ld_fill_bank_r;
                  state   <= S_LOAD_REQ;
                end else begin
                  error_code  <= g_code_r;
                  error_valid <= 1'b1;
                  busy        <= 1'b0;
                  state       <= S_IDLE;
                end
              end
            end
            default: begin
              if (ld_bank == 2'd3) begin
                state <= S_COMPUTE_PRE;
              end else if (b_cols[ld_bank] == 4'd0) begin
                ld_bank <= ld_bank + 2'd1;
              end else begin
                if (g_ok_r) begin
                  ld_addr <= ld_aligned64_r[31:0];
                  ld_len  <= ld_len_r;
                  ld_e    <= ld_e_c_r;
                  ld_w    <= ld_w_c_r;
                  ld_fill_bank <= ld_fill_bank_r;
                  state   <= S_LOAD_REQ;
                end else begin
                  error_code  <= g_code_r;
                  error_valid <= 1'b1;
                  busy        <= 1'b0;
                  state       <= S_IDLE;
                end
              end
            end
          endcase
          end
          endcase
        end

        S_LOAD_REQ: begin
          if (req_valid && req_ready) begin
            ld_bi <= 7'd0;
            req_r_ready_r <= 1'b1;
            state <= S_LOAD_RECV;
          end
        end

        S_LOAD_RECV: begin
          if (req_r_valid && req_r_ready) begin
            ld_beat_data <= req_r_data;
            scat_j       <= 3'd0;
            req_r_ready_r <= 1'b0;
            state        <= S_LOAD_SCAT;
          end else begin
            req_r_ready_r <= 1'b1;
          end
        end

        S_LOAD_SCAT: begin
          if (scat_j == 3'd7) begin
            if (ld_bi == ld_len - 16'd1) begin
              // Advance to the next load unit.
              case (ld_kind)
                2'd0: begin
                  if (head_mode) begin
                    if (ld_idx == a_rows - 16'd1) begin
                      ld_idx  <= 16'd0;
                      ld_bank <= ld_bank + 2'd1;
                    end else begin
                      ld_idx <= ld_idx + 16'd1;
                    end
                  end else begin
                    ld_idx <= ld_idx + 16'd1;
                  end
                end
                2'd1: begin
                  if (ld_idx == b_count[ld_bank] - 16'd1) begin
                    ld_idx  <= 16'd0;
                    ld_bank <= ld_bank + 2'd1;
                  end else begin
                    ld_idx <= ld_idx + 16'd1;
                  end
                end
                default: begin
                  ld_bank <= ld_bank + 2'd1;
                end
              endcase
              state <= S_LOAD_SETUP;
            end else begin
              ld_bi  <= ld_bi + 7'd1;
              req_r_ready_r <= 1'b1;
              state  <= S_LOAD_RECV;
            end
          end else begin
            scat_j <= scat_j + 3'd1;
          end
        end

        S_COMPUTE_PRE: begin
          clear_accum <= 1'b1;
          kc          <= 16'd0;
          for (int b = 0; b < 3; b++) begin
            a_rd_addr[b] <= 13'd0;
            b_rd_addr[b] <= 13'd0;
          end
          state <= S_COMPUTE_WARM;
        end

        S_COMPUTE_WARM: begin
          for (int b = 0; b < 3; b++) begin
            a_rd_addr[b] <= 13'd8;
            b_rd_addr[b] <= 13'd8;
          end
          state <= S_COMPUTE_ACC;
        end

        S_COMPUTE_ACC: begin
          accum_valid <= 1'b1;
          row_mask    <= (a_rows == 4'd8) ? 8'hff : ((8'h01 << a_rows[2:0]) - 8'h01);
          for (int b = 0; b < 3; b++) begin
            col_mask[b] <= (b_cols[b] == 4'd8) ? 8'hff :
                           ((8'h01 << b_cols[b][2:0]) - 8'h01);
            for (int r = 0; r < 8; r++) a_lane[b][r] <= a_rd_data[b][8*r +: 8];
            for (int c = 0; c < 8; c++) b_lane[b][c] <= b_rd_data[b][8*c +: 8];
            mac_active_cycles[b] <= mac_active_cycles[b] + 32'd1;
            if (kc < k_eff - 16'd2) begin
              a_rd_addr[b] <= (kc + 16'd2) << 3;
              b_rd_addr[b] <= (kc + 16'd2) << 3;
            end
          end
          if (kc == k_eff - 16'd1) begin
            if (post_op == POST_GELU) begin
              gb    <= 2'd0;
              gr    <= 3'd0;
              gc    <= 3'd0;
              cr    <= 3'd0;
              cc    <= 3'd0;
              // Collect starts at the first non-empty bank (mirrors the
              // feed side, which skips empty banks one cycle each).
              cb    <= (b_cols[0] != 4'd0) ? 2'd0 :
                       (b_cols[1] != 4'd0) ? 2'd1 :
                       (b_cols[2] != 4'd0) ? 2'd2 : 2'd3;
              // One settle cycle: the MAC bank lands the last K partial
              // one cycle after the final S_COMPUTE_ACC, so the first
              // lane's accumulator read must wait (the old serial GELU
              // had this implicitly via its stage-0 sampling latency).
              state <= S_GELU_SETTLE;
            end else if (post_op == POST_PLAN) begin
              gb    <= 2'd0;
              gr    <= 3'd0;
              gc    <= 3'd0;
              // The final MAC partial lands one cycle after the last
              // S_COMPUTE_ACC edge.  Raw activation fetch must wait for it.
              state <= S_PLAN_SETTLE;
            end else begin
              wb_b  <= 2'd0;
              wb_r  <= 3'd0;
              state <= S_WB_NEXT;
            end
          end else begin
            kc <= kc + 16'd1;
          end
        end

        S_GELU_SETTLE: begin
          state <= S_GELU_NEXT;
        end

        S_PLAN_SETTLE: begin
          state <= S_PLAN_NEXT;
        end

        S_GELU_NEXT: begin
          // Pipelined GELU: feed one lane per cycle on (gb,gr,gc) and
          // collect finished lanes on (cb,cr,cc) in feed order; results
          // arrive 41 cycles after their start, possibly while feeding.
          if (gelu_done) begin
            gelu_buf[cb][cr][cc] <= gelu_out;
            if (cc == b_cols[cb] - 4'd1) begin
              cc <= 3'd0;
              if (cr == a_rows - 4'd1) begin
                cr <= 3'd0;
                // skip empty banks, mirroring the feed order
                if (cb == 2'd0 && b_cols[1] != 4'd0) cb <= 2'd1;
                else if (cb != 2'd2 && b_cols[2] != 4'd0) cb <= 2'd2;
                else cb <= 2'd3;
              end else begin
                cr <= cr + 3'd1;
              end
            end else begin
              cc <= cc + 3'd1;
            end
          end
          if (gb == 2'd3) begin
            // Feeding complete; drain the remaining in-flight lanes.
            if (cb == 2'd3) begin
              wb_b  <= 2'd0;
              wb_r  <= 3'd0;
              state <= S_WB_NEXT;
            end else begin
              state <= S_GELU_DRAIN;
            end
          end else if (b_cols[gb] == 4'd0) begin
            gb <= gb + 2'd1;
          end else begin
            act_acc_r         <= bank_accum[gb][gr][gc];
            act_bias_r        <= bias_all[8*int'(gb) + int'(gc)];
            act_fetch_valid_r <= 1'b1;
            if (gc == b_cols[gb] - 4'd1) begin
              gc <= 3'd0;
              if (gr == a_rows - 4'd1) begin
                gr <= 3'd0;
                gb <= gb + 2'd1;
              end else begin
                gr <= gr + 3'd1;
              end
            end else begin
              gc <= gc + 3'd1;
            end
          end
        end

        S_GELU_DRAIN: begin
          if (gelu_done) begin
            gelu_buf[cb][cr][cc] <= gelu_out;
            if (cc == b_cols[cb] - 4'd1) begin
              cc <= 3'd0;
              if (cr == a_rows - 4'd1) begin
                cr <= 3'd0;
                if (cb == 2'd0 && b_cols[1] != 4'd0) cb <= 2'd1;
                else if (cb != 2'd2 && b_cols[2] != 4'd0) cb <= 2'd2;
                else cb <= 2'd3;
              end else begin
                cr <= cr + 3'd1;
              end
            end else begin
              cc <= cc + 3'd1;
            end
          end
          if (cb == 2'd3) begin
            wb_b  <= 2'd0;
            wb_r  <= 3'd0;
            state <= S_WB_NEXT;
          end
        end

        S_PLAN_NEXT: begin
          if (gb == 2'd3) begin
            wb_b  <= 2'd0;
            wb_r  <= 3'd0;
            state <= S_WB_NEXT;
          end else if (b_cols[gb] == 4'd0) begin
            gb <= gb + 2'd1;
          end else begin
            act_acc_r         <= bank_accum[gb][gr][gc];
            act_bias_r        <= bias_all[8*int'(gb) + int'(gc)];
            act_fetch_valid_r <= 1'b1;
            state             <= S_PLAN_WAIT;
          end
        end

        S_PLAN_WAIT: begin
          if (plan_done) begin
            plan_buf[gb][gr][gc] <= plan_out;
            if (gc == b_cols[gb] - 4'd1) begin
              gc <= 3'd0;
              if (gr == a_rows - 4'd1) begin
                gr <= 3'd0;
                gb <= gb + 2'd1;
              end else begin
                gr <= gr + 3'd1;
              end
            end else begin
              gc <= gc + 3'd1;
            end
            state <= S_PLAN_NEXT;
          end
        end

        // Write-back window selection. The address guard is deep
        // Write-back window selection with the same four-phase guard
        // pipeline as S_LOAD_SETUP (P7-5): stage 0 registers the address
        // multiply cone, stage 1 the cover/end/len chain, stage 2 the
        // addr-guard compare, stage 3 decides from registered values only.
        S_WB_NEXT: begin
          case (wb_settle)
          2'd0: begin
            wb_settle <= 2'd1;
            wb_addr64_r <= wb_aligned64_c;
            wb_e_c_r    <= wb_e_c;
            wb_w_c_r    <= wb_w_c;
            wb_rbase_r  <= dst_region_base;
            wb_rbytes_r <= dst_region_bytes;
          end
          2'd1: begin
            wb_settle <= 2'd2;
            wb_len_c_r <= 16'((wb_end64_p1 - wb_addr64_r) >> 3);
          end
          2'd2: begin
            wb_settle <= 2'd3;
            g_ok_r    <= g_ok;
            g_code_r  <= g_code;
          end
          default: begin
            wb_settle <= 2'd0;
            if (wb_b == 2'd3) begin
              // Advance to the next column group / row tile.
              if (head_mode) begin
                if (n0 + 16'd8 < nph) n0 <= n0 + 16'd8;
                else begin
                  n0 <= 16'd0;
                  m0 <= m0 + 16'd8;
                end
              end else begin
                if (n0 + 16'd24 < n_eff) n0 <= n0 + 16'd24;
                else begin
                  n0 <= 16'd0;
                  m0 <= m0 + 16'd8;
                end
              end
              if ((head_mode && (n0 + 16'd8 >= nph)) || (!head_mode && (n0 + 16'd24 >= n_eff))) begin
                if (m0 + 16'd8 >= m_eff) begin
                  state <= S_DONE;
                end else begin
                  ld_kind <= 2'd0;
                  ld_bank <= 2'd0;
                  ld_idx  <= 16'd0;
                  state   <= S_LOAD_SETUP;
                end
              end else begin
                ld_kind <= 2'd0;
                ld_bank <= 2'd0;
                ld_idx  <= 16'd0;
                state   <= S_LOAD_SETUP;
              end
            end else if (b_cols[wb_b] == 4'd0) begin
              wb_b <= wb_b + 2'd1;
            end else if (wb_r < a_rows) begin
              if (g_ok_r) begin
                wb_bi   <= 3'd0;
                wb_addr <= wb_addr64_r[31:0];
                wb_len  <= wb_len_c_r;
                wb_e    <= wb_e_c_r;
                wb_w    <= wb_w_c_r;
                state   <= S_WB_REQ;
              end else begin
                error_code  <= g_code_r;
                error_valid <= 1'b1;
                busy        <= 1'b0;
                state       <= S_IDLE;
              end
            end else begin
              wb_b <= wb_b + 2'd1;
              wb_r <= 3'd0;
            end
          end
          endcase
        end

        S_WB_REQ: begin
          if (req_valid && req_ready) begin
            wb_bi            <= 3'd0;
            wb_comp_valid_r  <= 1'b0;
            wb_value_valid_r <= 1'b0;
            wb_exit_pipe     <= 2'b00;
            state            <= S_WB_COMPOSE;
          end
        end

        // Compose the burst into the staging RAM at one beat per cycle:
        // A decodes byte positions, B selects and registers source values,
        // C requants/packs, and the registered RAM port captures the result.
        // wb_exit_pipe drains B/C/write before the synchronous RAM read is
        // exposed to the memory master.
        S_WB_COMPOSE: begin
          wb_comp_valid_r <= (wb_bi <= wb_last_idx);
          wb_comp_idx_r   <= wb_bi;
          wb_hit_r        <= wb_hit_a;
          for (int j = 0; j < 8; j++) begin
            wb_c_r[j]    <= wb_c_a[j];
            wb_lane_r[j] <= wb_lane_a[j];
            wb_kind_r[j] <= wb_kind_a[j];
            wb_value_lane_r[j] <= wb_lane_r[j];
            wb_value_kind_r[j] <= wb_kind_r[j];
            wb_sum_r[j]  <= wb_sum_c[j];
            wb_gelu_r[j] <= wb_gelu_c[j];
            wb_plan_r[j] <= wb_plan_c[j];
          end
          wb_value_valid_r <= wb_comp_valid_r;
          wb_value_idx_r   <= wb_comp_idx_r;
          wb_value_hit_r   <= wb_hit_r;
          if (wb_comp_valid_r && (wb_comp_idx_r == wb_last_idx)) begin
            wb_exit_pipe     <= 2'b01;
            wb_comp_valid_r  <= 1'b0;
            wb_bi            <= 3'd0;
          end else if (wb_exit_pipe[0]) begin
            wb_exit_pipe     <= 2'b10;
            wb_comp_valid_r  <= 1'b0;
          end else if (wb_exit_pipe[1]) begin
            wb_exit_pipe     <= 2'b00;
            wb_rd            <= 3'd0;
            wb_valid_r       <= 1'b0;
            wb_comp_valid_r  <= 1'b0;
            wb_value_valid_r <= 1'b0;
            state            <= S_WB_BEAT;
            wb_bi            <= 3'd0;
          end else if (wb_bi == wb_last_idx || wb_bi == 3'd7) begin
            wb_bi <= 3'd0;
          end else begin
            wb_bi <= wb_bi + 3'd1;
          end
        end

        // Stream the composed beats out of the staging RAM. The presented
        // beat is wb_rd; the RAM read is registered with raddr tracking the
        // presented beat (advanced only on the accepted beat), so the data
        // holds under backpressure. The first cycle is a warm-up: valid
        // rises at its edge together with rdata = mem[0].
        S_WB_BEAT: begin
          if (!wb_valid_r) begin
            wb_valid_r <= 1'b1;
          end else if (req_w_ready) begin
            if (wb_rd == wb_last_idx) begin
              wb_valid_r <= 1'b0;
              wb_r  <= wb_r + 3'd1;
              state <= S_WB_NEXT;
            end else begin
              wb_rd <= (wb_rd == wb_last_idx) ? wb_last_idx : (wb_rd + 3'd1);
            end
          end
        end

        S_DONE: begin
          done  <= 1'b1;
          busy  <= 1'b0;
          state <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
