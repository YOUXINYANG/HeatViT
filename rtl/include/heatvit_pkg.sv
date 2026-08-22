`ifndef HEATVIT_PKG_SV
`define HEATVIT_PKG_SV

// HeatViT-T shared fixed-point types, constants, descriptor, and numeric
// helpers. This file is the single source of truth for the numeric contract.
package heatvit_pkg;

  // ---------------------------------------------------------------------
  // Fixed-width scalar types
  // ---------------------------------------------------------------------
  typedef logic signed [7:0]   heatvit_s8_t;
  typedef logic signed [23:0]  heatvit_q8_16_t;
  typedef logic signed [31:0]  heatvit_s32_t;
  typedef logic signed [47:0]  heatvit_s48_t;
  typedef logic signed [127:0] heatvit_s128_t;
  typedef logic signed [5:0]   heatvit_scale_t;
  typedef logic        [16:0]  heatvit_uq0_16_t;

  // ---------------------------------------------------------------------
  // 320-bit operation descriptor
  // ---------------------------------------------------------------------
  typedef struct packed {
    logic [7:0]      opcode;
    logic [23:0]     flags;
    logic [15:0]     m;
    logic [15:0]     n;
    logic [15:0]     k;
    logic [3:0]      heads;
    logic [31:0]     src0_offset;
    logic [31:0]     src1_offset;
    logic [31:0]     bias_offset;
    logic [31:0]     aux_offset;
    logic [31:0]     dst_offset;
    heatvit_scale_t  src0_scale_exp;
    heatvit_scale_t  src1_scale_exp;
    heatvit_scale_t  aux_scale_exp;
    heatvit_scale_t  dst_scale_exp;
    logic [15:0]     next_index;
    logic [15:0]     param0;
    logic [15:0]     param1;
    logic [3:0]      reserved;
  } heatvit_desc_t;

  // ---------------------------------------------------------------------
  // Opcodes
  // ---------------------------------------------------------------------
  typedef enum logic [7:0] {
    OP_NOP                 = 8'd0,
    OP_PATCHIFY            = 8'd1,
    OP_COPY_ADD_POS        = 8'd2,
    OP_GEMM                = 8'd3,
    OP_LAYERNORM           = 8'd4,
    OP_RESIDUAL            = 8'd5,
    OP_QKV_UNPACK          = 8'd6,
    OP_HEAD_CONCAT         = 8'd7,
    OP_ATTN_SOFTMAX        = 8'd8,
    OP_SELECTOR_SOFTMAX    = 8'd9,
    OP_REDUCE_MEAN         = 8'd10,
    OP_CONCAT_LOCAL_GLOBAL = 8'd11,
    OP_HEAD_FUSE           = 8'd12,
    OP_SELECTOR_FINALIZE   = 8'd13,
    OP_FINISH              = 8'd14
  } heatvit_opcode_e;

  // ---------------------------------------------------------------------
  // Post-ops
  // ---------------------------------------------------------------------
  typedef enum logic [2:0] {
    POST_NONE             = 3'd0,
    POST_REQUANT          = 3'd1,
    POST_GELU             = 3'd2,
    POST_ATTN_SOFTMAX     = 3'd3,
    POST_SELECTOR_SOFTMAX = 3'd4,
    POST_PLAN             = 3'd5,
    POST_LAYERNORM        = 3'd6
  } heatvit_postop_e;

  // ---------------------------------------------------------------------
  // Descriptor flag bit positions
  // ---------------------------------------------------------------------
  localparam int FLAG_RHS_TRANSPOSE = 0;
  localparam int FLAG_BIAS_ENABLE   = 1;
  localparam int FLAG_AUX_ENABLE    = 2;
  localparam int FLAG_DYNAMIC_M     = 3;
  localparam int FLAG_SWAP_ACTIVATION = 4;
  localparam int FLAG_HEAD_MODE     = 5;
  localparam int FLAG_HEAD_CONCAT   = 6;
  localparam int FLAG_OUTPUT_INT32  = 7;
  localparam int FLAG_SRC0_INPUT    = 11;
  localparam int FLAG_SRC1_SCRATCH  = 12;
  localparam int FLAG_BIAS_SCRATCH  = 13;
  localparam int FLAG_AUX_WEIGHT    = 14;
  localparam int FLAG_DST_OUTPUT    = 15;
  localparam int FLAG_TOKEN_TAIL    = 16;
  localparam int FLAG_CHANNEL_TAIL  = 17;
  localparam int FLAG_SRC0_UNSIGNED = 18;
  localparam int FLAG_DYNAMIC_N     = 19;
  localparam int FLAG_DYNAMIC_K     = 20;
  localparam int FLAG_SRC0_CAND_MAJOR = 21;

  localparam logic [1:0] DYN_M_CURRENT    = 2'b00;
  localparam logic [1:0] DYN_M_CANDIDATES = 2'b01;
  localparam logic [1:0] REDUCE_AXIS_CANDIDATES = 2'b00;
  localparam logic [1:0] REDUCE_AXIS_HEAD_LANES = 2'b01;

  // ---------------------------------------------------------------------
  // Errors and warnings
  // ---------------------------------------------------------------------
  typedef enum logic [7:0] {
    ERR_NONE              = 8'd0,
    ERR_OPCODE            = 8'd1,
    ERR_DIMENSION         = 8'd2,
    ERR_ADDRESS           = 8'd3,
    ERR_TOKEN_COUNT       = 8'd4,
    ERR_MEMORY_PROTOCOL   = 8'd5,
    ERR_SOFTMAX_ZERO_SUM  = 8'd6,
    ERR_BUSY_START        = 8'd7
  } heatvit_error_e;

  localparam int WARN_HEAD_DEN_ZERO    = 0;
  localparam int WARN_PACKAGE_DEN_ZERO = 1;
  localparam int WARN_LN_NEGATIVE_VARIANCE = 2;

  // ---------------------------------------------------------------------
  // Fixed-point constants
  // ---------------------------------------------------------------------
  localparam int GELU_A_Q16        = -18927;
  localparam int GELU_B_Q16        = -115933;
  localparam int GELU_DELTA_Q16    = 32768;
  localparam int INV_SQRT2_Q16     = 46341;
  localparam int EXP_LN2_Q16       = 45426;
  localparam int EXP_QUAD_Q16      = 23495;
  localparam int EXP_OFFSET_Q16    = 88670;
  localparam int EXP_CONST_Q16     = 22544;
  localparam int SOFTMAX_DELTA_Q16_ATTENTION = 32768;
  localparam int SOFTMAX_DELTA_Q16_SELECTOR  = 65536;
  localparam logic [47:0] LN_EPS_Q32 = 48'd4295;

  localparam heatvit_s128_t S128_MIN = 128'sh80000000000000000000000000000000;
  localparam heatvit_s128_t S128_MAX = 128'sh7fffffffffffffffffffffffffffffff;

  // ---------------------------------------------------------------------
  // Numeric helper functions
  // ---------------------------------------------------------------------
  function automatic heatvit_s128_t round_shift_away_s128(
    input heatvit_s128_t value,
    input logic [6:0] shift
  );
    logic [128:0] magnitude;
    logic [128:0] rounded_mag;
    heatvit_s128_t result;
    if (shift == 7'd0) return value;
    magnitude = (value < 0) ? (129'd0 - {1'b1, value}) : {1'b0, value};
    rounded_mag = (magnitude + (129'd1 << (shift - 7'd1))) >> shift;
    result = $signed(rounded_mag[127:0]);
    return (value < 0) ? -result : result;
  endfunction

  function automatic heatvit_s48_t round_shift_away_s48(
    input heatvit_s48_t value,
    input logic [5:0] shift
  );
    logic [48:0] magnitude;
    logic [48:0] rounded_mag;
    heatvit_s48_t result;
    if (shift == 6'd0) return value;
    magnitude = (value < 0) ? (49'd0 - {1'b1, value}) : {1'b0, value};
    rounded_mag = (magnitude + (49'd1 << (shift - 6'd1))) >> shift;
    result = $signed(rounded_mag[47:0]);
    return (value < 0) ? -result : result;
  endfunction

  function automatic heatvit_s128_t scale_to_exp_s128(
    input heatvit_s128_t value,
    input heatvit_scale_t src_exp,
    input heatvit_scale_t dst_exp
  );
    int diff;
    heatvit_s128_t shifted;
    diff = int'(dst_exp) - int'(src_exp);
    if (diff == 0) return value;
    if (diff > 0) return round_shift_away_s128(value, diff[6:0]);
    shifted = value <<< (-diff);
    if ((shifted >>> (-diff)) != value)
      return (value < 0) ? S128_MIN : S128_MAX;
    return shifted;
  endfunction

  function automatic heatvit_s8_t sat_s8(input heatvit_s128_t value);
    if (value > 128'sh0000000000000000000000000000007F) return 8'sd127;
    if (value < -128'sh00000000000000000000000000000080) return -8'sd128;
    return heatvit_s8_t'(value[7:0]);
  endfunction

  function automatic heatvit_s32_t sat_s32(input heatvit_s128_t value);
    if (value > 128'sh0000000000000000000000007FFFFFFF) return 32'sd2147483647;
    if (value < -128'sh00000000000000000000000080000000) return -32'sd2147483648;
    return heatvit_s32_t'(value[31:0]);
  endfunction

endpackage

`endif
