// Single-descriptor tensor executor for Phase 3.
//
// Exactly one GEMM engine, one memory master, one restoring divider and one
// three-client divider arbiter are instantiated. Opcodes dispatch to GEMM,
// the layout engine or the vector engine; the external memory pins are muxed
// between the GEMM engine's own master and the executor master used by the
// layout/vector children. Dynamic M/N/K are resolved here, and every operand
// is pre-checked against its selected region before the child starts.
module heatvit_tensor_executor
  import heatvit_pkg::*;
(
  input  logic                 clk,
  input  logic                 rst_n,
  input  logic                 abort,
  input  logic                 desc_valid,
  output logic                 desc_ready,
  input  heatvit_desc_t        desc,
  input  logic [7:0]           current_token_count,
  input  logic                 current_package_present,
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
  output logic                 abort_done,
  output logic [2:0]           warning_pulse,
  output logic                 state_update_valid,
  output logic [7:0]           next_token_count,
  output logic                 next_package_present,
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

  typedef enum logic [3:0] {
    CHILD_NONE,
    CHILD_GEMM,
    CHILD_LAYOUT,
    CHILD_VECTOR,
    CHILD_REDUCE,
    CHILD_CONCAT,
    CHILD_HEAD_FUSE,
    CHILD_SELECTOR_FINALIZE,
    CHILD_SELECTOR_SOFTMAX
  } child_t;

  typedef enum logic [2:0] {
    S_IDLE,
    S_CHECK,
    S_RUN,
    S_COMPLETE,
    S_ERROR,
    S_ABORT_DRAIN
  } state_t;

  state_t state;
  child_t child_sel;
  child_t child_sel_c;
  heatvit_desc_t desc_reg;

  logic [15:0] m_eff_c;
  logic [15:0] n_eff_c;
  logic [15:0] k_eff_c;
  logic [7:0]  err_code;

  logic [31:0] src0_base_c;
  logic [31:0] src1_base_c;
  logic [31:0] aux_base_c;
  logic [31:0] dst_base_c;
  logic [31:0] src0_bytes_c;
  logic [31:0] src1_bytes_c;
  logic [31:0] aux_bytes_c;
  logic [31:0] dst_bytes_c;

  logic [31:0] src0_field_bytes;
  logic [31:0] src1_field_bytes;
  logic [31:0] aux_field_bytes;
  logic [31:0] dst_field_bytes;
  logic        src0_en;
  logic        src1_en;
  logic        aux_en;
  logic        dst_en;
  logic        op_supported;
  logic        dyn_param_ok;
  logic        token_range_ok;
  logic [7:0]  v_error;
  // P7-5: S_CHECK settles the deep validation cone into registers before
  // the state decision, so the child start pulses and the child-register
  // clock enables (which used to fan out from v_error / the guards) are
  // register-driven.  Three phases: 0 captures the field-byte products
  // (the DSP cone), 1 runs the guards from the captured products, 2
  // decides and starts the child.
  logic [7:0]  v_error_r;
  logic [1:0]  s_check_phase;
  logic [31:0] src0_fb_r;
  logic [31:0] src1_fb_r;
  logic [31:0] aux_fb_r;
  logic [31:0] dst_fb_r;

  logic [63:0] src0_abs64;
  logic [63:0] src1_abs64;
  logic [63:0] aux_abs64;
  logic [63:0] dst_abs64;
  logic [15:0] src0_beats;
  logic [15:0] src1_beats;
  logic [15:0] aux_beats;
  logic [15:0] dst_beats;
  logic        addr_ok_all;
  logic        ga_ok, gb_ok, gau_ok, gd_ok;
  logic [7:0]  ga_code, gb_code, gau_code, gd_code;

  // Memory master owned by the executor (layout/vector children).
  logic        mm_req_valid;
  logic        mm_req_ready;
  logic        mm_req_write;
  logic [31:0] mm_req_addr;
  logic [31:0] mm_req_bytes;
  logic        mm_req_w_valid;
  logic        mm_req_w_ready;
  logic [63:0] mm_req_w_data;
  logic [7:0]  mm_req_w_strb;
  logic        mm_req_w_last;
  logic        mm_req_r_valid;
  logic        mm_req_r_ready;
  logic [63:0] mm_req_r_data;
  logic        mm_req_r_last;
  logic        mm_abort;
  logic        mm_done;
  logic        mm_perr;
  logic        mm_abort_done;

  logic        mm_mem_cmd_valid;
  logic        mm_mem_cmd_ready;
  logic        mm_mem_cmd_write;
  logic [31:0] mm_mem_cmd_addr;
  logic [15:0] mm_mem_cmd_len;
  logic        mm_mem_w_valid;
  logic        mm_mem_w_ready;
  logic [63:0] mm_mem_w_data;
  logic [7:0]  mm_mem_w_strb;
  logic        mm_mem_w_last;
  logic        mm_mem_r_valid;
  logic        mm_mem_r_ready;
  logic [63:0] mm_mem_r_data;
  logic        mm_mem_r_last;

  heatvit_mem_master u_master (
    .clk            (clk),
    .rst_n          (rst_n),
    .req_valid      (mm_req_valid),
    .req_ready      (mm_req_ready),
    .req_write      (mm_req_write),
    .req_addr       (mm_req_addr),
    .req_bytes      (mm_req_bytes),
    .req_w_valid    (mm_req_w_valid),
    .req_w_ready    (mm_req_w_ready),
    .req_w_data     (mm_req_w_data),
    .req_w_strb     (mm_req_w_strb),
    .req_w_last     (mm_req_w_last),
    .req_r_valid    (mm_req_r_valid),
    .req_r_ready    (mm_req_r_ready),
    .req_r_data     (mm_req_r_data),
    .req_r_last     (mm_req_r_last),
    .abort          (mm_abort),
    .done           (mm_done),
    .protocol_error (mm_perr),
    .abort_done     (mm_abort_done),
    .mem_cmd_valid  (mm_mem_cmd_valid),
    .mem_cmd_ready  (mm_mem_cmd_ready),
    .mem_cmd_write  (mm_mem_cmd_write),
    .mem_cmd_addr   (mm_mem_cmd_addr),
    .mem_cmd_len    (mm_mem_cmd_len),
    .mem_w_valid    (mm_mem_w_valid),
    .mem_w_ready    (mm_mem_w_ready),
    .mem_w_data     (mm_mem_w_data),
    .mem_w_strb     (mm_mem_w_strb),
    .mem_w_last     (mm_mem_w_last),
    .mem_r_valid    (mm_mem_r_valid),
    .mem_r_ready    (mm_mem_r_ready),
    .mem_r_data     (mm_mem_r_data),
    .mem_r_last     (mm_mem_r_last)
  );

  // GEMM child with the dynamically overridden dimensions.
  logic        gemm_cmd_valid;
  logic        gemm_cmd_ready;
  heatvit_desc_t gemm_desc_c;
  logic        gemm_busy;
  logic        gemm_done;
  logic        gemm_error_valid;
  logic [7:0]  gemm_error_code;
  logic        gemm_mem_cmd_valid;
  logic        gemm_mem_cmd_ready;
  logic        gemm_mem_cmd_write;
  logic [31:0] gemm_mem_cmd_addr;
  logic [15:0] gemm_mem_cmd_len;
  logic        gemm_mem_w_valid;
  logic        gemm_mem_w_ready;
  logic [63:0] gemm_mem_w_data;
  logic [7:0]  gemm_mem_w_strb;
  logic        gemm_mem_w_last;
  logic        gemm_mem_r_valid;
  logic        gemm_mem_r_ready;
  logic [63:0] gemm_mem_r_data;
  logic        gemm_mem_r_last;
  logic [2:0][31:0] gemm_mac_active;

  heatvit_gemm_engine u_gemm (
    .clk              (clk),
    .rst_n            (rst_n),
    .cmd_valid        (gemm_cmd_valid),
    .cmd_ready        (gemm_cmd_ready),
    .desc             (gemm_desc_c),
    .input_base       (input_base),
    .input_bytes      (input_bytes),
    .weight_base      (weight_base),
    .weight_bytes     (weight_bytes),
    .scratch_base     (scratch_base),
    .scratch_bytes    (scratch_bytes),
    .output_base      (output_base),
    .output_bytes     (output_bytes),
    .busy             (gemm_busy),
    .done             (gemm_done),
    .error_valid      (gemm_error_valid),
    .error_code       (gemm_error_code),
    .mac_active_cycles(gemm_mac_active),
    .mem_cmd_valid    (gemm_mem_cmd_valid),
    .mem_cmd_ready    (gemm_mem_cmd_ready),
    .mem_cmd_write    (gemm_mem_cmd_write),
    .mem_cmd_addr     (gemm_mem_cmd_addr),
    .mem_cmd_len      (gemm_mem_cmd_len),
    .mem_w_valid      (gemm_mem_w_valid),
    .mem_w_ready      (gemm_mem_w_ready),
    .mem_w_data       (gemm_mem_w_data),
    .mem_w_strb       (gemm_mem_w_strb),
    .mem_w_last       (gemm_mem_w_last),
    .mem_r_valid      (gemm_mem_r_valid),
    .mem_r_ready      (gemm_mem_r_ready),
    .mem_r_data       (gemm_mem_r_data),
    .mem_r_last       (gemm_mem_r_last)
  );

  // Layout child.
  logic        lay_start;
  logic        lay_busy;
  logic        lay_done;
  logic        lay_error_valid;
  logic [7:0]  lay_error_code;
  logic [1:0]  lay_op;
  logic        lay_req_valid;
  logic        lay_req_ready;
  logic        lay_req_write;
  logic [31:0] lay_req_addr;
  logic [31:0] lay_req_bytes;
  logic        lay_req_w_valid;
  logic        lay_req_w_ready;
  logic [63:0] lay_req_w_data;
  logic [7:0]  lay_req_w_strb;
  logic        lay_req_w_last;
  logic        lay_req_r_valid;
  logic        lay_req_r_ready;
  logic [63:0] lay_req_r_data;
  logic        lay_req_r_last;

  heatvit_layout_engine u_layout (
    .clk          (clk),
    .rst_n        (rst_n),
    .start        (lay_start),
    .busy         (lay_busy),
    .done         (lay_done),
    .error_valid  (lay_error_valid),
    .error_code   (lay_error_code),
    .op           (lay_op),
    .m_eff        (m_eff_c),
    .n_eff        (n_eff_c),
    .src0_base    (src0_abs64[31:0]),
    .src1_base    (src1_abs64[31:0]),
    .aux_base     (aux_abs64[31:0]),
    .dst_base     (dst_abs64[31:0]),
    .src0_scale   (desc_reg.src0_scale_exp),
    .src1_scale   (desc_reg.src1_scale_exp),
    .aux_scale    (desc_reg.aux_scale_exp),
    .dst_scale    (desc_reg.dst_scale_exp),
    .req_valid    (lay_req_valid),
    .req_ready    (lay_req_ready),
    .req_write    (lay_req_write),
    .req_addr     (lay_req_addr),
    .req_bytes    (lay_req_bytes),
    .req_w_valid  (lay_req_w_valid),
    .req_w_ready  (lay_req_w_ready),
    .req_w_data   (lay_req_w_data),
    .req_w_strb   (lay_req_w_strb),
    .req_w_last   (lay_req_w_last),
    .req_r_valid  (lay_req_r_valid),
    .req_r_ready  (lay_req_r_ready),
    .req_r_data   (lay_req_r_data),
    .req_r_last   (lay_req_r_last)
  );

  // Vector child.
  logic        vec_start;
  logic        vec_busy;
  logic        vec_done;
  logic        vec_error_valid;
  logic [7:0]  vec_error_code;
  logic [1:0]  vec_op;
  logic        vec_warn;
  logic        vec_req_valid;
  logic        vec_req_ready;
  logic        vec_req_write;
  logic [31:0] vec_req_addr;
  logic [31:0] vec_req_bytes;
  logic        vec_req_w_valid;
  logic        vec_req_w_ready;
  logic [63:0] vec_req_w_data;
  logic [7:0]  vec_req_w_strb;
  logic        vec_req_w_last;
  logic        vec_req_r_valid;
  logic        vec_req_r_ready;
  logic [63:0] vec_req_r_data;
  logic        vec_req_r_last;
  logic        sm_div_req_valid;
  logic        sm_div_req_ready;
  logic [63:0] sm_div_num;
  logic [63:0] sm_div_den;
  logic        sm_div_rsp_valid;
  logic [63:0] sm_div_quot;
  logic [63:0] sm_div_rem;
  logic        sm_div_div_zero;
  logic        ln_div_req_valid;
  logic        ln_div_req_ready;
  logic [63:0] ln_div_num;
  logic [63:0] ln_div_den;
  logic        ln_div_rsp_valid;
  logic [63:0] ln_div_quot;
  logic [63:0] ln_div_rem;
  logic        ln_div_div_zero;

  heatvit_vector_engine u_vector (
    .clk            (clk),
    .rst_n          (rst_n),
    .start          (vec_start),
    .busy           (vec_busy),
    .done           (vec_done),
    .error_valid    (vec_error_valid),
    .error_code     (vec_error_code),
    .warn_negative_variance(vec_warn),
    .op             (vec_op),
    .m_eff          (m_eff_c),
    .src0_base      (src0_abs64[31:0]),
    .src1_base      (src1_abs64[31:0]),
    .aux_base       (aux_abs64[31:0]),
    .dst_base       (dst_abs64[31:0]),
    .src0_end       (src0_base_c + src0_bytes_c),
    .dst_end        (dst_base_c + dst_bytes_c),
    .src0_scale     (desc_reg.src0_scale_exp),
    .src1_scale     (desc_reg.src1_scale_exp),
    .aux_scale      (desc_reg.aux_scale_exp),
    .dst_scale      (desc_reg.dst_scale_exp),
    .ln_div_req_valid(ln_div_req_valid),
    .ln_div_req_ready(ln_div_req_ready),
    .ln_div_num      (ln_div_num),
    .ln_div_den      (ln_div_den),
    .ln_div_rsp_valid(ln_div_rsp_valid),
    .ln_div_quot     (ln_div_quot),
    .ln_div_rem      (ln_div_rem),
    .ln_div_div_zero (ln_div_div_zero),
    .sm_div_req_valid(sm_div_req_valid),
    .sm_div_req_ready(sm_div_req_ready),
    .sm_div_num      (sm_div_num),
    .sm_div_den      (sm_div_den),
    .sm_div_rsp_valid(sm_div_rsp_valid),
    .sm_div_quot     (sm_div_quot),
    .sm_div_rem      (sm_div_rem),
    .sm_div_div_zero (sm_div_div_zero),
    .req_valid    (vec_req_valid),
    .req_ready    (vec_req_ready),
    .req_write    (vec_req_write),
    .req_addr     (vec_req_addr),
    .req_bytes    (vec_req_bytes),
    .req_w_valid  (vec_req_w_valid),
    .req_w_ready  (vec_req_w_ready),
    .req_w_data   (vec_req_w_data),
    .req_w_strb   (vec_req_w_strb),
    .req_w_last   (vec_req_w_last),
    .req_r_valid  (vec_req_r_valid),
    .req_r_ready  (vec_req_r_ready),
    .req_r_data   (vec_req_r_data),
    .req_r_last   (vec_req_r_last)
  );

  // Reduce-mean child (selector client 2 divider user).
  logic        red_start;
  logic        red_busy;
  logic        red_done;
  logic        red_error_valid;
  logic [7:0]  red_error_code;
  logic        red_axis;
  logic        red_req_valid;
  logic        red_req_ready;
  logic        red_req_write;
  logic [31:0] red_req_addr;
  logic [31:0] red_req_bytes;
  logic        red_req_w_valid;
  logic        red_req_w_ready;
  logic [63:0] red_req_w_data;
  logic [7:0]  red_req_w_strb;
  logic        red_req_w_last;
  logic        red_req_r_valid;
  logic        red_req_r_ready;
  logic [63:0] red_req_r_data;
  logic        red_req_r_last;
  logic        red_div_req_valid;
  logic        red_div_req_ready;
  logic [63:0] red_div_num;
  logic [63:0] red_div_den;
  logic        red_div_rsp_valid;
  logic [63:0] red_div_quot;
  logic [63:0] red_div_rem;
  logic        red_div_div_zero;

  heatvit_reduce_mean u_reduce (
    .clk          (clk),
    .rst_n        (rst_n),
    .start        (red_start),
    .busy         (red_busy),
    .done         (red_done),
    .error_valid  (red_error_valid),
    .error_code   (red_error_code),
    .axis         (red_axis),
    .m_eff        (m_eff_c),
    .src0_base    (src0_abs64[31:0]),
    .dst_base     (dst_abs64[31:0]),
    .req_valid    (red_req_valid),
    .req_ready    (red_req_ready),
    .req_write    (red_req_write),
    .req_addr     (red_req_addr),
    .req_bytes    (red_req_bytes),
    .req_w_valid  (red_req_w_valid),
    .req_w_ready  (red_req_w_ready),
    .req_w_data   (red_req_w_data),
    .req_w_strb   (red_req_w_strb),
    .req_w_last   (red_req_w_last),
    .req_r_valid  (red_req_r_valid),
    .req_r_ready  (red_req_r_ready),
    .req_r_data   (red_req_r_data),
    .req_r_last   (red_req_r_last),
    .div_req_valid(red_div_req_valid),
    .div_req_ready(red_div_req_ready),
    .div_num      (red_div_num),
    .div_den      (red_div_den),
    .div_rsp_valid(red_div_rsp_valid),
    .div_quot     (red_div_quot),
    .div_rem      (red_div_rem),
    .div_div_zero (red_div_div_zero)
  );

  // Feature concat child.
  logic        con_start;
  logic        con_busy;
  logic        con_done;
  logic        con_error_valid;
  logic [7:0]  con_error_code;
  logic        con_req_valid;
  logic        con_req_ready;
  logic        con_req_write;
  logic [31:0] con_req_addr;
  logic [31:0] con_req_bytes;
  logic        con_req_w_valid;
  logic        con_req_w_ready;
  logic [63:0] con_req_w_data;
  logic [7:0]  con_req_w_strb;
  logic        con_req_w_last;
  logic        con_req_r_valid;
  logic        con_req_r_ready;
  logic [63:0] con_req_r_data;
  logic        con_req_r_last;

  heatvit_feature_concat u_concat (
    .clk          (clk),
    .rst_n        (rst_n),
    .start        (con_start),
    .busy         (con_busy),
    .done         (con_done),
    .error_valid  (con_error_valid),
    .error_code   (con_error_code),
    .m_eff        (m_eff_c),
    .src0_base    (src0_abs64[31:0]),
    .src1_base    (src1_abs64[31:0]),
    .dst_base     (dst_abs64[31:0]),
    .req_valid    (con_req_valid),
    .req_ready    (con_req_ready),
    .req_write    (con_req_write),
    .req_addr     (con_req_addr),
    .req_bytes    (con_req_bytes),
    .req_w_valid  (con_req_w_valid),
    .req_w_ready  (con_req_w_ready),
    .req_w_data   (con_req_w_data),
    .req_w_strb   (con_req_w_strb),
    .req_w_last   (con_req_w_last),
    .req_r_valid  (con_req_r_valid),
    .req_r_ready  (con_req_r_ready),
    .req_r_data   (con_req_r_data),
    .req_r_last   (con_req_r_last)
  );

  // Head-fuse child (selector client 2 divider user).
  logic        hf_start;
  logic        hf_busy;
  logic        hf_done;
  logic        hf_error_valid;
  logic [7:0]  hf_error_code;
  logic        hf_warn;
  logic        hf_req_valid;
  logic        hf_req_ready;
  logic        hf_req_write;
  logic [31:0] hf_req_addr;
  logic [31:0] hf_req_bytes;
  logic        hf_req_w_valid;
  logic        hf_req_w_ready;
  logic [63:0] hf_req_w_data;
  logic [7:0]  hf_req_w_strb;
  logic        hf_req_w_last;
  logic        hf_req_r_valid;
  logic        hf_req_r_ready;
  logic [63:0] hf_req_r_data;
  logic        hf_req_r_last;
  logic        hf_div_req_valid;
  logic        hf_div_req_ready;
  logic [63:0] hf_div_num;
  logic [63:0] hf_div_den;
  logic        hf_div_rsp_valid;
  logic [63:0] hf_div_quot;
  logic [63:0] hf_div_rem;
  logic        hf_div_div_zero;

  heatvit_head_fuse u_head_fuse (
    .clk               (clk),
    .rst_n             (rst_n),
    .start             (hf_start),
    .busy              (hf_busy),
    .done              (hf_done),
    .error_valid       (hf_error_valid),
    .error_code        (hf_error_code),
    .m_eff             (m_eff_c),
    .src0_base         (src0_abs64[31:0]),
    .src1_base         (src1_abs64[31:0]),
    .dst_base          (dst_abs64[31:0]),
    .warn_head_den_zero(hf_warn),
    .req_valid         (hf_req_valid),
    .req_ready         (hf_req_ready),
    .req_write         (hf_req_write),
    .req_addr          (hf_req_addr),
    .req_bytes         (hf_req_bytes),
    .req_w_valid       (hf_req_w_valid),
    .req_w_ready       (hf_req_w_ready),
    .req_w_data        (hf_req_w_data),
    .req_w_strb        (hf_req_w_strb),
    .req_w_last        (hf_req_w_last),
    .req_r_valid       (hf_req_r_valid),
    .req_r_ready       (hf_req_r_ready),
    .req_r_data        (hf_req_r_data),
    .req_r_last        (hf_req_r_last),
    .div_req_valid     (hf_div_req_valid),
    .div_req_ready     (hf_div_req_ready),
    .div_num           (hf_div_num),
    .div_den           (hf_div_den),
    .div_rsp_valid     (hf_div_rsp_valid),
    .div_quot          (hf_div_quot),
    .div_rem           (hf_div_rem),
    .div_div_zero      (hf_div_div_zero)
  );

  // Selector finalize child (the only state-updating child).
  logic        fin_start;
  logic        fin_busy;
  logic        fin_done;
  logic        fin_error_valid;
  logic [7:0]  fin_error_code;
  logic        fin_warn;
  logic [7:0]  fin_next_token_count;
  logic        fin_next_package_present;
  logic        fin_update_pending;
  logic        fin_req_valid;
  logic        fin_req_ready;
  logic        fin_req_write;
  logic [31:0] fin_req_addr;
  logic [31:0] fin_req_bytes;
  logic        fin_req_w_valid;
  logic        fin_req_w_ready;
  logic [63:0] fin_req_w_data;
  logic [7:0]  fin_req_w_strb;
  logic        fin_req_w_last;
  logic        fin_req_r_valid;
  logic        fin_req_r_ready;
  logic [63:0] fin_req_r_data;
  logic        fin_req_r_last;
  logic        fin_div_req_valid;
  logic        fin_div_req_ready;
  logic [63:0] fin_div_num;
  logic [63:0] fin_div_den;
  logic        fin_div_rsp_valid;
  logic [63:0] fin_div_quot;
  logic [63:0] fin_div_rem;
  logic        fin_div_div_zero;

  heatvit_selector_finalize u_finalize (
    .clk                   (clk),
    .rst_n                 (rst_n),
    .start                 (fin_start),
    .busy                  (fin_busy),
    .done                  (fin_done),
    .error_valid           (fin_error_valid),
    .error_code            (fin_error_code),
    .m_eff                 (m_eff_c),
    .current_package_present(current_package_present),
    .src0_base             (src0_abs64[31:0]),
    .src1_base             (src1_abs64[31:0]),
    .dst_base              (dst_abs64[31:0]),
    .warn_package_den_zero (fin_warn),
    .next_token_count      (fin_next_token_count),
    .next_package_present  (fin_next_package_present),
    .req_valid             (fin_req_valid),
    .req_ready             (fin_req_ready),
    .req_write             (fin_req_write),
    .req_addr              (fin_req_addr),
    .req_bytes             (fin_req_bytes),
    .req_w_valid           (fin_req_w_valid),
    .req_w_ready           (fin_req_w_ready),
    .req_w_data            (fin_req_w_data),
    .req_w_strb            (fin_req_w_strb),
    .req_w_last            (fin_req_w_last),
    .req_r_valid           (fin_req_r_valid),
    .req_r_ready           (fin_req_r_ready),
    .req_r_data            (fin_req_r_data),
    .req_r_last            (fin_req_r_last),
    .div_req_valid         (fin_div_req_valid),
    .div_req_ready         (fin_div_req_ready),
    .div_num               (fin_div_num),
    .div_den               (fin_div_den),
    .div_rsp_valid         (fin_div_rsp_valid),
    .div_quot              (fin_div_quot),
    .div_rem               (fin_div_rem),
    .div_div_zero          (fin_div_div_zero)
  );

  // Selector softmax child (logits -> keep scores, divider client 0).
  logic        ss_start;
  logic        ss_busy;
  logic        ss_done;
  logic        ss_error_valid;
  logic [7:0]  ss_error_code;
  logic        ss_req_valid;
  logic        ss_req_ready;
  logic        ss_req_write;
  logic [31:0] ss_req_addr;
  logic [31:0] ss_req_bytes;
  logic        ss_req_w_valid;
  logic        ss_req_w_ready;
  logic [63:0] ss_req_w_data;
  logic [7:0]  ss_req_w_strb;
  logic        ss_req_w_last;
  logic        ss_req_r_valid;
  logic        ss_req_r_ready;
  logic [63:0] ss_req_r_data;
  logic        ss_req_r_last;
  logic        ss_div_req_valid;
  logic        ss_div_req_ready;
  logic [63:0] ss_div_num;
  logic [63:0] ss_div_den;
  logic        ss_div_rsp_valid;
  logic [63:0] ss_div_quot;
  logic [63:0] ss_div_rem;
  logic        ss_div_div_zero;

  heatvit_selector_softmax u_selector_softmax (
    .clk           (clk),
    .rst_n         (rst_n),
    .start         (ss_start),
    .busy          (ss_busy),
    .done          (ss_done),
    .error_valid   (ss_error_valid),
    .error_code    (ss_error_code),
    .m_eff         (m_eff_c),
    .src0_base     (src0_abs64[31:0]),
    .dst_base      (dst_abs64[31:0]),
    .src0_scale    (desc_reg.src0_scale_exp),
    .req_valid     (ss_req_valid),
    .req_ready     (ss_req_ready),
    .req_write     (ss_req_write),
    .req_addr      (ss_req_addr),
    .req_bytes     (ss_req_bytes),
    .req_w_valid   (ss_req_w_valid),
    .req_w_ready   (ss_req_w_ready),
    .req_w_data    (ss_req_w_data),
    .req_w_strb    (ss_req_w_strb),
    .req_w_last    (ss_req_w_last),
    .req_r_valid   (ss_req_r_valid),
    .req_r_ready   (ss_req_r_ready),
    .req_r_data    (ss_req_r_data),
    .req_r_last    (ss_req_r_last),
    .div_req_valid (ss_div_req_valid),
    .div_req_ready (ss_div_req_ready),
    .div_num       (ss_div_num),
    .div_den       (ss_div_den),
    .div_rsp_valid (ss_div_rsp_valid),
    .div_quot      (ss_div_quot),
    .div_rem       (ss_div_rem),
    .div_div_zero  (ss_div_div_zero)
  );

  // Shared restoring divider + arbiter: Softmax client 0, LayerNorm
  // client 1, reduce/head-fuse share client 2 (one child at a time).
  logic [2:0]  a_req_valid;
  logic [2:0]  a_req_ready;
  logic [63:0] a_num [2:0];
  logic [63:0] a_den [2:0];
  logic [2:0]  a_rsp_valid;
  logic [63:0] a_quot [2:0];
  logic [63:0] a_rem [2:0];
  logic [2:0]  a_div_zero;

  assign a_req_valid[0] = (child_sel == CHILD_VECTOR) ? sm_div_req_valid :
                          (child_sel == CHILD_SELECTOR_SOFTMAX)
                              ? ss_div_req_valid : 1'b0;
  assign a_req_valid[1] = ln_div_req_valid;
  assign a_req_valid[2] = (child_sel == CHILD_REDUCE) ? red_div_req_valid :
                          (child_sel == CHILD_HEAD_FUSE) ? hf_div_req_valid :
                          (child_sel == CHILD_SELECTOR_FINALIZE)
                              ? fin_div_req_valid : 1'b0;
  assign a_num[0] = (child_sel == CHILD_SELECTOR_SOFTMAX) ? ss_div_num
                                                          : sm_div_num;
  assign a_num[1] = ln_div_num;
  assign a_num[2] = (child_sel == CHILD_REDUCE) ? red_div_num :
                    (child_sel == CHILD_SELECTOR_FINALIZE) ? fin_div_num
                                                           : hf_div_num;
  assign a_den[0] = (child_sel == CHILD_SELECTOR_SOFTMAX) ? ss_div_den
                                                          : sm_div_den;
  assign a_den[1] = ln_div_den;
  assign a_den[2] = (child_sel == CHILD_REDUCE) ? red_div_den :
                    (child_sel == CHILD_SELECTOR_FINALIZE) ? fin_div_den
                                                           : hf_div_den;
  assign sm_div_req_ready = (child_sel == CHILD_VECTOR) ? a_req_ready[0]
                                                        : 1'b0;
  assign ss_div_req_ready = (child_sel == CHILD_SELECTOR_SOFTMAX)
                                ? a_req_ready[0] : 1'b0;
  assign ln_div_req_ready = a_req_ready[1];
  assign red_div_req_ready = (child_sel == CHILD_REDUCE) ? a_req_ready[2]
                                                         : 1'b0;
  assign hf_div_req_ready = (child_sel == CHILD_HEAD_FUSE) ? a_req_ready[2]
                                                           : 1'b0;
  assign fin_div_req_ready = (child_sel == CHILD_SELECTOR_FINALIZE)
                                 ? a_req_ready[2] : 1'b0;
  assign sm_div_rsp_valid = (child_sel == CHILD_VECTOR) ? a_rsp_valid[0]
                                                        : 1'b0;
  assign ss_div_rsp_valid = (child_sel == CHILD_SELECTOR_SOFTMAX)
                                ? a_rsp_valid[0] : 1'b0;
  assign ln_div_rsp_valid = a_rsp_valid[1];
  assign red_div_rsp_valid = (child_sel == CHILD_REDUCE) ? a_rsp_valid[2]
                                                         : 1'b0;
  assign hf_div_rsp_valid = (child_sel == CHILD_HEAD_FUSE) ? a_rsp_valid[2]
                                                           : 1'b0;
  assign fin_div_rsp_valid = (child_sel == CHILD_SELECTOR_FINALIZE)
                                 ? a_rsp_valid[2] : 1'b0;
  assign sm_div_quot = a_quot[0];
  assign ss_div_quot = a_quot[0];
  assign ln_div_quot = a_quot[1];
  assign red_div_quot = a_quot[2];
  assign hf_div_quot = a_quot[2];
  assign fin_div_quot = a_quot[2];
  assign sm_div_rem  = a_rem[0];
  assign ss_div_rem  = a_rem[0];
  assign ln_div_rem  = a_rem[1];
  assign red_div_rem  = a_rem[2];
  assign hf_div_rem  = a_rem[2];
  assign fin_div_rem  = a_rem[2];
  assign sm_div_div_zero = a_div_zero[0];
  assign ss_div_div_zero = a_div_zero[0];
  assign ln_div_div_zero = a_div_zero[1];
  assign red_div_div_zero = a_div_zero[2];
  assign hf_div_div_zero = a_div_zero[2];
  assign fin_div_div_zero = a_div_zero[2];

  heatvit_div_arbiter #(.NUM_W(64), .DEN_W(64), .QUOT_W(64)) u_arb (
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

  // ------------------------------------------------------------------
  // Validation and address pre-checks.
  // ------------------------------------------------------------------
  always_comb begin
    op_supported = (desc_reg.opcode == OP_NOP) ||
                   (desc_reg.opcode == OP_PATCHIFY) ||
                   (desc_reg.opcode == OP_COPY_ADD_POS) ||
                   (desc_reg.opcode == OP_GEMM) ||
                   (desc_reg.opcode == OP_LAYERNORM) ||
                   (desc_reg.opcode == OP_RESIDUAL) ||
                   (desc_reg.opcode == OP_QKV_UNPACK) ||
                   (desc_reg.opcode == OP_HEAD_CONCAT) ||
                   (desc_reg.opcode == OP_ATTN_SOFTMAX) ||
                   (desc_reg.opcode == OP_REDUCE_MEAN) ||
                   (desc_reg.opcode == OP_CONCAT_LOCAL_GLOBAL) ||
                   (desc_reg.opcode == OP_HEAD_FUSE) ||
                   (desc_reg.opcode == OP_SELECTOR_FINALIZE) ||
                   (desc_reg.opcode == OP_SELECTOR_SOFTMAX);
  end

  always_comb begin
    m_eff_c = desc_reg.m;
    n_eff_c = desc_reg.n;
    k_eff_c = desc_reg.k;
    if (desc_reg.flags[FLAG_DYNAMIC_M]) begin
      case (desc_reg.param0[1:0])
        DYN_M_CURRENT:    m_eff_c = {8'd0, current_token_count};
        DYN_M_CANDIDATES: m_eff_c = {8'd0, current_token_count} - 16'd1;
        default:          m_eff_c = desc_reg.m;
      endcase
    end
    if (desc_reg.flags[FLAG_DYNAMIC_N]) n_eff_c = {8'd0, current_token_count};
    if (desc_reg.flags[FLAG_DYNAMIC_K]) k_eff_c = {8'd0, current_token_count};
  end

  always_comb begin
    src0_base_c  = desc_reg.flags[FLAG_SRC0_INPUT] ? input_base : scratch_base;
    src0_bytes_c = desc_reg.flags[FLAG_SRC0_INPUT] ? input_bytes : scratch_bytes;
    src1_base_c  = desc_reg.flags[FLAG_SRC1_SCRATCH] ? scratch_base : weight_base;
    src1_bytes_c = desc_reg.flags[FLAG_SRC1_SCRATCH] ? scratch_bytes : weight_bytes;
    aux_base_c   = desc_reg.flags[FLAG_AUX_WEIGHT] ? weight_base : scratch_base;
    aux_bytes_c  = desc_reg.flags[FLAG_AUX_WEIGHT] ? weight_bytes : scratch_bytes;
    dst_base_c   = desc_reg.flags[FLAG_DST_OUTPUT] ? output_base : scratch_base;
    dst_bytes_c  = desc_reg.flags[FLAG_DST_OUTPUT] ? output_bytes : scratch_bytes;
  end

  always_comb begin
    src0_field_bytes = 32'd0;
    src1_field_bytes = 32'd0;
    aux_field_bytes  = 32'd0;
    dst_field_bytes  = 32'd0;
    src0_en = 1'b0;
    src1_en = 1'b0;
    aux_en  = 1'b0;
    dst_en  = 1'b0;
    case (desc_reg.opcode)
      OP_PATCHIFY: begin
        src0_field_bytes = 32'd196 * 32'd768;
        dst_field_bytes  = 32'd196 * 32'd768;
        src0_en = 1'b1; dst_en = 1'b1;
      end
      OP_COPY_ADD_POS: begin
        src0_field_bytes = 32'd196 * 32'd192;
        src1_field_bytes = 32'd197 * 32'd192;
        aux_field_bytes  = 32'd192;
        dst_field_bytes  = 32'd197 * 32'd192;
        src0_en = 1'b1; src1_en = 1'b1; aux_en = 1'b1; dst_en = 1'b1;
      end
      OP_QKV_UNPACK: begin
        src0_field_bytes = {16'd0, m_eff_c} * 32'd576;
        dst_field_bytes  = {16'd0, m_eff_c} * 32'd576;
        src0_en = 1'b1; dst_en = 1'b1;
      end
      OP_HEAD_CONCAT: begin
        src0_field_bytes = 32'd3 * {16'd0, m_eff_c} * 32'd64;
        dst_field_bytes  = {16'd0, m_eff_c} * 32'd192;
        src0_en = 1'b1; dst_en = 1'b1;
      end
      OP_LAYERNORM: begin
        src0_field_bytes = {16'd0, m_eff_c} * 32'd192;
        src1_field_bytes = 32'd192;
        aux_field_bytes  = 32'd192;
        dst_field_bytes  = {16'd0, m_eff_c} * 32'd192;
        src0_en = 1'b1; src1_en = 1'b1; aux_en = 1'b1; dst_en = 1'b1;
      end
      OP_RESIDUAL: begin
        src0_field_bytes = {16'd0, m_eff_c} * 32'd192;
        aux_field_bytes  = {16'd0, m_eff_c} * 32'd192;
        dst_field_bytes  = {16'd0, m_eff_c} * 32'd192;
        src0_en = 1'b1; aux_en = 1'b1; dst_en = 1'b1;
      end
      OP_ATTN_SOFTMAX: begin
        src0_field_bytes = 32'd3 * {16'd0, m_eff_c} * {16'd0, m_eff_c} * 32'd4;
        dst_field_bytes  = 32'd3 * {16'd0, m_eff_c} * {16'd0, m_eff_c};
        src0_en = 1'b1; dst_en = 1'b1;
      end
      OP_REDUCE_MEAN: begin
        if (desc_reg.param0[3:2] == REDUCE_AXIS_CANDIDATES) begin
          src0_field_bytes = 32'd3 * {16'd0, m_eff_c} * 32'd32;
          dst_field_bytes  = 32'd96;
        end else begin
          src0_field_bytes = {16'd0, m_eff_c} * 32'd192;
          dst_field_bytes  = {16'd0, m_eff_c} * 32'd3;
        end
        src0_en = 1'b1; dst_en = 1'b1;
      end
      OP_CONCAT_LOCAL_GLOBAL: begin
        src0_field_bytes = 32'd3 * {16'd0, m_eff_c} * 32'd32;
        src1_field_bytes = 32'd96;
        dst_field_bytes  = 32'd3 * {16'd0, m_eff_c} * 32'd64;
        src0_en = 1'b1; src1_en = 1'b1; dst_en = 1'b1;
      end
      OP_HEAD_FUSE: begin
        src0_field_bytes = 32'd3 * {16'd0, m_eff_c} * 32'd4;
        src1_field_bytes = 32'd3 * {16'd0, m_eff_c} * 32'd4;
        dst_field_bytes  = {16'd0, m_eff_c} * 32'd4;
        src0_en = 1'b1; src1_en = 1'b1; dst_en = 1'b1;
      end
      OP_SELECTOR_FINALIZE: begin
        src0_field_bytes = {16'd0, m_eff_c} * 32'd192;
        src1_field_bytes = ({16'd0, m_eff_c} - 16'd1) * 32'd4;
        dst_field_bytes  = {16'd0, m_eff_c} * 32'd192;
        src0_en = 1'b1; src1_en = 1'b1; dst_en = 1'b1;
      end
      OP_SELECTOR_SOFTMAX: begin
        src0_field_bytes = 32'd3 * {16'd0, m_eff_c} * 32'd2;
        dst_field_bytes  = 32'd3 * {16'd0, m_eff_c} * 32'd4;
        src0_en = 1'b1; dst_en = 1'b1;
      end
      default: begin
      end
    endcase
  end

  always_comb begin
    src0_abs64 = {32'd0, src0_base_c} + {32'd0, desc_reg.src0_offset};
    src1_abs64 = {32'd0, src1_base_c} + {32'd0, desc_reg.src1_offset};
    aux_abs64  = {32'd0, aux_base_c} + {32'd0, desc_reg.aux_offset};
    dst_abs64  = {32'd0, dst_base_c} + {32'd0, desc_reg.dst_offset};
    // P7-5: the beats derive from the registered field-byte products, so
    // the guard comb is a register-to-register cone (the DSP products are
    // captured in S_CHECK phase 0).
    src0_beats = 16'((src0_fb_r + 32'd7) >> 3);
    src1_beats = 16'((src1_fb_r + 32'd7) >> 3);
    aux_beats  = 16'((aux_fb_r + 32'd7) >> 3);
    dst_beats  = 16'((dst_fb_r + 32'd7) >> 3);
    addr_ok_all =
      (!src0_en || ((src0_abs64[63:32] == 32'd0) && ga_ok)) &&
      (!src1_en || ((src1_abs64[63:32] == 32'd0) && gb_ok)) &&
      (!aux_en  || ((aux_abs64[63:32] == 32'd0) && gau_ok)) &&
      (!dst_en  || ((dst_abs64[63:32] == 32'd0) && gd_ok));
  end

  heatvit_addr_guard u_guard_src0 (
    .region_base    (src0_base_c),
    .region_bytes   (src0_bytes_c),
    .cmd_addr       (src0_abs64[31:0]),
    .cmd_len        (src0_beats),
    .addr_ok        (ga_ok),
    .addr_error_code(ga_code)
  );
  heatvit_addr_guard u_guard_src1 (
    .region_base    (src1_base_c),
    .region_bytes   (src1_bytes_c),
    .cmd_addr       (src1_abs64[31:0]),
    .cmd_len        (src1_beats),
    .addr_ok        (gb_ok),
    .addr_error_code(gb_code)
  );
  heatvit_addr_guard u_guard_aux (
    .region_base    (aux_base_c),
    .region_bytes   (aux_bytes_c),
    .cmd_addr       (aux_abs64[31:0]),
    .cmd_len        (aux_beats),
    .addr_ok        (gau_ok),
    .addr_error_code(gau_code)
  );
  heatvit_addr_guard u_guard_dst (
    .region_base    (dst_base_c),
    .region_bytes   (dst_bytes_c),
    .cmd_addr       (dst_abs64[31:0]),
    .cmd_len        (dst_beats),
    .addr_ok        (gd_ok),
    .addr_error_code(gd_code)
  );

  always_comb begin
    dyn_param_ok = 1'b1;
    token_range_ok = 1'b1;
    if (desc_reg.flags[FLAG_DYNAMIC_M]) begin
      if (desc_reg.param0[1:0] == 2'd2 || desc_reg.param0[1:0] == 2'd3)
        dyn_param_ok = 1'b0;
    end
    if (desc_reg.flags[FLAG_DYNAMIC_M] || desc_reg.flags[FLAG_DYNAMIC_N] ||
        desc_reg.flags[FLAG_DYNAMIC_K]) begin
      if (current_token_count < 8'd2 || current_token_count > 8'd197)
        token_range_ok = 1'b0;
    end
  end

  always_comb begin
    v_error = ERR_NONE;
    if (!op_supported) v_error = ERR_OPCODE;
    else if (desc_reg.reserved != 4'd0) v_error = ERR_DIMENSION;
    else if (desc_reg.opcode == OP_PATCHIFY &&
             (desc_reg.m != 16'd196 || desc_reg.n != 16'd768))
      v_error = ERR_DIMENSION;
    else if (desc_reg.opcode == OP_COPY_ADD_POS &&
             (desc_reg.m != 16'd197 || desc_reg.n != 16'd192))
      v_error = ERR_DIMENSION;
    else if (desc_reg.opcode == OP_QKV_UNPACK &&
             (desc_reg.n != 16'd576 || desc_reg.heads != 4'd3))
      v_error = ERR_DIMENSION;
    else if (desc_reg.opcode == OP_HEAD_CONCAT &&
             (desc_reg.n != 16'd192 || desc_reg.heads != 4'd3))
      v_error = ERR_DIMENSION;
    else if (desc_reg.opcode == OP_LAYERNORM &&
             (desc_reg.n != 16'd192 || desc_reg.src0_scale_exp > 6'sd0))
      v_error = ERR_DIMENSION;
    else if (desc_reg.opcode == OP_RESIDUAL && desc_reg.n != 16'd192)
      v_error = ERR_DIMENSION;
    else if (desc_reg.opcode == OP_ATTN_SOFTMAX && desc_reg.heads != 4'd3)
      v_error = ERR_DIMENSION;
    else if (desc_reg.opcode == OP_REDUCE_MEAN &&
             (desc_reg.heads != 4'd3 ||
              desc_reg.param0[3:2] > REDUCE_AXIS_HEAD_LANES ||
              !desc_reg.flags[FLAG_DYNAMIC_M] ||
              desc_reg.param0[1:0] != DYN_M_CANDIDATES ||
              (desc_reg.param0[3:2] == REDUCE_AXIS_CANDIDATES &&
               desc_reg.n != 16'd32) ||
              (desc_reg.param0[3:2] == REDUCE_AXIS_HEAD_LANES &&
               desc_reg.n != 16'd64)))
      v_error = ERR_DIMENSION;
    else if (desc_reg.opcode == OP_CONCAT_LOCAL_GLOBAL &&
             (desc_reg.heads != 4'd3 || desc_reg.n != 16'd64 ||
              !desc_reg.flags[FLAG_DYNAMIC_M] ||
              desc_reg.param0[1:0] != DYN_M_CANDIDATES))
      v_error = ERR_DIMENSION;
    else if (desc_reg.opcode == OP_HEAD_FUSE &&
             (desc_reg.heads != 4'd3 || desc_reg.n != 16'd3 ||
              !desc_reg.flags[FLAG_DYNAMIC_M] ||
              desc_reg.param0[1:0] != DYN_M_CANDIDATES))
      v_error = ERR_DIMENSION;
    else if (desc_reg.opcode == OP_SELECTOR_FINALIZE &&
             (desc_reg.n != 16'd192 ||
              !desc_reg.flags[FLAG_DYNAMIC_M] ||
              desc_reg.param0[1:0] != DYN_M_CURRENT))
      v_error = ERR_DIMENSION;
    else if (desc_reg.opcode == OP_SELECTOR_SOFTMAX &&
             (desc_reg.n != 16'd2 || desc_reg.heads != 4'd3 ||
              !desc_reg.flags[FLAG_DYNAMIC_M] ||
              desc_reg.param0[1:0] != DYN_M_CANDIDATES))
      v_error = ERR_DIMENSION;
    else if (!dyn_param_ok) v_error = ERR_DIMENSION;
    else if (!token_range_ok) v_error = ERR_TOKEN_COUNT;
    else if (desc_reg.opcode != OP_GEMM && desc_reg.opcode != OP_NOP && !addr_ok_all)
      v_error = ERR_ADDRESS;
  end

  // ------------------------------------------------------------------
  // Child config and memory routing.
  // ------------------------------------------------------------------
  always_comb begin
    gemm_desc_c        = desc_reg;
    gemm_desc_c.m      = m_eff_c;
    gemm_desc_c.n      = n_eff_c;
    gemm_desc_c.k      = k_eff_c;
  end

  always_comb begin
    case (desc_reg.opcode)
      OP_GEMM:          child_sel_c = CHILD_GEMM;
      OP_PATCHIFY,
      OP_COPY_ADD_POS,
      OP_QKV_UNPACK,
      OP_HEAD_CONCAT:   child_sel_c = CHILD_LAYOUT;
      OP_LAYERNORM,
      OP_RESIDUAL,
      OP_ATTN_SOFTMAX:  child_sel_c = CHILD_VECTOR;
      OP_REDUCE_MEAN:   child_sel_c = CHILD_REDUCE;
      OP_CONCAT_LOCAL_GLOBAL: child_sel_c = CHILD_CONCAT;
      OP_HEAD_FUSE:     child_sel_c = CHILD_HEAD_FUSE;
      OP_SELECTOR_FINALIZE: child_sel_c = CHILD_SELECTOR_FINALIZE;
      OP_SELECTOR_SOFTMAX:  child_sel_c = CHILD_SELECTOR_SOFTMAX;
      default:          child_sel_c = CHILD_NONE;
    endcase
  end

  always_comb begin
    lay_op = 2'd0;
    case (desc_reg.opcode)
      OP_PATCHIFY:     lay_op = 2'd0;
      OP_COPY_ADD_POS: lay_op = 2'd1;
      OP_QKV_UNPACK:   lay_op = 2'd2;
      OP_HEAD_CONCAT:  lay_op = 2'd3;
      default:         lay_op = 2'd0;
    endcase
  end

  always_comb begin
    vec_op = 2'd0;
    case (desc_reg.opcode)
      OP_LAYERNORM:     vec_op = 2'd0;
      OP_RESIDUAL:      vec_op = 2'd1;
      OP_ATTN_SOFTMAX:  vec_op = 2'd2;
      default:          vec_op = 2'd0;
    endcase
  end

  // Child start/command pulses (P7-5: driven from the registered S_CHECK
  // phase 2 and v_error_r, so no guard/validation cone reaches the
  // children).
  assign lay_start = (state == S_CHECK) && (s_check_phase == 2'd2) &&
                     (v_error_r == ERR_NONE) && (child_sel == CHILD_LAYOUT);
  assign vec_start = (state == S_CHECK) && (s_check_phase == 2'd2) &&
                     (v_error_r == ERR_NONE) && (child_sel == CHILD_VECTOR);
  assign red_start = (state == S_CHECK) && (s_check_phase == 2'd2) &&
                     (v_error_r == ERR_NONE) && (child_sel == CHILD_REDUCE);
  assign con_start = (state == S_CHECK) && (s_check_phase == 2'd2) &&
                     (v_error_r == ERR_NONE) && (child_sel == CHILD_CONCAT);
  assign hf_start = (state == S_CHECK) && (s_check_phase == 2'd2) &&
                    (v_error_r == ERR_NONE) && (child_sel == CHILD_HEAD_FUSE);
  assign fin_start = (state == S_CHECK) && (s_check_phase == 2'd2) &&
                     (v_error_r == ERR_NONE) &&
                     (child_sel == CHILD_SELECTOR_FINALIZE);
  assign ss_start = (state == S_CHECK) && (s_check_phase == 2'd2) &&
                    (v_error_r == ERR_NONE) &&
                    (child_sel == CHILD_SELECTOR_SOFTMAX);
  assign gemm_cmd_valid = (state == S_CHECK) && (s_check_phase == 2'd2) &&
                          (v_error_r == ERR_NONE) && (child_sel == CHILD_GEMM);

  // Reduce axis from the descriptor: 0 candidate axis, 1 head-lane axis.
  assign red_axis = (desc_reg.param0[3:2] == REDUCE_AXIS_HEAD_LANES);

  // Executor memory master client routing.
  assign mm_req_valid   = (child_sel == CHILD_LAYOUT) ? lay_req_valid :
                          (child_sel == CHILD_VECTOR) ? vec_req_valid :
                          (child_sel == CHILD_REDUCE) ? red_req_valid :
                          (child_sel == CHILD_CONCAT) ? con_req_valid :
                          (child_sel == CHILD_HEAD_FUSE) ? hf_req_valid :
                          (child_sel == CHILD_SELECTOR_FINALIZE)
                              ? fin_req_valid :
                          (child_sel == CHILD_SELECTOR_SOFTMAX)
                              ? ss_req_valid : 1'b0;
  assign mm_req_write   = (child_sel == CHILD_LAYOUT) ? lay_req_write :
                          (child_sel == CHILD_VECTOR) ? vec_req_write :
                          (child_sel == CHILD_REDUCE) ? red_req_write :
                          (child_sel == CHILD_CONCAT) ? con_req_write :
                          (child_sel == CHILD_HEAD_FUSE) ? hf_req_write :
                          (child_sel == CHILD_SELECTOR_FINALIZE)
                              ? fin_req_write :
                          (child_sel == CHILD_SELECTOR_SOFTMAX)
                              ? ss_req_write : 1'b0;
  assign mm_req_addr    = (child_sel == CHILD_LAYOUT) ? lay_req_addr :
                          (child_sel == CHILD_VECTOR) ? vec_req_addr :
                          (child_sel == CHILD_REDUCE) ? red_req_addr :
                          (child_sel == CHILD_CONCAT) ? con_req_addr :
                          (child_sel == CHILD_HEAD_FUSE) ? hf_req_addr :
                          (child_sel == CHILD_SELECTOR_FINALIZE)
                              ? fin_req_addr :
                          (child_sel == CHILD_SELECTOR_SOFTMAX)
                              ? ss_req_addr : 32'd0;
  assign mm_req_bytes   = (child_sel == CHILD_LAYOUT) ? lay_req_bytes :
                          (child_sel == CHILD_VECTOR) ? vec_req_bytes :
                          (child_sel == CHILD_REDUCE) ? red_req_bytes :
                          (child_sel == CHILD_CONCAT) ? con_req_bytes :
                          (child_sel == CHILD_HEAD_FUSE) ? hf_req_bytes :
                          (child_sel == CHILD_SELECTOR_FINALIZE)
                              ? fin_req_bytes :
                          (child_sel == CHILD_SELECTOR_SOFTMAX)
                              ? ss_req_bytes : 32'd0;
  assign mm_req_w_valid = (child_sel == CHILD_LAYOUT) ? lay_req_w_valid :
                          (child_sel == CHILD_VECTOR) ? vec_req_w_valid :
                          (child_sel == CHILD_REDUCE) ? red_req_w_valid :
                          (child_sel == CHILD_CONCAT) ? con_req_w_valid :
                          (child_sel == CHILD_HEAD_FUSE) ? hf_req_w_valid :
                          (child_sel == CHILD_SELECTOR_FINALIZE)
                              ? fin_req_w_valid :
                          (child_sel == CHILD_SELECTOR_SOFTMAX)
                              ? ss_req_w_valid : 1'b0;
  assign mm_req_w_data  = (child_sel == CHILD_LAYOUT) ? lay_req_w_data :
                          (child_sel == CHILD_VECTOR) ? vec_req_w_data :
                          (child_sel == CHILD_REDUCE) ? red_req_w_data :
                          (child_sel == CHILD_CONCAT) ? con_req_w_data :
                          (child_sel == CHILD_HEAD_FUSE) ? hf_req_w_data :
                          (child_sel == CHILD_SELECTOR_FINALIZE)
                              ? fin_req_w_data :
                          (child_sel == CHILD_SELECTOR_SOFTMAX)
                              ? ss_req_w_data : 64'd0;
  assign mm_req_w_strb  = (child_sel == CHILD_LAYOUT) ? lay_req_w_strb :
                          (child_sel == CHILD_VECTOR) ? vec_req_w_strb :
                          (child_sel == CHILD_REDUCE) ? red_req_w_strb :
                          (child_sel == CHILD_CONCAT) ? con_req_w_strb :
                          (child_sel == CHILD_HEAD_FUSE) ? hf_req_w_strb :
                          (child_sel == CHILD_SELECTOR_FINALIZE)
                              ? fin_req_w_strb :
                          (child_sel == CHILD_SELECTOR_SOFTMAX)
                              ? ss_req_w_strb : 8'h00;
  assign mm_req_w_last  = (child_sel == CHILD_LAYOUT) ? lay_req_w_last :
                          (child_sel == CHILD_VECTOR) ? vec_req_w_last :
                          (child_sel == CHILD_REDUCE) ? red_req_w_last :
                          (child_sel == CHILD_CONCAT) ? con_req_w_last :
                          (child_sel == CHILD_HEAD_FUSE) ? hf_req_w_last :
                          (child_sel == CHILD_SELECTOR_FINALIZE)
                              ? fin_req_w_last :
                          (child_sel == CHILD_SELECTOR_SOFTMAX)
                              ? ss_req_w_last : 1'b0;
  assign mm_req_r_ready = (child_sel == CHILD_LAYOUT) ? lay_req_r_ready :
                          (child_sel == CHILD_VECTOR) ? vec_req_r_ready :
                          (child_sel == CHILD_REDUCE) ? red_req_r_ready :
                          (child_sel == CHILD_CONCAT) ? con_req_r_ready :
                          (child_sel == CHILD_HEAD_FUSE) ? hf_req_r_ready :
                          (child_sel == CHILD_SELECTOR_FINALIZE)
                              ? fin_req_r_ready :
                          (child_sel == CHILD_SELECTOR_SOFTMAX)
                              ? ss_req_r_ready : 1'b0;
  assign lay_req_ready  = (child_sel == CHILD_LAYOUT) ? mm_req_ready : 1'b0;
  assign vec_req_ready  = (child_sel == CHILD_VECTOR) ? mm_req_ready : 1'b0;
  assign red_req_ready  = (child_sel == CHILD_REDUCE) ? mm_req_ready : 1'b0;
  assign con_req_ready  = (child_sel == CHILD_CONCAT) ? mm_req_ready : 1'b0;
  assign hf_req_ready   = (child_sel == CHILD_HEAD_FUSE) ? mm_req_ready : 1'b0;
  assign fin_req_ready  = (child_sel == CHILD_SELECTOR_FINALIZE)
                              ? mm_req_ready : 1'b0;
  assign ss_req_ready   = (child_sel == CHILD_SELECTOR_SOFTMAX)
                              ? mm_req_ready : 1'b0;
  assign lay_req_w_ready = (child_sel == CHILD_LAYOUT) ? mm_req_w_ready : 1'b0;
  assign vec_req_w_ready = (child_sel == CHILD_VECTOR) ? mm_req_w_ready : 1'b0;
  assign red_req_w_ready = (child_sel == CHILD_REDUCE) ? mm_req_w_ready : 1'b0;
  assign con_req_w_ready = (child_sel == CHILD_CONCAT) ? mm_req_w_ready : 1'b0;
  assign hf_req_w_ready  = (child_sel == CHILD_HEAD_FUSE) ? mm_req_w_ready
                                                          : 1'b0;
  assign fin_req_w_ready = (child_sel == CHILD_SELECTOR_FINALIZE)
                               ? mm_req_w_ready : 1'b0;
  assign ss_req_w_ready  = (child_sel == CHILD_SELECTOR_SOFTMAX)
                               ? mm_req_w_ready : 1'b0;
  assign lay_req_r_valid = (child_sel == CHILD_LAYOUT) ? mm_req_r_valid : 1'b0;
  assign vec_req_r_valid = (child_sel == CHILD_VECTOR) ? mm_req_r_valid : 1'b0;
  assign red_req_r_valid = (child_sel == CHILD_REDUCE) ? mm_req_r_valid : 1'b0;
  assign con_req_r_valid = (child_sel == CHILD_CONCAT) ? mm_req_r_valid : 1'b0;
  assign hf_req_r_valid  = (child_sel == CHILD_HEAD_FUSE) ? mm_req_r_valid
                                                          : 1'b0;
  assign fin_req_r_valid = (child_sel == CHILD_SELECTOR_FINALIZE)
                               ? mm_req_r_valid : 1'b0;
  assign ss_req_r_valid  = (child_sel == CHILD_SELECTOR_SOFTMAX)
                               ? mm_req_r_valid : 1'b0;
  assign lay_req_r_data  = mm_req_r_data;
  assign vec_req_r_data  = mm_req_r_data;
  assign red_req_r_data  = mm_req_r_data;
  assign con_req_r_data  = mm_req_r_data;
  assign hf_req_r_data   = mm_req_r_data;
  assign fin_req_r_data  = mm_req_r_data;
  assign ss_req_r_data   = mm_req_r_data;
  assign lay_req_r_last  = mm_req_r_last;
  assign vec_req_r_last  = mm_req_r_last;
  assign red_req_r_last  = mm_req_r_last;
  assign con_req_r_last  = mm_req_r_last;
  assign hf_req_r_last   = mm_req_r_last;
  assign fin_req_r_last  = mm_req_r_last;
  assign ss_req_r_last   = mm_req_r_last;

  // External memory mux between the GEMM master and the executor master.
  assign mem_cmd_valid = (child_sel == CHILD_GEMM) ? gemm_mem_cmd_valid : mm_mem_cmd_valid;
  assign mem_cmd_write = (child_sel == CHILD_GEMM) ? gemm_mem_cmd_write : mm_mem_cmd_write;
  assign mem_cmd_addr  = (child_sel == CHILD_GEMM) ? gemm_mem_cmd_addr : mm_mem_cmd_addr;
  assign mem_cmd_len   = (child_sel == CHILD_GEMM) ? gemm_mem_cmd_len : mm_mem_cmd_len;
  assign mem_w_valid   = (child_sel == CHILD_GEMM) ? gemm_mem_w_valid : mm_mem_w_valid;
  assign mem_w_data    = (child_sel == CHILD_GEMM) ? gemm_mem_w_data : mm_mem_w_data;
  assign mem_w_strb    = (child_sel == CHILD_GEMM) ? gemm_mem_w_strb : mm_mem_w_strb;
  assign mem_w_last    = (child_sel == CHILD_GEMM) ? gemm_mem_w_last : mm_mem_w_last;
  assign mem_r_ready   = (child_sel == CHILD_GEMM) ? gemm_mem_r_ready : mm_mem_r_ready;

  assign gemm_mem_cmd_ready = (child_sel == CHILD_GEMM) ? mem_cmd_ready : 1'b0;
  assign mm_mem_cmd_ready   = (child_sel == CHILD_GEMM) ? 1'b0 : mem_cmd_ready;
  assign gemm_mem_w_ready   = (child_sel == CHILD_GEMM) ? mem_w_ready : 1'b0;
  assign mm_mem_w_ready     = (child_sel == CHILD_GEMM) ? 1'b0 : mem_w_ready;
  assign gemm_mem_r_valid   = (child_sel == CHILD_GEMM) ? mem_r_valid : 1'b0;
  assign mm_mem_r_valid     = (child_sel == CHILD_GEMM) ? 1'b0 : mem_r_valid;
  assign gemm_mem_r_data    = mem_r_data;
  assign mm_mem_r_data      = mem_r_data;
  assign gemm_mem_r_last    = mem_r_last;
  assign mm_mem_r_last      = mem_r_last;

  assign desc_ready          = (state == S_IDLE);
  assign warning_pulse       = {(vec_warn && (child_sel == CHILD_VECTOR)),
                                (fin_warn && (child_sel == CHILD_SELECTOR_FINALIZE)),
                                (hf_warn && (child_sel == CHILD_HEAD_FUSE))};

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state      <= S_IDLE;
      child_sel  <= CHILD_NONE;
      v_error_r  <= 8'd0;
      s_check_phase <= 2'd0;
      src0_fb_r  <= 32'd0;
      src1_fb_r  <= 32'd0;
      aux_fb_r   <= 32'd0;
      dst_fb_r   <= 32'd0;
      busy       <= 1'b0;
      done       <= 1'b0;
      error_valid <= 1'b0;
      error_code  <= 8'd0;
      abort_done  <= 1'b0;
      desc_reg    <= '0;
      mm_abort    <= 1'b0;
      err_code    <= 8'd0;
      state_update_valid   <= 1'b0;
      next_token_count     <= 8'd0;
      next_package_present <= 1'b0;
      fin_update_pending   <= 1'b0;
    end else begin
      done        <= 1'b0;
      error_valid <= 1'b0;
      abort_done  <= 1'b0;
      state_update_valid <= 1'b0;

      // The shared memory master's protocol violations (bad last framing)
      // abort the current child: any running descriptor is poisoned.
      if (mm_perr && (state == S_RUN || state == S_ABORT_DRAIN)) begin
        err_code <= ERR_MEMORY_PROTOCOL;
        state    <= S_ERROR;
      end else begin
      case (state)
        S_IDLE: begin
          if (abort) begin
            abort_done <= 1'b1;
          end else if (desc_valid && desc_ready) begin
            desc_reg <= desc;
            busy     <= 1'b1;
            state    <= S_CHECK;
          end
        end

        S_CHECK: begin
          case (s_check_phase)
          2'd0: begin
            // P7-5 phase 0: capture the field-byte DSP products and decode
            // the child selector unconditionally.  The products end the
            // desc_reg -> multiply cone here.
            s_check_phase <= 2'd1;
            src0_fb_r <= src0_field_bytes;
            src1_fb_r <= src1_field_bytes;
            aux_fb_r  <= aux_field_bytes;
            dst_fb_r  <= dst_field_bytes;
            case (desc_reg.opcode)
              OP_GEMM:          child_sel <= CHILD_GEMM;
              OP_PATCHIFY,
              OP_COPY_ADD_POS,
              OP_QKV_UNPACK,
              OP_HEAD_CONCAT:   child_sel <= CHILD_LAYOUT;
              OP_LAYERNORM,
              OP_RESIDUAL,
              OP_ATTN_SOFTMAX:  child_sel <= CHILD_VECTOR;
              OP_REDUCE_MEAN:   child_sel <= CHILD_REDUCE;
              OP_CONCAT_LOCAL_GLOBAL: child_sel <= CHILD_CONCAT;
              OP_HEAD_FUSE:     child_sel <= CHILD_HEAD_FUSE;
              OP_SELECTOR_FINALIZE: child_sel <= CHILD_SELECTOR_FINALIZE;
              OP_SELECTOR_SOFTMAX:  child_sel <= CHILD_SELECTOR_SOFTMAX;
              default:          child_sel <= CHILD_NONE;
            endcase
          end
          2'd1: begin
            // Phase 1: the guards run from the registered products (and
            // shallow address adds); the full validation cone ends here.
            s_check_phase <= 2'd2;
            v_error_r     <= v_error;
          end
          default: begin
            // Phase 2: decide from the registered validation result.  The
            // child start pulses assert during this phase.
            s_check_phase <= 2'd0;
            if (v_error_r != ERR_NONE) begin
              error_code  <= v_error_r;
              error_valid <= 1'b1;
              busy        <= 1'b0;
              state       <= S_IDLE;
            end else begin
              state <= S_RUN;
            end
          end
          endcase
        end

        S_RUN: begin
          if (abort) begin
            if (child_sel == CHILD_GEMM) begin
              if (gemm_done || gemm_error_valid)
                state <= S_ABORT_DRAIN;
            end else if (child_sel != CHILD_NONE) begin
              mm_abort <= 1'b1;
              state    <= S_ABORT_DRAIN;
            end else begin
              state <= S_ABORT_DRAIN;
            end
          end else case (child_sel)
            CHILD_NONE: state <= S_COMPLETE;
            CHILD_GEMM: begin
              if (gemm_done) state <= S_COMPLETE;
              if (gemm_error_valid) begin
                err_code <= gemm_error_code;
                state    <= S_ERROR;
              end
            end
            CHILD_LAYOUT: begin
              if (lay_done) state <= S_COMPLETE;
              if (lay_error_valid) begin
                err_code <= lay_error_code;
                state    <= S_ERROR;
              end
            end
            CHILD_VECTOR: begin
              if (vec_done) state <= S_COMPLETE;
              if (vec_error_valid) begin
                err_code <= vec_error_code;
                state    <= S_ERROR;
              end
            end
            CHILD_REDUCE: begin
              if (red_done) state <= S_COMPLETE;
              if (red_error_valid) begin
                err_code <= red_error_code;
                state    <= S_ERROR;
              end
            end
            CHILD_CONCAT: begin
              if (con_done) state <= S_COMPLETE;
              if (con_error_valid) begin
                err_code <= con_error_code;
                state    <= S_ERROR;
              end
            end
            CHILD_HEAD_FUSE: begin
              if (hf_done) state <= S_COMPLETE;
              if (hf_error_valid) begin
                err_code <= hf_error_code;
                state    <= S_ERROR;
              end
            end
            CHILD_SELECTOR_FINALIZE: begin
              if (fin_done) begin
                // Only the finalize child may update Token/Package state.
                // The update is held until S_COMPLETE so that
                // state_update_valid asserts on the same cycle as done
                // (the scheduler samples both together).
                fin_update_pending   <= 1'b1;
                next_token_count     <= fin_next_token_count;
                next_package_present <= fin_next_package_present;
                state <= S_COMPLETE;
              end
              if (fin_error_valid) begin
                err_code <= fin_error_code;
                state    <= S_ERROR;
              end
            end
            CHILD_SELECTOR_SOFTMAX: begin
              if (ss_done) state <= S_COMPLETE;
              if (ss_error_valid) begin
                err_code <= ss_error_code;
                state    <= S_ERROR;
              end
            end
            default: state <= S_ERROR;
          endcase
        end

        S_COMPLETE: begin
          done               <= 1'b1;
          state_update_valid <= fin_update_pending;
          fin_update_pending <= 1'b0;
          busy               <= 1'b0;
          state              <= S_IDLE;
        end

        S_ERROR: begin
          error_valid        <= 1'b1;
          error_code         <= err_code;
          fin_update_pending <= 1'b0;
          busy               <= 1'b0;
          state              <= S_IDLE;
        end

        S_ABORT_DRAIN: begin
          if (child_sel == CHILD_GEMM || child_sel == CHILD_NONE) begin
            abort_done <= 1'b1;
            busy       <= 1'b0;
            state      <= S_IDLE;
          end else if (mm_abort_done) begin
            mm_abort   <= 1'b0;
            abort_done <= 1'b1;
            busy       <= 1'b0;
            state      <= S_IDLE;
          end
        end

        default: state <= S_IDLE;
      endcase
      end
    end
  end

endmodule
