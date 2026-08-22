"""Row-major integer GEMM reference and exact int8/int32 writeback.

The scalar path loops M, N, K in that fixed order with per-product int8
(or unsigned x signed) semantics and unbounded Python integers. The NumPy
path is an independent full-size implementation that must agree bitwise;
it asserts the pre-bias accumulator stays inside int32.
"""

import numpy as np

from .fixed import requant


def _check_int(name, value):
    if isinstance(value, float):
        raise TypeError(f"{name} must be an integer, got float")


def _matrix(name, value, lo, hi):
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list of rows")
    rows = len(value)
    cols = len(value[0]) if rows else 0
    out = []
    for r, row in enumerate(value):
        if not isinstance(row, (list, tuple)) or len(row) != cols:
            raise ValueError(f"{name} rows must have equal length")
        out_row = []
        for c, item in enumerate(row):
            _check_int(f"{name}[{r}][{c}]", item)
            item = int(item)
            if item < lo or item > hi:
                raise ValueError(f"{name}[{r}][{c}]={item} outside [{lo}, {hi}]")
            out_row.append(item)
        out.append(out_row)
    return out


def _bias_vector(name, value, n):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise ValueError(f"{name} must have length {n}")
    out = []
    for c, item in enumerate(value):
        _check_int(f"{name}[{c}]", item)
        item = int(item)
        if item < -(1 << 31) or item > (1 << 31) - 1:
            raise ValueError(f"{name}[{c}] outside int32")
        out.append(item)
    return out


def _b_element(b, transpose_b, k, c):
    return b[c][k] if transpose_b else b[k][c]


def gemm(a, b, bias, transpose_b, a_unsigned=False):
    """Exact scalar reference: A[M][K] * B[K][N] (+ bias), row-major."""
    lo, hi = (0, 255) if a_unsigned else (-128, 127)
    a = _matrix("a", a, lo, hi)
    b = _matrix("b", b, -128, 127)
    m = len(a)
    k = len(a[0]) if m else 0
    b_rows = len(b)
    b_cols = len(b[0]) if b_rows else 0
    if transpose_b:
        n, k_b = b_rows, b_cols
    else:
        k_b, n = b_rows, b_cols
    if k != k_b:
        raise ValueError(f"K mismatch: a has {k}, b has {k_b}")
    bias = _bias_vector("bias", bias, n)

    result = []
    for r in range(m):
        row = []
        for c in range(n):
            acc = 0
            for kk in range(k):
                if a_unsigned:
                    product = (a[r][kk] & 0xFF) * _b_element(b, transpose_b, kk, c)
                else:
                    product = a[r][kk] * _b_element(b, transpose_b, kk, c)
                acc += product
            if bias is not None:
                acc += bias[c]
            row.append(acc)
        result.append(row)
    return result


def gemm_numpy(a, b, bias, transpose_b, a_unsigned=False):
    """Independent NumPy int64 path used to cross-check the scalar loop."""
    lo, hi = (0, 255) if a_unsigned else (-128, 127)
    a = _matrix("a", a, lo, hi)
    b = _matrix("b", b, -128, 127)
    m = len(a)
    k = len(a[0]) if m else 0
    b_rows = len(b)
    b_cols = len(b[0]) if b_rows else 0
    if transpose_b:
        n, k_b = b_rows, b_cols
    else:
        k_b, n = b_rows, b_cols
    if k != k_b:
        raise ValueError(f"K mismatch: a has {k}, b has {k_b}")
    bias = _bias_vector("bias", bias, n)

    a64 = np.asarray(a, dtype=np.int64)
    if a_unsigned:
        a64 = a64 & 0xFF
    b64 = np.asarray(b, dtype=np.int64)
    rhs = b64.T if transpose_b else b64
    accum = a64 @ rhs
    if np.any(accum < -(1 << 31)) or np.any(accum > (1 << 31) - 1):
        raise OverflowError("GEMM accumulator exceeds int32")
    if bias is not None:
        accum = accum + np.asarray(bias, dtype=np.int64)[None, :]
    return accum.tolist()


def gemm_writeback(accum, src_exp, dst_exp, output_bits):
    """Write back a GEMM accumulator matrix as int8 or int32."""
    _check_int("src_exp", src_exp)
    _check_int("dst_exp", dst_exp)
    _check_int("output_bits", output_bits)
    accum = _matrix("accum", accum, -(1 << 63), (1 << 63) - 1)
    if output_bits == 32:
        return [
            [requant(int(value), int(src_exp), int(dst_exp), 32) for value in row]
            for row in accum
        ]
    if output_bits == 8:
        return [
            [requant(int(value), int(src_exp), int(dst_exp), 8) for value in row]
            for row in accum
        ]
    raise ValueError("output_bits must be 8 or 32")
