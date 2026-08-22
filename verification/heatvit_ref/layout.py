"""NHWC patchify, CLS/position add, QKV unpack and Head concat layouts."""

from .fixed import sat_signed
from .memory import _check_int


def _int8_list(name, values):
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be a list")
    out = []
    for i, value in enumerate(values):
        _check_int(f"{name}[{i}]", value)
        value = int(value)
        if value < -128 or value > 127:
            raise ValueError(f"{name}[{i}]={value} outside int8")
        out.append(value)
    return out


def _int8_matrix(name, rows, cols):
    if not isinstance(rows, (list, tuple)):
        raise TypeError(f"{name} must be a list of rows")
    out = []
    for r, row in enumerate(rows):
        out.append(_int8_list(f"{name}[{r}]", row))
        if len(out[-1]) != cols:
            raise ValueError(f"{name} row {r} must have {cols} elements")
    return out


def patchify(image, width, patch):
    """Convert a flat NHWC image into row-major patch vectors.

    Patches are emitted in raster order (top to bottom, left to right); each
    patch is flattened as (in_row, in_col, channel), so a pixel's R/G/B bytes
    stay consecutive.
    """
    _check_int("width", width)
    _check_int("patch", patch)
    width = int(width)
    patch = int(patch)
    if width <= 0 or patch <= 0 or width % patch != 0:
        raise ValueError("width must be a positive multiple of patch")
    image = _int8_list("image", image)
    if len(image) % (width * 3) != 0:
        raise ValueError("image length is not width*3*height")
    height = len(image) // (width * 3)
    if height % patch != 0:
        raise ValueError("height must be a multiple of patch")
    patches_r = height // patch
    patches_c = width // patch

    def image_index(row, col, channel):
        return ((row * width) + col) * 3 + channel

    def patch_index(in_row, in_col, channel):
        # Within-patch flatten: the doc's formula uses in-row/in-col indices
        # here; the patch's raster position contributes the outer block.
        return ((in_row * patch) + in_col) * 3 + channel

    patches = []
    for pr in range(patches_r):
        for pc in range(patches_c):
            vector = [0] * (patch * patch * 3)
            for in_row in range(patch):
                for in_col in range(patch):
                    for channel in range(3):
                        vector[patch_index(in_row, in_col, channel)] = image[
                            image_index(pr * patch + in_row, pc * patch + in_col, channel)
                        ]
            patches.append(vector)
    return patches


def add_pos(patch_embed, cls, pos):
    """Build [197][D]: row 0 = cls + pos[0], row i+1 = patch i + pos[i+1]."""
    cls = _int8_list("cls", cls)
    pos = _int8_matrix("pos", pos, len(cls))
    if len(pos) != len(patch_embed) + 1:
        raise ValueError("pos must have one more row than patch_embed")
    patch_embed = _int8_matrix("patch_embed", patch_embed, len(cls))

    out = []
    out.append([sat_signed(cls[c] + pos[0][c], 8) for c in range(len(cls))])
    for i, patch_row in enumerate(patch_embed):
        out.append(
            [sat_signed(patch_row[c] + pos[i + 1][c], 8) for c in range(len(cls))]
        )
    return out


def qkv_unpack(fused_qkv, tokens):
    """Unpack [token][Q192,K192,V192] into [kind][head][token][64]."""
    _check_int("tokens", tokens)
    tokens = int(tokens)
    if tokens < 0:
        raise ValueError("tokens must be non-negative")
    fused = _int8_matrix("fused_qkv", fused_qkv, 576)
    if len(fused) != tokens:
        raise ValueError("fused_qkv row count must equal tokens")

    total = 3 * 3 * tokens * 64
    flat = [0] * total

    def qkv_lane_index(kind, head, lane):
        return kind * 192 + head * 64 + lane

    def head_major_index(kind, head, token, lane):
        return ((kind * 3 + head) * tokens + token) * 64 + lane

    for token in range(tokens):
        for kind in range(3):
            for head in range(3):
                for lane in range(64):
                    flat[head_major_index(kind, head, token, lane)] = (
                        fused[token][qkv_lane_index(kind, head, lane)]
                    )

    result = []
    for kind in range(3):
        kind_list = []
        for head in range(3):
            head_list = []
            for token in range(tokens):
                base = head_major_index(kind, head, token, 0)
                head_list.append(flat[base:base + 64])
            kind_list.append(head_list)
        result.append(kind_list)
    return result


def head_concat(context, tokens):
    """Merge [head][token][64] back into [token][192].

    This inverts the head-major layout of one kind (Q, K or V); the full QKV
    matrix is recovered by concatenating the three kinds in order.
    """
    _check_int("tokens", tokens)
    tokens = int(tokens)
    if tokens < 0:
        raise ValueError("tokens must be non-negative")
    if not isinstance(context, (list, tuple)) or len(context) != 3:
        raise ValueError("context must have 3 heads")

    out = []
    for token in range(tokens):
        row = [0] * 192
        for head in range(3):
            token_rows = context[head]
            if len(token_rows) != tokens:
                raise ValueError("context head row count must equal tokens")
            for lane in range(64):
                row[head * 64 + lane] = token_rows[token][lane]
        out.append(_int8_list(f"context token {token}", row))
    return out
