"""8-byte-aligned tensor arena and explicit little-endian packing helpers.

No ``struct`` or native byte-order assumptions are used: every byte position
is written with explicit shifts so the golden model stays independent of host
endianness and alignment.
"""


def _check_int(name, value):
    if isinstance(value, float):
        raise TypeError(f"{name} must be an integer, got float")


class TensorArena:
    """Linear allocator that rounds every allocation to an 8-byte offset."""

    def __init__(self):
        self._end = 0
        self._allocations = {}

    def allocate(self, name, byte_count):
        _check_int("byte_count", byte_count)
        byte_count = int(byte_count)
        if byte_count < 0:
            raise ValueError("byte_count must be non-negative")
        if name in self._allocations:
            raise ValueError(f"duplicate allocation name: {name}")
        offset = (self._end + 7) & ~7
        self._allocations[name] = (offset, byte_count)
        self._end = offset + byte_count
        return offset

    def offset_of(self, name):
        if name not in self._allocations:
            raise KeyError(f"unknown allocation: {name}")
        return self._allocations[name][0]

    def byte_count_of(self, name):
        if name not in self._allocations:
            raise KeyError(f"unknown allocation: {name}")
        return self._allocations[name][1]

    @property
    def total_bytes(self):
        return self._end

    @property
    def padded_bytes(self):
        return (self._end + 7) & ~7


def _check_int8s(name, values):
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


def _check_int32s(name, values):
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be a list")
    out = []
    for i, value in enumerate(values):
        _check_int(f"{name}[{i}]", value)
        value = int(value)
        if value < -(1 << 31) or value > (1 << 31) - 1:
            raise ValueError(f"{name}[{i}] outside int32")
        out.append(value)
    return out


def pack_int8_le(values):
    """Pack signed int8 values into little-endian bytes."""
    values = _check_int8s("values", values)
    return bytes(value & 0xFF for value in values)


def unpack_int8_le(data, count):
    """Unpack ``count`` little-endian int8 values from a bytes-like object."""
    _check_int("count", count)
    count = int(count)
    if count < 0 or len(data) < count:
        raise ValueError("count out of range for data length")
    out = []
    for i in range(count):
        byte = data[i]
        out.append(byte - 256 if byte >= 128 else byte)
    return out


def pack_int32_le(values):
    """Pack signed int32 values into little-endian bytes."""
    values = _check_int32s("values", values)
    out = bytearray()
    for value in values:
        value &= 0xFFFFFFFF
        out.append(value & 0xFF)
        out.append((value >> 8) & 0xFF)
        out.append((value >> 16) & 0xFF)
        out.append((value >> 24) & 0xFF)
    return bytes(out)


def unpack_int32_le(data, count):
    """Unpack ``count`` little-endian int32 values from a bytes-like object."""
    _check_int("count", count)
    count = int(count)
    if count < 0 or len(data) < count * 4:
        raise ValueError("count out of range for data length")
    out = []
    for i in range(count):
        word = (
            data[4 * i]
            | (data[4 * i + 1] << 8)
            | (data[4 * i + 2] << 16)
            | (data[4 * i + 3] << 24)
        )
        out.append(word - (1 << 32) if word >= (1 << 31) else word)
    return out
