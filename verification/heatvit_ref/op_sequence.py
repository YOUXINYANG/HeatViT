"""Component-level 320-bit descriptor sequence emission.

Task 3 and later build per-component sequences (Patch, MHSA, FFN, Block) here;
the shared primitive is the exact 80-hex-digit .mem writer defined by Task 1.
"""

from pathlib import Path

from .descriptor import format_desc_hex


def write_descriptors_mem(path, descriptors):
    """Write one 80-hex-digit line per packed 320-bit descriptor."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [format_desc_hex(descriptor.pack()) for descriptor in descriptors]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
