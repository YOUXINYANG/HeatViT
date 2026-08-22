"""P2 (real DeiT-T weights) tooling.

Pipeline overview (see docs/heatvit.md Part 2 Section 13):

  float DeiT-T checkpoint (timm, cached locally)
      -> p2_quantize.py      : per-tensor power-of-2 PTQ + activation scale
                               calibration  ->  scale_table.json + int8 params
      -> p2_train_selector.py: Gumbel-Softmax selector training on the frozen
                               quantized backbone (blocks 4/7/10, 197->88->45->32)
      -> p2_export_weights.py: quantized tensors -> HeatViTParams-compatible
                               layout (Part 4 weight table) + .mem/descriptor
                               scale table for generate_descriptors.py
      -> generate_e2e_vectors.py / XSim bit-exact regression

None of these modules may import torch at module import time so that the
pure-integer golden model stays torch-free; only the torch-facing entry
points (p2_quantize, p2_train_selector) import it inside their main().
"""
