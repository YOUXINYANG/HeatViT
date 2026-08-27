#!/usr/bin/env python3
"""P3 QAT: ImageNet data pipeline + HeatViT<->timm tensor mapping.

  * :func:`make_train_loader`  ImageNet train loader with the DeiT-style
    preprocessing (RandomResizedCrop + hflip + optional RandAugment);
    the subset selection uses the fixed repo seed for reproducibility.
  * :func:`heatvit_to_timm_state`  the inverse of p2_quantize.
    to_heatvit_tensors: maps QAT-trained HeatViT-layout tensors back into
    the timm DeiT-T state_dict layout, so the existing calibration hooks
    (p2_quantize.collect_float_histograms) can run on the trained model
    without any new hook code.

Data loading uses num_workers > 0 (spawn on Windows): callers must run
under an ``if __name__ == "__main__":`` guard, like p2_qat.py does.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

TRAIN_DIR = r"D:\SEU_Liubo\prj\HeatViT\data\imagenet\train"

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def make_train_loader(max_images=0, batch_size=128, num_workers=4,
                      shuffle=True, seed=20260815, randaug=False):
    """ImageNet train loader; max_images=0 means the full 1.28M set.

    The subset (when max_images > 0) is a fixed randperm head so runs are
    reproducible; shuffle then randomizes order within the subset per
    epoch. ``drop_last=True`` keeps the step count deterministic.
    """
    ops = [
        transforms.RandomResizedCrop(
            224, scale=(0.08, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
    ]
    if randaug:
        ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
    dataset = datasets.ImageFolder(TRAIN_DIR, transform=transforms.Compose(ops))
    if max_images and max_images < len(dataset):
        gen = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(dataset), generator=gen)[:max_images].tolist()
        dataset = Subset(dataset, idx)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=num_workers > 0, drop_last=True)


def heatvit_to_timm_state(floats):
    """Inverse of p2_quantize.to_heatvit_tensors.

    Returns a state_dict that timm's ``deit_tiny_patch16_224`` accepts
    (strict load), built from HeatViT-layout float tensors. Round-trips
    through to_heatvit_tensors exactly (tested in test_qat.py).
    """
    st = {}
    pe = floats["patch_w"].t().reshape(192, 16, 16, 3) \
        .permute(0, 3, 1, 2).contiguous()
    st["patch_embed.proj.weight"] = pe
    st["patch_embed.proj.bias"] = floats["patch_b"]
    st["cls_token"] = floats["cls"].reshape(1, 1, 192)
    st["pos_embed"] = floats["pos"].unsqueeze(0)
    for n in range(12):
        p = f"blocks.{n}."
        st[p + "norm1.weight"] = floats[f"b{n + 1}_gamma1"]
        st[p + "norm1.bias"] = floats[f"b{n + 1}_beta1"]
        st[p + "attn.qkv.weight"] = floats[f"b{n + 1}_wqkv"].t().contiguous()
        st[p + "attn.qkv.bias"] = floats[f"b{n + 1}_bqkv"]
        st[p + "attn.proj.weight"] = floats[f"b{n + 1}_wproj"].t().contiguous()
        st[p + "attn.proj.bias"] = floats[f"b{n + 1}_bproj"]
        st[p + "norm2.weight"] = floats[f"b{n + 1}_gamma2"]
        st[p + "norm2.bias"] = floats[f"b{n + 1}_beta2"]
        st[p + "mlp.fc1.weight"] = floats[f"b{n + 1}_w1"].t().contiguous()
        st[p + "mlp.fc1.bias"] = floats[f"b{n + 1}_b1"]
        st[p + "mlp.fc2.weight"] = floats[f"b{n + 1}_w2"].t().contiguous()
        st[p + "mlp.fc2.bias"] = floats[f"b{n + 1}_b2"]
    st["norm.weight"] = floats["final_gamma"]
    st["norm.bias"] = floats["final_beta"]
    st["head.weight"] = floats["head_w"].t().contiguous()
    st["head.bias"] = floats["head_b"]
    return st
