#!/usr/bin/env python3
"""P2-A: verify the official DeiT-T checkpoint and reproduce the baseline.

Loads the locally cached timm checkpoint
(~/.cache/torch/hub/checkpoints/deit_tiny_patch16_224-a1311bcf.pth) through
the timm model, sanity-checks tensor shapes, and evaluates ImageNet Top-1
on a subset of the local val set (D:\\SEU_Liubo\\prj\\HeatViT\\data\\imagenet\\val).

Usage (run with the torch venv):

  .venv-torch\\Scripts\\python tools/p2/p2a_checkpoint.py --max-images 2000

Expected results: ~72.2% Top-1 on the full val set (timm official number).
"""

import argparse
import time

import torch


def build_timm_deit_tiny(device):
    import timm
    model = timm.create_model("deit_tiny_patch16_224", pretrained=False)
    raw = torch.load(
        r"C:\Users\Youxi\.cache\torch\hub\checkpoints"
        r"\deit_tiny_patch16_224-a1311bcf.pth",
        map_location="cpu", weights_only=False,
    )
    state = raw.get("model", raw) if isinstance(raw, dict) else raw
    extra = {k: v for k, v in raw.items() if k != "model"} \
        if isinstance(raw, dict) else {}
    if extra:
        print(f"checkpoint metadata: {extra}")
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model = model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=2000,
                        help="evaluate on the first N val images")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-dir", default=r"D:\SEU_Liubo\prj\HeatViT\data\imagenet\val")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available()
                          else "cpu")
    print(f"device={device}")

    model = build_timm_deit_tiny(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"DeiT-T parameter count: {total:,}")

    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    indices = range(min(args.max_images, len(dataset)))
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=0)

    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            pred = logits.argmax(dim=1).cpu()
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    elapsed = time.time() - start
    print(f"Top-1 on {total} val images: {100.0 * correct / total:.2f}% "
          f"({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
