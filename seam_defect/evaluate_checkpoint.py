"""Re-evaluate a saved seven-run checkpoint using the current common reporting schema."""
import argparse
from hashlib import sha256
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import VARIANTS, ROOT, SeamImages, load_manifest
from .evaluation import evaluate
from .factory import build_model
from .training import BACKBONES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("architecture", choices=("resnet18_baseline", *BACKBONES))
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    net = build_model(args.architecture, dropout=0, pretrained=False)
    net.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    frame = load_manifest()
    digest = sha256(args.checkpoint.read_bytes()).hexdigest()[:16]
    folder = ROOT / "reports/model_evaluation/checkpoints" / args.architecture / args.variant / digest
    for split in ("validation", "test"):
        dataset = SeamImages(frame.loc[frame.split == split], args.variant)
        evaluate(net, DataLoader(dataset, batch_size=args.batch_size), device, folder / split)
    print(folder)


if __name__ == "__main__":
    main()
