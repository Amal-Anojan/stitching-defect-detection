"""Train one architecture/input pair and evaluate its best validation checkpoint."""

import json
import random
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from .data import ROOT, MEAN, STD, SeamImages, load_manifest
from .evaluation import evaluate
from .factory import build_model

CONFIGS = json.loads((ROOT / "seam_defect/training_config.json").read_text())
BACKBONES = (
    "resnet18_fpn",
    "densenet121_fpn",
    "convnext_tiny_fpn",
    "efficientnet_b0_fpn",
)


def train(name, variant="original_rgb"):
    is_resnet = name in ("resnet18_baseline", "resnet18_fpn")
    cfg = CONFIGS["resnet18_fpn" if is_resnet else name]
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = load_manifest()
    loaders = {}
    for split in ("train", "validation", "test"):
        dataset = SeamImages(
            frame.loc[frame.split == split], variant, training=split == "train"
        )
        loaders[split] = DataLoader(
            dataset, batch_size=cfg["batch_size"], shuffle=split == "train"
        )
    model = build_model(name, cfg["dropout"]).to(device)
    head_prefixes = ("fc.", "classifier.", "lateral", "smooth", "fpn_head.")
    backbone, head = [], []
    for key, parameter in model.named_parameters():
        (head if key.startswith(head_prefixes) else backbone).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone, "lr": cfg["backbone_lr"]},
            {"params": head, "lr": cfg["head_lr"]},
        ],
        weight_decay=cfg["weight_decay"],
    )
    weights = None
    if is_resnet:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[cfg["backbone_lr"], cfg["head_lr"]],
            epochs=cfg["max_epochs"],
            steps_per_epoch=len(loaders["train"]),
        )
    else:
        counts = torch.bincount(
            torch.tensor(loaders["train"].dataset.labels), minlength=11
        ).float()
        weights = counts.rsqrt()
        weights = (weights / weights.mean()).to(device)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=cfg["scheduler_patience"],
            factor=cfg["scheduler_factor"],
            min_lr=[cfg["min_backbone_lr"], cfg["min_head_lr"]],
        )
    criterion = nn.CrossEntropyLoss(
        weight=weights, label_smoothing=cfg["label_smoothing"]
    )
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    )
    folder = ROOT / "runs" / name / variant / run_id
    folder.mkdir(parents=True)
    metadata = dict(
        architecture=name,
        input_variant=variant,
        config=cfg,
        seed=seed,
        scheduler=type(scheduler).__name__,
        image_size=224,
        imagenet_mean=MEAN,
        imagenet_std=STD,
        num_classes=11,
        class_names=frame.sort_values("class_id").folder_2.drop_duplicates().tolist(),
    )
    (folder / "config.json").write_text(json.dumps(metadata, indent=2))
    frame[["sample_id", "s3_key", "split", "class_id"]].to_csv(
        folder / "split_manifest.csv", index=False
    )
    best, best_epoch = -1, 0
    history = []
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        batches = len(loaders["train"])
        accumulation = cfg["accumulation_steps"]
        for step, (images, labels) in enumerate(loaders["train"]):
            logits = model(images.to(device))
            group_size = min(
                accumulation, batches - (step // accumulation) * accumulation
            )
            (criterion(logits, labels.to(device)) / group_size).backward()
            if (step + 1) % accumulation == 0 or step + 1 == batches:
                optimizer.step()
                optimizer.zero_grad()
                if is_resnet:
                    scheduler.step()
        model.eval()
        with torch.inference_mode():
            predictions = torch.cat(
                [
                    model(images.to(device)).argmax(1).cpu()
                    for images, _ in loaders["validation"]
                ]
            )
        score = f1_score(
            loaders["validation"].dataset.labels, predictions, average="macro"
        )
        if not is_resnet:
            scheduler.step(score)
        history.append({"epoch": epoch, "validation_macro_f1": score})
        pd.DataFrame(history).to_csv(folder / "training_history.csv", index=False)
        if score > best:
            best, best_epoch = score, epoch
            torch.save(
                dict(
                    **metadata,
                    model_state_dict=model.state_dict(),
                    epoch=epoch,
                    validation_macro_f1=float(score),
                ),
                folder / "best_model.pth",
            )
        if (
            epoch >= cfg.get("min_epochs", 1)
            and epoch - best_epoch >= cfg["early_stopping_patience"]
        ):
            break
    checkpoint = torch.load(
        folder / "best_model.pth", map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    for split in ("validation", "test"):
        evaluate(model, loaders[split], device, folder / "evaluation" / split)
    return folder


def leading_rgb_backbone():
    scores = {}
    for name in BACKBONES:
        paths = sorted(
            (ROOT / "runs" / name / "original_rgb").glob(
                "*/evaluation/validation/metrics.csv"
            )
        )
        if not paths:
            raise FileNotFoundError(f"Complete the original RGB run for {name} first.")
        scores[name] = pd.read_csv(paths[-1]).macro_f1.iloc[0]
    return max(scores, key=scores.get)


def completed_comparison():
    """Summarize the latest completed runs in the three-stage protocol."""
    leader = leading_rgb_backbone()
    pairs = [(name, "original_rgb") for name in ("resnet18_baseline", *BACKBONES)]
    pairs.extend((leader, variant) for variant in ("lab_clahe", "bilateral_clahe"))
    records = []
    for name, variant in pairs:
        paths = sorted(
            (ROOT / "runs" / name / variant).glob("*/evaluation/test/metrics.csv")
        )
        if not paths:
            raise FileNotFoundError(f"Complete the run for {name}/{variant} first.")
        run = paths[-1].parents[2]
        row = pd.read_csv(run / "evaluation/validation/metrics.csv").iloc[0].to_dict()
        records.append(dict(model=name, input_variant=variant, run=run.name, **row))
    board = pd.DataFrame(records).sort_values(
        ["macro_f1", "defects_called_normal", "wrong"], ascending=[False, True, True]
    )
    board.to_csv(ROOT / "runs/validation_leaderboard.csv", index=False)
    return board
