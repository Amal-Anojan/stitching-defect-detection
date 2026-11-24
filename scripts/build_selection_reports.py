"""Build a validation leaderboard and deployment-model error reports from saved runs."""
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports/model_evaluation"
OUTPUT = REPORTS / "deployment_selection"
RGB = REPORTS / "experiment19_fpn_backbones"
ENHANCED = REPORTS / "experiment20_enhanced_fpn"
CHOSEN = "efficientnet_b0_fpn"
OUTPUT.mkdir(exist_ok=True)
reference = pd.read_csv(RGB / f"{CHOSEN}_validation_predictions.csv").set_index("sample_id").class_id.sort_index()
assert reference.index.is_unique
rows = []
for directory, names in [(RGB, ["efficientnet_b0_fpn", "convnext_tiny_fpn", "densenet121_fpn"]),
                         (ENHANCED, ["lab_clahe", "bilateral_clahe"])]:
    for name in names:
        path = directory / f"{name}_validation_predictions.csv"
        df = pd.read_csv(path)
        assert df.set_index("sample_id").class_id.sort_index().equals(reference)
        rows.append(dict(model=name if directory == RGB else CHOSEN,
                         input="original_rgb" if directory == RGB else name,
                         macro_f1=f1_score(df.class_id, df.predicted_class_id, average="macro"),
                         wrong=int((df.class_id != df.predicted_class_id).sum()), images=len(df),
                         source=str(path.relative_to(ROOT))))
for name in ["resnet18_baseline", "resnet18_fpn"]:
    path = ROOT / f"Notebooks/models/exp18_resnet18_fpn/{name}_best.pth"
    metrics = torch.load(path, map_location="cpu", weights_only=True)["validation_metrics"]
    rows.append(dict(model=name, input="original_rgb", macro_f1=metrics["macro_f1"],
                     wrong=round(len(reference)*(1-metrics["accuracy"])), images=len(reference),
                     source=str(path.relative_to(ROOT))))
ranking = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
ranking.to_csv(OUTPUT / "validation_leaderboard.csv", index=False)
if ranking.iloc[0].model != CHOSEN or ranking.iloc[0].input != "original_rgb":
    raise ValueError("Review the new winner before updating deployment.")
for split in ["validation", "test"]:
    df = pd.read_csv(RGB / f"{CHOSEN}_{split}_predictions.csv").rename(
        columns={"folder_1": "fabric", "folder_2": "actual_class"})
    df["correct"] = df.class_id == df.predicted_class_id
    folder = OUTPUT / CHOSEN / "original_rgb" / split
    folder.mkdir(parents=True, exist_ok=True)
    predictions = df[["sample_id", "s3_key", "fabric", "actual_class", "predicted_class", "confidence", "correct"]]
    predictions.sort_values("confidence").to_csv(folder / "all_predictions.csv", index=False)
    predictions.loc[~df.correct].sort_values("confidence", ascending=False).to_csv(folder / "wrong_predictions.csv", index=False)
    for group in ["fabric", "actual_class"]:
        summary = df.groupby(group).correct.agg(images="size", correct="sum")
        summary["wrong"] = summary.images - summary.correct
        summary["recall" if group == "actual_class" else "accuracy"] = summary.correct / summary.images
        summary.to_csv(folder / f"performance_by_{group}.csv")
    print(split, len(df), "errors", (~df.correct).sum(), "defects called normal", ((df.class_id != 0) & (df.predicted_class_id == 0)).sum())
