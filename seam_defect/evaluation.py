from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss


@torch.inference_mode()
def evaluate(model, loader, device, folder):
    """Collect predictions once and derive every report from those rows."""
    model.eval()
    batches = []
    inference_seconds = 0.0
    for images, _ in loader:
        start = perf_counter()
        batches.append(model(images.to(device)).float().softmax(1).cpu())
        inference_seconds += perf_counter() - start
    probabilities = torch.cat(batches).numpy()
    rows = (
        loader.dataset.frame[
            ["sample_id", "s3_key", "folder_1", "folder_2", "class_id"]
        ]
        .rename(columns={"folder_1": "fabric", "folder_2": "actual_class"})
        .copy()
    )
    rows["predicted_class_id"] = probabilities.argmax(1)
    rows["predicted_class"] = rows.predicted_class_id.map(
        dict(zip(rows.class_id, rows.actual_class))
    )
    rows["confidence"] = probabilities.max(1)
    rows["correct"] = rows.class_id == rows.predicted_class_id
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(probabilities.shape[1]):
        rows[f"prob_class_{i}"] = probabilities[:, i]
    rows.to_csv(folder / "all_predictions.csv", index=False)
    rows.loc[~rows.correct].sort_values("confidence", ascending=False).to_csv(
        folder / "wrong_predictions.csv", index=False
    )
    metrics = dict(
        images=len(rows),
        device=str(device),
        batch_size=loader.batch_size,
        inference_images_per_second=len(rows) / inference_seconds,
        wrong=int((~rows.correct).sum()),
        defects_called_normal=int(
            ((rows.class_id != 0) & (rows.predicted_class_id == 0)).sum()
        ),
        accuracy=accuracy_score(rows.class_id, rows.predicted_class_id),
        balanced_accuracy=balanced_accuracy_score(
            rows.class_id, rows.predicted_class_id
        ),
        macro_f1=f1_score(rows.class_id, rows.predicted_class_id, average="macro"),
        log_loss=log_loss(
            rows.class_id, probabilities, labels=np.arange(probabilities.shape[1])
        ),
    )
    for group in ["actual_class", "fabric"]:
        summaries = []
        for name, part in rows.groupby(group):
            summaries.append(
                {
                    group: name,
                    "images": len(part),
                    "wrong": int((~part.correct).sum()),
                    (
                        "recall" if group == "actual_class" else "accuracy"
                    ): part.correct.mean(),
                }
            )
        pd.DataFrame(summaries).to_csv(
            folder / f"performance_by_{group}.csv", index=False
        )
    pd.DataFrame([metrics]).to_csv(folder / "metrics.csv", index=False)
    return metrics
