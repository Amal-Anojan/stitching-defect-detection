# stitching-defect-detection
An Intelligent Visual Quality and Process-Anomaly Detection System for Apparel Manufacturing

## Model selection and local deployment

The planned seven runs comprise two ResNet18 classifiers on original RGB (plain and FPN), three additional FPN backbones on original RGB, and two enhanced inputs for the leading backbone. FPN is part of the network. Original RGB remains the reference input. The two historical enhanced ResNet runs used a different head and scheduler and are exploratory.

Use validation macro F1 to rank checkpoints. Inspect defect-to-normal errors, per-class recall, fabric performance, and single-image latency on the deployment hardware before production acceptance. The local demo uses EfficientNet-B0 + FPN with original RGB. Historical test results have already been inspected. A fresh labeled production batch is needed for an independent field estimate.

From the repository root in the existing disaster-env conda environment:

```powershell
python scripts/build_selection_reports.py
python -m streamlit run deployment/app.py
```

The report builder reads the original local experiment artifacts. It writes the seven-run validation leaderboard under `reports/model_evaluation/deployment_selection/`. Within that directory, `efficientnet_b0_fpn/original_rgb/{validation,test}/` holds individual predictions, errors ordered by confidence, and summaries by fabric and true class. ResNet validation error counts use checkpoint accuracy and the reference split size. The other five runs have matching validation sample IDs and labels.

The app loads the trusted local checkpoint at `models/experiment19_fpn_backbones/efficientnet_b0_fpn/best_model.pth`. Include this separately when moving the app to another host. Checkpoints and generated reports are git-ignored. The tested local runtime uses CPU PyTorch 2.0.1, torchvision 0.15.2, Pillow 10.4.0 and Streamlit 1.39.0.
## Active experiment workflow

Run notebooks 01–04 for inventory, pixel analysis, representation screening and dataset preparation. These notebooks use S3 and the existing AWS configuration. They write local outputs relative to the repository root. Historical training notebooks, including the working-copy changes to the architecture comparison, are under `Notebooks/archive/` for provenance.

Run notebook 05 for the plain and FPN ResNet comparison, 06 for the remaining RGB backbones, and 07 for the two enhanced variants of the validation leader. Each notebook starts in a fresh kernel. Training uses the corrected split manifest, cached RGB and enhanced images, and ImageNet initialization. First-time initialization requires cached torchvision weights or network access. The shared data loader also reads the historical original-image cache.

The active model definitions, dataset, training loop, evaluation function and hyperparameters live in `seam_defect/`. All future ResNet18 + FPN runs use the original 128-channel head, batch size 64, AdamW and OneCycleLR. Enhanced inputs change only the image source. This is the reference configuration for controlled comparisons. Historical scores describe the original implementations.

Each new training invocation creates `runs/<architecture>/<input_variant>/<run_id>/`. It contains the resolved configuration, split manifest, training history, best checkpoint, and separate `evaluation/validation/` and `evaluation/test/` folders. Both splits have metrics, per-image probabilities, errors, and class and fabric summaries. Notebook 07 writes the seven-run validation leaderboard to `runs/validation_leaderboard.csv`. The deployment demo continues to use its measured historical checkpoint.

Use Python 3.11 and install `requirements.txt` in a project environment. To restore missing images after preparing the split manifest, run `python -m seam_defect.cache original_rgb`, `python -m seam_defect.cache lab_clahe`, or `python -m seam_defect.cache bilateral_clahe` with access to the project bucket.

To produce the same class, fabric and error reports for a saved checkpoint without retraining, run the following command from the root. Outputs go under `reports/model_evaluation/checkpoints/<architecture>/<input_variant>/<checkpoint_hash>/`.

```powershell
python -m seam_defect.evaluate_checkpoint models/experiment19_fpn_backbones/efficientnet_b0_fpn/best_model.pth efficientnet_b0_fpn original_rgb
```
