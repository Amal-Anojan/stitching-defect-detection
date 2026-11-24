# Stitching defect detection

This project classifies a photograph of a sewn seam into one of 11 StitchingNet classes. The AWS Jupyter notebooks prepare the images, train and compare classifiers, and export mistakes for inspection. A Streamlit app serves the selected **EfficientNet-B0 + Feature Pyramid Network (FPN)** checkpoint and shows the three highest-scoring classes for an uploaded photo.

**Measured performance:** the selected checkpoint classified 2,199 of 2,208 internal test images correctly (99.59%). Real garment photos exposed confident errors, including a loose thread classified as stain and damage. Treat the app as an inspection demo until performance is measured on independently labeled garments from the intended camera setup.

## Dataset and inputs

Download **StitchingNet** from the authors. It contains 14,565 images at 224 × 224 pixels across 11 fabric types. Class 0 is normal. Classes 1–10 cover skipped stitch, broken stitch, pinched fabric, crooked seam, thread sagging, puckering, stain and damage, needle mark, bobbin thread pulling up, and overlapped stitch. [Dataset download](https://data.mendeley.com/datasets/6tdthsjgfc/1) · [Original publication](https://doi.org/10.1093/jcde/qwaf037)

Extract the archive and preserve the fabric and numbered class directory names. The notebooks expect object keys such as `data/A. Cotton-Poly/0. Normal/A_00_001.jpg`. Upload the directory **containing the fabric folders** to the bucket's `data/` prefix. The project uses original RGB and two derived representations: contrast-limited adaptive histogram equalization in LAB color space (LAB CLAHE), and bilateral filtering with CLAHE.

## Run on AWS

1. Open a SageMaker JupyterLab environment with Python 3.11, Git, persistent storage for three image caches and checkpoints, and an execution role with access to your S3 bucket. The notebooks list the bucket, read input images, and write inventory and enhanced data under `eda/` and `processed-data/`. Use a CUDA-capable PyTorch environment for training when available. CPU inference works for the demo.
2. In a JupyterLab terminal, clone the working branch and install the dependencies into the Python environment used by your kernel. In an existing notebook kernel, `%pip install -r requirements.txt` from the repository root targets that kernel. Restart it after installation.

   ```bash
   git clone --branch development https://github.com/Amal-Anojan/stitching-defect-detection.git
   cd stitching-defect-detection
   python -m pip install -r requirements.txt
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```

3. Upload the extracted fabric folders. Set the source directory and bucket name for your account:

   ```bash
   aws s3 sync "/path/to/extracted/fabric-folders" "s3://YOUR-BUCKET/data/"
   aws s3 ls "s3://YOUR-BUCKET/data/"
   ```

4. In notebooks 01–05, replace the project-specific bucket and region wherever they occur, including the default bucket and S3 client in `restore_images`. The original configuration uses `ap-southeast-2`. Run **01 → 02 → 03 → 04** to build the inventory, inspect the data, and create the corrected split manifest and enhanced S3 images. Notebook 00 loads automatically.
5. Run **05** to load the shared definitions. On a fresh training machine, run this cell in 05 after the corrected manifest and enhanced objects exist:

   ```python
   for variant in ("original_rgb", "lab_clahe", "bilateral_clahe"):
       restore_images(variant)
   ```

6. Run **06 → 07 → 08** for training and selection. Each stage loads the saved definitions in 05 into its kernel. Save changes to 05 before rerunning a stage. First-time ImageNet initialization needs cached torchvision weights or network access. Open **09** for optional checkpoint evaluation or real-photo comparisons.

The notebooks use PyTorch and torchvision for transfer learning, OpenCV and Pillow for image processing, pandas and NumPy for data preparation, scikit-learn for metrics, Matplotlib for plots, and boto3 for S3. The pinned core environment includes PyTorch 2.0.1 and torchvision 0.15.2. [Dependencies](requirements.txt)

## How the model was chosen

The corrected split has 10,154 training, 2,203 validation, and 2,208 test images. Splitting is stratified by fabric and class, with identical file hashes assigned to the same split. For a field evaluation, keep photos of the same garment together.

1. Compare plain ResNet18 and ResNet18 + FPN using original RGB. The FPN combines features from multiple resolutions.
2. Train three additional RGB backbones with FPN and compare all four FPN classifiers using validation macro F1.
3. Train the leading backbone with each enhanced representation. Rank the seven candidate runs by validation macro F1, which gives each class equal weight. Inspect mistakes, fabric performance and inference cost before choosing a deployment candidate.

| Historical candidate | Input | Validation macro F1 | Validation errors |
|---|---|---:|---:|
| **EfficientNet-B0 + FPN** | **Original RGB** | **0.99877** | **3** |
| ResNet18 baseline | Original RGB | 0.99767 | 6 |
| EfficientNet-B0 + FPN | Bilateral + CLAHE | 0.99755 | 6 |
| ResNet18 + FPN | Original RGB | 0.99752 | 6 |
| ConvNeXt-Tiny + FPN | Original RGB | 0.99706 | 7 |
| DenseNet121 + FPN | Original RGB | 0.99681 | 7 |
| EfficientNet-B0 + FPN | LAB CLAHE | 0.99654 | 8 |

These values describe the saved historical experiments. A fresh run of the consolidated workflow produces its own leaderboard. The two original ResNet validation summaries come from their checkpoints; the remaining five candidates have matched per-image validation records.

## Reading the selected model's test results

| Historical test metric | Result | How to use it |
|---|---:|---|
| Accuracy | 99.59% | Overall agreement across 2,208 images. Inspect the nine errors individually. |
| Macro precision / macro recall | 0.99637 / 0.99654 | Average class-level precision and recall. Per-class recall shows which defects the classifier misses. |
| Macro F1 / weighted F1 | 0.99645 / 0.99592 | Macro F1 gives rare classes equal weight; weighted F1 reflects the observed class frequencies. |
| Defects predicted as normal | 3 images | Count faulty seams that could pass inspection. |
| Worst fabric macro F1 | 0.98918 | Check whether an overall average hides a difficult fabric. |
| Log loss / expected calibration error | 0.06951 / 0.04309 | Examine predicted probabilities, especially confident mistakes. |
| Batched inference throughput | 760.65 images/s | Historical benchmark for this model and runtime. Measure single-photo latency on the intended deployment machine. |

New runs write `runs/<architecture>/<input_variant>/<run_id>/` with the configuration, split manifest, training history and best checkpoint. Under `evaluation/validation/` and `evaluation/test/`, open `wrong_predictions.csv` for individual mistakes and the class and fabric summaries for patterns. Notebook 08 writes `runs/validation_leaderboard.csv`. Notebook 09 can compare the five historical RGB checkpoints on a field CSV containing `image_path,garment_id,class_id`.

## Run the Streamlit demo

From the repository root, supply the historical EfficientNet-B0 + FPN checkpoint at `models/experiment19_fpn_backbones/efficientnet_b0_fpn/best_model.pth`, then run:

```bash
python -m streamlit run deployment/app.py
```

Open the local URL printed by Streamlit and upload a JPEG or PNG. The app corrects orientation, converts to RGB, resizes the **whole photo** to 224 × 224, applies the checkpoint's normalization, and performs CPU inference. A new clone contains source only. To use a newly trained RGB EfficientNet-B0 + FPN checkpoint, set `WEIGHTS` in `deployment/app.py` to its `best_model.pth`. Other architectures or enhanced inputs require matching inference code and preprocessing.

## File guide

| File | Purpose |
|---|---|
| [00 · Inventory helpers](Notebooks/00_inventory_helpers.ipynb) | Reusable inventory and sampling definitions for exploration. |
| [01 · Dataset inventory](Notebooks/01_dataset_inventory.ipynb) | Index S3 objects and summarize class and fabric coverage. |
| [02 · Pixel and ROI analysis](Notebooks/02_pixel_and_roi_analysis.ipynb) | Inspect pixel values, seam regions and representative images. |
| [03 · Representation screening](Notebooks/03_image_representation_screening.ipynb) | Compare preprocessing variants and select the 121-image screening sample. |
| [04 · Enhancement and splits](Notebooks/04_dataset_enhancement_and_splits.ipynb) | Generate enhanced inputs, check duplicates and save the corrected manifest. |
| [05 · Classification setup](Notebooks/05_classification_setup.ipynb) | Edit data loading, S3 restoration, model definitions, training settings and evaluation. |
| [06 · ResNet comparison](Notebooks/06_resnet18_baseline_and_fpn.ipynb) | Train the plain and FPN ResNet18 baselines. |
| [07 · Backbone comparison](Notebooks/07_fpn_backbone_comparison.ipynb) | Train the three additional RGB backbones. |
| [08 · Enhanced-input comparison](Notebooks/08_input_enhancement_comparison.ipynb) | Train two enhanced variants and rank the seven candidates. |
| [09 · Checkpoint and field evaluation](Notebooks/09_checkpoint_and_field_evaluation.ipynb) | Re-evaluate a checkpoint, rebuild historical reports or compare field predictions. |
| [Archived ResNet experiment](Notebooks/archive/historical_resnet18_experiment.ipynb) | Original baseline, FPN training and evaluation code. |
| [Archived FPN and enhancement experiments](Notebooks/archive/historical_fpn_and_enhancement_experiments.ipynb) | Original architecture and input comparisons, including exploratory enhanced ResNet runs. |
| [Streamlit app](deployment/app.py) | Accept uploads, load weights, run inference and display scores and latency. |
| [Deployment model](deployment/model.py) | Define the FPN classifier used to load the serving checkpoint. |
| [Requirements](requirements.txt) | Python dependencies for training, analysis and serving. |
| [Git ignore rules](.gitignore) | Keep datasets, checkpoints, generated reports, CSVs and local caches out of Git. |
