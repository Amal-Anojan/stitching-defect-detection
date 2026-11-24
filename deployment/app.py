from pathlib import Path
from time import perf_counter
import sys

import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from seam_defect.models import FPNClassifier
from seam_defect.data import evaluation_transform
WEIGHTS = ROOT / "models/experiment19_fpn_backbones/efficientnet_b0_fpn/best_model.pth"


@st.cache_resource
def load_classifier():
    checkpoint = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    network = FPNClassifier(checkpoint["architecture"], checkpoint["num_classes"], pretrained=False)
    network.load_state_dict(checkpoint["model_state_dict"])
    size = checkpoint["image_size"]
    preprocess = evaluation_transform(size, checkpoint["imagenet_mean"], checkpoint["imagenet_std"])
    return network.eval(), preprocess, checkpoint["class_names"]


st.title("Seam defect inspection")
st.caption("EfficientNet-B0 + FPN · Original RGB · 11 seam classes")
upload = st.file_uploader("Choose a seam photograph", type=["jpg", "jpeg", "png"])
if upload:
    try:
        with Image.open(upload) as source:
            photo = ImageOps.exif_transpose(source).convert("RGB")
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
        st.error("This image could not be opened.")
        st.stop()
    network, preprocess, labels = load_classifier()
    start = perf_counter()
    with torch.inference_mode():
        scores, indices = network(preprocess(photo).unsqueeze(0)).softmax(1)[0].topk(3)
    elapsed = perf_counter() - start
    st.image(photo, width=360)
    st.subheader(labels[indices[0].item()])
    for score, index in zip(scores.tolist(), indices.tolist()):
        st.write(f"{labels[index]} — {score:.1%}")
    st.caption(f"Model scores · Single-image preprocessing and inference {elapsed * 1000:.0f} ms")
