from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(
    "reports/eda/exp16_duplicate_detection/dataset_split_manifest_corrected.csv"
)
VARIANTS = ("original_rgb", "lab_clahe", "bilateral_clahe")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_manifest():
    path = ROOT / MANIFEST
    if not path.exists():
        path = ROOT / "Notebooks" / MANIFEST
    df = pd.read_csv(path)
    df["class_id"] = df.folder_2.str.extract(r"^(\d+)")[0].astype(int)
    if (
        not df.sample_id.is_unique
        or not df.s3_key.is_unique
        or set(df.split) != {"train", "validation", "test"}
        or set(df.class_id) != set(range(11))
    ):
        raise ValueError(
            "Invalid sample identifiers, splits or classes in corrected manifest"
        )
    return df


def evaluation_transform(size=224, mean=MEAN, std=STD):
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


class SeamImages(Dataset):
    def __init__(self, frame, variant, training=False):
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.frame = frame.reset_index(drop=True)
        suffixes = [Path(key).relative_to("data") for key in frame.s3_key]
        roots = (
            [
                ROOT / "data",
                ROOT / "data/training_cache/original/data",
                ROOT / "Notebooks/data/training_cache/original_rgb",
            ]
            if variant == "original_rgb"
            else [ROOT / "data/training_cache/enhanced" / variant]
        )
        self.paths = []
        for suffix in suffixes:
            path = next(
                (base / suffix for base in roots if (base / suffix).is_file()), None
            )
            if path is None:
                raise FileNotFoundError(roots[0] / suffix)
            self.paths.append(path)
        steps = evaluation_transform().transforms
        if training:
            steps[1:1] = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(
                    degrees=4, translate=(0.02, 0.02), scale=(0.97, 1.03), fill=128
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.08, 0.08, 0.05, 0.01)], p=0.5
                ),
            ]
        self.transform = transforms.Compose(steps)
        self.labels = self.frame.class_id.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return (
                self.transform(ImageOps.exif_transpose(image).convert("RGB")),
                self.labels[index],
            )
