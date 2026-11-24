"""Restore missing experiment images from the dataset's S3 bucket."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

from .data import ROOT, VARIANTS, load_manifest


def main():
    import boto3

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument(
        "--bucket", default="amal-stitchingnet-data-903473565738-ap-southeast-2-an"
    )
    args = parser.parse_args()
    s3 = boto3.client("s3", region_name="ap-southeast-2")

    def download(key):
        relative = Path(key).relative_to("data")
        if args.variant == "original_rgb":
            destination = ROOT / key
            if any(
                (base / relative).is_file()
                for base in (
                    ROOT / "data/training_cache/original/data",
                    ROOT / "Notebooks/data/training_cache/original_rgb",
                )
            ):
                return
        else:
            destination = (
                ROOT / "data/training_cache/enhanced" / args.variant / relative
            )
            key = f"processed-data/exp17_multiview/{args.variant}/{relative.as_posix()}"
        if destination.is_file() and destination.stat().st_size:
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix("." + uuid4().hex + ".part")
        try:
            s3.download_file(args.bucket, key, str(temporary))
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(download, load_manifest().s3_key))


if __name__ == "__main__":
    main()
