"""Classify dermoscopy images into folders based on predicted hair count.

Example:
    python run_batch_classification.py --input data/Test --model unet_512.h5 --output Classify --threshold 20
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from run_demo import count_hair_from_mask, segment_hair

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def classify_images(input_folder: Path, output_folder: Path, model_path: Path, threshold: int) -> None:
    """Classify images into folders according to estimated hair count."""
    output_folder.mkdir(parents=True, exist_ok=True)

    image_paths = [path for path in input_folder.iterdir() if path.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not image_paths:
        print(f"No supported images found in {input_folder}")
        return

    for image_path in image_paths:
        _, predicted_mask = segment_hair(image_path, model_path)
        hair_count, _ = count_hair_from_mask(predicted_mask)

        if hair_count >= threshold:
            target_folder = output_folder / f"{threshold}_plus"
        else:
            target_folder = output_folder / str(hair_count)

        target_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_folder / image_path.name)
        print(f"{image_path.name}: {hair_count} hairs -> {target_folder}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify images by predicted hair count.")
    parser.add_argument("--input", required=True, type=Path, help="Folder containing input images")
    parser.add_argument("--model", default=Path("unet_512.h5"), type=Path, help="Path to trained model file")
    parser.add_argument("--output", default=Path("Classify"), type=Path, help="Output classification folder")
    parser.add_argument("--threshold", default=20, type=int, help="Maximum folder class before using threshold_plus")
    args = parser.parse_args()

    classify_images(args.input, args.output, args.model, args.threshold)


if __name__ == "__main__":
    main()
