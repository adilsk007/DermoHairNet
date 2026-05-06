"""Run DermoHairNet on one dermoscopy image.

Example:
    python run_demo.py --image temp.png --model unet_512.h5 --output outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from keras.models import load_model
from keras.utils import normalize
from PIL import Image

IMAGE_SIZE = 512


def segment_hair(image_path: Path, model_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an image, predict the hair mask, and return the RGB image and binary mask."""
    model = load_model(model_path)

    image_rgb = cv2.imread(str(image_path))
    if image_rgb is None:
        raise FileNotFoundError(f"Could not read input image: {image_path}")

    image_rgb = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

    image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    image_gray = cv2.resize(image_gray, (IMAGE_SIZE, IMAGE_SIZE))

    model_input = np.expand_dims(image_gray, axis=[0, 3])
    model_input = normalize(model_input, axis=1)

    prediction = model.predict(model_input)
    predicted_mask = np.argmax(prediction, axis=3)[0, :, :]

    return image_rgb, predicted_mask


def count_hair_from_mask(mask: np.ndarray) -> tuple[int, np.ndarray]:
    """Estimate hair count using the OpenCV contour logic from the project notebook."""
    mask_uint8 = np.uint8(mask.copy())
    mask_uint8[mask_uint8 >= 1] = 255

    _, inverted = cv2.threshold(mask_uint8, 10, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(inverted, 10, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    valid_contours = [contour for contour in contours if cv2.contourArea(contour) > 0]
    contour_image = cv2.drawContours(inverted.copy(), valid_contours, -1, (0, 0, 255), 7)

    edges = cv2.Canny(contour_image, 10, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    valid_contours = [contour for contour in contours if cv2.contourArea(contour) > 0]

    return len(valid_contours), contour_image


def save_overlay(image_rgb: np.ndarray, mask: np.ndarray, output_path: Path) -> None:
    """Save a transparent hair-mask overlay on top of the original image."""
    background = Image.fromarray(image_rgb).convert("RGBA")
    mask_image = Image.fromarray(np.uint8(mask * 255)).convert("L")

    overlay = Image.new("RGBA", background.size, (255, 255, 255, 0))
    overlay_pixels = overlay.load()
    mask_pixels = mask_image.load()

    for y in range(mask_image.height):
        for x in range(mask_image.width):
            if mask_pixels[x, y] > 0:
                overlay_pixels[x, y] = (255, 255, 255, 180)

    result = Image.alpha_composite(background, overlay).convert("RGB")
    result.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment and count hair artifacts in a dermoscopy image.")
    parser.add_argument("--image", required=True, type=Path, help="Path to input image")
    parser.add_argument("--model", default=Path("unet_512.h5"), type=Path, help="Path to trained model file")
    parser.add_argument("--output", default=Path("outputs"), type=Path, help="Output folder")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    image_rgb, predicted_mask = segment_hair(args.image, args.model)
    hair_count, contour_image = count_hair_from_mask(predicted_mask)

    mask_path = args.output / f"{args.image.stem}_predicted_mask.png"
    overlay_path = args.output / f"{args.image.stem}_overlay.png"
    contour_path = args.output / f"{args.image.stem}_hair_count_contours.png"

    cv2.imwrite(str(mask_path), np.uint8(predicted_mask * 255))
    cv2.imwrite(str(contour_path), contour_image)
    save_overlay(image_rgb, predicted_mask, overlay_path)

    print(f"Estimated hair count: {hair_count}")
    print(f"Saved predicted mask: {mask_path}")
    print(f"Saved overlay: {overlay_path}")
    print(f"Saved contour image: {contour_path}")


if __name__ == "__main__":
    main()
