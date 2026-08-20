"""Robust, shared image decoding for private orchid datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock

from PIL import Image, ImageFile, ImageOps


_TRUNCATED_IMAGE_LOCK = Lock()


def load_orchid_rgb(path: str | Path) -> Image.Image:
    """Decode an image as detached RGB pixels for training and leakage audit.

    Palette and transparency-bearing PNG files are valid image inputs. They are
    normalized to RGB so torchvision transforms and perceptual hashing receive
    consistent pixels. A PNG-only compatibility retry handles files that Pillow
    can render only when its conservative truncated-stream check is disabled.
    """
    image_path = Path(path)
    try:
        return _decode_rgb(image_path)
    except (OSError, ValueError, SyntaxError) as original_error:
        if image_path.suffix.lower() != ".png":
            raise
        try:
            with _TRUNCATED_IMAGE_LOCK:
                previous_setting = ImageFile.LOAD_TRUNCATED_IMAGES
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                try:
                    decoded = _decode_rgb(image_path)
                finally:
                    ImageFile.LOAD_TRUNCATED_IMAGES = previous_setting
            logging.warning("Decoded PNG with Pillow compatibility retry: %s", image_path)
            return decoded
        except (OSError, ValueError, SyntaxError) as retry_error:
            raise OSError(f"Unable to decode PNG {image_path}: {original_error}; retry failed: {retry_error}") from retry_error


def _decode_rgb(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        image.load()
        if "A" in image.getbands():
            background = Image.new("RGBA", image.size, "white")
            background.alpha_composite(image.convert("RGBA"))
            return background.convert("RGB")
        return image.convert("RGB")
