from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image

from .models import ImageRecord
from .utils import SUPPORTED_IMAGE_EXTS


def scan_images(image_dir: str | Path) -> List[Path]:
    root = Path(image_dir)
    if not root.exists():
        return []
    return sorted(
        [path for path in root.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS],
        key=lambda item: item.name.lower(),
    )


def read_image_size(image_path: str | Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def build_image_records(image_dir: str | Path) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    for image_path in scan_images(image_dir):
        width, height = read_image_size(image_path)
        records.append(
            ImageRecord(
                image_path=str(image_path),
                image_width=width,
                image_height=height,
            )
        )
    return records
