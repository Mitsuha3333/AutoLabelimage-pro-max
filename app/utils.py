from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Iterable, Tuple

from .models import BBox


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLOR_PALETTE = [
    "#ff6b6b",
    "#4ecdc4",
    "#ffe66d",
    "#5dade2",
    "#a569bd",
    "#58d68d",
    "#f5b041",
    "#ec7063",
]


def pick_color(index: int) -> str:
    return COLOR_PALETTE[index % len(COLOR_PALETTE)]


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def clamp_bbox(bbox: BBox, image_width: int, image_height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    left = clamp(min(x1, x2), 0, image_width)
    top = clamp(min(y1, y2), 0, image_height)
    right = clamp(max(x1, x2), 0, image_width)
    bottom = clamp(max(y1, y2), 0, image_height)
    return left, top, right, bottom


def bbox_area(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def rel1000_to_xyxy(rel_bbox: BBox, image_width: int, image_height: int) -> BBox:
    x1, y1, x2, y2 = rel_bbox
    pixel_bbox = (
        image_width * x1 / 1000.0,
        image_height * y1 / 1000.0,
        image_width * x2 / 1000.0,
        image_height * y2 / 1000.0,
    )
    return clamp_bbox(pixel_bbox, image_width, image_height)


def xyxy_to_rel1000(bbox: BBox, image_width: int, image_height: int) -> BBox:
    x1, y1, x2, y2 = clamp_bbox(bbox, image_width, image_height)
    if image_width <= 0 or image_height <= 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        x1 / image_width * 1000.0,
        y1 / image_height * 1000.0,
        x2 / image_width * 1000.0,
        y2 / image_height * 1000.0,
    )


def xyxy_to_yolo(bbox: BBox, image_width: int, image_height: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = clamp_bbox(bbox, image_width, image_height)
    if image_width <= 0 or image_height <= 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        ((x1 + x2) / 2.0) / image_width,
        ((y1 + y2) / 2.0) / image_height,
        (x2 - x1) / image_width,
        (y2 - y1) / image_height,
    )


def yolo_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> BBox:
    box_width = width * image_width
    box_height = height * image_height
    center_x = x_center * image_width
    center_y = y_center * image_height
    return clamp_bbox(
        (
            center_x - box_width / 2.0,
            center_y - box_height / 2.0,
            center_x + box_width / 2.0,
            center_y + box_height / 2.0,
        ),
        image_width,
        image_height,
    )


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def image_to_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def format_bbox_text(bbox: BBox) -> str:
    x1, y1, x2, y2 = bbox
    return f"{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}"


def unique_names(names: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw_name in names:
        name = raw_name.strip()
        if not name or name in seen:
            continue
        result.append(name)
        seen.add(name)
    return result
