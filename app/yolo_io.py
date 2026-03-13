from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .models import BoxAnnotation, make_id
from .utils import ensure_dir, xyxy_to_rel1000, xyxy_to_yolo, yolo_to_xyxy


def annotation_path_for_image(image_path: str | Path, output_dir: str | Path) -> Path:
    image = Path(image_path)
    return ensure_dir(output_dir) / f"{image.stem}.txt"


def save_classes_txt(path: str | Path, class_names: Iterable[str]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    content = "\n".join(name.strip() for name in class_names if name.strip())
    target.write_text(content, encoding="utf-8")


def load_classes_txt(path: str | Path) -> List[str]:
    source = Path(path)
    if not source.exists():
        return []
    return [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_yolo_annotation(
    boxes: Iterable[BoxAnnotation],
    class_names: List[str],
    image_width: int,
    image_height: int,
    output_path: str | Path,
) -> None:
    lines: list[str] = []
    label_to_id = {name: index for index, name in enumerate(class_names)}
    for box in boxes:
        class_id = label_to_id.get(box.label, -1)
        if class_id < 0:
            continue
        x_center, y_center, width, height = xyxy_to_yolo(box.bbox_xyxy, image_width, image_height)
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    target = Path(output_path)
    ensure_dir(target.parent)
    target.write_text("\n".join(lines), encoding="utf-8")


def load_yolo_annotation(
    label_path: str | Path,
    class_names: List[str],
    image_width: int,
    image_height: int,
) -> List[BoxAnnotation]:
    source = Path(label_path)
    if not source.exists():
        return []

    result: List[BoxAnnotation] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(parts[0])
            if class_id < 0 or class_id >= len(class_names):
                continue
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError:
            continue

        bbox = yolo_to_xyxy(x_center, y_center, width, height, image_width, image_height)
        result.append(
            BoxAnnotation(
                id=make_id("box"),
                label=class_names[class_id],
                class_id=class_id,
                source="final",
                bbox_xyxy=bbox,
                bbox_rel_1000=xyxy_to_rel1000(bbox, image_width, image_height),
            )
        )
    return result
