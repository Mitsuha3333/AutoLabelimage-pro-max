from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image

from .config import SamConfig
from .models import BBox, BoxAnnotation, make_id
from .utils import bbox_area, bbox_center, clamp_bbox, xyxy_to_rel1000


class SamRefiner:
    def __init__(self, config: SamConfig) -> None:
        self.config = config
        self._predictor = None
        self._model_loaded = False
        self._load_error = ""
        self._runtime_device = config.device

    @property
    def last_error(self) -> str:
        return self._load_error

    def refine_boxes(self, image_path: str | Path, boxes: Iterable[BoxAnnotation]) -> List[BoxAnnotation]:
        source_boxes = list(boxes)
        if not source_boxes:
            return []

        if not self._ensure_model_loaded():
            return [self._fallback_annotation(box) for box in source_boxes]

        with Image.open(image_path) as raw_image:
            image = np.array(raw_image.convert("RGB"))
        image_height, image_width = image.shape[:2]
        self._predictor.set_image(image)

        result: List[BoxAnnotation] = []
        for box in source_boxes:
            result.append(self._refine_single(box, image_width, image_height))
        return result

    def _ensure_model_loaded(self) -> bool:
        if self._model_loaded and self._predictor is not None:
            return True
        if self._load_error:
            return False

        checkpoint_path = Path(self.config.checkpoint)
        if not checkpoint_path.exists():
            self._load_error = f"SAM checkpoint 不存在: {checkpoint_path}"
            return False

        try:
            import torch
            from segment_anything import SamPredictor, sam_model_registry
        except Exception as exc:
            self._load_error = f"SAM 依赖不可用: {exc}"
            return False

        device = self.config.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self._runtime_device = device

        try:
            sam_model = sam_model_registry[self.config.model_type](checkpoint=str(checkpoint_path))
            sam_model.to(device=device)
            self._predictor = SamPredictor(sam_model)
            self._model_loaded = True
            return True
        except Exception as exc:
            self._load_error = f"SAM 模型加载失败: {exc}"
            return False

    def _refine_single(self, box: BoxAnnotation, image_width: int, image_height: int) -> BoxAnnotation:
        expanded_box = self._expand_box(box.bbox_xyxy, image_width, image_height)
        try:
            masks, scores, _ = self._predictor.predict(
                box=np.array(expanded_box, dtype=np.float32),
                multimask_output=True,
            )
        except Exception:
            return self._fallback_annotation(box)

        if masks is None or len(masks) == 0:
            return self._fallback_annotation(box)

        best_index = int(np.argmax(scores)) if len(scores) else 0
        mask = masks[best_index]
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return self._fallback_annotation(box)

        candidate = clamp_bbox(
            (
                float(xs.min()),
                float(ys.min()),
                float(xs.max() + 1),
                float(ys.max() + 1),
            ),
            image_width,
            image_height,
        )
        if not self._passes_quality_check(box.bbox_xyxy, candidate):
            candidate = box.bbox_xyxy

        return box.clone(
            id=make_id("box"),
            source="sam",
            bbox_xyxy=candidate,
            bbox_rel_1000=xyxy_to_rel1000(candidate, image_width, image_height),
            selected=False,
        )

    def _expand_box(self, bbox: BBox, image_width: int, image_height: int) -> BBox:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        dx = width * self.config.expand_ratio
        dy = height * self.config.expand_ratio
        return clamp_bbox((x1 - dx, y1 - dy, x2 + dx, y2 + dy), image_width, image_height)

    def _passes_quality_check(self, original: BBox, candidate: BBox) -> bool:
        original_area = bbox_area(original)
        candidate_area = bbox_area(candidate)
        if candidate_area < self.config.min_box_area:
            return False
        if original_area <= 0:
            return True

        ratio = candidate_area / original_area
        if ratio < self.config.min_area_ratio or ratio > self.config.max_area_ratio:
            return False

        original_center = bbox_center(original)
        candidate_center = bbox_center(candidate)
        max_reference = max(original[2] - original[0], original[3] - original[1], 1.0)
        center_shift = np.hypot(
            candidate_center[0] - original_center[0],
            candidate_center[1] - original_center[1],
        )
        return center_shift / max_reference <= self.config.max_center_shift_ratio

    def _fallback_annotation(self, box: BoxAnnotation) -> BoxAnnotation:
        return box.clone(
            id=make_id("box"),
            source="sam",
            bbox_xyxy=box.bbox_xyxy,
            bbox_rel_1000=box.bbox_rel_1000,
            selected=False,
        )
