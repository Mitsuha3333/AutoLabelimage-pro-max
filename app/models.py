from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple
from uuid import uuid4


BBox = Tuple[float, float, float, float]


def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:8]}"


class ImageStatus(str, Enum):
    UNPROCESSED = "未处理"
    ROUGH_DONE = "粗标完成"
    REFINED = "精修完成"
    CONFIRMED = "已确认"
    SAVED = "已保存"
    FAILED = "失败"


@dataclass
class ClassItem:
    id: str
    name: str
    color: str
    visible: bool = True


@dataclass
class BoxAnnotation:
    id: str
    label: str
    class_id: int
    source: str
    bbox_xyxy: BBox
    bbox_rel_1000: Optional[BBox] = None
    edited_by_user: bool = False
    visible: bool = True
    selected: bool = False

    def clone(self, **changes: object) -> "BoxAnnotation":
        return replace(self, **changes)


@dataclass
class ImageRecord:
    image_path: str
    image_width: int
    image_height: int
    raw_qwen_boxes: List[BoxAnnotation] = field(default_factory=list)
    sam_refined_boxes: List[BoxAnnotation] = field(default_factory=list)
    final_boxes: List[BoxAnnotation] = field(default_factory=list)
    status: ImageStatus = ImageStatus.UNPROCESSED
    saved: bool = False
    last_error: str = ""

    @property
    def image_name(self) -> str:
        return Path(self.image_path).name


@dataclass
class ProjectState:
    image_dir: str = ""
    output_dir: str = ""
    classes: List[ClassItem] = field(default_factory=list)
    prompt_template: str = ""
    generated_prompt: str = ""
    current_index: int = 0
    image_records: List[ImageRecord] = field(default_factory=list)

    def current_record(self) -> Optional[ImageRecord]:
        if not self.image_records:
            return None
        if self.current_index < 0 or self.current_index >= len(self.image_records):
            return None
        return self.image_records[self.current_index]
