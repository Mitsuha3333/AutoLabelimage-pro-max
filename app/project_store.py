from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .class_manager import ClassManager
from .config import AppConfig
from .image_loader import build_image_records
from .models import BoxAnnotation, ImageRecord, ImageStatus, ProjectState
from .prompt_builder import DEFAULT_PROMPT_TEMPLATE, build_prompt
from .utils import ensure_dir, xyxy_to_rel1000
from .yolo_io import (
    annotation_path_for_image,
    load_classes_txt,
    load_yolo_annotation,
    save_classes_txt,
    save_yolo_annotation,
)


class ProjectStore:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.class_manager = ClassManager()
        self.state = ProjectState(prompt_template=config.qwen.prompt_template)
        self.refresh_prompt()

    def load_image_dir(self, image_dir: str) -> List[ImageRecord]:
        self.state.image_dir = image_dir
        self.state.image_records = build_image_records(image_dir)
        self.state.current_index = 0
        if self.state.output_dir:
            self.load_all_existing_annotations()
        return self.state.image_records

    def set_output_dir(self, output_dir: str) -> None:
        self.state.output_dir = str(ensure_dir(output_dir))
        classes_path = self.classes_file_path()
        if classes_path.exists():
            self.set_class_names(load_classes_txt(classes_path))
        elif self.state.classes:
            self.save_classes()
        if self.state.image_records:
            self.load_all_existing_annotations()

    def classes_file_path(self) -> Path:
        return Path(self.state.output_dir) / self.config.project.classes_file

    def class_names(self) -> list[str]:
        return [item.name for item in self.state.classes]

    def set_class_names(self, names: Iterable[str]) -> None:
        self.state.classes = self.class_manager.set_classes_from_names(names)
        self._sync_all_boxes_class_ids()
        self.refresh_prompt()

    def add_class(self, name: str) -> None:
        self.state.classes = self.class_manager.add_class(name)
        self._sync_all_boxes_class_ids()
        self.refresh_prompt()

    def remove_class(self, index: int) -> None:
        self.state.classes = self.class_manager.remove_class(index)
        self._sync_all_boxes_class_ids()
        self.refresh_prompt()

    def rename_class(self, index: int, name: str) -> None:
        self.state.classes = self.class_manager.rename_class(index, name)
        self._sync_all_boxes_class_ids()
        self.refresh_prompt()

    def move_class_up(self, index: int) -> None:
        self.state.classes = self.class_manager.move_up(index)
        self._sync_all_boxes_class_ids()
        self.refresh_prompt()

    def move_class_down(self, index: int) -> None:
        self.state.classes = self.class_manager.move_down(index)
        self._sync_all_boxes_class_ids()
        self.refresh_prompt()

    def set_prompt_template(self, template: str) -> None:
        self.state.prompt_template = template.strip() or DEFAULT_PROMPT_TEMPLATE
        self.refresh_prompt()

    def restore_default_prompt_template(self) -> None:
        self.state.prompt_template = DEFAULT_PROMPT_TEMPLATE
        self.refresh_prompt()

    def refresh_prompt(self) -> None:
        self.state.generated_prompt = build_prompt(self.state.prompt_template, self.class_names())

    def current_record(self) -> ImageRecord | None:
        return self.state.current_record()

    def set_current_index(self, index: int) -> None:
        if 0 <= index < len(self.state.image_records):
            self.state.current_index = index

    def class_id_for_label(self, label: str) -> int:
        try:
            return self.class_names().index(label)
        except ValueError:
            return -1

    def merge_record(self, index: int, record: ImageRecord) -> None:
        if 0 <= index < len(self.state.image_records):
            self.state.image_records[index] = record

    def update_final_boxes(
        self,
        index: int,
        boxes: Iterable[BoxAnnotation],
        status: ImageStatus = ImageStatus.CONFIRMED,
    ) -> None:
        if not 0 <= index < len(self.state.image_records):
            return
        record = self.state.image_records[index]
        normalized_boxes: list[BoxAnnotation] = []
        for box in boxes:
            class_id = self.class_id_for_label(box.label)
            normalized_boxes.append(
                box.clone(
                    class_id=class_id,
                    bbox_rel_1000=xyxy_to_rel1000(box.bbox_xyxy, record.image_width, record.image_height),
                    selected=False,
                )
            )
        record.final_boxes = normalized_boxes
        record.saved = False
        record.status = status
        record.last_error = ""

    def save_classes(self) -> None:
        if not self.state.output_dir:
            return
        save_classes_txt(self.classes_file_path(), self.class_names())

    def save_record(self, index: int) -> Path:
        if not self.state.output_dir:
            raise ValueError("请先设置输出文件夹")
        record = self.state.image_records[index]
        output_path = annotation_path_for_image(record.image_path, self.state.output_dir)
        save_yolo_annotation(
            record.final_boxes,
            self.class_names(),
            record.image_width,
            record.image_height,
            output_path,
        )
        record.saved = True
        record.status = ImageStatus.SAVED
        record.last_error = ""
        self.save_classes()
        return output_path

    def save_all_records(self) -> int:
        count = 0
        for index in range(len(self.state.image_records)):
            self.save_record(index)
            count += 1
        return count

    def load_annotation_for_record(self, index: int) -> None:
        if not self.state.output_dir or not self.state.classes:
            return
        if not 0 <= index < len(self.state.image_records):
            return
        record = self.state.image_records[index]
        label_path = annotation_path_for_image(record.image_path, self.state.output_dir)
        if not label_path.exists():
            return
        boxes = load_yolo_annotation(label_path, self.class_names(), record.image_width, record.image_height)
        record.final_boxes = boxes
        record.saved = True
        record.status = ImageStatus.SAVED
        record.last_error = ""

    def load_all_existing_annotations(self) -> None:
        for index in range(len(self.state.image_records)):
            self.load_annotation_for_record(index)

    def has_unsaved_changes(self) -> bool:
        return any(record.final_boxes and not record.saved for record in self.state.image_records)

    def save_intermediate_json(self, index: int) -> None:
        if not self.state.output_dir or not self.config.project.save_intermediate_json:
            return
        if not 0 <= index < len(self.state.image_records):
            return

        record = self.state.image_records[index]
        target_dir = ensure_dir(Path(self.state.output_dir) / "intermediate")
        payload = {
            "image_path": record.image_path,
            "status": record.status.value,
            "saved": record.saved,
            "last_error": record.last_error,
            "raw_qwen_boxes": [self._box_to_dict(box) for box in record.raw_qwen_boxes],
            "sam_refined_boxes": [self._box_to_dict(box) for box in record.sam_refined_boxes],
            "final_boxes": [self._box_to_dict(box) for box in record.final_boxes],
        }
        target = target_dir / f"{Path(record.image_path).stem}.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _sync_all_boxes_class_ids(self) -> None:
        for record in self.state.image_records:
            record.raw_qwen_boxes = self._sync_box_collection(record, record.raw_qwen_boxes)
            record.sam_refined_boxes = self._sync_box_collection(record, record.sam_refined_boxes)
            record.final_boxes = self._sync_box_collection(record, record.final_boxes)

    def _sync_box_collection(self, record: ImageRecord, boxes: Iterable[BoxAnnotation]) -> list[BoxAnnotation]:
        result: list[BoxAnnotation] = []
        for box in boxes:
            result.append(
                box.clone(
                    class_id=self.class_id_for_label(box.label),
                    bbox_rel_1000=xyxy_to_rel1000(box.bbox_xyxy, record.image_width, record.image_height),
                )
            )
        return result

    def _box_to_dict(self, box: BoxAnnotation) -> dict[str, object]:
        return {
            "id": box.id,
            "label": box.label,
            "class_id": box.class_id,
            "source": box.source,
            "bbox_xyxy": list(box.bbox_xyxy),
            "bbox_rel_1000": list(box.bbox_rel_1000) if box.bbox_rel_1000 else None,
            "edited_by_user": box.edited_by_user,
            "visible": box.visible,
        }
