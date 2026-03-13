from __future__ import annotations

import copy
import json
import threading
from pathlib import Path
from typing import Iterable, List

from PyQt5 import QtCore

from .config import AppConfig
from .models import BoxAnnotation, ImageRecord, ImageStatus, make_id
from .project_store import ProjectStore
from .qwen_client import QwenClient
from .sam_refiner import SamRefiner
from .utils import ensure_dir, xyxy_to_rel1000
from .yolo_io import annotation_path_for_image, save_yolo_annotation


class TaskWorker(QtCore.QObject):
    progress_changed = QtCore.pyqtSignal(int, int, str)
    record_processed = QtCore.pyqtSignal(int, object)
    status_message = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(
        self,
        config: AppConfig,
        records: List[ImageRecord],
        indices: Iterable[int],
        class_names: List[str],
        prompt: str,
        output_dir: str,
        auto_save: bool,
    ) -> None:
        super().__init__()
        self.config = config
        self.records = records
        self.indices = list(indices)
        self.class_names = class_names
        self.prompt = prompt
        self.output_dir = output_dir
        self.auto_save = auto_save
        self.qwen_client = QwenClient(config.qwen)
        self.sam_refiner = SamRefiner(config.sam)
        self._stop_requested = False
        self._paused = False
        self._pause_condition = threading.Condition()

    @QtCore.pyqtSlot()
    def run(self) -> None:
        total = len(self.indices)
        try:
            for position, index in enumerate(self.indices, start=1):
                self._wait_if_paused()
                if self._stop_requested:
                    break

                record = copy.deepcopy(self.records[index])
                self.status_message.emit(f"处理中: {record.image_name}")
                try:
                    rough_boxes = self.qwen_client.detect(record.image_path, self.class_names, self.prompt)
                    record.raw_qwen_boxes = rough_boxes
                    record.status = ImageStatus.ROUGH_DONE

                    refined_boxes = self.sam_refiner.refine_boxes(record.image_path, rough_boxes)
                    record.sam_refined_boxes = refined_boxes
                    record.final_boxes = self._build_final_boxes(refined_boxes, record)
                    record.status = ImageStatus.REFINED
                    record.saved = False
                    record.last_error = self.sam_refiner.last_error

                    if self.auto_save and self.output_dir:
                        self._save_yolo(record)
                        record.saved = True
                        record.status = ImageStatus.SAVED

                    self._save_intermediate_json(record)
                except Exception as exc:
                    record.status = ImageStatus.FAILED
                    record.last_error = str(exc)

                self.record_processed.emit(index, record)
                self.progress_changed.emit(position, total, record.status.value)
        finally:
            self.finished.emit()

    def pause(self) -> None:
        with self._pause_condition:
            self._paused = True

    def resume(self) -> None:
        with self._pause_condition:
            self._paused = False
            self._pause_condition.notify_all()

    def stop(self) -> None:
        self._stop_requested = True
        self.resume()

    def _wait_if_paused(self) -> None:
        with self._pause_condition:
            while self._paused and not self._stop_requested:
                self._pause_condition.wait(0.2)

    def _build_final_boxes(self, refined_boxes: Iterable[BoxAnnotation], record: ImageRecord) -> list[BoxAnnotation]:
        result: list[BoxAnnotation] = []
        for box in refined_boxes:
            result.append(
                box.clone(
                    id=make_id("box"),
                    source="final",
                    bbox_rel_1000=xyxy_to_rel1000(box.bbox_xyxy, record.image_width, record.image_height),
                    selected=False,
                )
            )
        return result

    def _save_yolo(self, record: ImageRecord) -> None:
        target = annotation_path_for_image(record.image_path, self.output_dir)
        save_yolo_annotation(
            record.final_boxes,
            self.class_names,
            record.image_width,
            record.image_height,
            target,
        )

    def _save_intermediate_json(self, record: ImageRecord) -> None:
        if not self.output_dir or not self.config.project.save_intermediate_json:
            return
        target_dir = ensure_dir(Path(self.output_dir) / "intermediate")
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

    def _box_to_dict(self, box: BoxAnnotation) -> dict[str, object]:
        return {
            "id": box.id,
            "label": box.label,
            "class_id": box.class_id,
            "source": box.source,
            "bbox_xyxy": list(box.bbox_xyxy),
            "bbox_rel_1000": list(box.bbox_rel_1000) if box.bbox_rel_1000 else None,
            "edited_by_user": box.edited_by_user,
        }


class TaskManager(QtCore.QObject):
    progress_changed = QtCore.pyqtSignal(int, int, str)
    record_processed = QtCore.pyqtSignal(int, object)
    status_message = QtCore.pyqtSignal(str)
    running_changed = QtCore.pyqtSignal(bool)

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self._thread: QtCore.QThread | None = None
        self._worker: TaskWorker | None = None
        self._store: ProjectStore | None = None
        self._paused = False

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    @property
    def is_paused(self) -> bool:
        return self._paused

    def start_semi_auto(self, store: ProjectStore) -> bool:
        return self._start(store, [store.state.current_index], auto_save=False)

    def start_full_auto(self, store: ProjectStore) -> bool:
        indices = list(range(len(store.state.image_records)))
        auto_save = bool(store.state.output_dir and self.config.project.auto_save)
        return self._start(store, indices, auto_save=auto_save)

    def pause(self) -> None:
        if self._worker and self.is_running:
            self._worker.pause()
            self._paused = True

    def resume(self) -> None:
        if self._worker and self.is_running:
            self._worker.resume()
            self._paused = False

    def stop(self) -> None:
        if self._worker and self.is_running:
            self._worker.stop()
            self._paused = False

    def _start(self, store: ProjectStore, indices: List[int], auto_save: bool) -> bool:
        if self.is_running or not indices:
            return False

        self._store = store
        self._paused = False
        self._thread = QtCore.QThread()
        self._worker = TaskWorker(
            config=self.config,
            records=copy.deepcopy(store.state.image_records),
            indices=indices,
            class_names=store.class_names(),
            prompt=store.state.generated_prompt,
            output_dir=store.state.output_dir,
            auto_save=auto_save,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress_changed.connect(self.progress_changed.emit)
        self._worker.status_message.connect(self.status_message.emit)
        self._worker.record_processed.connect(self._handle_record_processed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()
        self.running_changed.emit(True)
        return True

    def _handle_record_processed(self, index: int, record: ImageRecord) -> None:
        if self._store is not None:
            self._store.merge_record(index, record)
        self.record_processed.emit(index, record)

    def _cleanup_thread(self) -> None:
        if self._thread is not None:
            self._thread.deleteLater()
        self._thread = None
        self._worker = None
        self._paused = False
        self.running_changed.emit(False)
