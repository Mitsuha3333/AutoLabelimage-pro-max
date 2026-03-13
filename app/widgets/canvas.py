from __future__ import annotations

import copy
from typing import Iterable

from PyQt5 import QtCore, QtGui, QtWidgets

from ..models import BoxAnnotation, ClassItem, ImageRecord, make_id
from ..utils import clamp, clamp_bbox


HANDLE_NAMES = ("nw", "n", "ne", "e", "se", "s", "sw", "w")


class AnnotationCanvas(QtWidgets.QWidget):
    boxes_changed = QtCore.pyqtSignal(object)
    selection_changed = QtCore.pyqtSignal(str)
    message_requested = QtCore.pyqtSignal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(640, 480)
        self._pixmap = QtGui.QPixmap()
        self._record: ImageRecord | None = None
        self._raw_boxes: list[BoxAnnotation] = []
        self._sam_boxes: list[BoxAnnotation] = []
        self._final_boxes: list[BoxAnnotation] = []
        self._class_map: dict[str, ClassItem] = {}
        self._current_class = ""
        self._show_raw = True
        self._show_sam = True
        self._show_final = True
        self._fit_scale = 1.0
        self._zoom = 1.0
        self._pan = QtCore.QPointF(0.0, 0.0)
        self._mode = ""
        self._selected_box_id = ""
        self._temp_bbox: tuple[float, float, float, float] | None = None
        self._drag_start_image = QtCore.QPointF()
        self._drag_start_view = QtCore.QPointF()
        self._original_bbox: tuple[float, float, float, float] | None = None
        self._resize_handle = ""

    def set_record(self, record: ImageRecord | None) -> None:
        self._record = copy.deepcopy(record)
        self._pixmap = QtGui.QPixmap(record.image_path) if record else QtGui.QPixmap()
        self._raw_boxes = self._copy_boxes(record.raw_qwen_boxes if record else [])
        self._sam_boxes = self._copy_boxes(record.sam_refined_boxes if record else [])
        self._final_boxes = self._copy_boxes(record.final_boxes if record else [])
        self._selected_box_id = ""
        self._zoom = 1.0
        self._pan = QtCore.QPointF(0.0, 0.0)
        self._recalculate_fit_scale()
        self.update()

    def set_classes(self, classes: Iterable[ClassItem]) -> None:
        self._class_map = {item.name: item for item in classes}
        self.update()

    def set_current_class(self, class_name: str) -> None:
        self._current_class = class_name.strip()

    def set_source_visibility(self, show_raw: bool, show_sam: bool, show_final: bool) -> None:
        self._show_raw = show_raw
        self._show_sam = show_sam
        self._show_final = show_final
        self.update()

    def select_box(self, box_id: str) -> None:
        if box_id == self._selected_box_id and all(
            box.selected == (box.id == box_id) for box in self._final_boxes
        ):
            return
        self._selected_box_id = box_id
        for box in self._final_boxes:
            box.selected = box.id == box_id
        self.selection_changed.emit(box_id)
        self.update()

    def apply_label_to_selected(self, label: str) -> bool:
        box = self._selected_box()
        if box is None:
            return False
        class_item = self._class_map.get(label)
        box.label = label
        box.class_id = list(self._class_map).index(label) if class_item else -1
        box.source = "manual"
        box.edited_by_user = True
        self._emit_boxes_changed()
        return True

    def delete_selected_box(self) -> None:
        if not self._selected_box_id:
            return
        self._final_boxes = [box for box in self._final_boxes if box.id != self._selected_box_id]
        self._selected_box_id = ""
        self.selection_changed.emit("")
        self._emit_boxes_changed()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#1f252b"))
        if self._pixmap.isNull():
            painter.setPen(QtGui.QColor("#8795a1"))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "请选择图片文件夹")
            return

        target_rect = self._image_rect()
        painter.drawPixmap(target_rect.toRect(), self._pixmap)

        if self._show_raw:
            for box in self._raw_boxes:
                self._draw_box(painter, box, target_rect, style="raw")
        if self._show_sam:
            for box in self._sam_boxes:
                self._draw_box(painter, box, target_rect, style="sam")
        if self._show_final:
            for box in self._final_boxes:
                self._draw_box(painter, box, target_rect, style="final")

        if self._temp_bbox is not None:
            temp_box = BoxAnnotation(
                id="temp",
                label=self._current_class or "new",
                class_id=-1,
                source="manual",
                bbox_xyxy=self._temp_bbox,
            )
            self._draw_box(painter, temp_box, target_rect, style="temp")

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._recalculate_fit_scale()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if self._pixmap.isNull():
            return
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 0.9
        self._zoom = clamp(self._zoom * factor, 0.2, 10.0)
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._pixmap.isNull():
            return
        self.setFocus()
        view_pos = QtCore.QPointF(event.pos())
        image_pos = self._view_to_image(view_pos)

        if event.button() == QtCore.Qt.RightButton:
            self._mode = "pan"
            self._drag_start_view = view_pos
            return

        if event.button() != QtCore.Qt.LeftButton:
            return

        handle = self._hit_handle(view_pos)
        if handle and self._selected_box():
            self._mode = "resize"
            self._resize_handle = handle
            self._drag_start_image = image_pos
            self._original_bbox = self._selected_box().bbox_xyxy
            return

        hit_box = self._hit_box(view_pos)
        if hit_box is not None:
            self.select_box(hit_box.id)
            self._mode = "move"
            self._drag_start_image = image_pos
            self._original_bbox = hit_box.bbox_xyxy
            return

        if not self._current_class:
            self.message_requested.emit("请先选择当前绘制类别")
            return

        self.select_box("")
        self._mode = "create"
        clamped_pos = self._clamp_point(image_pos)
        self._drag_start_image = clamped_pos
        self._temp_bbox = (clamped_pos.x(), clamped_pos.y(), clamped_pos.x(), clamped_pos.y())
        self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._pixmap.isNull() or not self._mode:
            return
        view_pos = QtCore.QPointF(event.pos())
        image_pos = self._clamp_point(self._view_to_image(view_pos))

        if self._mode == "pan":
            delta = view_pos - self._drag_start_view
            self._pan += delta
            self._drag_start_view = view_pos
            self.update()
            return

        if self._mode == "create":
            self._temp_bbox = (
                self._drag_start_image.x(),
                self._drag_start_image.y(),
                image_pos.x(),
                image_pos.y(),
            )
            self.update()
            return

        box = self._selected_box()
        if box is None or self._original_bbox is None:
            return

        if self._mode == "move":
            dx = image_pos.x() - self._drag_start_image.x()
            dy = image_pos.y() - self._drag_start_image.y()
            width = self._original_bbox[2] - self._original_bbox[0]
            height = self._original_bbox[3] - self._original_bbox[1]
            image_width, image_height = self._image_size()
            left = clamp(self._original_bbox[0] + dx, 0.0, max(0.0, image_width - width))
            top = clamp(self._original_bbox[1] + dy, 0.0, max(0.0, image_height - height))
            box.bbox_xyxy = (left, top, left + width, top + height)
            box.source = "manual"
            box.edited_by_user = True
            self.update()
            return

        if self._mode == "resize":
            box.bbox_xyxy = self._resize_bbox(self._original_bbox, image_pos, self._resize_handle)
            box.source = "manual"
            box.edited_by_user = True
            self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._pixmap.isNull():
            return

        if self._mode == "create" and self._temp_bbox is not None:
            bbox = clamp_bbox(self._temp_bbox, *self._image_size())
            if bbox[2] - bbox[0] >= 4 and bbox[3] - bbox[1] >= 4:
                class_names = list(self._class_map)
                class_id = class_names.index(self._current_class) if self._current_class in class_names else -1
                new_box = BoxAnnotation(
                    id=make_id("box"),
                    label=self._current_class,
                    class_id=class_id,
                    source="manual",
                    bbox_xyxy=bbox,
                    edited_by_user=True,
                    selected=True,
                )
                self._final_boxes.append(new_box)
                self.select_box(new_box.id)
                self._emit_boxes_changed()
            self._temp_bbox = None

        if self._mode in {"move", "resize"}:
            self._emit_boxes_changed()

        self._mode = ""
        self._resize_handle = ""
        self._original_bbox = None
        self.update()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_Delete:
            self.delete_selected_box()
            return
        super().keyPressEvent(event)

    def _copy_boxes(self, boxes: Iterable[BoxAnnotation]) -> list[BoxAnnotation]:
        return [box.clone() for box in boxes]

    def _emit_boxes_changed(self) -> None:
        for box in self._final_boxes:
            box.selected = box.id == self._selected_box_id
        self.boxes_changed.emit(self._copy_boxes(self._final_boxes))
        self.update()

    def _selected_box(self) -> BoxAnnotation | None:
        for box in self._final_boxes:
            if box.id == self._selected_box_id:
                return box
        return None

    def _image_size(self) -> tuple[int, int]:
        if self._record is None:
            return 0, 0
        return self._record.image_width, self._record.image_height

    def _recalculate_fit_scale(self) -> None:
        if self._pixmap.isNull():
            self._fit_scale = 1.0
            return
        width = max(1, self.width() - 20)
        height = max(1, self.height() - 20)
        self._fit_scale = min(width / self._pixmap.width(), height / self._pixmap.height())
        self.update()

    def _image_rect(self) -> QtCore.QRectF:
        if self._pixmap.isNull():
            return QtCore.QRectF()
        scale = self._fit_scale * self._zoom
        width = self._pixmap.width() * scale
        height = self._pixmap.height() * scale
        x = (self.width() - width) / 2.0 + self._pan.x()
        y = (self.height() - height) / 2.0 + self._pan.y()
        return QtCore.QRectF(x, y, width, height)

    def _image_to_view(self, point: QtCore.QPointF) -> QtCore.QPointF:
        rect = self._image_rect()
        scale = self._fit_scale * self._zoom
        return QtCore.QPointF(rect.x() + point.x() * scale, rect.y() + point.y() * scale)

    def _view_to_image(self, point: QtCore.QPointF) -> QtCore.QPointF:
        rect = self._image_rect()
        scale = self._fit_scale * self._zoom
        if scale <= 0:
            return QtCore.QPointF()
        return QtCore.QPointF((point.x() - rect.x()) / scale, (point.y() - rect.y()) / scale)

    def _clamp_point(self, point: QtCore.QPointF) -> QtCore.QPointF:
        image_width, image_height = self._image_size()
        return QtCore.QPointF(
            clamp(point.x(), 0.0, float(image_width)),
            clamp(point.y(), 0.0, float(image_height)),
        )

    def _draw_box(self, painter: QtGui.QPainter, box: BoxAnnotation, target_rect: QtCore.QRectF, style: str) -> None:
        if not box.visible:
            return
        x1, y1, x2, y2 = box.bbox_xyxy
        top_left = self._image_to_view(QtCore.QPointF(x1, y1))
        bottom_right = self._image_to_view(QtCore.QPointF(x2, y2))
        view_rect = QtCore.QRectF(top_left, bottom_right).normalized()

        color = QtGui.QColor(self._class_map.get(box.label, ClassItem("", "", "#5dade2")).color or "#5dade2")
        pen = QtGui.QPen(color, 2)
        if style == "raw":
            pen.setStyle(QtCore.Qt.DashLine)
            color.setAlpha(180)
        elif style == "sam":
            pen.setStyle(QtCore.Qt.DashDotLine)
            color.setAlpha(200)
        elif style == "temp":
            pen.setColor(QtGui.QColor("#f8f9fa"))
            pen.setStyle(QtCore.Qt.DashLine)
        else:
            pen.setStyle(QtCore.Qt.SolidLine)
            if box.id == self._selected_box_id:
                pen.setWidth(3)

        painter.setPen(pen)
        painter.drawRect(view_rect)
        label_text = f"{box.label} [{box.source}]"
        painter.fillRect(
            QtCore.QRectF(view_rect.x(), view_rect.y() - 20, max(90, len(label_text) * 8), 18),
            QtGui.QColor(0, 0, 0, 140),
        )
        painter.setPen(QtGui.QColor("#f8f9fa"))
        painter.drawText(int(view_rect.x() + 4), int(view_rect.y() - 6), label_text)

        if style == "final" and box.id == self._selected_box_id:
            for handle_rect in self._handle_rects(view_rect).values():
                painter.fillRect(handle_rect, QtGui.QColor("#ffffff"))

    def _hit_box(self, view_pos: QtCore.QPointF) -> BoxAnnotation | None:
        if not self._show_final:
            return None
        for box in reversed(self._final_boxes):
            rect = QtCore.QRectF(
                self._image_to_view(QtCore.QPointF(box.bbox_xyxy[0], box.bbox_xyxy[1])),
                self._image_to_view(QtCore.QPointF(box.bbox_xyxy[2], box.bbox_xyxy[3])),
            ).normalized()
            if rect.contains(view_pos):
                return box
        return None

    def _hit_handle(self, view_pos: QtCore.QPointF) -> str:
        box = self._selected_box()
        if box is None:
            return ""
        rect = QtCore.QRectF(
            self._image_to_view(QtCore.QPointF(box.bbox_xyxy[0], box.bbox_xyxy[1])),
            self._image_to_view(QtCore.QPointF(box.bbox_xyxy[2], box.bbox_xyxy[3])),
        ).normalized()
        for name, handle_rect in self._handle_rects(rect).items():
            if handle_rect.contains(view_pos.toPoint()):
                return name
        return ""

    def _handle_rects(self, rect: QtCore.QRectF) -> dict[str, QtCore.QRect]:
        points = {
            "nw": rect.topLeft(),
            "n": QtCore.QPointF(rect.center().x(), rect.top()),
            "ne": rect.topRight(),
            "e": QtCore.QPointF(rect.right(), rect.center().y()),
            "se": rect.bottomRight(),
            "s": QtCore.QPointF(rect.center().x(), rect.bottom()),
            "sw": rect.bottomLeft(),
            "w": QtCore.QPointF(rect.left(), rect.center().y()),
        }
        size = 8
        return {
            name: QtCore.QRect(int(point.x() - size / 2), int(point.y() - size / 2), size, size)
            for name, point in points.items()
        }

    def _resize_bbox(
        self,
        original_bbox: tuple[float, float, float, float],
        image_pos: QtCore.QPointF,
        handle: str,
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = original_bbox
        minimum_size = 4.0
        if "w" in handle:
            x1 = min(image_pos.x(), x2 - minimum_size)
        if "e" in handle:
            x2 = max(image_pos.x(), x1 + minimum_size)
        if "n" in handle:
            y1 = min(image_pos.y(), y2 - minimum_size)
        if "s" in handle:
            y2 = max(image_pos.y(), y1 + minimum_size)
        return clamp_bbox((x1, y1, x2, y2), *self._image_size())
