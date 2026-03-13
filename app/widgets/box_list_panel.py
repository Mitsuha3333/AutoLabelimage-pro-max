from __future__ import annotations

from typing import Iterable

from PyQt5 import QtCore, QtWidgets

from ..models import BoxAnnotation, ClassItem
from ..utils import format_bbox_text


class BoxListPanel(QtWidgets.QWidget):
    selection_changed = QtCore.pyqtSignal(str)
    apply_label_requested = QtCore.pyqtSignal(str)
    delete_requested = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QtWidgets.QLabel("标注框列表")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)

        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["类别", "坐标", "来源", "人工修改"])
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.table, stretch=1)

        controls = QtWidgets.QHBoxLayout()
        self.class_combo = QtWidgets.QComboBox()
        self.apply_button = QtWidgets.QPushButton("应用类别")
        self.delete_button = QtWidgets.QPushButton("删除选中")
        controls.addWidget(self.class_combo, stretch=1)
        controls.addWidget(self.apply_button)
        controls.addWidget(self.delete_button)
        layout.addLayout(controls)

    def _connect_signals(self) -> None:
        self.table.itemSelectionChanged.connect(self._emit_selection_changed)
        self.apply_button.clicked.connect(self._emit_apply_label_requested)
        self.delete_button.clicked.connect(self.delete_requested.emit)

    def set_classes(self, classes: Iterable[ClassItem]) -> None:
        self.class_combo.blockSignals(True)
        current_text = self.class_combo.currentText()
        self.class_combo.clear()
        for item in classes:
            self.class_combo.addItem(item.name)
        index = self.class_combo.findText(current_text)
        if index >= 0:
            self.class_combo.setCurrentIndex(index)
        self.class_combo.blockSignals(False)

    def set_boxes(self, boxes: Iterable[BoxAnnotation]) -> None:
        items = list(boxes)
        self.table.setRowCount(len(items))
        for row, box in enumerate(items):
            label_item = QtWidgets.QTableWidgetItem(box.label)
            label_item.setData(QtCore.Qt.UserRole, box.id)
            self.table.setItem(row, 0, label_item)
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(format_bbox_text(box.bbox_xyxy)))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(box.source))
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem("是" if box.edited_by_user else "否"))

    def select_box(self, box_id: str) -> None:
        if not box_id:
            self.table.clearSelection()
            return
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.data(QtCore.Qt.UserRole) == box_id:
                self.table.selectRow(row)
                return

    def selected_box_id(self) -> str:
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return ""
        item = self.table.item(selected_rows[0].row(), 0)
        return item.data(QtCore.Qt.UserRole) if item else ""

    def _emit_selection_changed(self) -> None:
        self.selection_changed.emit(self.selected_box_id())

    def _emit_apply_label_requested(self) -> None:
        label = self.class_combo.currentText().strip()
        if label:
            self.apply_label_requested.emit(label)
