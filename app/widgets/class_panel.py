from __future__ import annotations

from typing import Iterable

from PyQt5 import QtCore, QtWidgets

from ..models import ClassItem


class ClassPanel(QtWidgets.QWidget):
    add_requested = QtCore.pyqtSignal()
    remove_requested = QtCore.pyqtSignal(int)
    rename_requested = QtCore.pyqtSignal(int)
    move_up_requested = QtCore.pyqtSignal(int)
    move_down_requested = QtCore.pyqtSignal(int)
    import_requested = QtCore.pyqtSignal()
    export_requested = QtCore.pyqtSignal()
    current_class_changed = QtCore.pyqtSignal(str)
    edit_template_requested = QtCore.pyqtSignal()
    restore_template_requested = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QtWidgets.QLabel("类别管理")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)

        self.class_list = QtWidgets.QListWidget()
        layout.addWidget(self.class_list, stretch=1)

        button_grid = QtWidgets.QGridLayout()
        self.add_button = QtWidgets.QPushButton("新增")
        self.remove_button = QtWidgets.QPushButton("删除")
        self.rename_button = QtWidgets.QPushButton("重命名")
        self.up_button = QtWidgets.QPushButton("上移")
        self.down_button = QtWidgets.QPushButton("下移")
        self.import_button = QtWidgets.QPushButton("导入")
        self.export_button = QtWidgets.QPushButton("导出")
        button_grid.addWidget(self.add_button, 0, 0)
        button_grid.addWidget(self.remove_button, 0, 1)
        button_grid.addWidget(self.rename_button, 1, 0)
        button_grid.addWidget(self.up_button, 1, 1)
        button_grid.addWidget(self.down_button, 2, 0)
        button_grid.addWidget(self.import_button, 2, 1)
        button_grid.addWidget(self.export_button, 3, 0, 1, 2)
        layout.addLayout(button_grid)

        layout.addWidget(QtWidgets.QLabel("当前绘制类别"))
        self.current_class_combo = QtWidgets.QComboBox()
        layout.addWidget(self.current_class_combo)

        prompt_header = QtWidgets.QHBoxLayout()
        prompt_header.addWidget(QtWidgets.QLabel("提示词预览"))
        prompt_header.addStretch(1)
        self.edit_template_button = QtWidgets.QPushButton("编辑模板")
        self.restore_button = QtWidgets.QPushButton("恢复默认")
        prompt_header.addWidget(self.edit_template_button)
        prompt_header.addWidget(self.restore_button)
        layout.addLayout(prompt_header)

        self.prompt_preview = QtWidgets.QPlainTextEdit()
        self.prompt_preview.setReadOnly(True)
        self.prompt_preview.setMinimumHeight(180)
        layout.addWidget(self.prompt_preview)

    def _connect_signals(self) -> None:
        self.add_button.clicked.connect(self.add_requested.emit)
        self.remove_button.clicked.connect(self._emit_remove_requested)
        self.rename_button.clicked.connect(self._emit_rename_requested)
        self.up_button.clicked.connect(self._emit_move_up_requested)
        self.down_button.clicked.connect(self._emit_move_down_requested)
        self.import_button.clicked.connect(self.import_requested.emit)
        self.export_button.clicked.connect(self.export_requested.emit)
        self.edit_template_button.clicked.connect(self.edit_template_requested.emit)
        self.restore_button.clicked.connect(self.restore_template_requested.emit)
        self.current_class_combo.currentTextChanged.connect(self.current_class_changed.emit)
        self.class_list.currentRowChanged.connect(self._sync_combo_from_list)

    def set_classes(self, classes: Iterable[ClassItem], current_name: str = "") -> None:
        items = list(classes)
        self.class_list.blockSignals(True)
        self.current_class_combo.blockSignals(True)

        self.class_list.clear()
        self.current_class_combo.clear()
        for item in items:
            list_item = QtWidgets.QListWidgetItem(item.name)
            list_item.setData(QtCore.Qt.UserRole, item.id)
            self.class_list.addItem(list_item)
            self.current_class_combo.addItem(item.name)

        selected_name = current_name or (items[0].name if items else "")
        if selected_name:
            combo_index = self.current_class_combo.findText(selected_name)
            if combo_index >= 0:
                self.current_class_combo.setCurrentIndex(combo_index)
            list_items = self.class_list.findItems(selected_name, QtCore.Qt.MatchExactly)
            if list_items:
                self.class_list.setCurrentItem(list_items[0])

        self.class_list.blockSignals(False)
        self.current_class_combo.blockSignals(False)

    def set_prompt_preview(self, prompt_text: str) -> None:
        self.prompt_preview.setPlainText(prompt_text)

    def current_row(self) -> int:
        return self.class_list.currentRow()

    def current_class_name(self) -> str:
        return self.current_class_combo.currentText().strip()

    def _sync_combo_from_list(self, row: int) -> None:
        if row < 0:
            return
        item = self.class_list.item(row)
        if item is None:
            return
        combo_index = self.current_class_combo.findText(item.text())
        if combo_index >= 0 and combo_index != self.current_class_combo.currentIndex():
            self.current_class_combo.setCurrentIndex(combo_index)

    def _emit_remove_requested(self) -> None:
        self.remove_requested.emit(self.current_row())

    def _emit_rename_requested(self) -> None:
        self.rename_requested.emit(self.current_row())

    def _emit_move_up_requested(self) -> None:
        self.move_up_requested.emit(self.current_row())

    def _emit_move_down_requested(self) -> None:
        self.move_down_requested.emit(self.current_row())
