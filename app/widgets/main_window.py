from __future__ import annotations

from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets

from ..config import load_config
from ..project_store import ProjectStore
from ..task_manager import TaskManager
from ..yolo_io import load_classes_txt, save_classes_txt
from .box_list_panel import BoxListPanel
from .canvas import AnnotationCanvas
from .class_panel import ClassPanel


class PromptTemplateDialog(QtWidgets.QDialog):
    def __init__(self, template_text: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("编辑提示词模板")
        self.resize(720, 520)
        layout = QtWidgets.QVBoxLayout(self)
        self.editor = QtWidgets.QPlainTextEdit()
        self.editor.setPlainText(template_text)
        layout.addWidget(self.editor)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def template_text(self) -> str:
        return self.editor.toPlainText()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config_path: str | Path, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.config_path = Path(config_path)
        self.config = load_config(config_path)
        self.store = ProjectStore(self.config)
        self.task_manager = TaskManager(self.config)
        self._selection_syncing = False
        self._build_ui()
        self._connect_signals()
        self._refresh_class_views()
        self._refresh_image_list()
        self._show_current_record()
        self._update_actions(False)

    def _build_ui(self) -> None:
        self.setWindowTitle("AutoLabelProMax")
        self.resize(1460, 900)

        self._build_toolbar()

        self.image_list = QtWidgets.QListWidget()
        self.image_list.setMinimumWidth(260)

        self.canvas = AnnotationCanvas()
        self.box_panel = BoxListPanel()
        self.class_panel = ClassPanel()

        left_group = QtWidgets.QGroupBox("图片列表")
        left_layout = QtWidgets.QVBoxLayout(left_group)
        left_layout.addWidget(self.image_list)

        center_group = QtWidgets.QGroupBox("标注画布")
        center_layout = QtWidgets.QVBoxLayout(center_group)
        center_layout.addWidget(self.canvas)

        right_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right_splitter.addWidget(self.box_panel)
        right_splitter.addWidget(self.class_panel)
        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 4)

        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_splitter.addWidget(left_group)
        main_splitter.addWidget(center_group)
        main_splitter.addWidget(right_splitter)
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 7)
        main_splitter.setStretchFactor(2, 4)
        self.setCentralWidget(main_splitter)

        self.statusBar().showMessage("就绪")

    def _build_toolbar(self) -> None:
        toolbar = self.addToolBar("主工具栏")
        toolbar.setMovable(False)

        self.open_images_action = QtWidgets.QAction("打开图片文件夹", self)
        self.open_output_action = QtWidgets.QAction("打开输出文件夹", self)
        self.save_current_action = QtWidgets.QAction("保存当前", self)
        self.save_all_action = QtWidgets.QAction("保存全部", self)
        self.prev_action = QtWidgets.QAction("上一张", self)
        self.next_action = QtWidgets.QAction("下一张", self)
        self.semi_action = QtWidgets.QAction("开始半自动", self)
        self.auto_action = QtWidgets.QAction("开始全自动", self)
        self.pause_action = QtWidgets.QAction("暂停", self)
        self.stop_action = QtWidgets.QAction("停止", self)
        self.reprocess_action = QtWidgets.QAction("重新处理当前图", self)
        self.show_raw_action = QtWidgets.QAction("粗框", self)
        self.show_sam_action = QtWidgets.QAction("精修框", self)
        self.show_final_action = QtWidgets.QAction("最终框", self)

        for action in (self.show_raw_action, self.show_sam_action, self.show_final_action):
            action.setCheckable(True)
            action.setChecked(True)

        for action in (
            self.open_images_action,
            self.open_output_action,
            self.save_current_action,
            self.save_all_action,
            self.prev_action,
            self.next_action,
            self.semi_action,
            self.auto_action,
            self.pause_action,
            self.stop_action,
            self.reprocess_action,
        ):
            toolbar.addAction(action)

        toolbar.addSeparator()
        toolbar.addAction(self.show_raw_action)
        toolbar.addAction(self.show_sam_action)
        toolbar.addAction(self.show_final_action)

    def _connect_signals(self) -> None:
        self.open_images_action.triggered.connect(self._choose_image_dir)
        self.open_output_action.triggered.connect(self._choose_output_dir)
        self.save_current_action.triggered.connect(self._save_current)
        self.save_all_action.triggered.connect(self._save_all)
        self.prev_action.triggered.connect(lambda: self._switch_image(-1))
        self.next_action.triggered.connect(lambda: self._switch_image(1))
        self.semi_action.triggered.connect(self._start_semi_auto)
        self.auto_action.triggered.connect(self._start_full_auto)
        self.pause_action.triggered.connect(self._toggle_pause)
        self.stop_action.triggered.connect(self.task_manager.stop)
        self.reprocess_action.triggered.connect(self._start_semi_auto)
        self.show_raw_action.toggled.connect(self._apply_visibility)
        self.show_sam_action.toggled.connect(self._apply_visibility)
        self.show_final_action.toggled.connect(self._apply_visibility)

        self.image_list.currentRowChanged.connect(self._on_image_row_changed)
        self.canvas.boxes_changed.connect(self._on_canvas_boxes_changed)
        self.canvas.selection_changed.connect(self._on_canvas_selection_changed)
        self.canvas.message_requested.connect(self.statusBar().showMessage)
        self.box_panel.selection_changed.connect(self._on_box_panel_selection_changed)
        self.box_panel.apply_label_requested.connect(self._on_apply_label_to_box)
        self.box_panel.delete_requested.connect(self.canvas.delete_selected_box)

        self.class_panel.add_requested.connect(self._add_class)
        self.class_panel.remove_requested.connect(self._remove_class)
        self.class_panel.rename_requested.connect(self._rename_class)
        self.class_panel.move_up_requested.connect(self._move_class_up)
        self.class_panel.move_down_requested.connect(self._move_class_down)
        self.class_panel.import_requested.connect(self._import_classes)
        self.class_panel.export_requested.connect(self._export_classes)
        self.class_panel.current_class_changed.connect(self.canvas.set_current_class)
        self.class_panel.edit_template_requested.connect(self._edit_prompt_template)
        self.class_panel.restore_template_requested.connect(self._restore_default_template)

        self.task_manager.progress_changed.connect(self._on_task_progress)
        self.task_manager.record_processed.connect(self._on_record_processed)
        self.task_manager.status_message.connect(self.statusBar().showMessage)
        self.task_manager.running_changed.connect(self._update_actions)

    def _choose_image_dir(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not directory:
            return
        try:
            records = self.store.load_image_dir(directory)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "加载失败", str(exc))
            return

        if not records:
            QtWidgets.QMessageBox.information(self, "提示", "所选目录中没有支持的图片文件")
        self._refresh_image_list()
        self._show_current_record()
        self.statusBar().showMessage(f"已加载图片目录: {directory}")

    def _choose_output_dir(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if not directory:
            return
        self.store.set_output_dir(directory)
        self._refresh_class_views()
        self._refresh_image_list()
        self._show_current_record()
        self.statusBar().showMessage(f"已设置输出目录: {directory}")

    def _save_current(self) -> None:
        record = self.store.current_record()
        if record is None:
            return
        if not self.store.state.output_dir:
            QtWidgets.QMessageBox.warning(self, "缺少输出目录", "请先设置输出文件夹")
            return
        self.store.save_classes()
        self.store.save_record(self.store.state.current_index)
        self.store.save_intermediate_json(self.store.state.current_index)
        self._refresh_image_list()
        self._show_current_record()
        self.statusBar().showMessage(f"已保存: {record.image_name}")

    def _save_all(self) -> None:
        if not self.store.state.output_dir:
            QtWidgets.QMessageBox.warning(self, "缺少输出目录", "请先设置输出文件夹")
            return
        self.store.save_classes()
        count = self.store.save_all_records()
        for index in range(len(self.store.state.image_records)):
            self.store.save_intermediate_json(index)
        self._refresh_image_list()
        self._show_current_record()
        self.statusBar().showMessage(f"已保存 {count} 张图片")

    def _switch_image(self, step: int) -> None:
        if not self.store.state.image_records:
            return
        target_index = self.store.state.current_index + step
        if 0 <= target_index < len(self.store.state.image_records):
            self.image_list.setCurrentRow(target_index)

    def _start_semi_auto(self) -> None:
        if not self._check_processing_ready(require_output=True):
            return
        self.store.save_classes()
        started = self.task_manager.start_semi_auto(self.store)
        if started:
            self.statusBar().showMessage("半自动处理已开始")

    def _start_full_auto(self) -> None:
        if not self._check_processing_ready(require_output=True):
            return
        self.store.save_classes()
        started = self.task_manager.start_full_auto(self.store)
        if started:
            self.statusBar().showMessage("全自动处理已开始")

    def _toggle_pause(self) -> None:
        if not self.task_manager.is_running:
            return
        if self.task_manager.is_paused:
            self.task_manager.resume()
            self.pause_action.setText("暂停")
            self.statusBar().showMessage("已继续处理")
        else:
            self.task_manager.pause()
            self.pause_action.setText("继续")
            self.statusBar().showMessage("已暂停")

    def _apply_visibility(self) -> None:
        self.canvas.set_source_visibility(
            self.show_raw_action.isChecked(),
            self.show_sam_action.isChecked(),
            self.show_final_action.isChecked(),
        )

    def _on_image_row_changed(self, row: int) -> None:
        if row < 0:
            return
        self.store.set_current_index(row)
        self._show_current_record()

    def _on_canvas_boxes_changed(self, boxes: object) -> None:
        if not isinstance(boxes, list):
            return
        self.store.update_final_boxes(self.store.state.current_index, boxes)
        self._refresh_image_list()
        self._show_current_record(refresh_canvas=False)

    def _on_canvas_selection_changed(self, box_id: str) -> None:
        if self._selection_syncing:
            return
        self._selection_syncing = True
        self.box_panel.select_box(box_id)
        self._selection_syncing = False

    def _on_box_panel_selection_changed(self, box_id: str) -> None:
        if self._selection_syncing:
            return
        self._selection_syncing = True
        self.canvas.select_box(box_id)
        self._selection_syncing = False

    def _on_apply_label_to_box(self, label: str) -> None:
        applied = self.canvas.apply_label_to_selected(label)
        if applied:
            self.statusBar().showMessage(f"已将选中框改为类别: {label}")

    def _on_task_progress(self, current: int, total: int, status_text: str) -> None:
        self.statusBar().showMessage(f"任务进度 {current}/{total}: {status_text}")

    def _on_record_processed(self, index: int, record: object) -> None:
        self._refresh_image_list()
        if index == self.store.state.current_index:
            self._show_current_record()
        if hasattr(record, "image_name"):
            self.statusBar().showMessage(f"处理完成: {record.image_name}")

    def _refresh_image_list(self) -> None:
        self.image_list.blockSignals(True)
        self.image_list.clear()
        for record in self.store.state.image_records:
            self.image_list.addItem(f"[{record.status.value}] {record.image_name}")
        if self.store.state.image_records:
            current_index = min(self.store.state.current_index, len(self.store.state.image_records) - 1)
            self.store.state.current_index = current_index
            self.image_list.setCurrentRow(current_index)
        self.image_list.blockSignals(False)

    def _show_current_record(self, refresh_canvas: bool = True) -> None:
        record = self.store.current_record()
        if refresh_canvas:
            self.canvas.set_record(record)
            self.canvas.set_classes(self.store.state.classes)
            self.canvas.set_current_class(self.class_panel.current_class_name())
            self._apply_visibility()
        self.box_panel.set_boxes(record.final_boxes if record else [])
        self.box_panel.set_classes(self.store.state.classes)
        self._refresh_class_views()
        self._update_window_title()

    def _refresh_class_views(self) -> None:
        current_name = self.class_panel.current_class_name()
        if not current_name and self.store.state.classes:
            current_name = self.store.state.classes[0].name
        self.class_panel.set_classes(self.store.state.classes, current_name)
        self.class_panel.set_prompt_preview(self.store.state.generated_prompt)
        self.box_panel.set_classes(self.store.state.classes)
        self.canvas.set_classes(self.store.state.classes)
        self.canvas.set_current_class(self.class_panel.current_class_name())

    def _update_window_title(self) -> None:
        record = self.store.current_record()
        image_name = record.image_name if record else "未加载图片"
        self.setWindowTitle(f"AutoLabelProMax - {image_name}")

    def _update_actions(self, running: bool) -> None:
        self.pause_action.setEnabled(running)
        self.stop_action.setEnabled(running)
        self.open_images_action.setEnabled(not running)
        self.open_output_action.setEnabled(not running)
        self.semi_action.setEnabled(not running)
        self.auto_action.setEnabled(not running)
        self.reprocess_action.setEnabled(not running)
        if not running:
            self.pause_action.setText("暂停")

    def _check_processing_ready(self, require_output: bool) -> bool:
        if not self.store.state.image_records:
            QtWidgets.QMessageBox.warning(self, "缺少图片", "请先加载图片文件夹")
            return False
        if not self.store.state.classes:
            QtWidgets.QMessageBox.warning(self, "缺少类别", "类别为空时不能启动自动标注")
            return False
        if require_output and not self.store.state.output_dir:
            QtWidgets.QMessageBox.warning(self, "缺少输出目录", "请先设置输出文件夹")
            return False
        return True

    def _add_class(self) -> None:
        text, accepted = QtWidgets.QInputDialog.getText(self, "新增类别", "类别名")
        if accepted and text.strip():
            self.store.add_class(text)
            self.store.save_classes()
            self._refresh_class_views()
            self._show_current_record(refresh_canvas=False)

    def _remove_class(self, index: int) -> None:
        if index < 0:
            return
        self.store.remove_class(index)
        self.store.save_classes()
        self._refresh_class_views()
        self._show_current_record(refresh_canvas=False)

    def _rename_class(self, index: int) -> None:
        if index < 0 or index >= len(self.store.state.classes):
            return
        current_name = self.store.state.classes[index].name
        text, accepted = QtWidgets.QInputDialog.getText(self, "重命名类别", "新类别名", text=current_name)
        if accepted and text.strip():
            self.store.rename_class(index, text)
            self.store.save_classes()
            self._refresh_class_views()
            self._show_current_record(refresh_canvas=False)

    def _move_class_up(self, index: int) -> None:
        if index < 0:
            return
        self.store.move_class_up(index)
        self.store.save_classes()
        self._refresh_class_views()
        self._show_current_record(refresh_canvas=False)

    def _move_class_down(self, index: int) -> None:
        if index < 0:
            return
        self.store.move_class_down(index)
        self.store.save_classes()
        self._refresh_class_views()
        self._show_current_record(refresh_canvas=False)

    def _import_classes(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "导入 classes.txt", "", "Text Files (*.txt)")
        if not path:
            return
        self.store.set_class_names(load_classes_txt(path))
        self.store.load_all_existing_annotations()
        self.store.save_classes()
        self._refresh_class_views()
        self._show_current_record(refresh_canvas=False)

    def _export_classes(self) -> None:
        default_path = str(self.store.classes_file_path()) if self.store.state.output_dir else "classes.txt"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "导出 classes.txt", default_path, "Text Files (*.txt)")
        if not path:
            return
        save_classes_txt(path, self.store.class_names())
        self.statusBar().showMessage(f"已导出类别文件: {path}")

    def _edit_prompt_template(self) -> None:
        dialog = PromptTemplateDialog(self.store.state.prompt_template, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.store.set_prompt_template(dialog.template_text())
            self._refresh_class_views()

    def _restore_default_template(self) -> None:
        self.store.restore_default_prompt_template()
        self._refresh_class_views()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.task_manager.is_running:
            QtWidgets.QMessageBox.warning(self, "任务仍在运行", "请先停止后台任务再退出")
            event.ignore()
            return
        if self.store.has_unsaved_changes():
            result = QtWidgets.QMessageBox.question(
                self,
                "存在未保存修改",
                "还有未保存的标注，确认退出吗？",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if result != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
        event.accept()
