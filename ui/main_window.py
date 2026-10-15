from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np

from PyQt5.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QAction,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QComboBox,
    QTextEdit,
    QLineEdit,
    QInputDialog,
    QToolButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from models import CaseRecord
from services.ai_service import AIService
from services.report_service import ReportService
from services.storage_service import StorageService
from ui.case_list_panel import CaseListPanel
from ui.fields_panel import FieldsPanel
from ui.image_canvas import ImageCanvas
from ui.video_panel import VideoPanel
from ui.pdf_report_window import PdfReportWindow
from ui.report_panel import ReportPanel
from ui.styles import APP_STYLE


class _AnalysisWorker(QObject):
    """Runs AIService inference in a background QThread."""

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, ai_service, image_path, video_path, manual_override, calibration):
        super().__init__()
        self._ai_service = ai_service
        self._image_path = image_path
        self._video_path = video_path
        self._manual_override = manual_override
        self._calibration = calibration

    def run(self):
        try:
            if self._video_path:
                result = self._ai_service.analyze_video(self._video_path)
            else:
                result = self._ai_service.analyze_image(
                    self._image_path, self._manual_override, self._calibration
                )
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("超声影像辅助报告系统")
        self.resize(1380, 860)
        icon_path = Path(__file__).resolve().parent.parent / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self.ai_service = AIService()
        self.report_service = ReportService()
        self.storage_service = StorageService()
        self.settings = self.storage_service.load_settings()
        if not self.settings.get("save_dir"):
            self.settings["save_dir"] = str(Path(__file__).resolve().parents[1] / "case_outputs")
            self.storage_service.save_settings(self.settings)
        self.ai_service.model_config = self.settings
        self.cases = self.storage_service.load_cases()
        self.current_case = None

        self.case_panel = CaseListPanel()
        self.image_canvas = ImageCanvas()
        self.fields_panel = FieldsPanel()
        self.video_panel = VideoPanel()
        self.report_panel = ReportPanel()
        self.status_label = QLabel("等待上传图像")
        self.report_windows = []
        self._analysis_thread = None
        self._analysis_worker = None
        self._calibration_points = []
        self._display_mode = self.settings.get("display_mode", "overlay")
        self._loading_report = False
        self._loading_ai_summary = False
        self._save_manual_calibration_as_default = False
        self.modality_combo = QComboBox()
        self.modality_combo.addItems(["自动", "B-mode", "CDFI"])
        self.modality_combo.setCurrentText(self.settings.get("default_modality", "自动"))
        self.modality_label = QLabel("AI判断：未分析")

        self._build_ui()
        self._connect_signals()
        self.report_panel.set_layout(self.settings.get("report_layout"))
        self._refresh_case_list()
        self.setStyleSheet(APP_STYLE)

    def _build_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        layout.addLayout(self._build_toolbar())
        layout.addWidget(self._build_main_splitter())
        layout.addWidget(self.status_label)
        self.setCentralWidget(central)

    def _build_toolbar(self):
        layout = QHBoxLayout()

        title = QLabel("AI辅助结构化报告系统")
        title.setObjectName("Title")
        layout.addWidget(title)
        layout.addStretch()

        self.upload_button = QPushButton("导入图像")
        self.upload_video_button = QPushButton("导入视频")
        self.analyze_button = QPushButton("AI分析")
        self.generate_button = QPushButton("生成报告")
        self.save_button = QPushButton("保存病例")
        self.settings_button = QPushButton("设置")

        for button in [self.upload_button, self.upload_video_button, self.analyze_button, self.generate_button, self.save_button]:
            button.setFixedSize(118, 40)
            layout.addWidget(button)
        layout.addStretch()
        layout.addWidget(self.settings_button)

        return layout

    def _build_main_splitter(self):
        image_panel = QGroupBox("超声图像 / AI检测框 / 分割结果")
        image_layout = QVBoxLayout(image_panel)
        image_header = QHBoxLayout()
        image_header.addWidget(QLabel("当前图像"))
        image_header.addStretch()
        self.image_tools_button = QToolButton(self)
        self.image_tools_button.setText("图像工具")
        self.image_tools_button.setFixedSize(118, 40)
        self.image_tools_button.setPopupMode(QToolButton.InstantPopup)
        from PyQt5.QtWidgets import QMenu
        menu = QMenu(self)
        self.image_tools_button.setMenu(menu)
        self._add_image_tool_actions(menu)
        image_header.addWidget(self.image_tools_button)
        image_layout.addLayout(image_header)
        image_layout.addWidget(self.image_canvas)

        hint = QLabel("红色框为AI识别区域，蓝色半透明区域为分割结果。")
        hint.setObjectName("Hint")
        image_layout.addWidget(hint)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)
        right_layout.addWidget(self.fields_panel, 1)
        right_layout.addWidget(self.video_panel, 4)

        upper_splitter = QSplitter(Qt.Horizontal)
        upper_splitter.addWidget(self.case_panel)
        upper_splitter.addWidget(image_panel)
        upper_splitter.addWidget(right_panel)
        upper_splitter.setSizes([200, 730, 400])
        upper_splitter.setStretchFactor(0, 15)
        upper_splitter.setStretchFactor(1, 55)
        upper_splitter.setStretchFactor(2, 30)

        ai_summary_panel = QGroupBox("AI超声所见 / 诊断建议（可人工修改）")
        ai_summary_layout = QHBoxLayout(ai_summary_panel)
        finding_layout = QVBoxLayout()
        finding_layout.addWidget(QLabel("AI超声所见"))
        self.ai_findings_edit = QTextEdit()
        self.ai_findings_edit.setPlaceholderText("AI分析后自动生成，可由医生修改。")
        finding_layout.addWidget(self.ai_findings_edit)
        diagnosis_layout = QVBoxLayout()
        diagnosis_layout.addWidget(QLabel("AI诊断 / 诊断建议"))
        self.ai_diagnosis_edit = QTextEdit()
        self.ai_diagnosis_edit.setPlaceholderText("AI分析后自动生成，可由医生审核修改。")
        diagnosis_layout.addWidget(self.ai_diagnosis_edit)
        ai_summary_layout.addLayout(finding_layout, 3)
        ai_summary_layout.addLayout(diagnosis_layout, 2)

        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(upper_splitter)
        main_splitter.addWidget(ai_summary_panel)
        main_splitter.setSizes([610, 240])
        main_splitter.setStretchFactor(0, 4)
        main_splitter.setStretchFactor(1, 1)
        return main_splitter

    def _connect_signals(self):
        self.upload_button.clicked.connect(self.upload_image)
        self.upload_video_button.clicked.connect(self.upload_video)
        self.analyze_button.clicked.connect(self.run_ai_analysis)
        self.generate_button.clicked.connect(self.generate_report)
        self.save_button.clicked.connect(self.save_current_case)
        self.case_panel.list_widget.itemSelectionChanged.connect(self.select_case)
        self.video_panel.frame_list.itemClicked.connect(self._on_lesion_frame_selected)
        self.image_canvas.lymphnode_selected.connect(self._on_lymphnode_selected)
        self.settings_button.clicked.connect(self.open_settings)
        self.image_canvas.calibration_line_finished.connect(self._on_calibration_line)
        self.report_panel.content_changed.connect(self._on_report_changed)
        self.report_panel.save_default_requested.connect(self._save_default_report_layout)
        self.report_panel.restore_default_requested.connect(self._restore_default_report_layout)
        self.report_panel.export_requested.connect(self.export_current_report)
        self.report_panel.images_visibility_changed.connect(self._on_report_images_visibility_changed)
        self.ai_findings_edit.textChanged.connect(self._on_ai_summary_changed)
        self.ai_diagnosis_edit.textChanged.connect(self._on_ai_summary_changed)

    def _add_image_tool_actions(self, menu):
        mode_menu = menu.addMenu("超声模式")
        for label, value in (("自动识别", "自动"), ("BUS", "B-mode"), ("CDFI", "CDFI")):
            action = QAction(label, self)
            action.triggered.connect(lambda checked=False, v=value: self._set_modality(v))
            mode_menu.addAction(action)

        calibration_menu = menu.addMenu("标尺校准")
        action = QAction("自动识别", self)
        action.triggered.connect(self._auto_calibrate)
        calibration_menu.addAction(action)
        action = QAction("人工两点校准", self)
        action.triggered.connect(lambda: self._start_manual_calibration(False))
        calibration_menu.addAction(action)
        action = QAction("修改实际刻度距离", self)
        action.triggered.connect(self._edit_scale_distance)
        calibration_menu.addAction(action)

        display_menu = menu.addMenu("显示")
        for label, value in (("原图", "original"), ("分割 Mask", "mask"), ("叠加显示", "overlay")):
            action = QAction(label, self)
            action.triggered.connect(lambda checked=False, v=value: self._set_display_mode(v))
            display_menu.addAction(action)

    def _set_modality(self, value):
        self.modality_combo.setCurrentText(value)
        if self.current_case:
            self.current_case.modality["manual_override"] = None if value == "自动" else value
        self.status_label.setText(f"当前超声模式：{value}")
        if self.current_case and self.current_case.image_path:
            self.run_ai_analysis()

    def _set_display_mode(self, value):
        self._display_mode = value
        self.image_canvas.set_display_mode(value)
        self.settings["display_mode"] = value

    def _auto_calibrate(self):
        if not self.current_case:
            return
        info = getattr(self.current_case, "image_info", {}) or {}
        info.pop("calibration", None)
        auto_scale = info.get("auto_mm_per_pixel")
        if auto_scale:
            self._apply_calibration(float(auto_scale), "auto")
        else:
            info.pop("mm_per_pixel", None)
            for instance in getattr(self.current_case, "ln_instances", []) or []:
                measurements = instance.get("measurements", {})
                measurements.pop("long_diameter_mm", None)
                measurements.pop("short_diameter_mm", None)
            if getattr(self.current_case, "ln_instances", None):
                self.video_panel.set_features(self._format_lymphnode_instances(self.current_case.ln_instances))
            self._refresh_report_if_allowed()
        self.current_case.image_info = info
        self.status_label.setText("已恢复自动标尺识别；无法识别时不显示毫米尺寸。")

    def _start_manual_calibration(self, save_as_default=False):
        if not self.current_case or not self.current_case.image_path:
            QMessageBox.information(self, "标尺校准", "请先选择一张静态原始图像。")
            return
        self._save_manual_calibration_as_default = bool(save_as_default)
        self._calibration_points = []
        self.image_canvas.set_calibration_active(True)
        suffix = "；本次结果将保存为后续新图的默认人工标尺" if save_as_default else ""
        self.status_label.setText(f"请在原始图像上拖动选择两个纵向标尺刻度{suffix}。")

    def _on_calibration_line(self, x1, y1, x2, y2):
        self.image_canvas.set_calibration_active(False)
        pixel_distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        distance, ok = self._get_calibration_distance(
            "人工深度标定", "两点实际距离（mm）：", 10.0
        )
        if not ok or pixel_distance <= 0:
            self._save_manual_calibration_as_default = False
            self.status_label.setText("标尺校准已取消或无效。")
            return
        mm_per_pixel = float(distance) / pixel_distance
        self._apply_calibration(mm_per_pixel, "manual_two_points")
        if self._save_manual_calibration_as_default:
            self.settings["default_manual_mm_per_pixel"] = mm_per_pixel
            self.storage_service.save_settings(self.settings)
            self.status_label.setText(f"默认人工标尺已保存：{mm_per_pixel:.6f} mm/px。")
        self._save_manual_calibration_as_default = False

    def _edit_scale_distance(self):
        if not self.current_case:
            return
        info = getattr(self.current_case, "image_info", {}) or {}
        pixels = info.get("depth_pixel_height")
        if not pixels or pixels <= 0:
            QMessageBox.information(self, "标尺校准", "当前图像未检测到有效标尺，无法修改实际距离。")
            return
        distance, ok = self._get_calibration_distance(
            "修改实际刻度距离",
            "检测到的整段刻度实际距离（mm）：",
            float(info.get("detected_depth_cm", 1) * 10),
            10000.0,
        )
        if ok:
            self._apply_calibration(float(distance) / pixels, "manual_scale_distance")

    def _get_calibration_distance(self, title, label, value, maximum=1000.0):
        dialog = QInputDialog(self)
        dialog.setWindowTitle(title)
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dialog.setMinimumWidth(430)
        dialog.resize(430, dialog.sizeHint().height())
        dialog.setLabelText(label)
        dialog.setInputMode(QInputDialog.DoubleInput)
        dialog.setDoubleDecimals(2)
        dialog.setDoubleRange(0.01, maximum)
        dialog.setDoubleValue(value)
        accepted = dialog.exec_() == QDialog.Accepted
        return dialog.doubleValue(), accepted

    def _apply_calibration(self, mm_per_pixel, method):
        if mm_per_pixel <= 0 or not self.current_case:
            return
        info = getattr(self.current_case, "image_info", {}) or {}
        info["mm_per_pixel"] = mm_per_pixel
        if method == "auto":
            info.pop("calibration", None)
        else:
            info["calibration"] = {"mm_per_pixel": mm_per_pixel, "method": method}
        self.current_case.image_info = info
        for instance in getattr(self.current_case, "ln_instances", []) or []:
            measurements = instance.get("measurements", {})
            for key in ("long_diameter_px", "short_diameter_px"):
                if measurements.get(key) is not None:
                    measurements[key.replace("_px", "_mm")] = measurements[key] * mm_per_pixel
        if getattr(self.current_case, "ln_instances", None):
            self.video_panel.set_features(self._format_lymphnode_instances(self.current_case.ln_instances))
        self._refresh_report_if_allowed()
        self.status_label.setText(f"标尺已校准：{mm_per_pixel:.6f} mm/px；可重新运行 AI 更新结果。")

    def _set_ai_summary(self, findings, diagnosis, manual=False):
        self._loading_ai_summary = True
        self.ai_findings_edit.setPlainText(findings or "")
        self.ai_diagnosis_edit.setPlainText(diagnosis or "")
        self._loading_ai_summary = False
        if self.current_case:
            self.current_case.ai_findings = findings or ""
            self.current_case.ai_diagnosis = diagnosis or ""
            self.current_case.ai_summary_manual = bool(manual)

    def _on_ai_summary_changed(self):
        if not self.current_case or self._loading_ai_summary:
            return
        self.current_case.ai_findings = self.ai_findings_edit.toPlainText()
        self.current_case.ai_diagnosis = self.ai_diagnosis_edit.toPlainText()
        self.current_case.ai_summary_manual = True
        self.status_label.setText("AI超声所见 / 诊断建议 · 已修改")

    def _refresh_ai_summary(self, force=False):
        if not self.current_case:
            return
        if getattr(self.current_case, "ai_summary_manual", False) and not force:
            self.status_label.setText("测量结果已更新；医生修改的超声所见/诊断未覆盖。")
            return
        self._set_ai_summary(
            self.report_service.generate_finding(self.current_case),
            self.report_service.generate_impression(self.current_case),
            manual=False,
        )

    def _on_report_changed(self):
        if self.current_case and not self._loading_report:
            self.current_case.report_manual = True
            self.status_label.setText("报告草稿 · 已修改")

    def _save_default_report_layout(self):
        self.settings["report_layout"] = self.report_panel.layout_values()
        self.storage_service.save_settings(self.settings)
        self.status_label.setText("默认报告版式已保存。")

    def _on_report_images_visibility_changed(self, visible):
        self.settings["report_layout"] = self.report_panel.layout_values()
        self.storage_service.save_settings(self.settings)
        self.report_panel.set_images_visible(visible)
        if self.current_case and self.current_case.report_manual:
            self.status_label.setText("图像显示设置已保存；当前报告含医生修改，未覆盖正文。")
        else:
            self.status_label.setText("图像显示设置已保存。")

    def _restore_default_report_layout(self):
        values = dict(ReportPanel.DEFAULT_LAYOUT)
        self.report_panel.apply_layout_to_editor(values)
        self.settings["report_layout"] = values
        self.storage_service.save_settings(self.settings)
        self.status_label.setText("已恢复默认报告版式。")

    def _refresh_report_if_allowed(self):
        self._refresh_ai_summary(force=False)

    def open_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("设置")
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        form = QFormLayout(dialog)
        mode = QComboBox(); mode.addItems(["自动", "B-mode", "CDFI"]); mode.setCurrentText(self.settings.get("default_modality", "自动"))
        calibration = QComboBox(); calibration.addItems(["自动识别", "人工两点校准"]); calibration.setCurrentText(self.settings.get("default_calibration", "自动识别"))
        manual_scale = QLineEdit()
        saved_manual_scale = self.settings.get("default_manual_mm_per_pixel")
        manual_scale.setText(f"{float(saved_manual_scale):.6f}" if saved_manual_scale else "")
        manual_scale.setPlaceholderText("首次人工标定后自动保存")
        save_dir = QLineEdit(self.settings["save_dir"])
        choose_dir = QPushButton("选择...")
        save_dir_row = QWidget()
        save_dir_layout = QHBoxLayout(save_dir_row)
        save_dir_layout.setContentsMargins(0, 0, 0, 0)
        save_dir_layout.addWidget(save_dir)
        save_dir_layout.addWidget(choose_dir)
        choose_dir.clicked.connect(lambda: self._choose_save_dir(dialog, save_dir))
        form.addRow("默认超声模式", mode); form.addRow("默认标尺方式", calibration)
        form.addRow("默认人工标尺 (mm/px)", manual_scale)
        form.addRow("保存目录", save_dir_row)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        form.addRow(buttons)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject)
        if not dialog.exec_():
            return
        manual_value = manual_scale.text().strip()
        if manual_value:
            try:
                manual_value = float(manual_value)
                if manual_value <= 0:
                    raise ValueError
            except ValueError:
                QMessageBox.warning(self, "设置", "默认人工标尺必须是大于 0 的 mm/px 数值。")
                return
        else:
            manual_value = None
        self.settings.update({
            "default_modality": mode.currentText(),
            "default_calibration": calibration.currentText(),
            "default_manual_mm_per_pixel": manual_value,
            "save_dir": save_dir.text() or self.settings["save_dir"],
        })
        self.storage_service.save_settings(self.settings)
        QMessageBox.information(self, "设置", "设置已保存。")

    def _choose_save_dir(self, parent, line_edit):
        selected = QFileDialog.getExistingDirectory(parent, "选择保存目录", line_edit.text())
        if selected:
            line_edit.setText(selected)

    def _apply_default_calibration_to_new_case(self):
        if not self.current_case or not self.current_case.image_path:
            return
        if self.settings.get("default_calibration", "自动识别") == "自动识别":
            self.current_case.image_info = {}
            self.status_label.setText("新图将使用自动深度刻度识别，可点击 AI分析。")
            return
        mm_per_pixel = self.settings.get("default_manual_mm_per_pixel")
        if mm_per_pixel and float(mm_per_pixel) > 0:
            self.current_case.image_info = {
                "calibration": {
                    "mm_per_pixel": float(mm_per_pixel),
                    "method": "default_manual",
                }
            }
            self.status_label.setText(
                f"已应用默认人工标尺：{float(mm_per_pixel):.6f} mm/px，可点击 AI分析。"
            )
            return
        self.current_case.image_info = {}
        self.status_label.setText("尚未设置默认人工标尺，请在原图上完成首次人工深度标定。")
        QTimer.singleShot(0, lambda: self._start_manual_calibration(True))


    def upload_image(self):
        image_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择超声图像",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not image_path:
            return

        if not self.image_canvas.load_image(image_path):
            return

        self.current_case = CaseRecord(
            case_id=datetime.now().strftime("CASE%Y%m%d%H%M%S"),
            image_path=image_path,
        )
        self.current_case.video_path = ""
        self.current_case.lesion_frame_paths = []
        self.current_case.lesion_frame_indices = []
        self.current_case.lesion_features = {}
        self.fields_panel.load_case(self.current_case)
        self.report_panel.set_report("")
        self._set_ai_summary("", "", manual=False)
        self.video_panel.clear()
        self._add_or_update_case(self.current_case)
        self.modality_combo.setCurrentText(self.settings.get("default_modality", "自动"))
        self._apply_default_calibration_to_new_case()

    def upload_video(self):
        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv)",
        )
        if not video_path:
            return

        self.current_case = CaseRecord(
            case_id=datetime.now().strftime("CASE%Y%m%d%H%M%S"),
            video_path=video_path,
        )
        self.current_case.image_path = ""
        self.current_case.lesion_frame_paths = []
        self.current_case.lesion_frame_indices = []
        self.current_case.lesion_features = {}
        self.fields_panel.load_case(self.current_case)
        self.report_panel.set_report("")
        self._set_ai_summary("", "", manual=False)
        self.video_panel.clear()
        self.image_canvas.play_video(video_path)
        self._add_or_update_case(self.current_case)
        self.modality_combo.setCurrentText(self.settings.get("default_modality", "自动"))
        self.status_label.setText("视频已上传，可点击 AI分析。")

    def run_ai_analysis(self):
        if not self._has_active_image():
            QMessageBox.information(self, "提示", "请先上传图像或视频。")
            return

        # Prevent concurrent inference threads
        if self._analysis_thread is not None and self._analysis_thread.isRunning():
            return

        # Disable button and show status during inference
        self.analyze_button.setEnabled(False)
        self.analyze_button.setText("分析中...")
        self.status_label.setText("AI 分析中，请稍候...")

        # Create worker and move to background thread
        self._analysis_worker = _AnalysisWorker(
            self.ai_service,
            self.current_case.image_path,
            self.current_case.video_path,
            {"B-mode": "B-mode", "CDFI": "CDFI"}.get(self.modality_combo.currentText()),
            (getattr(self.current_case, "image_info", {}) or {}).get("calibration"),
        )
        self._analysis_thread = QThread()
        self._analysis_worker.moveToThread(self._analysis_thread)

        self._analysis_thread.started.connect(self._analysis_worker.run)
        self._analysis_worker.finished.connect(self._on_analysis_finished)
        self._analysis_worker.error.connect(self._on_analysis_error)
        self._analysis_worker.finished.connect(self._analysis_thread.quit)
        self._analysis_worker.error.connect(self._analysis_thread.quit)

        self._analysis_thread.start()

    def _on_analysis_finished(self, result):
        mask = result.pop("mask", None)
        result.pop("effective_image", None)
        image_info = result.pop("image_info", {})
        previous_calibration = (getattr(self.current_case, "image_info", {}) or {}).get("calibration")
        if previous_calibration and image_info.get("mm_per_pixel"):
            image_info["calibration"] = previous_calibration
        lesion_frames = result.pop("lesion_frames", [])
        lesion_features = result.pop("lesion_features", {})
        instances = result.pop("instances", [])
        self.current_case.modality = result.get("modality", {})
        modality = self.current_case.modality
        override_text = "（人工修改）" if modality.get("manual_override") else ""
        self.modality_label.setText(
            f"AI判断：{modality.get('ai_mode', '未知')} "
            f"{modality.get('confidence', 0) * 100:.1f}% | "
            f"最终模式：{modality.get('final_mode', '未知')}{override_text}"
        )
        self.current_case.lesion_frame_indices = [item["frame_index"] for item in lesion_frames]
        self.current_case.lesion_frame_paths = [item["thumbnail_path"] for item in lesion_frames]
        self.current_case.lesion_features = lesion_features
        self.current_case.ln_instances = instances
        self.current_case.image_info = image_info
        self.current_case.ai_mask = mask
        self.current_case.ai_score = result.get("ai_score", "")
        ai_diagnosis = result.get("diagnosis", self.current_case.diagnosis)
        self.current_case.diagnosis = ai_diagnosis
        self.current_case.size = result.get("size", self.current_case.size)

        self.fields_panel.apply_ai_result(result)
        self.fields_panel.update_case(self.current_case)
        # The legacy diagnosis combo is hidden and cannot represent the current
        # detection summary; keep the actual AI result instead of its fallback item.
        self.current_case.diagnosis = ai_diagnosis
        if instances:
            self.video_panel.set_features(
                self._format_lymphnode_instances(
                    [instances[0]]
                )
            )
        else:
            self.video_panel.set_features(
                self._format_lesion_features(lesion_features)
            )
        self.video_panel.set_frames(
            [{"frame_index": idx, "thumbnail_path": path} for idx, path in zip(self.current_case.lesion_frame_indices, self.current_case.lesion_frame_paths)]
        )
        if self.current_case.image_path and Path(self.current_case.image_path).exists():
            self.image_canvas.load_image(self.current_case.image_path)
            if instances:
                self.image_canvas.draw_lymphnode_instances(instances)
            elif mask is not None:
                self.image_canvas.draw_ai_result(mask)
        elif self.current_case.lesion_frame_paths:
            self.image_canvas.load_image(self.current_case.lesion_frame_paths[0])
            if not instances:
                self.image_canvas.draw_ai_result(mask)
        elif mask is not None and not instances:
            self.image_canvas.draw_ai_result(mask)
        if not instances:
            self.status_label.setText("未检测到淋巴结：请检查图像质量并进行人工复核，不能据此判定无淋巴结。")
            QMessageBox.information(self, "AI 分析提示", "当前图像未检测到淋巴结。请检查图像质量并允许人工复核；这不是“无淋巴结”的诊断结论。")
        else:
            self.status_label.setText("AI分析完成：已显示检测框、分割结果和结构化字段。")
        self._refresh_ai_summary(force=True)

        # Restore button
        self.analyze_button.setEnabled(True)
        self.analyze_button.setText("AI分析")

    def _on_analysis_error(self, error_msg):
        QMessageBox.warning(self, "AI 分析失败", f"推理出错：{error_msg}")
        self.status_label.setText("AI 分析失败。")
        self.analyze_button.setEnabled(True)
        self.analyze_button.setText("AI分析")

    def generate_report(self):
        if not self.current_case:
            QMessageBox.information(self, "提示", "请先上传图像并建立病例。")
            return

        if self.current_case.report_manual and self.current_case.report:
            answer = QMessageBox.question(self, "报告已修改", "当前报告包含医生修改，重新生成会覆盖这些修改，是否继续？", QMessageBox.Yes | QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
            self.current_case.report_manual = False
        self._generate_report_content(open_pdf=True)

    def export_current_report(self):
        if not self.current_case or not self.report_panel.report():
            QMessageBox.information(self, "提示", "当前没有可导出的报告。")
            return
        self.current_case.report = self.report_panel.report()
        self.current_case.report_html = self.report_panel.html()
        file_path = PdfReportWindow.export_html(
            self,
            self.current_case.case_id,
            self.current_case.report_html,
        )
        if file_path:
            self.status_label.setText(f"PDF已导出：{file_path}")

    def _generate_report_content(self, open_pdf=True):
        self.fields_panel.update_case(self.current_case)
        if self.current_case.ai_score:
            self.image_canvas.draw_ai_result()

        finding = self.ai_findings_edit.toPlainText().strip()
        impression = self.ai_diagnosis_edit.toPlainText().strip()
        if not finding:
            finding = self.report_service.generate_finding(self.current_case)
        if not impression:
            impression = self.report_service.generate_impression(self.current_case)
        self.current_case.ai_findings = finding
        self.current_case.ai_diagnosis = impression
        text_report = f"超声所见：\n{finding}\n\n超声提示：\n{impression}"
        self.current_case.report = text_report
        self._loading_report = True

        report_asset_dir = Path(self.settings.get("save_dir") or "report_assets")
        if self.settings.get("save_dir"):
            report_asset_dir = report_asset_dir / "report_assets"
        report_asset_dir.mkdir(exist_ok=True)
        ai_image_path = report_asset_dir / f"{self.current_case.case_id}_ai.png"
        self.image_canvas.export_snapshot(ai_image_path)

        report_html = self.report_service.generate_html(
            self.current_case,
            ai_image_path=str(ai_image_path),
            layout=self.report_panel.layout_values(),
            finding=finding,
            impression=impression,
        )
        self.current_case.report_html = report_html
        self.report_panel.set_report(text_report, report_html)
        self._loading_report = False
        if open_pdf:
            self._open_pdf_report_window(report_html)
            self.status_label.setText("报告已生成，已在独立窗口打开，可编辑后导出PDF。")
        else:
            self.status_label.setText("报告草稿已更新。")

    def save_current_case(self):
        if not self.current_case:
            QMessageBox.information(self, "提示", "当前没有可保存的病例。")
            return

        self.fields_panel.update_case(self.current_case)
        self.current_case.ai_findings = self.ai_findings_edit.toPlainText()
        self.current_case.ai_diagnosis = self.ai_diagnosis_edit.toPlainText()
        self.current_case.report = self.report_panel.report()
        self.current_case.report_html = self.report_panel.html()
        self.current_case.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._add_or_update_case(self.current_case)
        self.storage_service.save_cases(self.cases)

        # 保存病例完整结果（基础信息 + 所有淋巴结实例 + mask PNG）
        try:
            case_dir = self._save_case_result_json()
            self.status_label.setText(
                f"病例数据保存成功：{case_dir.resolve()}"
            )
            QMessageBox.information(
                self,
                "保存成功",
                f"病例数据保存成功\n保存目录：\n{case_dir.resolve()}",
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "保存失败",
                f"病例结果保存失败：{exc}",
            )

    def _save_case_result_json(self):
        """保存 case_result.json 与每个 instance 的 mask PNG。"""
        case = self.current_case
        case_id = case.case_id
        output_root = Path(self.settings.get("save_dir") or (Path(__file__).resolve().parents[1] / "case_outputs"))
        case_dir = output_root / case.record_id
        masks_dir = case_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        instances_out = []
        for idx, inst in enumerate(getattr(case, "ln_instances", []) or []):
            seg = inst.get("segmentation", {}) or {}
            mask = seg.get("mask")

            mask_rel = None
            if mask is not None:
                ln_tag = f"LN_{idx + 1:02d}"
                mask_path = masks_dir / f"{ln_tag}.png"
                mask_u8 = np.asarray(mask, dtype=np.uint8)
                if mask_u8.ndim == 3:
                    mask_u8 = mask_u8[0]
                mask_u8 = (mask_u8 > 0).astype(np.uint8) * 255
                cv2.imwrite(str(mask_path), mask_u8)
                mask_rel = str(mask_path)

            seg_out = {
                "confidence": seg.get("confidence"),
                "bbox_xyxy": seg.get("bbox_xyxy"),
                "yolo_bbox_xyxy": seg.get("yolo_bbox_xyxy"),
                "sym15_bbox_xyxy": seg.get("sym15_bbox_xyxy"),
            }
            if mask_rel is not None:
                seg_out["mask_path"] = mask_rel

            instances_out.append({
                "instance_id": inst.get("instance_id", f"LN_{idx + 1:02d}"),
                "segmentation": seg_out,
                "measurements": inst.get("measurements", {}),
                "phenotypes": inst.get("phenotypes", {}),
                "vascularity": inst.get("vascularity", {}),
            })

        payload = {
            "case_id": case.case_id,
            "patient_name": case.patient_name,
            "age": case.age,
            "gender": case.gender,
            "site": case.site,
            "image_path": case.image_path,
            "video_path": case.video_path,
            "image_info": getattr(case, "image_info", {}),
            "instances": instances_out,
            "modality": getattr(case, "modality", {}),
            "report": case.report,
            "report_html": getattr(case, "report_html", ""),
            "ai_findings": getattr(case, "ai_findings", ""),
            "ai_diagnosis": getattr(case, "ai_diagnosis", ""),
        }

        json_path = case_dir / "case_result.json"
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return case_dir

    def select_case(self):
        record_id = self.case_panel.selected_case_id()
        if not record_id:
            return

        selected = next((case for case in self.cases if case.record_id == record_id), None)
        if not selected:
            return

        self.current_case = selected
        self.fields_panel.load_case(selected)
        findings = getattr(selected, "ai_findings", "")
        diagnosis = getattr(selected, "ai_diagnosis", "")
        if not findings and getattr(selected, "ln_instances", None):
            findings = self.report_service.generate_finding(selected)
        if not diagnosis and getattr(selected, "ai_score", ""):
            diagnosis = self.report_service.generate_impression(selected)
        self._set_ai_summary(
            findings,
            diagnosis,
            manual=getattr(selected, "ai_summary_manual", False),
        )
        self._loading_report = True
        self.report_panel.set_report(selected.report, getattr(selected, "report_html", ""))
        self._loading_report = False
        modality = getattr(selected, "modality", {}) or {}
        self.modality_combo.setCurrentText(modality.get("manual_override") or self.settings.get("default_modality", "自动"))

        if selected.video_path and Path(selected.video_path).exists():
            self.image_canvas.play_video(selected.video_path)
            self.video_panel.set_features(self._format_lesion_features(selected.lesion_features))
            self.video_panel.set_frames(
                [{"frame_index": idx, "thumbnail_path": path}
                 for idx, path in zip(selected.lesion_frame_indices, selected.lesion_frame_paths)]
            )
        elif selected.image_path and Path(selected.image_path).exists():
            self.image_canvas.load_image(selected.image_path)
            instances = getattr(selected, "ln_instances", [])
            if instances:
                self.image_canvas.draw_lymphnode_instances(instances)
                self.video_panel.set_features(
                    self._format_lymphnode_instances(
                        [instances[0]]
                    )
                )
            else:
                if selected.ai_score:
                    self.image_canvas.draw_ai_result(getattr(selected, "ai_mask", None))
                self.video_panel.clear()

        self.status_label.setText(f"已选择病例：{selected.case_id}")

    def _on_lymphnode_selected(self, index):
        if not self.current_case:
            return

        instances = getattr(
            self.current_case,
            "ln_instances",
            [],
        )

        if index < 0 or index >= len(instances):
            return

        text = self._format_lymphnode_instances(
            [instances[index]]
        )

        self.video_panel.set_features(text)

    def _has_active_image(self):
        return bool(self.current_case and (self.current_case.image_path or self.current_case.video_path))

    @staticmethod
    def _format_lesion_features(lesion_features):
        if not lesion_features or lesion_features.get("total_lesion_frames", 0) == 0:
            return "未检测到可疑病灶。"

        class_counts = lesion_features.get("class_counts", {})
        total_frames = lesion_features.get("total_lesion_frames", 0)
        total_lesions = lesion_features.get("total_lesions", 0)
        class_text = ", ".join(
            f"类别{class_id}:{count}" for class_id, count in sorted(class_counts.items(), key=lambda x: -x[1])
        )
        return (
            f"检测到 {total_frames} 帧可疑病灶，共计 {total_lesions} 处病变。"
            f" 主要类别：{class_text}。"
        )

    # ---- Lymph node phenotype label mappings ----
    _LN_LABELS = {
        "posterior": {
            "field_name": "后方回声",
            "classes": {
                "non_enhancement": "无增强",
                "enhancement": "增强",
            },
        },
        "liquefaction": {
            "field_name": "液化",
            "classes": {
                "absent": "无",
                "present": "有",
            },
        },
        "hilum": {
            "field_name": "淋巴门",
            "classes": {
                "disappearance": "消失",
                "destruction": "破坏",
            },
        },
        "margin": {
            "field_name": "边缘",
            "classes": {
                "blurred": "模糊",
                "partial": "部分清晰",
                "clear": "清晰",
            },
        },
        "shape": {
            "field_name": "形状",
            "classes": {
                "ellipse": "椭圆形",
                "round": "圆形",
                "fusion": "融合",
            },
        },
    }

    def _format_lymphnode_instances(self, instances):
        if not instances:
            return "未检测到淋巴结"

        blocks = []
        for i, inst in enumerate(instances):
            ln_id = inst.get("instance_id", f"LN_{i + 1:02d}")

            seg = inst.get("segmentation", {})
            conf = seg.get("confidence", 0)

            meas = inst.get("measurements", {})
            long_axis = meas.get("long_diameter_px", meas.get("long_axis_px"))
            short_axis = meas.get("short_diameter_px", meas.get("short_axis_px"))
            long_mm = meas.get("long_diameter_mm")
            short_mm = meas.get("short_diameter_mm")
            ls_ratio = meas.get("ls_ratio", 0)

            lines = []
            lines.append(f"【{ln_id}】")
            # 分割置信度保留在数据中，暂不在 UI 显示
            # lines.append(f"分割置信度：{conf * 100:.1f}%")
            lines.append(f"长径（Feret）：{long_axis:.1f} px")
            lines.append(f"短径（垂直）：{short_axis:.1f} px")
            lines.append(
                f"实际长径：{long_mm:.1f} mm" if long_mm is not None else "实际长径：不可用"
            )
            lines.append(
                f"实际短径：{short_mm:.1f} mm" if short_mm is not None else "实际短径：不可用"
            )
            lines.append(f"L/S：{ls_ratio:.2f}")
            vascularity = inst.get("vascularity", {})
            lines.append(
                f"血供：{vascularity.get('label', '未知')}"
                if vascularity.get("available")
                else "血供：未进行血供分类"
            )
            lines.append("")

            phenotypes = inst.get("phenotypes", {})
            for head_key in ["posterior", "liquefaction", "hilum", "margin", "shape"]:
                ph = phenotypes.get(head_key, {})
                label = ph.get("label", "未知")
                prob = ph.get("probability", 0)

                field_name = self._LN_LABELS[head_key]["field_name"]
                cn_label = self._LN_LABELS[head_key]["classes"].get(label, label)

                # probability 保留在底层数据中，暂不在 UI 显示百分数
                lines.append(f"{field_name}：{cn_label}")

            blocks.append("\n".join(lines))

        return "\n\n────────────\n\n".join(blocks)

    def _on_lesion_frame_selected(self, item):
        data = item.data(Qt.UserRole)
        if not data:
            return
        thumb_path = data.get("thumbnail_path")
        if thumb_path and Path(thumb_path).exists():
            self.image_canvas.load_image(thumb_path)
            try:
                # Run per-frame analysis and map results to structured fields
                result = self.ai_service.analyze_image(thumb_path)
            except Exception as e:
                self.status_label.setText(f"帧分析失败：{e}")
                if getattr(self.current_case, "ai_mask", None) is not None:
                    self.image_canvas.draw_ai_result(self.current_case.ai_mask)
                return

            mask = result.pop("mask", None)
            lesion_counter = result.pop("lesion_counter", {})

            # Update current case and UI fields
            self.current_case.ai_mask = mask
            self.current_case.ai_score = result.get("ai_score", self.current_case.ai_score)
            self.current_case.diagnosis = result.get("diagnosis", self.current_case.diagnosis)
            self.current_case.size = result.get("size", self.current_case.size)
            self.current_case.lesion_features = {"class_counts": lesion_counter}

            self.fields_panel.apply_ai_result(result)
            self.fields_panel.update_case(self.current_case)

            if mask is not None:
                self.image_canvas.draw_ai_result(mask)

            self.status_label.setText(f"已分析帧 {data.get('frame_index')}，结构化字段已更新。")

    def _add_or_update_case(self, case):
        for index, saved_case in enumerate(self.cases):
            if saved_case.record_id == case.record_id:
                self.cases[index] = case
                self._refresh_case_list()
                return

        self.cases.insert(0, case)
        self._refresh_case_list()

    def _refresh_case_list(self):
        selected_id = self.current_case.record_id if self.current_case else ""
        self.case_panel.refresh(self.cases, selected_id)

    def _open_pdf_report_window(self, report_html):
        window = PdfReportWindow(
            self.current_case.case_id,
            report_html,
            on_save_edits=self._save_pdf_window_edits,
            on_content_changed=self._on_report_window_content_changed,
            on_layout_changed=self._on_report_window_layout_changed,
            on_save_default=self._save_report_layout_values,
            on_restore_default=self._restore_report_window_layout,
            on_exported=self._on_report_exported,
            layout=self.report_panel.layout_values(),
        )
        window.destroyed.connect(lambda: self._remove_report_window(window))
        self.report_windows.append(window)
        window.show()

    def _on_report_window_content_changed(self, plain_text, html):
        if not self.current_case:
            return
        self.current_case.report = plain_text
        self.current_case.report_html = html
        self.current_case.report_manual = True
        self._loading_report = True
        self.report_panel.set_report(plain_text, html)
        self._loading_report = False
        self.status_label.setText("报告草稿 · 已修改")

    def _on_report_window_layout_changed(self, values):
        self.status_label.setText("当前报告版式已应用。")

    def _save_report_layout_values(self, values):
        self.settings["report_layout"] = values
        self.storage_service.save_settings(self.settings)
        self.report_panel.set_layout(values)
        self.status_label.setText("默认报告版式已保存。")

    def _restore_report_window_layout(self, window):
        self.status_label.setText("当前报告已恢复初始版式。")

    def _on_report_exported(self, file_path):
        self.status_label.setText(f"PDF已导出：{file_path}")

    def _save_pdf_window_edits(self, plain_text, html):
        if not self.current_case:
            return
        self.current_case.report = plain_text
        self.current_case.report_html = html
        self.current_case.report_manual = True
        self._loading_report = True
        self.report_panel.set_report(plain_text, html)
        self._loading_report = False
        self.current_case.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._add_or_update_case(self.current_case)
        self.storage_service.save_cases(self.cases)
        self.status_label.setText("PDF窗口中的报告修改已保存到当前病例。")

    def _remove_report_window(self, window):
        if window in self.report_windows:
            self.report_windows.remove(window)
