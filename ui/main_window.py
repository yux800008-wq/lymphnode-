from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np

from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QComboBox,
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

    def __init__(self, ai_service, image_path, video_path, manual_override):
        super().__init__()
        self._ai_service = ai_service
        self._image_path = image_path
        self._video_path = video_path
        self._manual_override = manual_override

    def run(self):
        try:
            if self._video_path:
                result = self._ai_service.analyze_video(self._video_path)
            else:
                result = self._ai_service.analyze_image(self._image_path, self._manual_override)
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
        self.modality_combo = QComboBox()
        self.modality_combo.addItems(["自动", "B-mode", "CDFI"])
        self.modality_label = QLabel("AI判断：未分析")

        self._build_ui()
        self._connect_signals()
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

        self.upload_button = QPushButton("上传图像")
        self.upload_video_button = QPushButton("上传视频")
        self.analyze_button = QPushButton("AI分析")
        self.generate_button = QPushButton("生成报告")
        self.save_button = QPushButton("保存数据")

        for button in [self.upload_button, self.upload_video_button, self.analyze_button, self.generate_button, self.save_button]:
            button.setMinimumHeight(36)
            layout.addWidget(button)

        layout.addWidget(QLabel("超声模式"))
        self.modality_combo.setMinimumHeight(36)
        layout.addWidget(self.modality_combo)
        layout.addWidget(self.modality_label)

        return layout

    def _build_main_splitter(self):
        image_panel = QGroupBox("超声图像 / AI检测框 / 分割结果")
        image_layout = QVBoxLayout(image_panel)
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

        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(upper_splitter)
        main_splitter.addWidget(self.report_panel)
        main_splitter.setSizes([600, 230])
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
        self.video_panel.clear()
        self._add_or_update_case(self.current_case)
        self.status_label.setText("图像已上传，可点击 AI分析。")

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
        self.video_panel.clear()
        self.image_canvas.play_video(video_path)
        self._add_or_update_case(self.current_case)
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
        self.current_case.diagnosis = result.get("diagnosis", self.current_case.diagnosis)
        self.current_case.size = result.get("size", self.current_case.size)

        self.fields_panel.apply_ai_result(result)
        self.fields_panel.update_case(self.current_case)
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
        self.status_label.setText("AI分析完成：已显示检测框、分割结果和结构化字段。")

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

        self.fields_panel.update_case(self.current_case)
        if self.current_case.ai_score:
            self.image_canvas.draw_ai_result()

        text_report = self.report_service.generate_text(self.current_case)
        self.current_case.report = text_report
        self.report_panel.set_report(text_report)

        report_asset_dir = Path("report_assets")
        report_asset_dir.mkdir(exist_ok=True)
        ai_image_path = report_asset_dir / f"{self.current_case.case_id}_ai.png"
        self.image_canvas.export_snapshot(ai_image_path)

        report_html = self.report_service.generate_html(
            self.current_case,
            ai_image_path=str(ai_image_path),
        )
        self._open_pdf_report_window(report_html)
        self.status_label.setText("文本报告已生成，PDF图示报告已在新窗口打开。")

    def save_current_case(self):
        if not self.current_case:
            QMessageBox.information(self, "提示", "当前没有可保存的病例。")
            return

        self.fields_panel.update_case(self.current_case)
        self.current_case.report = self.report_panel.report()
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
        case_dir = Path(__file__).resolve().parents[1] / "case_outputs" / case_id
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
        }

        json_path = case_dir / "case_result.json"
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return case_dir

    def select_case(self):
        case_id = self.case_panel.selected_case_id()
        if not case_id:
            return

        selected = next((case for case in self.cases if case.case_id == case_id), None)
        if not selected:
            return

        self.current_case = selected
        self.fields_panel.load_case(selected)
        self.report_panel.set_report(selected.report)

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

        self.status_label.setText(f"已选择病例：{case_id}")

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
            if saved_case.case_id == case.case_id:
                self.cases[index] = case
                self._refresh_case_list()
                return

        self.cases.insert(0, case)
        self._refresh_case_list()

    def _refresh_case_list(self):
        selected_id = self.current_case.case_id if self.current_case else ""
        self.case_panel.refresh(self.cases, selected_id)

    def _open_pdf_report_window(self, report_html):
        window = PdfReportWindow(
            self.current_case.case_id,
            report_html,
            on_save_edits=self._save_pdf_window_edits,
        )
        window.destroyed.connect(lambda: self._remove_report_window(window))
        self.report_windows.append(window)
        window.show()

    def _save_pdf_window_edits(self, plain_text, html):
        if not self.current_case:
            return
        self.current_case.report = plain_text
        self.report_panel.set_report(plain_text)
        self.current_case.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._add_or_update_case(self.current_case)
        self.storage_service.save_cases(self.cases)
        self.status_label.setText("PDF窗口中的报告修改已保存到当前病例。")

    def _remove_report_window(self, window):
        if window in self.report_windows:
            self.report_windows.remove(window)
