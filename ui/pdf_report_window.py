import math

from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QPainter, QTextDocument
from PyQt5.QtPrintSupport import QPrinter
from PyQt5.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)
from ui.report_panel import ReportPanel


class PdfReportWindow(QWidget):
    def __init__(
        self,
        case_id,
        html,
        on_save_edits=None,
        on_content_changed=None,
        on_layout_changed=None,
        on_save_default=None,
        on_restore_default=None,
        on_exported=None,
        layout=None,
        parent=None,
    ):
        super().__init__(None)
        self.case_id = case_id
        self.on_save_edits = on_save_edits
        self.on_content_changed = on_content_changed
        self.on_layout_changed = on_layout_changed
        self.on_save_default = on_save_default
        self.on_restore_default = on_restore_default
        self.on_exported = on_exported
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle(f"报告编辑 - {case_id}")
        self.resize(980, 1040)

        self.report_editor = ReportPanel()
        self.report_editor.set_layout(layout)
        self.report_editor.set_report("", html)
        self.editor = self.report_editor.editor
        self.report_editor.content_changed.connect(self._content_changed)
        self.report_editor.images_visibility_changed.connect(self._layout_changed)
        self.report_editor.save_current_requested.connect(self._save_current_layout)
        self.report_editor.save_default_requested.connect(self._save_default_layout)
        self.report_editor.restore_default_requested.connect(self._restore_default_layout)
        self.report_editor.export_requested.connect(self.export_pdf)

        layout = QVBoxLayout(self)
        layout.addWidget(self.report_editor)

    def _content_changed(self):
        if self.on_content_changed:
            self.on_content_changed(self.report_editor.report(), self.report_editor.html())

    def _layout_changed(self, visible):
        if self.on_layout_changed:
            self.on_layout_changed(self.report_editor.layout_values())
        if self.on_content_changed:
            self.on_content_changed(self.report_editor.report(), self.report_editor.html())

    def _save_default_layout(self):
        values = self._apply_current_layout()
        if self.on_save_default:
            self.on_save_default(values)

    def _save_current_layout(self):
        self._apply_current_layout()
        if self.on_layout_changed:
            self.on_layout_changed(self.report_editor.layout_values())

    def _apply_current_layout(self):
        values = self.report_editor.layout_values()
        self.report_editor.apply_layout_to_editor(values)
        if self.on_content_changed:
            self.on_content_changed(self.report_editor.report(), self.report_editor.html())
        return values

    def _restore_default_layout(self):
        self.report_editor.apply_layout_to_editor(dict(ReportPanel.DEFAULT_LAYOUT))
        if self.on_content_changed:
            self.on_content_changed(self.report_editor.report(), self.report_editor.html())
        if self.on_restore_default:
            self.on_restore_default(self)

    def save_edits(self):
        if self.on_save_edits:
            self.on_save_edits(self.report_editor.report(), self.report_editor.html())
        QMessageBox.information(self, "保存成功", "报告修改已保存到当前病例。")

    def export_pdf(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存PDF报告",
            f"{self.case_id}_超声检查报告.pdf",
            "PDF Files (*.pdf)",
        )
        if not file_path:
            return
        if not file_path.lower().endswith(".pdf"):
            file_path += ".pdf"

        self._write_pdf(file_path, self.report_editor.editor.document())
        if self.on_exported:
            self.on_exported(file_path)
        QMessageBox.information(self, "导出成功", f"PDF报告已保存到：\n{file_path}")

    @staticmethod
    def export_html(parent, case_id, html):
        file_path, _ = QFileDialog.getSaveFileName(
            parent,
            "保存PDF报告",
            f"{case_id}_超声检查报告.pdf",
            "PDF Files (*.pdf)",
        )
        if not file_path:
            return ""
        if not file_path.lower().endswith(".pdf"):
            file_path += ".pdf"
        PdfReportWindow._write_pdf(file_path, html)
        return file_path

    @staticmethod
    def _write_pdf(file_path, content):
        printer = QPrinter(QPrinter.ScreenResolution)
        printer.setOutputFormat(QPrinter.PdfFormat)
        printer.setOutputFileName(file_path)
        printer.setPageSize(QPrinter.A4)
        printer.setOrientation(QPrinter.Portrait)
        printer.setPageMargins(12, 12, 12, 12, QPrinter.Millimeter)

        printer.setResolution(96)
        page_rect = printer.pageRect(QPrinter.DevicePixel)
        page_size = page_rect.size()

        if isinstance(content, QTextDocument):
            document = content
        else:
            document = QTextDocument()
            document.setDocumentMargin(0)
            document.setHtml(content)
            document.setTextWidth(page_size.width())

        last_block = document.lastBlock()
        content_height = (
            document.documentLayout().blockBoundingRect(last_block).bottom()
            + document.documentMargin() * 2
        )
        page_count = max(1, int(math.ceil(content_height / page_size.height())))

        painter = QPainter(printer)
        for page_index in range(page_count):
            if page_index:
                printer.newPage()
            painter.save()
            painter.translate(page_rect.left(), page_rect.top())
            painter.setClipRect(QRectF(0, 0, page_size.width(), page_size.height()))
            painter.translate(0, -page_index * page_size.height())
            document.drawContents(
                painter,
                QRectF(0, page_index * page_size.height(), page_size.width(), page_size.height()),
            )
            painter.restore()
        painter.end()
