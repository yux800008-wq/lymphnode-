from PyQt5.QtCore import QRectF, QSizeF, Qt
from PyQt5.QtGui import QPainter, QTextDocument
from PyQt5.QtPrintSupport import QPrinter
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class PdfReportWindow(QWidget):
    def __init__(self, case_id, html, on_save_edits=None, parent=None):
        super().__init__(None)
        self.case_id = case_id
        self.on_save_edits = on_save_edits
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle(f"PDF图示报告 - {case_id}")
        self.resize(980, 1040)

        self.editor = QTextEdit()
        self.editor.setHtml(html)
        self.editor.setMinimumSize(880, 820)

        self.save_edit_button = QPushButton("保存修改")
        self.export_pdf_button = QPushButton("导出PDF")
        self.close_button = QPushButton("关闭")

        self.save_edit_button.clicked.connect(self.save_edits)
        self.export_pdf_button.clicked.connect(self.export_pdf)
        self.close_button.clicked.connect(self.close)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.save_edit_button)
        button_layout.addWidget(self.export_pdf_button)
        button_layout.addWidget(self.close_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.editor)
        layout.addLayout(button_layout)

    def save_edits(self):
        if self.on_save_edits:
            self.on_save_edits(self.editor.toPlainText(), self.editor.toHtml())
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

        self._write_pdf(file_path)
        QMessageBox.information(self, "导出成功", f"PDF报告已保存到：\n{file_path}")

    def _write_pdf(self, file_path):
        printer = QPrinter(QPrinter.HighResolution)
        printer.setOutputFormat(QPrinter.PdfFormat)
        printer.setOutputFileName(file_path)
        printer.setPageSize(QPrinter.A4)
        printer.setPageMargins(0, 0, 0, 0, QPrinter.Millimeter)

        document = QTextDocument()
        document.setHtml(self.editor.toHtml())
        page_size = printer.pageRect(QPrinter.Point).size()
        margin = 24
        target_width = page_size.width() - margin * 2
        target_height = page_size.height() - margin * 2

        document.setTextWidth(target_width)
        document.setPageSize(QSizeF(page_size.width(), 10000))
        document.adjustSize()

        content_size = document.size()
        width_scale = target_width / max(content_size.width(), 1)
        height_scale = target_height / max(content_size.height(), 1)
        scale = min(width_scale, height_scale, 1.0)
        draw_width = content_size.width() * scale
        draw_height = content_size.height() * scale
        offset_x = (page_size.width() - draw_width) / 2
        offset_y = max(margin / 2, (page_size.height() - draw_height) / 2)

        painter = QPainter(printer)
        painter.translate(offset_x, offset_y)
        painter.scale(scale, scale)
        document.drawContents(
            painter,
            QRectF(0, 0, content_size.width(), content_size.height()),
        )
        painter.end()
