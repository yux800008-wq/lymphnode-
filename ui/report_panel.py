import math

from PyQt5.QtCore import pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import (
    QGroupBox, QTextEdit, QVBoxLayout, QHBoxLayout, QPushButton,
    QLineEdit, QCheckBox, QLabel, QGraphicsScene, QGraphicsView,
)


class ReportPanel(QGroupBox):
    IMAGE_HIDDEN_STYLE = '<style>img { display: none; }</style>'
    DEFAULT_IMAGE_SIZE = (450.0, 336.0)
    HIDDEN_IMAGE_SIZE = 0.01
    DEFAULT_LAYOUT = {
        "hospital_name": "凉山彝族自治州第一人民医院",
        "title": "彩色超声检查报告",
        "footer": "注意：医生签字有效，此报告仅反映当时被检者情况，仅供临床医师参考。",
        "show_images": True,
    }
    save_current_requested = pyqtSignal()
    save_default_requested = pyqtSignal()
    restore_default_requested = pyqtSignal()
    export_requested = pyqtSignal()
    content_changed = pyqtSignal()
    images_visibility_changed = pyqtSignal(bool)

    def __init__(self):
        super().__init__("自动生成报告 / 医生修改")
        self.editor = QTextEdit()
        self.editor.setAcceptRichText(True)
        self.editor.setPlaceholderText("点击“生成报告”后自动生成，医生可在此修改。")
        self.editor.setMinimumHeight(150)
        self.editor.setFixedSize(794, 1123)
        self.editor.setStyleSheet("QTextEdit { background: white; border: 1px solid #cbd5e1; padding: 45px; }")
        self.editor.setLineWrapMode(QTextEdit.WidgetWidth)
        self.editor.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._page_width = 794
        self._page_height = 1123
        self._zoom = 1.0
        self._fit_width = True
        self._page_count = 1
        self._updating_page_extent = False
        self._images_visible = True
        self._image_sizes = {}
        self._active_layout = dict(self.DEFAULT_LAYOUT)
        self.page_scene = QGraphicsScene(self)
        self.page_view = QGraphicsView(self.page_scene)
        self.page_view.setAlignment(Qt.AlignCenter)
        self.page_view.setStyleSheet("QGraphicsView { background: #e5e7eb; border: none; }")
        self.page_proxy = self.page_scene.addWidget(self.editor)

        self.zoom_out_button = QPushButton("缩小")
        self.zoom_in_button = QPushButton("放大")
        self.fit_width_button = QPushButton("适合宽度")
        self.zoom_out_button.clicked.connect(lambda: self._change_zoom(-0.1))
        self.zoom_in_button.clicked.connect(lambda: self._change_zoom(0.1))
        self.fit_width_button.clicked.connect(self.fit_width)

        self.hospital_edit = QLineEdit()
        self.title_edit = QLineEdit()
        self.footer_edit = QLineEdit()
        self.show_images = QCheckBox("显示图像区域")
        self.show_images.setChecked(True)
        layout_fields = QHBoxLayout()
        layout_fields.addWidget(QLabel("医院")); layout_fields.addWidget(self.hospital_edit)
        layout_fields.addWidget(QLabel("标题")); layout_fields.addWidget(self.title_edit)
        layout_fields.addWidget(self.show_images)
        layout_fields.addWidget(QLabel("页脚")); layout_fields.addWidget(self.footer_edit)

        self.save_current_button = QPushButton("保存当前版式")
        self.save_default_button = QPushButton("保存为默认版式")
        self.restore_default_button = QPushButton("恢复初始版式")
        self.export_button = QPushButton("导出 PDF")
        buttons = QHBoxLayout()
        buttons.addWidget(self.save_current_button)
        buttons.addWidget(self.save_default_button)
        buttons.addWidget(self.restore_default_button)
        buttons.addWidget(self.export_button)
        buttons.addWidget(self.zoom_out_button)
        buttons.addWidget(self.zoom_in_button)
        buttons.addWidget(self.fit_width_button)
        buttons.addStretch()
        self.save_current_button.clicked.connect(self.save_current_requested)
        self.save_default_button.clicked.connect(self.save_default_requested)
        self.restore_default_button.clicked.connect(self.restore_default_requested)
        self.export_button.clicked.connect(self.export_requested)
        self.editor.textChanged.connect(self.content_changed)
        self.editor.textChanged.connect(lambda: QTimer.singleShot(0, self._update_page_extent))

        layout = QVBoxLayout(self)
        layout.addLayout(layout_fields)
        layout.addLayout(buttons)
        layout.addWidget(self.page_view, 1)
        self.show_images.toggled.connect(self.set_images_visible)
        self.show_images.toggled.connect(self.images_visibility_changed)
        self._apply_zoom()

    def set_report(self, report, html=""):
        self.editor.blockSignals(True)
        self._image_sizes = {}
        if html:
            self.editor.setHtml(html)
        else:
            self.editor.setPlainText(report)
        self.editor.blockSignals(False)
        self.set_images_visible(self.show_images.isChecked())
        QTimer.singleShot(0, lambda: self._update_page_extent(False))

    def report(self):
        return self.editor.toPlainText()

    def html(self):
        html = self.editor.toHtml()
        html = html.replace('<style>.report-images { display: none; }</style>', '')
        html = html.replace(self.IMAGE_HIDDEN_STYLE, '')
        return html

    def set_images_visible(self, visible):
        self._images_visible = bool(visible)
        document = self.editor.document()
        signals_blocked = self.editor.blockSignals(True)
        block = document.begin()
        while block.isValid():
            iterator = block.begin()
            while not iterator.atEnd():
                fragment = iterator.fragment()
                if fragment.isValid() and fragment.charFormat().isImageFormat():
                    image_format = fragment.charFormat().toImageFormat()
                    image_name = image_format.name()
                    if self._images_visible:
                        width, height = self._image_sizes.get(
                            image_name, self.DEFAULT_IMAGE_SIZE
                        )
                    else:
                        if image_format.width() > 1 and image_format.height() > 1:
                            self._image_sizes[image_name] = (
                                image_format.width(), image_format.height()
                            )
                        width = height = self.HIDDEN_IMAGE_SIZE
                    cursor = QTextCursor(document)
                    cursor.setPosition(fragment.position())
                    cursor.setPosition(
                        fragment.position() + fragment.length(), QTextCursor.KeepAnchor
                    )
                    image_format.setWidth(width)
                    image_format.setHeight(height)
                    cursor.setCharFormat(image_format)
                iterator += 1
            block = block.next()
        self.editor.blockSignals(signals_blocked)
        document.markContentsDirty(0, document.characterCount())
        QTimer.singleShot(0, self._update_page_extent)

    def _change_zoom(self, delta):
        self._fit_width = False
        self._zoom = max(0.5, min(2.0, round(self._zoom + delta, 2)))
        self._apply_zoom()

    def fit_width(self):
        self._fit_width = True
        available = max(1, self.page_view.viewport().width() - 24)
        self._zoom = max(0.5, min(2.0, available / self._page_width))
        self._apply_zoom()
        self.page_view.verticalScrollBar().setValue(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit_width:
            self.fit_width()

    def _update_scene_rect(self):
        self.page_scene.setSceneRect(
            0,
            0,
            self._page_width * self._zoom,
            self._page_height * self._page_count * self._zoom,
        )

    def _apply_zoom(self):
        self.page_proxy.setScale(self._zoom)
        self._update_scene_rect()
        self.page_view.centerOn(self.page_scene.sceneRect().center())

    def _update_page_extent(self, preserve_view=True):
        if self._updating_page_extent:
            return
        self._updating_page_extent = True
        try:
            document = self.editor.document()
            last_block = document.lastBlock()
            content_height = (
                document.documentLayout().blockBoundingRect(last_block).bottom()
                + document.documentMargin() * 2
                + 90
            )
            page_count = max(1, int(math.ceil(content_height / self._page_height)))
            if page_count == self._page_count:
                return
            vertical = self.page_view.verticalScrollBar().value()
            horizontal = self.page_view.horizontalScrollBar().value()
            cursor = self.editor.textCursor()
            self._page_count = page_count
            self.editor.setFixedHeight(self._page_height * page_count)
            self._update_scene_rect()
            if preserve_view:
                self.editor.setTextCursor(cursor)
                QTimer.singleShot(0, lambda: self._restore_view_position(vertical, horizontal))
        finally:
            self._updating_page_extent = False

    def _restore_view_position(self, vertical, horizontal):
        self.page_view.verticalScrollBar().setValue(vertical)
        self.page_view.horizontalScrollBar().setValue(horizontal)

    def layout_values(self):
        return {"hospital_name": self.hospital_edit.text(), "title": self.title_edit.text(), "footer": self.footer_edit.text(), "show_images": self.show_images.isChecked()}

    def set_layout(self, values):
        values = {**self.DEFAULT_LAYOUT, **(values or {})}
        self.hospital_edit.setText(values["hospital_name"])
        self.title_edit.setText(values["title"])
        self.footer_edit.setText(values["footer"])
        self.show_images.setChecked(values.get("show_images", True))
        self._active_layout = dict(values)

    def apply_layout_to_editor(self, values):
        """Apply layout-only changes without regenerating or replacing medical text."""
        values = {**self.DEFAULT_LAYOUT, **(values or {})}
        old = self._active_layout
        saved_cursor = self.editor.textCursor()
        vertical = self.page_view.verticalScrollBar().value()
        horizontal = self.page_view.horizontalScrollBar().value()
        replacements = (
            (old.get("hospital_name", ""), values["hospital_name"]),
            (old.get("title", ""), values["title"]),
            (old.get("footer", ""), values["footer"]),
        )
        for previous, current in replacements:
            if previous and previous != current:
                match_cursor = self.editor.document().find(previous)
                if not match_cursor.isNull():
                    match_cursor.insertText(current)
        self.set_layout(values)
        self.editor.setTextCursor(saved_cursor)
        QTimer.singleShot(0, lambda: self._restore_view_position(vertical, horizontal))
