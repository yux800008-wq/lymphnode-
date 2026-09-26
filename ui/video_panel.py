from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QFont, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QGroupBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QVBoxLayout,
)


class VideoPanel(QGroupBox):
    def __init__(self):
        super().__init__("当前淋巴结结果")
        self.features_label = QLabel("暂无淋巴结结果")
        self.features_label.setWordWrap(True)
        self.features_label.setMinimumHeight(240)
        self.features_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.features_label.setFont(QFont("Microsoft YaHei", 15))
        self.features_label.setStyleSheet("""
            QLabel {
                background: white;
                border: 1px solid #d7e0ea;
                border-radius: 6px;
                padding: 12px;
            }
        """)
        self.features_label.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        self.frame_list = QListWidget()
        self.frame_list.setViewMode(QListWidget.IconMode)
        self.frame_list.setIconSize(QSize(150, 100))
        self.frame_list.setResizeMode(QListWidget.Adjust)
        self.frame_list.setMovement(QListWidget.Static)
        self.frame_list.setSpacing(8)
        self.frame_list.setSelectionMode(QListWidget.SingleSelection)
        self.frame_list.setMaximumHeight(160)
        self.frame_list.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self.features_label)
        layout.addWidget(self.frame_list)

    def set_features(self, text):
        self.features_label.setText(text or "暂无淋巴结结果")

    def clear(self):
        self.features_label.setText("暂无淋巴结结果")
        self.frame_list.clear()
        self.frame_list.hide()

    def set_frames(self, frame_items):
        self.frame_list.clear()

        if not frame_items:
            self.frame_list.hide()
            return

        self.frame_list.show()

        for item in frame_items:
            index = item.get("frame_index")
            thumb_path = item.get("thumbnail_path")
            title = f"帧 {index}"
            list_item = QListWidgetItem(title)
            pixmap = QPixmap(thumb_path)

            if not pixmap.isNull():
                list_item.setIcon(
                    QIcon(
                        pixmap.scaled(
                            150,
                            100,
                            Qt.KeepAspectRatio,
                            Qt.SmoothTransformation,
                        )
                    )
                )

            list_item.setData(Qt.UserRole, item)
            self.frame_list.addItem(list_item)
