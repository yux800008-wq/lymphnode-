APP_STYLE = """
QWidget {
    font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size: 16px;
    color: #172033;
    background: #f6f8fb;
}
#Title {
    font-size: 24px;
    font-weight: 700;
    color: #0f172a;
}
QGroupBox {
    background: #ffffff;
    border: 1px solid #d8dee9;
    border-radius: 8px;
    margin-top: 12px;
    padding: 12px;
    font-weight: 700;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
}
QPushButton {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: 600;
}
QPushButton:hover {
    background: #1d4ed8;
}
QPushButton:pressed {
    background: #1e40af;
}
QListWidget, QTextEdit, QLineEdit, QComboBox, QGraphicsView {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
}
QListWidget::item {
    padding: 8px;
    border-bottom: 1px solid #edf2f7;
}
QListWidget::item:selected {
    background: #dbeafe;
    color: #1e3a8a;
}
#Hint {
    color: #64748b;
    font-size: 14px;
}
"""
