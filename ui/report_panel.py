from PyQt5.QtWidgets import QGroupBox, QTextEdit, QVBoxLayout


class ReportPanel(QGroupBox):
    def __init__(self):
        super().__init__("自动生成报告 / 医生修改")
        self.editor = QTextEdit()
        self.editor.setPlaceholderText("点击“生成报告”后自动生成，医生可在此修改。")
        self.editor.setMinimumHeight(150)

        layout = QVBoxLayout(self)
        layout.addWidget(self.editor)

    def set_report(self, report):
        self.editor.setPlainText(report)

    def report(self):
        return self.editor.toPlainText()
