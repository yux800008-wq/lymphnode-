from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QComboBox, QFormLayout, QGroupBox, QLineEdit


FIELD_LABELS = {
    "case_id": "病例号",
    "patient_name": "姓名",
    "age": "年龄",
    "gender": "性别",
    "site": "部位",
    "size": "大小",
    "shape": "形态",
    "boundary": "边界",
    "echo": "回声",
    "calcification": "钙化",
    "necrosis": "液化/坏死",
    "vascularity": "血流",
    "ai_score": "AI置信度",
    "diagnosis": "AI提示",
}

# Fields shown in the UI. Hidden fields remain in self.fields for data persistence.
VISIBLE_FIELDS = ["case_id", "patient_name", "age", "gender", "site"]


class FieldsPanel(QGroupBox):
    def __init__(self):
        super().__init__("病例基础信息")
        self.fields = {
            "case_id": QLineEdit(),
            "patient_name": QLineEdit(),
            "age": QLineEdit(),
            "gender": QComboBox(),
            "site": QComboBox(),
            "size": QLineEdit(),
            "shape": QComboBox(),
            "boundary": QComboBox(),
            "echo": QComboBox(),
            "calcification": QComboBox(),
            "necrosis": QComboBox(),
            "vascularity": QComboBox(),
            "ai_score": QLineEdit(),
            "diagnosis": QComboBox(),
        }
        self._setup_options()
        self._build_layout()

    def _setup_options(self):
        self.fields["gender"].addItems(["女", "男"])
        self.fields["site"].addItems(["颈部", "腋窝", "锁骨上", "腹股沟", "纵隔", "其他"])
        self.fields["shape"].addItems(["椭圆形", "圆形", "不规则"])
        self.fields["boundary"].addItems(["清晰", "欠清晰", "不清晰"])
        self.fields["echo"].addItems(["低回声", "等回声", "高回声", "混合回声"])
        self.fields["calcification"].addItems(["无", "点状", "粗大", "弧形"])
        self.fields["necrosis"].addItems(["无", "可疑", "有"])
        self.fields["vascularity"].addItems(["少量", "中等", "丰富"])
        self.fields["diagnosis"].addItems(["倾向淋巴结结核", "炎性淋巴结", "肿瘤转移待排", "建议结合临床"])
        self.fields["ai_score"].setReadOnly(True)

        # 病例基础信息编辑权限：
        # 病例号是医生可编辑的显示编号；CaseRecord.record_id 保持内部关联稳定。
        self.fields["case_id"].setReadOnly(False)

        for key in ("patient_name", "age"):
            line_edit = self.fields[key]
            line_edit.setReadOnly(False)
            line_edit.setEnabled(True)
            line_edit.setFocusPolicy(Qt.StrongFocus)

        for key in ("gender", "site"):
            combo = self.fields[key]
            combo.setEnabled(True)
            combo.setFocusPolicy(Qt.StrongFocus)

    def _build_layout(self):
        layout = QFormLayout(self)
        layout.setLabelAlignment(Qt.AlignRight)

        for key in VISIBLE_FIELDS:
            label = FIELD_LABELS[key]
            widget = self.fields[key]
            widget.setMinimumHeight(30)
            layout.addRow(label, widget)

    def load_case(self, case):
        for key in self.fields:
            self.set_value(key, getattr(case, key, ""))

    def update_case(self, case):
        for key, widget in self.fields.items():
            if isinstance(widget, QComboBox):
                setattr(case, key, widget.currentText())
            else:
                setattr(case, key, widget.text())

    def apply_ai_result(self, result):
        for key, value in result.items():
            if key in self.fields:
                self.set_value(key, value)

    def set_value(self, key, value):
        widget = self.fields[key]
        if isinstance(widget, QComboBox):
            index = widget.findText(str(value))
            widget.setCurrentIndex(index if index >= 0 else 0)
        else:
            widget.setText(str(value))
