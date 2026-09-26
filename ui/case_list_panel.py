from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGroupBox, QListWidget, QListWidgetItem, QVBoxLayout


class CaseListPanel(QGroupBox):
    def __init__(self):
        super().__init__("病例列表")
        self.list_widget = QListWidget()

        layout = QVBoxLayout(self)
        layout.addWidget(self.list_widget)

    def refresh(self, cases, selected_case_id=""):
        self.list_widget.blockSignals(True)
        self.list_widget.clear()

        for case in cases:
            text = f"{case.case_id}  {case.patient_name}".strip()
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, case.case_id)
            self.list_widget.addItem(item)
            if case.case_id == selected_case_id:
                item.setSelected(True)

        self.list_widget.blockSignals(False)

    def selected_case_id(self):
        items = self.list_widget.selectedItems()
        if not items:
            return ""
        return items[0].data(Qt.UserRole)
