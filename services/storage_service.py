import json
from pathlib import Path

from models import CaseRecord


class StorageService:
    def __init__(self, data_file="cases.json"):
        self.data_file = Path(data_file)

    def load_cases(self):
        if not self.data_file.exists():
            return []

        try:
            raw_cases = json.loads(self.data_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

        return [CaseRecord.from_dict(item) for item in raw_cases]

    def save_cases(self, cases):
        payload = [case.to_dict() for case in cases]
        self.data_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
