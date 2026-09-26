from dataclasses import asdict, dataclass, field


@dataclass
class CaseRecord:
    case_id: str
    patient_name: str = ""
    age: str = ""
    gender: str = "女"
    image_path: str = ""
    video_path: str = ""
    lesion_frame_paths: list = field(default_factory=list)
    lesion_frame_indices: list = field(default_factory=list)
    lesion_features: dict = field(default_factory=dict)
    site: str = "颈部"
    size: str = ""
    shape: str = "椭圆形"
    boundary: str = "清晰"
    echo: str = "低回声"
    calcification: str = "无"
    necrosis: str = "无"
    vascularity: str = "少量"
    ai_score: str = ""
    diagnosis: str = "建议结合临床"
    report: str = ""
    updated_at: str = ""
    ln_instances: list = field(default_factory=list)
    modality: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data):
        valid_keys = {field.name for field in cls.__dataclass_fields__.values()}
        values = {key: value for key, value in data.items() if key in valid_keys}
        return cls(**values)

    def to_dict(self):
        # cases.json 不保存 ln_instances（含 numpy ndarray，无法 JSON 序列化，
        # 且完整 AI 结果已单独写入 case_outputs/<case_id>/case_result.json）
        data = asdict(self)
        data.pop("ln_instances", None)
        return data
