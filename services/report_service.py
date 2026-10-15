from datetime import datetime
from html import escape
from pathlib import Path


class ReportService:
    PHENOTYPE_LABELS = {
        "posterior": ("后方回声", {"non_enhancement": "无增强", "enhancement": "增强"}),
        "liquefaction": ("液化", {"absent": "无", "present": "有"}),
        "hilum": ("淋巴门", {"disappearance": "消失", "destruction": "破坏"}),
        "margin": ("边缘", {"blurred": "模糊", "partial": "部分清晰", "clear": "清晰"}),
        "shape": ("形状", {"ellipse": "椭圆形", "round": "圆形", "fusion": "融合"}),
    }
    VASCULARITY_LABELS = {
        "avascular": "未见血流",
        "mixed": "混合型血流",
        "peripheral": "周边血流",
        "hilar": "门部血流",
    }
    def generate_text(self, case, template=None):
        instances = getattr(case, "ln_instances", []) or []
        if instances:
            return self._generate_instance_text(case, instances)
        return "超声所见：当前图像未获得有效淋巴结检测结果，建议结合原图进行人工复核。"

    def generate_finding(self, case):
        text = self.generate_text(case)
        return text[len("超声所见："):] if text.startswith("超声所见：") else text

    def generate_impression(self, case):
        diagnosis = str(getattr(case, "diagnosis", "") or "").strip("。 ")
        if diagnosis == "未检测到淋巴结":
            return "当前图像未获得有效淋巴结检测结果，建议检查图像并进行人工复核。"
        if diagnosis and diagnosis != "建议结合临床":
            return f"{diagnosis}。建议结合病史、实验室检查及必要时穿刺/病理结果综合判断。"
        return "建议结合病史、实验室检查及必要时穿刺/病理结果综合判断。"
    def _generate_instance_text(self, case, instances):
        features = [self._instance_features(instance) for instance in instances]
        signatures = [tuple(sorted((key, value) for key, value in item.items() if key != "size")) for item in features]
        count = len(instances)
        prefix = f"超声所见：当前图像{case.site}检测到{count}个淋巴结"

        if len(set(signatures)) == 1:
            common = self._feature_sentence(dict(signatures[0]))
            sizes = [item.get("size") for item in features if item.get("size")]
            size_text = f"，大小分别为{'、'.join(sizes)}" if sizes else ""
            return prefix + (f"，{common}" if common else "") + size_text + "。"

        common_keys = set(features[0])
        for item in features[1:]:
            common_keys.intersection_update(item)
        common = {
            key: features[0][key] for key in common_keys
            if all(item.get(key) == features[0][key] for item in features[1:]) and key != "size"
        }
        text = prefix + (f"，共同表现为{self._feature_sentence(common)}" if common else "")
        differing = []
        for index, item in enumerate(features, 1):
            individual = {key: value for key, value in item.items() if key not in common or key == "size"}
            sentence = self._feature_sentence(individual)
            if sentence:
                differing.append(f"目标{index}{sentence}")
        return text + ("；" + "；".join(differing) if differing else "") + "。"

    def _instance_features(self, instance):
        features = {}
        measurements = instance.get("measurements", {}) or {}
        long_mm = measurements.get("long_diameter_mm")
        short_mm = measurements.get("short_diameter_mm")
        if long_mm is not None and short_mm is not None:
            features["size"] = f"{long_mm:.1f} mm × {short_mm:.1f} mm"
        phenotypes = instance.get("phenotypes", {}) or {}
        for key in ("shape", "margin", "hilum", "liquefaction", "posterior"):
            value = phenotypes.get(key, {}) or {}
            field_name, labels = self.PHENOTYPE_LABELS[key]
            label = labels.get(value.get("label"))
            if label:
                features[field_name] = label
        vascularity = instance.get("vascularity", {}) or {}
        if vascularity.get("available") and vascularity.get("label"):
            features["血供"] = self.VASCULARITY_LABELS.get(vascularity["label"], vascularity["label"])
        return features

    @staticmethod
    def _feature_sentence(features):
        if not features:
            return ""
        ordered = [
            ("size", "大小约"), ("形状", "形态"), ("边缘", "边缘"),
            ("淋巴门", "淋巴门"), ("液化", "液化"), ("后方回声", "后方回声"),
            ("血供", "血供"),
        ]
        return "，".join(f"{label}{features[key]}" for key, label in ordered if key in features)

    def generate_html(self, case, ai_image_path="", layout=None, finding=None, impression=None):
        layout = layout or {}
        finding = self.generate_finding(case) if finding is None else finding
        impression = self.generate_impression(case) if impression is None else impression
        hospital_name = escape(layout.get("hospital_name", "凉山彝族自治州第一人民医院"))
        report_title = escape(layout.get("title", "彩色超声检查报告"))
        footer = escape(layout.get("footer", "注意：医生签字有效，此报告仅反映当时被检者情况，仅供临床医师参考。"))
        source_image = self._image_uri(case.image_path)
        image_block = f'''<div class="report-images"><table width="100%" style="margin-top: 8px; margin-bottom: 16px;">\n        <tr><td align="center"><img src="{source_image}" width="450" height="336"></td></tr>\n    </table></div>'''
        report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
body {{
    margin: 0;
    padding: 0;
    color: #000;
    background: #fff;
    font-family: SimSun, "Microsoft YaHei", serif;
    font-size: 12pt;
}}
table {{
    border-collapse: collapse;
    max-width: 100%;
    table-layout: fixed;
}}
.page {{
    width: 100%;
    max-width: 100%;
    box-sizing: border-box;
    margin: 0;
    padding: 12px 0 0;
}}
img {{
    max-width: 100%;
    height: auto;
}}
.hospital {{
    margin: 0;
    font-size: 17pt;
    font-weight: normal;
    text-align: center;
    letter-spacing: 1px;
}}
.title {{
    margin: 0;
    font-size: 17pt;
    font-weight: normal;
    text-align: center;
    letter-spacing: 2px;
}}
.logo {{
    width: 58px;
    height: 58px;
    border: 2px solid #0f5f99;
    border-radius: 29px;
    color: #0f5f99;
    font-weight: bold;
    font-size: 10pt;
    text-align: center;
    line-height: 58px;
}}
.report-no {{
    width: 100%;
    box-sizing: border-box;
    text-align: right;
    border-bottom: 2px solid #000;
    padding-right: 8px;
    font-size: 10.5pt;
    white-space: nowrap;
}}
.info {{
    width: 100%;
    border-bottom: 1px solid #000;
    margin-top: 2px;
    font-size: 11.5pt;
}}
.info td {{
    height: 24px;
    vertical-align: middle;
    padding: 2px 4px;
    text-align: left;
}}
.label {{
    white-space: nowrap;
    font-weight: bold;
    font-size: 10.5pt;
}}
.value {{
    white-space: normal;
}}
.short-value {{
    white-space: nowrap;
    font-size: 10.5pt;
}}
.barcode-value {{
    white-space: nowrap;
    font-size: 9.5pt;
    letter-spacing: -0.2px;
}}
.section-title {{
    color: #174bff;
    font-weight: bold;
    font-size: 11.5pt;
}}
.paragraph {{
    font-size: 13pt;
    line-height: 1.5;
}}
.footer {{
    width: 100%;
    border-top: 2px solid #000;
    font-size: 11pt;
}}
.notice {{
    width: 100%;
    border-top: 2px solid #000;
    font-size: 9.5pt;
    padding-top: 3px;
}}
</style>
</head>
<body>
<div class="page">
    <table width="100%">
        <tr>
            <td width="11%" valign="top"><div class="logo">医院</div></td>
            <td width="89%" align="center" style="padding-right: 154px;">
                <div class="hospital">{hospital_name}</div>
                <div class="title">{report_title}</div>
            </td>
        </tr>
    </table>

    <div class="report-no">超声号：{escape(case.case_id)}</div>

    <table class="info" width="100%">
        <tr>
            <td class="label" width="12%" nowrap="nowrap">姓名</td><td class="value short-value" width="24%" nowrap="nowrap">{escape(case.patient_name or "--")}</td>
            <td class="label" width="12%" nowrap="nowrap">性别</td><td class="value short-value" width="18%" nowrap="nowrap">{escape(case.gender)}</td>
            <td class="label" width="12%" nowrap="nowrap">年龄</td><td class="value short-value" width="22%" nowrap="nowrap">{escape(str(case.age or "--"))}</td>
        </tr>
        <tr>
            <td class="label" width="12%" nowrap="nowrap">科室</td><td class="value short-value" width="24%" nowrap="nowrap">感染儿科</td>
            <td class="label" width="12%" nowrap="nowrap">床号</td><td class="value short-value" width="18%" nowrap="nowrap">--</td>
            <td class="label" width="12%" nowrap="nowrap">住院号</td><td class="value short-value" width="22%" nowrap="nowrap">--</td>
        </tr>
        <tr>
            <td class="label" width="12%" nowrap="nowrap">临床诊断</td><td class="value" width="54%" colspan="3">医生审核后填写</td>
            <td class="label" width="12%" nowrap="nowrap">条码号</td><td class="value barcode-value" width="22%" nowrap="nowrap">{escape(case.case_id[-8:])}</td>
        </tr>
        <tr>
            <td class="label" width="12%" nowrap="nowrap">检查部位</td><td class="value" width="88%" colspan="5">{escape(case.site)}</td>
        </tr>
    </table>

    {image_block}

    <table width="100%">
        <tr><td class="section-title">超声所见：</td></tr>
        <tr>
            <td height="125" valign="top" class="paragraph">
                {escape(finding)}
            </td>
        </tr>
        <tr><td class="section-title">超声提示：</td></tr>
        <tr>
            <td height="70" valign="top" class="paragraph">
                {escape(impression)}
            </td>
        </tr>
    </table>

    <table class="footer">
        <tr>
            <td width="50%" height="28">报告医生：AI辅助系统</td>
            <td width="50%">审核医生：--</td>
        </tr>
        <tr>
            <td height="28">记录者：--</td>
            <td>报告时间：{report_time}</td>
        </tr>
    </table>
    <div class="notice">{footer}</div>
</div>
</body>
</html>
"""

    def generate(self, case):
        return self.generate_text(case)

    def _info_cell(self, label, value, width):
        width_pct = f"{width / 7:.2f}%"
        return (
            f'<td width="{width_pct}">'
            f'<table width="100%"><tr>'
            f'<td class="label">{escape(label)}</td>'
            f'<td class="value">{escape(str(value))}</td>'
            f"</tr></table>"
            f"</td>"
        )

    def _image_uri(self, image_path):
        if not image_path:
            return ""
        path = Path(image_path)
        if not path.exists():
            return ""
        return path.resolve().as_uri()
