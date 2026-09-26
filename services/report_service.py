from datetime import datetime
from html import escape
from pathlib import Path


class ReportService:
    def generate_text(self, case):
        return (
            f"超声所见：{case.site}见淋巴结样结构，大小约{case.size}，"
            f"形态呈{case.shape}，边界{case.boundary}，内部回声为{case.echo}，"
            f"钙化：{case.calcification}，液化/坏死：{case.necrosis}，"
            f"血流信号：{case.vascularity}。\n\n"
            f"AI分析：疑似病灶已完成检测与分割，AI置信度 {case.ai_score}。\n\n"
            f"超声提示：{case.diagnosis}。建议结合病史、实验室检查及必要时穿刺/病理结果综合判断。"
        )

    def generate_html(self, case, ai_image_path=""):
        finding = (
            f"{case.site}可见淋巴结样回声，大小约{case.size}，形态{case.shape}，"
            f"边界{case.boundary}，内部呈{case.echo}，钙化：{case.calcification}，"
            f"液化/坏死：{case.necrosis}，CDFI：其内可见{case.vascularity}血流信号。"
        )
        impression = f"{case.site}淋巴结长大：{case.diagnosis}。"
        source_image = self._image_uri(case.image_path)
        ai_image = self._image_uri(ai_image_path or case.image_path)
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
    font-size: 11pt;
}}
table {{
    border-collapse: collapse;
}}
.page {{
    width: 560px;
    margin: 0 auto;
    padding: 18px 0 0;
}}
.hospital {{
    margin: 0;
    font-size: 24pt;
    font-weight: normal;
    text-align: center;
    letter-spacing: 1px;
}}
.title {{
    margin: 0;
    font-size: 23pt;
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
.qr {{
    width: 62px;
    height: 62px;
    background: #d0d0d0;
    border: 1px solid #999;
}}
.report-no {{
    width: 560px;
    text-align: right;
    border-bottom: 2px solid #000;
    padding-right: 64px;
    font-size: 10.5pt;
}}
.info {{
    width: 560px;
    border-bottom: 1px solid #000;
    margin-top: 2px;
    font-size: 11pt;
}}
.info td {{
    height: 20px;
    vertical-align: middle;
}}
.label {{
    width: 62px;
    white-space: nowrap;
}}
.value {{
    white-space: nowrap;
}}
.section-title {{
    color: #174bff;
    font-weight: bold;
    font-size: 11.5pt;
}}
.paragraph {{
    font-size: 12pt;
    line-height: 1.55;
}}
.footer {{
    width: 560px;
    border-top: 2px solid #000;
    font-size: 11pt;
}}
.notice {{
    width: 560px;
    border-top: 2px solid #000;
    font-size: 9.5pt;
    padding-top: 3px;
}}
</style>
</head>
<body>
<div class="page">
    <table width="560">
        <tr>
            <td width="75" valign="top"><div class="logo">医院</div></td>
            <td width="410" align="center">
                <div class="hospital">凉山彝族自治州第一人民医院</div>
                <div class="title">彩色超声检查报告</div>
            </td>
            <td width="75" align="right" valign="top"><div class="qr"></div></td>
        </tr>
    </table>

    <div class="report-no">超声号：{escape(case.case_id)}</div>

    <table class="info">
        <tr>
            {self._info_cell("姓　　名", case.patient_name or "--", 185)}
            {self._info_cell("性　　别", case.gender, 150)}
            {self._info_cell("年　　龄", case.age or "--", 225)}
        </tr>
        <tr>
            {self._info_cell("科　　室", "感染儿科", 185)}
            {self._info_cell("床　　号", "--", 150)}
            {self._info_cell("住 院 号", "--", 225)}
        </tr>
        <tr>
            {self._info_cell("临床诊断", case.diagnosis, 335)}
            {self._info_cell("条 码 号", case.case_id[-8:], 225)}
        </tr>
        <tr>
            {self._info_cell("检查部位", case.site, 560)}
        </tr>
    </table>

    <table width="560" style="margin-top: 8px; margin-bottom: 16px;">
        <tr>
            <td width="280" align="center">
                <img src="{source_image}" width="225" height="168">
            </td>
            <td width="280" align="center">
                <img src="{ai_image}" width="225" height="168">
            </td>
        </tr>
    </table>

    <table width="560">
        <tr><td class="section-title">超声所见：</td></tr>
        <tr>
            <td height="185" valign="top" class="paragraph">
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{escape(finding)}
            </td>
        </tr>
        <tr><td class="section-title">超声提示：</td></tr>
        <tr>
            <td height="112" valign="top" class="paragraph">
                {escape(impression)}
            </td>
        </tr>
    </table>

    <table class="footer">
        <tr>
            <td width="280" height="28">报告医生：AI辅助系统</td>
            <td width="280">审核医生：--</td>
        </tr>
        <tr>
            <td height="28">记录者：--</td>
            <td>报告时间：{report_time}</td>
        </tr>
    </table>
    <div class="notice">注意：医生签字有效，此报告仅反映当时被检者情况，仅供临床医师参考。</div>
</div>
</body>
</html>
"""

    def generate(self, case):
        return self.generate_text(case)

    def _info_cell(self, label, value, width):
        return (
            f'<td width="{width}">'
            f'<table width="{width}"><tr>'
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
