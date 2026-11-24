"""Conservative draft extraction from explicitly labeled archival fields."""

import re

from local_rag.artifacts.models import LABELS

LABEL_MAP = {**{v: k for k, v in LABELS.items()}, "文物名称": "name", "正式名称": "name",
             "时代": "era", "材质": "material", "质地": "material", "收藏单位": "museum",
             "保管单位": "museum", "收藏博物馆": "museum", "馆藏登记号": "accession_number"}
PATTERN = re.compile(r"(?m)^[ \t]*(" + "|".join(map(re.escape, LABEL_MAP)) + r")[：:][ \t]*([^\n]+)")


def draft_records(index, request):
    document_id = str(request.document_id)
    index.get(document_id)
    text = (index.documents.directory(document_id) / "text.txt").read_text(encoding="utf-8")
    records, claims = [], []
    warnings = ["这是待核对草稿。请确认字段属于同一文物、来源类型正确，并区分收藏地点与展出地点后保存。",
                "只提取带明确字段标签的行；未标注的叙述段落需要人工摘录，不推断材料、年代或馆藏编号。"]
    for match in PATTERN.finditer(text):
        field, value = LABEL_MAP[match[1]], match[2].strip()
        if field == "name" and any(c["field"] == "name" for c in claims):
            records.append({"claims": claims, "build_index": True})
            claims = []
        if not value or re.search(r"暂无|待核对|不详|未公开|未知", value):
            continue
        quote = match[0].lstrip()
        claims.append({"field": field, "value": value, "document_id": document_id,
                       "quote": quote, "source_type": request.source_type,
                       "start_index": match.start() + len(match[0]) - len(quote)})
    if any(c["field"] == "name" for c in claims):
        records.append({"claims": claims, "build_index": True})
    if not records:
        warnings.append("没有识别到“名称：……”格式的文物记录，请使用建档模板填写来源摘录。")
    return {"records": records, "warnings": warnings}
